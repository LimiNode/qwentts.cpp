"""Bounded CMP short-utterance quality controls and separate unobserved timing.

Audio duration is not evidence of correct words, absence of cough, or release quality.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from run_eos_corpus import environment, gpu_identity, identity, source_identity
from run_eos_research import quantiles, wait_for_probe, write_json
from run_prefill_differential import dispatch_issues
from run_prefill_research import GPU, INPUT_HASHES, collect, compare

VARIANTS = ("baseline", "prefix-default", "prefix-mmq")
SEEDS = (64, 65, 72, 75)
SHORT = {
    "russian": {"ru-yes": "Да!", "ru-no": "Нет.", "ru-thanks": "Спасибо.", "ru-ready": "Готово."},
    "english": {"en-yes": "Yes.", "en-no": "No.", "en-hi": "Hi!", "en-thanks": "Thank you."},
}
EN_PRIMER = "The engineer inspected the equipment and handed the next shift a detailed report about the completed work."
CUSTOM_HASHES = ("cab2cff67a0a557310febe558dc83076b28ed790e491867eb2751759f4cd89fa",
                 "9a0bf7e0d16f04232a852bcd41fd2e485ea546acda1dbf3ce5e59dee76ae8c24")


def cases(phase: str, language: str, warm: int = 8) -> list[dict]:
    if phase == "timing":
        return ([dict(text_label="original", seed=64, kind="first_use")]
                + [dict(text_label="original", seed=64, kind="warm_pair") for _ in range(warm)])
    primer = "original" if language == "russian" else "en-primer"
    return ([dict(text_label=primer, seed=64, kind="primer_start")]
            + [dict(text_label=label, seed=seed, kind="short") for label in SHORT[language] for seed in SEEDS]
            + [dict(text_label=primer, seed=64, kind="primer_end")])


def blocks(phase: str) -> list[tuple[int, str, str]]:
    if phase == "quality":
        return [(0, language, variant) for language in SHORT for variant in VARIANTS]
    return [(repeat, "russian", variant) for repeat, order in enumerate((VARIANTS, VARIANTS[::-1]))
            for variant in order]


def configured_environment(phase: str, variant: str, dependencies: list[Path]) -> dict[str, str]:
    env = environment(GPU, dependencies, False)
    env = {k: v for k, v in env.items() if not k.startswith("QWEN_PREFILL_")}
    env.update(QWEN_PREFILL_PROFILE="1" if phase == "quality" else "0", GGML_CUDA_DISABLE_FUSION="1")
    if variant != "baseline":
        env.update(QWEN_PREFILL_PREFIX_ROWS="9", QWEN_PREFILL_PREFIX_SOURCE="full")
    if variant == "prefix-mmq":
        env["GGML_RESEARCH_FORCE_MMQ_Q8"] = "1"
    if phase == "quality":
        env["GGML_RESEARCH_MATMUL_TRACE"] = "1"
    return env


def manifest_for(source: dict, inputs: Path, planned: list[dict], language: str, phase: str) -> dict:
    return {"talker": source["talker"], "codec": source["codec"], "speaker": "ryan",
            "max_batch": 1, "max_tokens": 256, "deadline_ms": 60000, "stream_max_chunk_frames": 8,
            "observe_decisions": False, "observe_prefill": phase == "quality",
            "groups": [{"text_file": str(inputs / (row["text_label"] + ".txt")),
                        "language": language, "seeds": [row["seed"]]} for row in planned]}


def freeze_runtime(probe: Path, dependencies: list[Path], target: Path) -> tuple[Path, list[dict]]:
    target.mkdir(exist_ok=False)
    payloads = [probe, *sorted(probe.parent.glob("*.dll"))]
    payloads += [p for folder in dependencies for pattern in ("cublas*.dll", "cudart*.dll")
                 for p in sorted(folder.glob(pattern))]
    seen = {}
    for path in payloads:
        item = identity(path)
        key = path.name.lower()
        if key in seen:
            if seen[key] != item["sha256"]:
                raise ValueError(f"conflicting runtime basename: {path.name}")
            continue
        seen[key] = item["sha256"]
        shutil.copy2(path, target / path.name)
        if identity(target / path.name)["sha256"] != item["sha256"]:
            raise ValueError("runtime changed while freezing")
    return target / probe.name, [identity(p) for p in sorted(target.iterdir())]


def runtime_signature(items: list[dict]) -> dict:
    return {Path(p["path"]).name.lower(): (p["bytes"], p["sha256"]) for p in items}


def outcome_valid(row: dict) -> bool:
    return (row.get("status") == 0 and row.get("finish_reason") == 1
            and not any(row.get(k, True) for k in ("invalid_pcm", "capacity_exceeded"))
            and not row.get("artifact_error") and row.get("samples", 0) > 0
            and "raw" in row and bool(row.get("chunks")))


def outcome_issues(rows: list[dict]) -> list[str]:
    return [f"group {index}: unsuccessful or non-natural terminal/audio; status={row.get('status')}, "
            f"finish_reason={row.get('finish_reason')}, samples={row.get('samples')}"
            for index, row in enumerate(rows) if not outcome_valid(row)]


def evidence_issues(rows: list[dict], phase: str, variant: str, stderr: str) -> list[str]:
    issues = []
    prefix = variant != "baseline"
    forbidden = ("ResearchPrefillDiagnostic", "ResearchDecision", "ResearchFrame", "ARTrace", "[FrameSeed]")
    if any(token in stderr for token in forbidden):
        issues.append("unexpected tensor/decision/frame diagnostic log")
    jobs = [line for line in stderr.splitlines() if "[ResearchJob]" in line]
    parsed_jobs = [re.search(r"\[ResearchJob\] serial=(\d+) seed=(\d+)", line) for line in jobs]
    if jobs and (len(jobs) != len(rows) or any(match is None for match in parsed_jobs)
                 or [(int(match[1]), int(match[2])) for match in parsed_jobs if match]
                 != [(index, row["seed"]) for index, row in enumerate(rows)]):
        issues.append("operational job serial/seed log does not match retained rows")
    samples = [line for line in stderr.splitlines() if "[Sample]" in line]
    parsed_samples = [re.search(r"\[Sample\] step=(\d+) c0=\d+ u=[\d.]+ subseq=(\d+)", line) for line in samples]
    if (len(samples) > 2 * len(rows) or any(match is None or int(match[1]) > 1 or int(match[2]) >= 32
                                           for match in parsed_samples)):
        issues.append("sampler log exceeds the frozen runtime's ordinary first-two-step logging")
    routes = [line for line in stderr.splitlines() if "RESEARCH_MATMUL" in line]
    prefix_logs = [json.loads(line.split("[ResearchPrefix] ", 1)[1]) for line in stderr.splitlines()
                   if "[ResearchPrefix] " in line]
    if phase == "timing":
        if routes or "[ResearchPrefill]" in stderr:
            issues.append("timing contaminated by profiler or route logs")
        if prefix and len(prefix_logs) != len(rows):
            issues.append("missing effective prefix-hit telemetry")
    else:
        issues.extend(dispatch_issues(routes, "forced-mmq" if variant == "prefix-mmq" else "default", True))
        if variant != "prefix-mmq" and any("route=forced-" in line for line in routes):
            issues.append("default control unexpectedly used a forced matmul route")
    for index, row in enumerate(rows):
        if row.get("decision_observation_enabled") is not False or row.get("decisions"):
            issues.append(f"group {index}: decision observer unexpectedly enabled")
        if row.get("prefill_observation_enabled") != (phase == "quality"):
            issues.append(f"group {index}: prefill observer setting was ignored")
        observation = row.get("prefill_observation")
        if phase == "quality":
            if not observation:
                issues.append(f"group {index}: missing prefill observation")
                continue
            if (any(observation.get(k) is not False for k in ("diagnostic", "kv_f16", "last_only", "retain"))
                    or observation.get("alignment") != 256 or observation.get("nonfinite_logits") != 0
                    or observation.get("prefill_use_flash_attn") is not True):
                issues.append(f"group {index}: unexpected effective prefill settings")
        else:
            if observation is not None:
                issues.append(f"group {index}: unexpected timing observation")
            observation = prefix_logs[index] if prefix and index < len(prefix_logs) else None
        if observation and (observation.get("prefix_rows") != (9 if prefix else 0)
                            or observation.get("prefix_hit") != (prefix and index > 0)
                            or (prefix and observation.get("prefix_source") != "full")):
            issues.append(f"group {index}: prefix rows/hit/source not effective")
    return issues


def complete_comparison(candidate: dict, baseline: dict, planned: int) -> dict:
    result = compare(candidate["rows"], baseline["rows"])
    result.update(planned=planned, complete=(all(value["state"] in ("completed", "completed_with_failures")
                                               for value in (candidate, baseline))
                  and candidate.get("returncode") == baseline.get("returncode") == 0
                  and not candidate.get("issues") and not baseline.get("issues")
                  and result["paired"] == planned and result["same_observed_denominator"]),
                  outcomes_successful=bool(candidate["rows"] and baseline["rows"])
                  and all(outcome_valid(row) for value in (candidate, baseline) for row in value["rows"]))
    return result


def timing_summary(result: dict, planned: int) -> dict:
    eligible = (result["state"] == "completed" and result.get("returncode") == 0 and not result["issues"]
                and len(result["rows"]) == planned and all(outcome_valid(r) for r in result["rows"]))
    warm = [r for r in result["rows"] if r["kind"] == "warm_pair"] if eligible else []
    values = {"first_pcm_ms": [r["chunks"][0]["ms"] for r in warm],
              "prefill_ms": [r["prefill_ms"] for r in warm], "total_ms": [r["total_ms"] for r in warm]}
    if not all(math.isfinite(v) and v >= 0 for values_for_key in values.values() for v in values_for_key):
        eligible, warm, values = False, [], {k: [] for k in values}
    return {"eligible": eligible, "planned": planned, "observed": len(result["rows"]),
            "first_use": [r for r in result["rows"] if r["kind"] == "first_use"],
            "warm": {k: quantiles(v) for k, v in values.items()}}


def validate_reference(path: Path, runtime: list[dict], model_inputs: list[dict]) -> dict:
    plan = json.loads((path / "plan.json").read_text(encoding="utf-8"))
    results = json.loads((path / "results.json").read_text(encoding="utf-8"))
    expected = {f"r0-{language}-{variant}" for _, language, variant in blocks("quality")}
    if (plan.get("phase") != "quality" or set(results) != expected
            or any(r["state"] not in ("completed", "completed_with_failures") or r.get("returncode") != 0 or r["issues"]
                   or len(r["rows"]) != 18 for r in results.values())
            or runtime_signature(runtime) != runtime_signature(plan["runtime"])
            or runtime_signature(model_inputs) != runtime_signature(plan["model_inputs"])):
        raise ValueError("quality reference is incomplete or runtime/model/codec identity differs")
    def verify(item: dict) -> Path:
        actual = identity(Path(item["path"]))
        if any(actual[key] != item[key] for key in ("bytes", "sha256")):
            raise ValueError(f"quality reference identity changed: {item['path']}")
        return Path(item["path"])

    for key in ("source_manifest", "source_original"):
        verify(plan[key])
    for key in ("runtime", "inputs", "scripts"):
        if not plan[key]:
            raise ValueError(f"quality reference missing {key}")
        for item in plan[key]:
            verify(item)
    if runtime_signature(plan["scripts"]) != runtime_signature([identity(p) for p in script_paths()]):
        raise ValueError("quality reference scripts differ from this runner")
    source = {name: item["path"] for name, item in zip(("talker", "codec"), model_inputs)}
    expected_texts = {"original": Path(plan["source_original"]["path"]).read_text(encoding="utf-8"),
                      "en-primer": EN_PRIMER, **SHORT["russian"], **SHORT["english"]}
    if ({Path(item["path"]).name for item in plan["inputs"]} != {name + ".txt" for name in expected_texts}
            or any((path / "inputs" / (name + ".txt")).read_text(encoding="utf-8") != text
                   for name, text in expected_texts.items())):
        raise ValueError("quality reference input corpus differs")
    for _, language, variant in blocks("quality"):
        name = f"r0-{language}-{variant}"
        rows = results[name]["rows"]
        planned = cases("quality", language)
        result = results[name]
        failures = outcome_issues(rows)
        if (result.get("outcome_issues", []) != failures
                or result["state"] != ("completed_with_failures" if failures else "completed")):
            raise ValueError("quality reference outcome failures are missing or mislabeled")
        for key in ("manifest", "environment", "stderr"):
            if verify(result[key]).resolve() != (path / name / (key + (".log" if key == "stderr" else ".json"))).resolve():
                raise ValueError("quality reference block artifact path differs")
        manifest = json.loads((path / name / "manifest.json").read_text(encoding="utf-8"))
        env = json.loads((path / name / "environment.json").read_text(encoding="utf-8"))
        def controls(values: dict) -> dict:
            return {k: v for k, v in values.items()
                    if k.startswith(("QWEN_PREFILL_", "QWEN_RESEARCH_", "QWEN_TTS_", "GGML_")) or k == "CUDA_VISIBLE_DEVICES"}
        if (manifest != manifest_for(source, path / "inputs", planned, language, "quality")
                or controls(env) != controls(configured_environment("quality", variant, []))):
            raise ValueError("quality reference manifest/environment settings differ")
        for row in rows:
            for key in ("raw", "wav", "logits"):
                verify(row[key])
            if row["raw"]["bytes"] != row["samples"] * 4:
                raise ValueError("quality reference PCM byte count differs")
        if (any(any(row.get(key) != control[key] for key in ("text_label", "seed", "kind"))
                for row, control in zip(rows, planned))
                or evidence_issues(rows, "quality", variant,
                                   (path / name / "stderr.log").read_text(encoding="utf-8", errors="replace"))):
            raise ValueError("quality reference has invalid effective settings")
    return {"plan": identity(path / "plan.json"), "results": identity(path / "results.json"),
            "scope": "complete mechanical reference, not successful speech quality",
            "outcome_failures": sum(len(outcome_issues(r["rows"])) for r in results.values())}


def script_paths() -> list[Path]:
    return [Path(__file__), Path(__file__).with_name("test_prefill_short.py"),
            *[Path(__file__).with_name(name) for name in ("run_eos_corpus.py", "run_eos_research.py",
                                                         "run_prefill_research.py", "run_prefill_differential.py")]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "probe", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("quality", "timing"), required=True)
    parser.add_argument("--quality-reference", type=Path)
    parser.add_argument("--dependency-directory", type=Path, action="append", default=[])
    parser.add_argument("--warm", type=int, choices=(6, 8), default=8)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800 or (args.phase == "timing" and not args.quality_reference):
        parser.error("seconds must be 1..1800; timing requires --quality-reference")
    deadline = time.monotonic() + args.seconds
    args.repo, args.output, args.probe = args.repo.resolve(), args.output.resolve(), args.probe.resolve()
    args.output.relative_to(args.repo / ".tmp" / "base-acceptance")
    if args.output.exists():
        raise ValueError("output already exists; use a new run directory")
    if shutil.disk_usage(args.repo).free < 2 * 1024**3:
        raise ValueError("requires at least 2 GiB free")
    source = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    model_inputs = [identity(Path(source[k])) for k in ("talker", "codec")]
    if source.get("speaker") != "ryan" or tuple(p["sha256"] for p in model_inputs) != CUSTOM_HASHES:
        raise ValueError("requires retained CustomVoice Q8/comparator codec/ryan identities")
    original = next(Path(g["text_file"]) for g in source["groups"] if Path(g["text_file"]).stem == "original")
    original_text = original.read_text(encoding="utf-8")
    if (not original_text.strip() or "\x00" in original_text
            or identity(original)["sha256"] != INPUT_HASHES["original.txt"]):
        raise ValueError("invalid retained original text")
    gpu = gpu_identity(GPU)
    args.output.mkdir(parents=True, exist_ok=False)
    probe, runtime = freeze_runtime(args.probe, args.dependency_directory, args.output / "runtime")
    inputs = args.output / "inputs"
    inputs.mkdir()
    texts = {"original": original_text, "en-primer": EN_PRIMER, **SHORT["russian"], **SHORT["english"]}
    for label, text in texts.items():
        with (inputs / (label + ".txt")).open("xb") as stream:
            stream.write(text.encode("utf-8"))
    reference = validate_reference(args.quality_reference, runtime, model_inputs) if args.phase == "timing" else None
    scripts = script_paths()
    schedule = blocks(args.phase)
    write_json(args.output / "plan.json", {"phase": args.phase, "source": source_identity(), "gpu": gpu,
               "source_manifest": identity(args.manifest), "source_probe": identity(args.probe), "runtime": runtime,
               "model_inputs": model_inputs, "scripts": [identity(p) for p in scripts], "seconds": args.seconds,
               "inputs": [identity(p) for p in sorted(inputs.iterdir())], "source_original": identity(original),
               "schedule": schedule, "warm": args.warm, "quality_reference": reference,
               "planned_requests": sum(len(cases(args.phase, language, args.warm)) for _, language, _ in schedule),
               "quality_scope": "complete outputs retained; word coverage and cough/voice quality unassessed",
               "timing_caveat": "PROFILE=0; ordinary job/seed, first-two-step Sample, and prefix-hit logs remain in the frozen engine; no kernel-event claims"})
    results = {}
    for repeat, language, variant in schedule:
        name = f"r{repeat}-{language}-{variant}"
        planned = cases(args.phase, language, args.warm)
        if time.monotonic() >= deadline:
            results[name] = {"state": "not_run", "rows": [], "issues": ["overall deadline"], "planned": len(planned)}
            write_json(args.output / (name + "-not-run.json"), results[name])
            continue
        directory = args.output / name
        directory.mkdir()
        manifest = manifest_for(source, inputs, planned, language, args.phase)
        write_json(directory / "manifest.json", manifest)
        env = configured_environment(args.phase, variant, [probe.parent])
        local_temp = directory / "temporary"
        local_temp.mkdir()
        env.update(TMP=str(local_temp), TEMP=str(local_temp))
        write_json(directory / "environment.json", {k: v for k, v in env.items() if k.startswith(("QWEN_", "GGML_", "CUDA_"))})
        code, state = None, "completed"
        print(f"START {name} ({len(planned)} requests)", flush=True)
        with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
            process = subprocess.Popen([str(probe), str(directory / "manifest.json"), str(directory / "probe")],
                                       cwd=probe.parent, env=env, stdout=out, stderr=err)
            try:
                code = wait_for_probe(process, max(0.01, min(180, deadline - time.monotonic())), directory / "watchdog.json")
            except subprocess.TimeoutExpired:
                state = "timeout"
        rows, issues = collect(directory, planned, False, "custom_voice")
        for row in rows:
            if "wav" in row:
                row["wav"].update(identity(directory / "probe" / row["wav"]["wav"]))
        stderr = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
        try:
            issues.extend(evidence_issues(rows, args.phase, variant, stderr))
        except (ValueError, KeyError, TypeError) as error:
            issues.append(f"malformed effective evidence: {error}")
        if code != 0 or len(rows) != len(planned) or issues:
            state = "incomplete" if state == "completed" else state
        failures = outcome_issues(rows)
        if state == "completed" and failures:
            state = "completed_with_failures" if args.phase == "quality" else "incomplete"
        result = {"state": state, "returncode": code, "rows": rows, "issues": issues, "planned": len(planned),
                  "outcome_issues": failures,
                  "human_listening": "not_run", "word_coverage": "unassessed", "voice_quality": "unassessed",
                  "manifest": identity(directory / "manifest.json"), "environment": identity(directory / "environment.json"),
                  "stderr": identity(directory / "stderr.log")}
        if args.phase == "timing":
            result["timing"] = timing_summary(result, len(planned))
        results[name] = result
        write_json(directory / "collected.json", result)
        print(f"DONE {name}: {state}, {len(rows)}/{len(planned)}", flush=True)
    for repeat, language, variant in schedule:
        name, baseline = f"r{repeat}-{language}-{variant}", f"r{repeat}-{language}-baseline"
        results[name]["comparison"] = complete_comparison(results[name], results[baseline], results[name]["planned"])
    write_json(args.output / "results.json", results)
    return int(any(value["state"] != "completed" for value in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
