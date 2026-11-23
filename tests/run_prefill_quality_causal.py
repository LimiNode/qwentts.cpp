"""Bounded fresh-process and trace-neutral CustomVoice controls; not acceptance."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

import run_prefill_short as short
from run_eos_corpus import gpu_identity, identity, source_identity
from run_eos_research import wait_for_probe, write_json
from run_prefill_research import GPU, collect

REFERENCE = Path(".tmp/base-acceptance/prefill-short-20261121/quality-002")
PROBE_SHA = "7e9ff122761e48b435272609980ae23692ed84f9794d7caa7feddc7ac0558443"
CUDA_SHA = "bd54280b060357d21d0345cd1ecbc3f58e220b169561cd44f4c3a42255661b30"
SAMPLER_PROBE_SHA = "91368b96d53972332a82d4ce98afa10db3116e373fe0ac121b6f866fa59f6cba"
MODEL_PATHS = (
    "models/prefill-customvoice-native-0c0e3051/qwen-talker-1.7b-customvoice-Q8_0.gguf",
    "models/prefill-customvoice-native-0c0e3051/qwen-tokenizer-12hz-F16-comparator.gguf",
)
COLD = (("russian", "ru-no", 64), ("russian", "ru-no", 75),
        ("russian", "ru-thanks", 65), ("russian", "ru-ready", 64),
        ("english", "en-no", 64), ("english", "en-hi", 64), ("english", "en-hi", 72))
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CODE_VECTOR = r"\d+(?:,\d+){15}"
EXPECTED_TRACE = (
    re.compile(r"\[ResearchFrame\] serial=\d+ slot=0 frame=\d+ codes=" + CODE_VECTOR),
    re.compile(r"\[ARTrace\] predictor step=\d+ codes=" + CODE_VECTOR),
)
SAMPLE = re.compile(r"\[Sample\] step=(\d+) c0=(\d+) u=(" + NUMBER + r") subseq=(\d+)")


def snapshot(runtime: Path, models: list[Path]) -> dict:
    """Hash every runtime file in place, never freeze/copy CUDA dependencies."""
    return {"runtime": [identity(p) for p in sorted(runtime.rglob("*")) if p.is_file()],
            "models": [identity(p) for p in models]}


def validate_reference(repo: Path) -> tuple[dict, dict, dict, dict]:
    reference = (repo / REFERENCE).resolve()
    plan = json.loads((reference / "plan.json").read_text(encoding="utf-8"))
    results = json.loads((reference / "results.json").read_text(encoding="utf-8"))
    runtime = reference / "runtime"
    models = [(repo / path).resolve() for path in MODEL_PATHS]
    before = snapshot(runtime, models)
    expected_runtime = {str((runtime / Path(item["path"]).name).resolve()) for item in plan["runtime"]}
    if (len(expected_runtime) != len(plan["runtime"])
            or {item["path"] for item in plan["runtime"]} != expected_runtime
            or {item["path"] for item in before["runtime"]} != expected_runtime
            or [item["path"] for item in plan["model_inputs"]] != [str(p) for p in models]
            or tuple(item["sha256"] for item in before["models"]) != short.CUSTOM_HASHES):
        raise ValueError("reference runtime/model paths or identities differ from the frozen layout")
    by_name = {Path(item["path"]).name.lower(): item for item in before["runtime"]}
    if (by_name["eos-research-probe.exe"]["sha256"] != PROBE_SHA
            or by_name["ggml-cuda.dll"]["sha256"] != CUDA_SHA
            or plan["source_probe"]["sha256"] != PROBE_SHA):
        raise ValueError("reference probe/CUDA identity differs from the pinned causal experiment")
    for item in plan["inputs"]:
        if Path(item["path"]).resolve() != (reference / "inputs" / Path(item["path"]).name).resolve():
            raise ValueError("reference input path escapes the retained corpus")
    for name, result in results.items():
        for index, row in enumerate(result["rows"]):
            for key, suffix in (("raw", ".f32"), ("wav", ".wav"), ("logits", "-prefill-logits.f32")):
                expected = reference / name / "probe" / f"group-{index}-lane-0{suffix}"
                if Path(row[key]["path"]).resolve() != expected.resolve():
                    raise ValueError("reference row artifact path differs from its group")
    receipt = short.validate_reference(reference, before["runtime"], before["models"])
    return plan, results, before, receipt


def schedule() -> list[dict]:
    jobs = []
    for language, label, seed in COLD:
        for repeat in range(2):
            jobs.append({"name": f"cold-{language}-{label}-s{seed}-r{repeat}", "phase": "cold",
                         "language": language, "variant": "baseline", "repeat": repeat,
                         "cases": [{"text_label": label, "seed": seed, "kind": "short"}]})
    for language in ("russian", "english"):
        for variant in ("baseline", "prefix-mmq"):
            jobs.append({"name": f"schedule-{language}-{variant}", "phase": "schedule",
                         "language": language, "variant": variant,
                         "cases": short.cases("quality", language)})
    for variant in ("prefix-default", "forced-mmq-no-prefix"):
        jobs.append({"name": f"schedule-english-{variant}", "phase": "schedule", "language": "english",
                     "variant": variant, "cases": short.cases("quality", "english")})
    return jobs


def sampler_schedule() -> list[dict]:
    jobs = []
    for language in ("russian", "english"):
        primer = "original" if language == "russian" else "en-primer"
        cases = ([dict(text_label=primer, seed=64, kind="primer_start")]
                 + [dict(text_label=label, seed=seed, kind="short", repeat=repeat)
                    for lang, label, seed in COLD if lang == language for repeat in range(2)]
                 + [dict(text_label=primer, seed=64, kind="primer_end")])
        for control in ("default", "talker-top1", "predictor-greedy"):
            jobs.append(dict(name=f"sampler-{language}-{control}", phase="sampler", language=language,
                             variant="baseline", sampler_control=control, cases=cases))
    return jobs


def sampler_issues(rows: list[dict], planned: list[dict], control: str) -> list[str]:
    issues = []
    for index, (row, case) in enumerate(zip(rows, planned)):
        effective = control if case["kind"] == "short" else "default"
        values = row.get("effective_sampler", {})
        expected = dict(do_sample=True, temperature=0.9, top_k=1 if effective == "talker-top1" else 50,
                        top_p=1.0, repetition_penalty=1.05, subtalker_do_sample=effective != "predictor-greedy",
                        subtalker_temperature=0.0 if effective == "predictor-greedy" else 0.9,
                        subtalker_top_k=50, subtalker_top_p=1.0, min_new_tokens=2, eos_guard_enabled=False)
        if row.get("sampler_control") != effective or any(
                (values.get(key) is not value if isinstance(value, bool) else
                 type(values.get(key)) not in (int, float) or not math.isfinite(values[key])
                 or abs(values[key] - value) > 1e-5)
                for key, value in expected.items()):
            issues.append(f"group {index}: applied sampler control/defaults differ from the plan")
    return issues


def configured_environment(variant: str, runtime: Path) -> dict[str, str]:
    env = short.configured_environment("quality", "baseline" if variant == "forced-mmq-no-prefix" else variant,
                                       [runtime])
    if variant == "forced-mmq-no-prefix":
        env["GGML_RESEARCH_FORCE_MMQ_Q8"] = "1"
    env["QWEN_TTS_FULL_AR_TRACE"] = "1"
    return env


def evidence_issues(rows: list[dict], variant: str, stderr: str) -> list[str]:
    """Permit only the prescribed bounded trace; keep all unexpected diagnostics."""
    retained, removed, samples = [], [], {}
    serial = None
    for line in stderr.splitlines():
        if job := re.search(r"\[ResearchJob\] serial=(\d+) seed=\d+", line):
            serial = int(job[1])
            samples.setdefault(serial, [])
        sample_marker = line.find("[Sample]")
        sample = SAMPLE.fullmatch(line[sample_marker:]) if sample_marker >= 0 else None
        if (sample and serial is not None and 0 <= int(sample[1]) < 128
                and int(sample[4]) == int(sample[1]) * 16 and 0 <= float(sample[3]) < 1):
            samples[serial].append((int(sample[1]), int(sample[2])))
            continue
        match = re.search(r"\[(?:ARTrace|ResearchFrame)\]", line)
        body = line[match.start():] if match else ""
        if match and any(pattern.fullmatch(body) for pattern in EXPECTED_TRACE):
            removed.append(body)
        else:
            retained.append(line)
    filtered = "\n".join(retained)
    if variant == "forced-mmq-no-prefix":
        # The no-cache control intentionally forces dispatch, not prefix telemetry.
        issues = [issue for issue in short.evidence_issues(rows, "quality", "baseline", filtered)
                  if issue != "default control unexpectedly used a forced matmul route"]
        issues.extend(short.dispatch_issues([line for line in retained if "RESEARCH_MATMUL" in line],
                                            "forced-mmq", True))
    else:
        issues = short.evidence_issues(rows, "quality", variant, filtered)
    expected_frames = sum(min(128, row.get("frames", 0)) for row in rows)
    for tag in ("[ResearchFrame] ", "[ARTrace] predictor "):
        if sum(line.startswith(tag) for line in removed) != expected_frames:
            issues.append(f"missing or extra bounded full16 frame trace: {tag}")
    for index, row in enumerate(rows):
        observed = samples.get(index, [])
        expected = min(128, row["frames"] + int(row.get("finish_reason") == 1))
        codes = row.get("codes", [])
        if (len(observed) != expected or any(step != position for position, (step, _) in enumerate(observed))
                or any(c0 != codes[step][0] for step, c0 in observed if step < len(codes))):
            issues.append(f"group {index}: missing, unordered or inconsistent bounded Sample trace")
    if set(samples) != set(range(len(rows))):
        issues.append("Sample trace job set differs from retained rows")
    return issues


def first_code_difference(left: dict, right: dict) -> dict:
    a, b = left.get("codes"), right.get("codes")
    if a is None or b is None:
        return {"state": "unavailable", "first_difference": None}
    for frame, (av, bv) in enumerate(zip(a, b)):
        for codebook, (x, y) in enumerate(zip(av, bv)):
            if x != y:
                return {"state": "different", "first_difference": {"frame": frame, "codebook": codebook,
                                                                     "candidate": x, "baseline": y}}
    return {"state": "equal_bounded_prefix" if len(a) == len(b) else "different_trace_lengths",
            "first_difference": None, "common_frames": min(len(a), len(b)),
            "candidate_trace_frames": len(a), "baseline_trace_frames": len(b),
            "scope": "at most first 128 frames; not evidence about untraced frames"}


def compare_rows(candidate: list[dict], baseline: list[dict], planned: int) -> dict:
    comparison = short.compare(candidate, baseline)
    for result, row, control in zip(comparison["rows"], candidate, baseline):
        result["full16_trace"] = first_code_difference(row, control)
    comparison["complete"] = len(candidate) == len(baseline) == planned
    comparison["all_raw_frames_terminal_equal"] = (comparison["complete"]
        and all(row["raw_equal"] is True and row["frames_equal"] and row["terminal_equal"]
                for row in comparison["rows"]))
    return comparison


def execute(job: dict, output: Path, source: dict, inputs: Path, runtime: Path, deadline: float,
            probe: Path | None = None) -> dict:
    planned = job["cases"]
    result = {"state": "not_run", "rows": [], "issues": [], "planned": len(planned),
              "not_run_groups": list(range(len(planned))), "returncode": None,
              "human_listening": "not_run", "word_coverage": "unassessed", "voice_quality": "unassessed"}
    if time.monotonic() >= deadline:
        result["issues"] = ["overall deadline"]
        return result
    directory = output / job["name"]
    directory.mkdir()
    manifest = short.manifest_for(source, inputs, planned, job["language"], "quality")
    if "sampler_control" in job:
        for group, case in zip(manifest["groups"], planned):
            group["sampler_control"] = job["sampler_control"] if case["kind"] == "short" else "default"
    write_json(directory / "manifest.json", manifest)
    env = configured_environment(job["variant"], runtime)
    temporary = directory / "temporary"
    temporary.mkdir()
    env.update(TMP=str(temporary), TEMP=str(temporary))
    write_json(directory / "environment.json", {k: v for k, v in env.items()
               if k.startswith(("QWEN_", "GGML_", "CUDA_"))})
    result["state"] = "completed"
    print(f"START {job['name']} ({len(planned)} requests)", flush=True)
    with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result["state"], result["issues"] = "not_run", ["overall deadline before launch"]
            else:
                process = subprocess.Popen([str(probe or runtime / "eos-research-probe.exe"),
                    str(directory / "manifest.json"), str(directory / "probe")],
                    cwd=runtime, env=env, stdout=out, stderr=err)
                try:
                    result["returncode"] = wait_for_probe(process, min(180, remaining), directory / "watchdog.json")
                except subprocess.TimeoutExpired:
                    result["state"] = "timeout"
                except KeyboardInterrupt:
                    result["state"] = "interrupted"
                    process.kill()
                    try:
                        result["returncode"] = process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        result["issues"].append("owned probe termination not observed within 15 seconds")
        except OSError as error:
            result["state"] = "launch_failed"
            result["issues"].append(str(error))
    try:
        rows, issues = collect(directory, planned, True, "custom_voice")
        result["rows"] = rows
        result["issues"].extend(issues)
        for row in rows:
            if "wav" in row:
                row["wav"].update(identity(directory / "probe" / row["wav"]["wav"]))
        stderr = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
        result["issues"].extend(evidence_issues(rows, job["variant"], stderr))
        if "sampler_control" in job:
            result["issues"].extend(sampler_issues(rows, planned, job["sampler_control"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["issues"].append(f"malformed/partial retained evidence: {error}")
    result["not_run_groups"] = list(range(len(result["rows"]), len(planned)))
    result["outcome_issues"] = short.outcome_issues(result["rows"])
    if result["state"] == "completed":
        if result["returncode"] != 0 or result["not_run_groups"] or result["issues"]:
            result["state"] = "incomplete"
        elif result["outcome_issues"]:
            result["state"] = "completed_with_failures"
    for key in ("manifest", "environment", "stderr"):
        result[key] = identity(directory / (key + (".log" if key == "stderr" else ".json")))
    write_json(directory / "collected.json", result)
    print(f"DONE {job['name']}: {result['state']}, {len(result['rows'])}/{len(planned)}", flush=True)
    return result


def comparisons(jobs: list[dict], results: dict, reference: dict) -> dict:
    output = {"cold_repeats": {}, "cold_vs_retained_warm": {}, "trace_neutral": {}, "prefix_vs_baseline": {},
              "isolated_english_vs_baseline": {}, "sampler_vs_baseline": {}}
    for job in jobs:
        name, language, rows = job["name"], job["language"], results[job["name"]]["rows"]
        if job["phase"] == "sampler":
            if job["sampler_control"] == "default":
                expected = [next(row for row in reference[f"r0-{language}-baseline"]["rows"]
                                 if (row["text_label"], row["seed"]) == (case["text_label"], case["seed"]))
                            for case in job["cases"]]
                output["trace_neutral"][name] = compare_rows(rows, expected, len(job["cases"]))
            else:
                output["sampler_vs_baseline"][name] = compare_rows(
                    rows, results[f"sampler-{language}-default"]["rows"], len(job["cases"]))
        elif job["phase"] == "cold":
            expected = [row for row in reference[f"r0-{language}-baseline"]["rows"]
                        if (row["text_label"], row["seed"]) == (job["cases"][0]["text_label"], job["cases"][0]["seed"])]
            output["cold_vs_retained_warm"][name] = compare_rows(rows, expected, 1)
            if job["repeat"] == 1:
                output["cold_repeats"][name] = compare_rows(rows, results[name[:-1] + "0"]["rows"], 1)
        else:
            if job["variant"] != "forced-mmq-no-prefix":
                output["trace_neutral"][name] = compare_rows(rows, reference[f"r0-{language}-{job['variant']}"]["rows"], 18)
            if job["variant"] == "prefix-mmq":
                output["prefix_vs_baseline"][language] = compare_rows(rows, results[f"schedule-{language}-baseline"]["rows"], 18)
            elif job["variant"] in ("prefix-default", "forced-mmq-no-prefix"):
                output["isolated_english_vs_baseline"][job["variant"]] = compare_rows(
                    rows, results["schedule-english-baseline"]["rows"], 18)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--sampler-probe", type=Path,
                        help="Pinned diagnostic probe, isolated from DLLs; use only no-prefix sampler controls")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error("seconds must be 1..1800")
    repo, output = args.repo.resolve(), args.output.resolve()
    output.relative_to(repo / ".tmp/base-acceptance")
    if output.exists():
        raise ValueError("output already exists; preserve the original run")
    plan, reference, before, receipt = validate_reference(repo)
    runtime, inputs = repo / REFERENCE / "runtime", repo / REFERENCE / "inputs"
    models = [Path(item["path"]) for item in before["models"]]
    source = {key: item["path"] for key, item in zip(("talker", "codec"), before["models"])}
    sampler_probe = args.sampler_probe.resolve() if args.sampler_probe else None
    sampler_before = identity(sampler_probe) if sampler_probe else None
    if sampler_probe and (sampler_before["sha256"] != SAMPLER_PROBE_SHA
                          or list(sampler_probe.parent.glob("*.dll"))):
        raise ValueError("sampler probe must match its pinned build and have no local DLL overrides")
    jobs, gpu = sampler_schedule() if sampler_probe else schedule(), gpu_identity(GPU)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", {"source": source_identity(), "gpu": gpu, "reference": receipt,
        "assets_before": before, "sampler_probe": sampler_before,
        "schedule": jobs, "seconds": args.seconds, "per_process_seconds": 180,
        "planned_requests": sum(len(job["cases"]) for job in jobs), "planned_processes": len(jobs),
        "inputs": plan["inputs"], "scripts": [identity(Path(__file__)), identity(Path(short.__file__))],
        "scope": "causal diagnostics, not latency evidence or a speech-quality acceptance claim",
        "trace_scope": "FULL_AR_TRACE=1; bounded complete16 codes for first128 frames; no tensor dumps"})
    results = {job["name"]: {"state": "not_run", "rows": [], "planned": len(job["cases"]),
                              "not_run_groups": list(range(len(job["cases"]))), "issues": ["not started"]}
               for job in jobs}
    deadline, interrupted = time.monotonic() + args.seconds, False
    after, integrity_issues = None, []
    try:
        for index, job in enumerate(jobs):
            results[job["name"]] = execute(job, output, source, inputs, runtime, deadline, sampler_probe)
            write_json(output / f"progress-{index:02d}.json", results)
            if results[job["name"]]["state"] == "interrupted":
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        try:
            after = snapshot(runtime, models)
            if after != before:
                integrity_issues.append("runtime/model file identity or file set changed during the experiment")
            if sampler_probe and identity(sampler_probe) != sampler_before:
                integrity_issues.append("sampler probe changed during the experiment")
        except OSError as error:
            integrity_issues.append(f"post-run asset hashing failed: {error}")
        paired = comparisons(jobs, results, reference)
        neutral = all(row["all_raw_frames_terminal_equal"] for row in paired["trace_neutral"].values())
        summary = {"results": results, "comparisons": paired, "assets_after": after,
                   "integrity_issues": integrity_issues, "trace_neutral_control_passed": neutral,
                   "interrupted": interrupted, "word_coverage": "unassessed", "voice_quality": "unassessed",
                   "human_listening": "not_run", "latency_eligible": False}
        write_json(output / "results.json", summary)
    return int(bool(interrupted or integrity_issues or not neutral
                    or any(result["state"] != "completed" for result in results.values())))


if __name__ == "__main__":
    raise SystemExit(main())
