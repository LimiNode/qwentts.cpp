"""Bounded, buffered checkpoint mechanics; not a live streaming recovery policy."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import time
from pathlib import Path

from run_eos_corpus import PREPARED_HASHES, environment, gpu_identity, identity, source_identity
from run_eos_research import ORIGINAL, make_wav, sha, wait_for_probe, write_json

GPU = "GPU-40361931-6cb5-ac58-a059-5ba3e70986fb"
KNOWN = {2: 3, 14: 3, 72: 6, 75: 2, 78: 3}
SEEDS = (*KNOWN, 42, 64)
MODELS = {
    "talker": ("models/ru-stress/qwen-talker-1.7b-ru-stress-Q8_0.gguf",
               "f4672741608d145fae169075cb0ec775a5382cbe9183eccb59c7ab9cc4797427"),
    "codec": ("models/common/qwen-tokenizer-12hz-Q8_0.gguf",
              "1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94"),
}
REFERENCE = Path(".tmp/base-acceptance/eos-corpus-decision-20261121/ru-corpus-001")
MAX_TOKENS = 256
RESERVE_BYTES = 512 * 1024**2


def schedule(smoke: bool = False) -> list[dict]:
    """Discovery mechanics only; do not call these held-out recovery successes."""
    controls = [{"seed": seed, "repeat": repeat} for repeat in range(2) for seed in SEEDS]
    replay = [{"seed": seed, "recovery": {"policy": "replay", "snapshot_frame": frame}}
              for frame in (0, 1, 2) for seed in SEEDS]
    recovery = [{"seed": seed, "recovery": {"policy": policy, "trigger_max_frames": 8,
                 "continuation_seed": 65 if seed == 64 else 64}}
                for policy in ("eos-resample", "rollback-1", "rollback-2") for seed in SEEDS]
    if smoke:
        return [{"name": name, "cases": [case for case in cases if case["seed"] in KNOWN]}
                for name, cases in (("controls", controls), ("replay", replay))]
    return [{"name": name, "cases": cases} for name, cases in
            (("controls", controls), ("replay", replay), ("recovery", recovery))]


def disk_bound(stages: list[dict], max_tokens: int = MAX_TOKENS, log_bytes: int = 32 * 1024**2) -> int:
    attempts = sum((2 if "recovery" in case else 1) for stage in stages for case in stage["cases"])
    # Every attempt: raw f32 + PCM16 WAV, full cap, with code/metadata/log allowance.
    return RESERVE_BYTES + attempts * max_tokens * 1920 * 6 + log_bytes


def make_manifest(repo: Path, inputs: Path, cases: list[dict], max_tokens: int = MAX_TOKENS) -> dict:
    return {**{key: str((repo / value[0]).resolve()) for key, value in MODELS.items()},
        "speaker_f32": str(inputs / "speaker.f32"), "reference_i32": str(inputs / "reference.i32"),
        "reference_text_file": str(inputs / "reference.txt"), "reference_frames": 226,
        "max_batch": 1, "max_tokens": max_tokens, "deadline_ms": 60000,
        "stream_max_chunk_frames": 8, "observe_decisions": True,
        "groups": [{"text_file": str(inputs / "original.txt"), "language": "russian",
                    "seeds": [case["seed"]],
                    **({"recovery": case["recovery"]} if "recovery" in case else {})}
                   for case in cases]}


def validate_pcm(path: Path, frames: int) -> None:
    if not 0 <= frames <= MAX_TOKENS or path.stat().st_size != frames * 1920 * 4:
        raise ValueError(f"invalid PCM frame count/size: {path.name}")
    values = struct.iter_unpack("<f", path.read_bytes())
    if any(not math.isfinite(value[0]) for value in values):
        raise ValueError(f"nonfinite PCM: {path.name}")


def validate_codes(path: Path, frames: int) -> None:
    if path.stat().st_size != frames * 16 * 4:
        raise ValueError(f"invalid complete-code size: {path.name}")
    if any(not 0 <= code[0] < 2048 for code in struct.iter_unpack("<i", path.read_bytes())):
        raise ValueError(f"invalid complete code: {path.name}")


def replay_passed(rows: list[dict], planned: int) -> bool:
    return len(rows) == planned and all(row.get("mechanical_valid") and row.get("research_recovery", {}).get("triggered")
        and row["research_recovery"].get("replay_codes_equal") is True
        and row["research_recovery"].get("replay_pcm_equal") is True for row in rows)


def validate_recovery(row: dict, case: dict, expected_frames: dict[int, int]) -> None:
    recovery, config = row["research_recovery"], case["recovery"]
    original_frames = recovery["original_frames"]
    if recovery["original_status"] != 0 or recovery["original_finish_reason"] != 1:
        raise ValueError("original did not complete with natural EOS")
    if case["seed"] in expected_frames and original_frames != expected_frames[case["seed"]]:
        raise ValueError("original frame count differs from retained control")
    threshold = config.get("trigger_max_frames", 8)
    if recovery["trigger_max_frames"] != threshold:
        raise ValueError("effective recovery trigger differs")
    replay = config["policy"] == "replay"
    triggered = replay or original_frames <= threshold
    if recovery["triggered"] is not triggered or recovery["attempt_count"] != (2 if triggered else 1):
        raise ValueError("effective intervention trigger/attempt count differs")
    if triggered:
        rewind = {"eos-resample": 0, "rollback-1": 1, "rollback-2": 2}
        target = config["snapshot_frame"] if replay else original_frames - rewind[config["policy"]]
        seed = case["seed"] if replay else config["continuation_seed"]
        if recovery["restore_frame"] != target or recovery["continuation_seed"] != seed:
            raise ValueError("effective restore boundary or continuation seed differs")
    elif recovery["restore_frame"] != -1:
        raise ValueError("non-intervened control claims restoration")


def collect(directory: Path, cases: list[dict], expected_frames: dict[int, int] | None = None) -> tuple[list[dict], list[str]]:
    rows, issues = [], []
    results = directory / "probe/results.jsonl"
    if not results.is_file():
        return [], ["no flushed results"]
    for index, line in enumerate(results.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
            case = cases[index]
            if row["group"] != index or row["lane"] != 0 or row["seed"] != case["seed"]:
                raise ValueError("result/case identity mismatch")
            if row["status"] != 0 or row["invalid_pcm"] or row["capacity_exceeded"] or row["decision_capture_invalid"]:
                raise ValueError("probe reports invalid/failed outcome")
            if "recovery" not in case and case["seed"] in KNOWN and row["frames"] != KNOWN[case["seed"]]:
                raise ValueError("known early-EOS frame count did not reproduce")
            stem = directory / "probe" / f"group-{index}-lane-0"
            raw = stem.with_suffix(".f32")
            validate_pcm(raw, row["frames"])
            row["raw"] = identity(raw)
            row["wav"] = make_wav(raw)
            recovery = row.get("research_recovery")
            if "recovery" in case:
                if not recovery or row["execution_mode"] != "buffered-uncommitted" or row["chunks"]:
                    raise ValueError("recovery falsely labeled streaming or missing metadata")
                if recovery["committed_samples"] != 0 or recovery["prefill_count"] != 1 or recovery["streaming_qualified"]:
                    raise ValueError("unsupported recovery/commit contract")
                if recovery["policy"] != case["recovery"]["policy"]:
                    raise ValueError("effective recovery policy differs")
                validate_recovery(row, case, expected_frames if expected_frames is not None else KNOWN)
                original = Path(str(stem) + "-original.f32")
                original_codes = Path(str(stem) + "-original.i32")
                final_codes = Path(str(stem) + "-codes.i32")
                validate_pcm(original, recovery["original_frames"])
                validate_codes(original_codes, recovery["original_frames"])
                validate_codes(final_codes, row["frames"])
                row["original_raw"] = identity(original)
                row["original_codes"] = identity(original_codes)
                row["final_codes"] = identity(final_codes)
                row["original_wav"] = make_wav(original)
                if case["recovery"]["policy"] == "replay":
                    actual_codes = original_codes.read_bytes() == final_codes.read_bytes()
                    actual_pcm = original.read_bytes() == raw.read_bytes()
                    if recovery["replay_codes_equal"] != actual_codes or recovery["replay_pcm_equal"] != actual_pcm:
                        raise ValueError("probe replay equality disagrees with independent artifact comparison")
            elif recovery is not None or row["execution_mode"] != "streaming":
                raise ValueError("streaming baseline has unexpected recovery state")
            row.update({"mechanical_valid": True, "case": case,
                        "word_coverage": "unassessed", "human_listening": "not_run"})
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            issues.append(f"group{index}: {error}")
            rows.append({"group": index, "mechanical_valid": False, "error": str(error)})
    return rows, issues


def controls_match(rows: list[dict], old: list[dict]) -> bool:
    for row in rows:
        if not row.get("mechanical_valid"):
            return False
        expected = next(item for item in old if item["text_label"] == "original" and item["seed"] == row["seed"])
        if (row["frames"], row["finish_reason"], row["raw"]["sha256"]) != (
                expected["frames"], expected["finish_reason"], expected["raw"]["sha256"]):
            return False
    return bool(rows)


def launch_owned(command, runtime, env, out, err, timeout, watchdog):
    """Kill/reap the owned process even when watchdog diagnostics cannot be written."""
    process = subprocess.Popen(command, cwd=runtime, env=env, stdout=out, stderr=err)
    try:
        return wait_for_probe(process, timeout, watchdog)
    finally:
        try:
            if process.poll() is None:
                process.kill()
        finally:
            process.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Only five known short-EOS controls and exact replays, cap8frames; no rescue/timing claim")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error("seconds must be 1..1800")
    repo, output = args.repo.resolve(), args.output.resolve()
    output.relative_to(repo / ".tmp/base-acceptance")
    if output.exists():
        raise ValueError("new exclusive output required")
    probe, runtime = args.probe.resolve(), args.runtime.resolve()
    if probe.parent != runtime and list(probe.parent.glob("*.dll")):
        raise ValueError("isolated probe directory must not contain local DLL overrides")
    stages = schedule(args.smoke)
    max_tokens, log_bytes = (8, 8 * 1024**2) if args.smoke else (MAX_TOKENS, 32 * 1024**2)
    free = shutil.disk_usage(repo).free
    required = disk_bound(stages, max_tokens, log_bytes)
    if args.execute and free < required:
        raise ValueError(f"disk preflight: need {required} bytes including512MiB reserve; have {free}")
    before = {"probe": identity(probe), "runtime": [identity(p) for p in sorted(runtime.glob("*.dll"))],
              "models": [identity(repo / value[0]) for value in MODELS.values()]}
    if [item["sha256"] for item in before["models"]] != [item[1] for item in MODELS.values()]:
        raise ValueError("original RU-stress/codec identities differ")
    retained = repo / REFERENCE
    for name, expected in PREPARED_HASHES.items():
        if sha(retained / "inputs" / name) != expected:
            raise ValueError(f"original conditioning differs: {name}")
    if (retained / "inputs/original.txt").read_bytes() != ORIGINAL.encode("utf-8"):
        raise ValueError("original spoken bytes differ")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in (*PREPARED_HASHES, "original.txt"):
        shutil.copyfile(retained / "inputs" / name, inputs / name)
    for stage in stages:
        write_json(output / (stage["name"] + "-manifest.json"), make_manifest(repo, inputs, stage["cases"], max_tokens))
    env = environment(GPU, [runtime], False)
    env = {key: value for key, value in env.items() if not key.startswith("QWEN_PREFILL_")}
    write_json(output / "environment.json", {k: v for k, v in env.items() if k.startswith(("QWEN_", "GGML_", "CUDA_"))})
    write_json(output / "plan.json", {"schedule": stages, "assets_before": before,
        "source": source_identity(), "script": identity(Path(__file__)),
        "inputs": [identity(p) for p in sorted(inputs.iterdir())],
        "gpu": gpu_identity(GPU) if args.execute else "not_queried_plan_only",
        "seconds": args.seconds, "disk_required_bytes": required, "disk_free_bytes": free,
        "smoke_only": args.smoke, "max_tokens": max_tokens,
        "prespecified_gate": "exact retained streaming controls, then all planned exact replay pairs, then interventions if planned",
        "scope": "buffered uncommitted mechanics; no fork/streaming latency or semantic recovery qualification"})
    if not args.execute:
        return 0
    old = json.loads((retained / "all-observations.json").read_text(encoding="utf-8"))
    expected_frames = {item["seed"]: item["frames"] for item in old if item["text_label"] == "original"}
    results = {stage["name"]: {"state": "not_run", "rows": []} for stage in stages}
    deadline = time.monotonic() + args.seconds
    interrupted = False
    try:
        for stage in stages:
            name, cases = stage["name"], stage["cases"]
            if time.monotonic() >= deadline:
                break
            directory = output / name
            directory.mkdir()
            print(f"START {name}: {len(cases)} groups", flush=True)
            with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
                try:
                    code = launch_owned([str(probe), str(output / (name + "-manifest.json")),
                        str(directory / "probe")], runtime, env, out, err,
                        min(600, max(0.001, deadline - time.monotonic())), directory / "watchdog.json")
                except subprocess.TimeoutExpired:
                    code = None
                except KeyboardInterrupt:
                    interrupted, code = True, None
            rows, issues = collect(directory, cases, expected_frames)
            passed = code == 0 and len(rows) == len(cases) and not issues
            if name == "controls":
                matched = controls_match(rows, old)
                if not matched:
                    issues.append("retained streaming control mismatch; replay/intervention interpretation blocked")
                passed = passed and matched
            elif name == "replay":
                matched = replay_passed(rows, len(cases))
                if not matched:
                    issues.append("unchanged replay did not pass all complete-code/PCM comparisons; interventions blocked")
                passed = passed and matched
            results[name] = {"state": "mechanical_pass" if passed else "failed", "returncode": code,
                             "rows": rows, "issues": issues, "missing_groups": len(cases) - len(rows)}
            write_json(output / (name + "-collected.json"), results[name])
            print(f"DONE {name}: {results[name]['state']}, {len(rows)}/{len(cases)}", flush=True)
            if not passed or interrupted:
                break
    finally:
        after = {"probe": identity(probe), "runtime": [identity(p) for p in sorted(runtime.glob("*.dll"))],
                 "models": [identity(repo / value[0]) for value in MODELS.values()]}
        write_json(output / "results.json", {"stages": results, "assets_after": after,
            "assets_unchanged": before == after, "interrupted": interrupted,
            "word_coverage": "unassessed", "human_listening": "not_run", "release": "blocked"})
    return int(interrupted or before != after or any(item["state"] != "mechanical_pass" for item in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
