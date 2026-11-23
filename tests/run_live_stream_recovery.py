"""Bounded single-lane live checkpoint study; no production EOS detector."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from run_eos_corpus import PREPARED_HASHES, environment, gpu_identity, identity, source_identity
from run_eos_recovery import (
    GPU, KNOWN, MAX_TOKENS, MODELS, REFERENCE, RESERVE_BYTES, SEEDS,
    launch_owned, validate_codes, validate_pcm,
)
from run_eos_research import ORIGINAL, make_wav, sha, write_json

PRIOR = Path(".tmp/base-acceptance/eos-retry-codec-20261122/retry-001/results.json")
HOP = 1920
HOLDBACK_FRAMES = 16
NEGATIVE_MODES = ("cancel", "deadline", "cap", "sink-fail", "stale", "newer-admission")


def schedule() -> list[dict]:
    """Known discovery seeds only; no held-out or speech-success denominator."""
    controls = [{"seed": seed, "mode": "baseline", "repeat": repeat}
                for repeat in range(2) for seed in SEEDS]
    replay = [{"seed": seed, "mode": "replay", "snapshot_frame": frame}
              for frame in (1, 2) for seed in SEEDS]
    recovery = [{"seed": seed, "mode": "rollback1",
                 "continuation_seed": 65 if seed == 64 else 64} for seed in SEEDS]
    negative = [{"seed": 2 if mode == "newer-admission" else 64, "mode": mode}
                for mode in NEGATIVE_MODES]
    return [{"name": name, "cases": cases} for name, cases in
            (("controls", controls), ("replay", replay), ("negative", negative), ("recovery", recovery))]


def disk_bound(stages: list[dict]) -> int:
    # Final/fresh/original/committed f32, final/original PCM16 WAV and per-case code,
    # commit-prefix/log allowance. Originals are budgeted even for controls.
    count = sum(len(stage["cases"]) for stage in stages)
    return RESERVE_BYTES + count * (MAX_TOKENS * HOP * 20 + 1024**2) + 64 * 1024**2


def make_manifest(repo: Path, inputs: Path, cases: list[dict]) -> dict:
    return {**{key: str((repo / value[0]).resolve()) for key, value in MODELS.items()},
        "speaker_f32": str(inputs / "speaker.f32"),
        "reference_i32": str(inputs / "reference.i32"),
        "reference_text_file": str(inputs / "reference.txt"), "reference_frames": 226,
        "max_tokens": MAX_TOKENS, "deadline_ms": 60000,
        "groups": [{"text_file": str(inputs / "original.txt"), "language": "russian",
                    **{key: value for key, value in case.items() if key != "repeat"}}
                   for case in cases]}


def require_clean_source(source: dict) -> None:
    if source["status"] or any(line.lstrip().startswith(("+", "-", "U"))
                               for line in source["submodules"].splitlines()):
        raise ValueError("execute requires a clean source checkout and pinned submodules")


def asset_identity(repo: Path, probe: Path, runtime: Path) -> dict:
    return {"probe": identity(probe),
            "runtime": [identity(path) for path in sorted(runtime.glob("*.dll"))],
            "models": [identity(repo / value[0]) for value in MODELS.values()],
            "conditioning": [identity(repo / REFERENCE / "inputs" / name)
                             for name in (*PREPARED_HASHES, "original.txt")]}


def retained_controls(repo: Path) -> dict[int, dict]:
    """Verify retained bytes, not just receipt flags or historical durations."""
    prior = json.loads((repo / PRIOR).read_text(encoding="utf-8"))
    stages = prior["stages"]
    if any(stages[name]["state"] != "mechanical_pass" for name in ("controls", "replay")):
        raise ValueError("retained controls/replay gate did not pass")
    result = {}
    for seed in SEEDS:
        controls = [row for row in stages["controls"]["rows"] if row["seed"] == seed]
        replay = [row for row in stages["replay"]["rows"] if row["seed"] == seed]
        if not controls or not replay:
            raise ValueError("retained discovery seed is missing")
        raw, codes = controls[0]["raw"], replay[0]["original_codes"]
        if (any(row["raw"]["sha256"] != raw["sha256"] for row in controls)
                or any(row["original_codes"]["sha256"] != codes["sha256"] for row in replay)):
            raise ValueError("retained discovery repeats disagree")
        if any(identity(Path(artifact["path"])) != artifact for artifact in (raw, codes)):
            raise ValueError("retained discovery artifacts changed")
        result[seed] = {"frames": controls[0]["frames"], "raw": raw, "codes": codes}
    return result


def bounded_integer(row: dict, key: str, low: int, high: int) -> int:
    value = row[key]
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid bounded integer: {key}")
    return value


def check_restore(row: dict, case: dict, original: bytes, final: bytes, prefix: bytes) -> None:
    """The commit cursor is irreversible; only owned held suffix can change."""
    decoded = bounded_integer(row, "decoded_frames", 0, 8)
    pending = bounded_integer(row, "pending_frames", 0, 7)
    frame = bounded_integer(row, "restore_frame", 0, 8)
    committed = bounded_integer(row, "committed_samples_at_restore", 0, MAX_TOKENS * HOP)
    if frame != decoded + pending or row["codec_target"] not in (1, 2, 4, 8):
        raise ValueError("incoherent AR/codec checkpoint boundary")
    if committed > decoded * HOP or committed % HOP or len(prefix) != committed * 4:
        raise ValueError("restore crosses committed PCM or has invalid prefix length")
    if prefix != original[:len(prefix)] or prefix != final[:len(prefix)]:
        raise ValueError("actual committed prefix changed")
    if row["committed_prefix_equal"] is not True:
        raise ValueError("probe reports committed-prefix mismatch")
    if case["mode"] == "replay":
        expected_frame = case["snapshot_frame"]
        if frame != expected_frame or (decoded, pending, row["codec_target"]) != (1, expected_frame - 1, 2):
            raise ValueError("replay checkpoint did not preserve pending/ramp state")
        # Seed75 reaches EOS at frame2: terminal drain freezes commits, so
        # its retired-to-active replay deliberately has C=0, unlike othercp2.
        expected_commit = HOP if expected_frame == 2 and case["seed"] != 75 else 0
        if committed != expected_commit:
            raise ValueError("replay did not exercise its planned partial commit")
    elif frame != row["original_frames"] - 1:
        raise ValueError("rollback did not restore one complete AR frame")


def collect_negative(directory: Path, index: int, row: dict, case: dict, control: dict) -> dict:
    """Intentional cancellations are guard evidence, never recovery candidates."""
    if (row["status"] != -5 or row["checks_passed"] is not True or row["negative_guard_pass"] is not True
            or row["restore_rejected"] is not True or row["triggered"] is not False):
        raise ValueError("negative case did not reject restoration and terminate cancelled")
    if row["restore_frame"] != -1 or row["committed_samples_at_restore"] != 0:
        raise ValueError("negative case claims successful restoration")
    frames = bounded_integer(row, "frames", 0, MAX_TOKENS)
    original_frames = bounded_integer(row, "original_frames", 0, MAX_TOKENS)
    samples = bounded_integer(row, "samples", 0, MAX_TOKENS * HOP)
    cap = HOP if case["mode"] == "cap" else HOLDBACK_FRAMES * HOP
    if samples % HOP or row["holdback_cap_samples"] != cap:
        raise ValueError("negative PCM shape/cap differs")
    bounded_integer(row, "held_peak_samples", 0, cap)
    committed = bounded_integer(row, "committed_samples", 0, samples)
    if committed % HOP:
        raise ValueError("negative commit cursor is not frame aligned")
    if case["mode"] in ("cancel", "deadline", "cap", "sink-fail") and committed:
        raise ValueError("negative case unexpectedly committed PCM")
    if case["mode"] == "stale" and committed <= HOP:
        raise ValueError("stale-checkpoint case did not pass its decoded boundary")
    stem = directory / "probe" / f"group-{index}"
    raw, codes, original, original_codes, sink = (Path(str(stem) + suffix) for suffix in
        (".f32", "-codes.i32", "-original.f32", "-original.i32", "-committed.f32"))
    validate_pcm(raw, samples // HOP)
    validate_pcm(sink, committed // HOP)
    validate_codes(codes, frames)
    if original.stat().st_size > original_frames * HOP * 4 or original.stat().st_size % (HOP * 4):
        raise ValueError("invalid negative original PCM size")
    validate_pcm(original, original.stat().st_size // (HOP * 4))
    validate_codes(original_codes, original_frames)
    expected_pcm = Path(control["raw"]["path"]).read_bytes()
    expected_codes = Path(control["codes"]["path"]).read_bytes()
    actual_original = original.read_bytes()
    actual_codes = original_codes.read_bytes()
    if sink.read_bytes() != actual_original[:committed * 4]:
        raise ValueError("negative sink bytes differ from observed committed prefix")
    if raw.read_bytes() != actual_original or row["committed_prefix_equal"] is not True:
        raise ValueError("negative guard changed already-observed PCM")
    if (actual_original != expected_pcm[:len(actual_original)]
            or actual_codes != expected_codes[:len(actual_codes)]):
        raise ValueError("negative case changed the original control prefix")
    if case["mode"] == "newer-admission":
        if frames or row["original_status"] != 0 or row["original_finish_reason"] != 1:
            raise ValueError("newer-admission case did not retire the original then cancel its replacement")
        if actual_original != expected_pcm or actual_codes != expected_codes:
            raise ValueError("newer-admission original differs from full control")
    elif codes.read_bytes() != actual_codes:
        raise ValueError("negative restoration changed final complete codes")
    row.update({"raw": identity(raw), "codes": identity(codes), "original_raw": identity(original),
                "original_codes": identity(original_codes), "wav": make_wav(raw),
                "sink_pcm": identity(sink),
                "mechanical_valid": True, "case": case, "word_coverage": "not_assessed_guard_case",
                "human_listening": "not_run"})
    return row


def collect(directory: Path, cases: list[dict], controls: dict[int, dict]) -> tuple[list[dict], list[str]]:
    rows, issues = [], []
    results = directory / "probe/results.jsonl"
    if not results.is_file():
        return [], ["no flushed results"]
    for index, line in enumerate(results.read_text(encoding="utf-8").splitlines()):
        try:
            row, case = json.loads(line), cases[index]
            if (row["group"], row["seed"], row["mode"]) != (index, case["seed"], case["mode"]):
                raise ValueError("result/case identity mismatch")
            if case["mode"] in NEGATIVE_MODES:
                rows.append(collect_negative(directory, index, row, case, controls[case["seed"]]))
                continue
            if row["status"] != 0 or row["finish_reason"] != 1 or row["checks_passed"] is not True:
                raise ValueError("probe reports failed/non-natural outcome")
            frames = bounded_integer(row, "frames", 1, MAX_TOKENS)
            bounded_integer(row, "held_peak_samples", 0, HOLDBACK_FRAMES * HOP)
            if (row["samples"] != frames * HOP or row["committed_samples"] != frames * HOP
                    or row["prefill_count"] != 1 or row["physical_playback"] is not False):
                raise ValueError("ordinary group did not commit complete PCM under one prefill")
            stem = directory / "probe" / f"group-{index}"
            raw, codes, fresh, sink = (Path(str(stem) + suffix)
                for suffix in (".f32", "-codes.i32", "-fresh-codec.f32", "-committed.f32"))
            validate_pcm(raw, frames)
            validate_pcm(fresh, frames)
            validate_pcm(sink, frames)
            validate_codes(codes, frames)
            final, final_codes = raw.read_bytes(), codes.read_bytes()
            if final != sink.read_bytes():
                raise ValueError("actual final sink bytes differ from full generated PCM")
            if final != fresh.read_bytes() or row["fresh_codec_pcm_equal"] is not True:
                raise ValueError("live PCM differs from independent full streaming-codec reconstruction")
            control = controls[case["seed"]]
            original = Path(control["raw"]["path"]).read_bytes()
            original_codes = Path(control["codes"]["path"]).read_bytes()
            if case["mode"] == "rollback1":
                original_raw, original_code = (Path(str(stem) + suffix)
                                                for suffix in ("-original.f32", "-original.i32"))
                original_frames = bounded_integer(row, "original_frames", 1, MAX_TOKENS)
                validate_pcm(original_raw, original_frames)
                validate_codes(original_code, original_frames)
                if original_raw.read_bytes() != original or original_code.read_bytes() != original_codes:
                    raise ValueError("recovery changed original control bytes")
                if row["original_status"] != 0 or row["original_finish_reason"] != 1:
                    raise ValueError("recovery original did not reach natural EOS")
                expected_trigger = case["seed"] in KNOWN
                if row["triggered"] is not expected_trigger:
                    raise ValueError("discovery trigger differs from frozen known-seed scope")
                if expected_trigger and row["continuation_seed"] != case["continuation_seed"]:
                    raise ValueError("continuation seed differs from frozen policy")
                row["original_raw"] = identity(original_raw)
                row["original_codes"] = identity(original_code)
                row["original_wav"] = make_wav(original_raw)
            else:
                expected_trigger = case["mode"] == "replay"
                if row["triggered"] is not expected_trigger:
                    raise ValueError("replay/baseline trigger mismatch")
            if case["mode"] != "rollback1" or not expected_trigger:
                if final != original or final_codes != original_codes or frames != control["frames"]:
                    raise ValueError("unchanged trajectory differs in full codes/PCM")
            if expected_trigger:
                prefix_path = Path(str(stem) + "-committed-prefix.f32")
                check_restore(row, case, original, final, prefix_path.read_bytes())
                code_prefix = row["restore_frame"] * 16 * 4
                if final_codes[:code_prefix] != original_codes[:code_prefix]:
                    raise ValueError("restoration changed complete-code prefix")
                row["committed_prefix"] = identity(prefix_path)
                if case["mode"] == "replay":
                    before, after = (Path(str(stem) + suffix) for suffix in
                        ("-replay-step-original.f32", "-replay-step-restored.f32"))
                    step_frames = 1 if case["snapshot_frame"] == 1 else min(control["frames"], 3)
                    validate_pcm(before, step_frames)
                    validate_pcm(after, step_frames)
                    if (before.read_bytes() != after.read_bytes()
                            or before.read_bytes() != original[:step_frames * HOP * 4]
                            or row["unchanged_step_pcm_equal"] is not True):
                        raise ValueError("actual one-step unchanged streaming replay differs")
                    row["replay_step_original"] = identity(before)
                    row["replay_step_restored"] = identity(after)
            elif row["restore_frame"] != -1 or row["committed_samples_at_restore"] != 0:
                raise ValueError("non-restored control claims a restore boundary")
            row.update({"raw": identity(raw), "codes": identity(codes), "fresh_codec": identity(fresh),
                        "sink_pcm": identity(sink),
                        "wav": make_wav(raw), "mechanical_valid": True, "case": case,
                        "word_coverage": "unassessed", "human_listening": "not_run"})
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            issues.append(f"group{index}: {error}")
            rows.append({"group": index, "mechanical_valid": False, "error": str(error)})
    return rows, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "probe", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error("seconds must be 1..1800")
    repo, output = args.repo.resolve(), args.output.resolve()
    output.relative_to(repo / ".tmp/base-acceptance")
    if output.exists():
        raise ValueError("new exclusive output directory required")
    probe, runtime = args.probe.resolve(), args.runtime.resolve()
    if probe.parent != runtime and list(probe.parent.glob("*.dll")):
        raise ValueError("local DLL overrides not allowed")
    stages = schedule()
    required, free = disk_bound(stages), shutil.disk_usage(repo).free
    if args.execute and free < required:
        raise ValueError(f"disk preflight needs {required} bytes including512MiB reserve; have {free}")
    source = source_identity()
    if args.execute:
        require_clean_source(source)
    before = asset_identity(repo, probe, runtime)
    if [item["sha256"] for item in before["models"]] != [value[1] for value in MODELS.values()]:
        raise ValueError("original model identities differ")
    retained = repo / REFERENCE / "inputs"
    if any(sha(retained / name) != expected for name, expected in PREPARED_HASHES.items()):
        raise ValueError("original conditioning differs")
    if (retained / "original.txt").read_bytes() != ORIGINAL.encode("utf-8"):
        raise ValueError("original spoken bytes differ")
    controls = retained_controls(repo)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in (*PREPARED_HASHES, "original.txt"):
        shutil.copyfile(retained / name, inputs / name)
    for stage in stages:
        write_json(output / (stage["name"] + "-manifest.json"), make_manifest(repo, inputs, stage["cases"]))
    env = {key: value for key, value in environment(GPU, [runtime], False).items()
           if not key.startswith("QWEN_PREFILL_")}
    write_json(output / "plan.json", {"schedule": stages, "assets_before": before,
        "source": source, "runner": identity(Path(__file__)),
        "inputs": [identity(path) for path in sorted(inputs.iterdir())],
        "retained_receipt": identity(repo / PRIOR), "retained_controls": controls,
        "gpu": gpu_identity(GPU) if args.execute else "not_queried_plan_only",
        "seconds": args.seconds, "disk_required_bytes": required, "disk_free_bytes": free,
        "environment": {key: value for key, value in env.items() if key.startswith(("QWEN_", "GGML_", "CUDA_"))},
        "prespecified_gate": "exact historical controls,14unchanged live replays,6negative guards, then interventions",
        "scope": "single-lane live AR/codec holdback mechanics; no detector/timing/listening/release qualification"})
    if not args.execute:
        return 0
    results = {stage["name"]: {"state": "not_run", "rows": []} for stage in stages}
    started = time.monotonic()
    deadline = started + args.seconds
    interrupted = False
    try:
        for stage in stages:
            name, cases = stage["name"], stage["cases"]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            directory = output / name
            directory.mkdir()
            print(f"START {name}: {len(cases)} groups", flush=True)
            launch_error = None
            with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
                try:
                    code = launch_owned([str(probe), str(output / (name + "-manifest.json")),
                        str(directory / "probe")], runtime, env, out, err,
                        min(600, remaining), directory / "watchdog.json")
                except (OSError, subprocess.SubprocessError) as error:
                    code, launch_error = None, str(error)
                except KeyboardInterrupt:
                    interrupted, code = True, None
            rows, issues = collect(directory, cases, controls)
            if launch_error is not None:
                issues.append("owned probe launch/wait failed: " + launch_error)
            passed = code == 0 and len(rows) == len(cases) and not issues
            results[name] = {"state": "mechanical_pass" if passed else "failed", "returncode": code,
                             "rows": rows, "issues": issues, "missing_groups": max(0, len(cases) - len(rows))}
            write_json(output / (name + "-collected.json"), results[name])
            print(f"DONE {name}: {results[name]['state']} {len(rows)}/{len(cases)}", flush=True)
            if not passed or interrupted:
                break
    finally:
        try:
            after = asset_identity(repo, probe, runtime)
            source_after = source_identity()
        except (OSError, subprocess.SubprocessError) as error:
            after, source_after = {"observation_error": str(error)}, None
        write_json(output / "results.json", {"stages": results, "assets_after": after,
            "source_after": source_after, "assets_unchanged": before == after,
            "source_unchanged": source == source_after,
            "elapsed_seconds": time.monotonic() - started, "interrupted": interrupted,
            "word_coverage": "unassessed", "human_listening": "not_run", "release": "blocked"})
    return int(interrupted or before != after or source != source_after
               or any(row["state"] != "mechanical_pass" for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
