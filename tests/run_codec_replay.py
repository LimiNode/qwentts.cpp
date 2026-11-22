"""Teacher-forced codec restoration on retained discovery histories; no AR recovery."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

from run_eos_corpus import PREPARED_HASHES, environment, gpu_identity, identity, source_identity
from run_eos_recovery import GPU, MODELS, REFERENCE, launch_owned, validate_codes, validate_pcm
from run_eos_research import make_wav, write_json

PRIOR = Path(".tmp/base-acceptance/eos-recovery-20261122/matrix-001")
SEEDS = (2, 14, 72, 75, 78, 64)


def transpose(payload: bytes) -> bytes:
    """Retained AR frames are [T,16]; the codec probe consumes [16,T]."""
    if not payload or len(payload) % 64:
        raise ValueError("expected complete16 frame-major input")
    values = struct.unpack(f"<{len(payload) // 4}i", payload)
    if any(not 0 <= value < 2048 for value in values):
        raise ValueError("code outside vocabulary")
    frames = len(payload) // 64
    return struct.pack(f"<{len(values)}i", *(values[t * 16 + k] for k in range(16) for t in range(frames)))


def inspect(directory: Path, streaming: dict) -> dict:
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (result["streaming_qualified"] or result["live_ar_recovery"] or result["committed_samples"] != 0
            or not result["checks_passed"] or not result["cases"]):
        raise ValueError("unsupported or failed codec replay")
    for case in result["cases"]:
        checks = ("restored_state_exact", "replay_final_state_exact", "reconstructed_boundary_state_exact",
                  "reconstruct_final_state_exact", "replay_pcm_exact", "reconstruct_pcm_exact",
                  "retained_pcm_prefix_exact", "changed_vs_fresh_pcm_exact", "changed_vs_fresh_state_exact",
                  "candidate_fresh_prefix_exact", "checks_passed")
        if any(case[key] is not True for key in checks):
            raise ValueError("codec case state/PCM check failed")
        root = directory / case["directory"]
        for name in ("original", "replay", "reconstruct", "changed", "candidate_fresh"):
            path = root / (name.replace("_", "-") + ".f32")
            frames = result["candidate_frames"] if name in ("changed", "candidate_fresh") else result["original_frames"]
            validate_pcm(path, frames)
            case[name]["raw"] = identity(path)
            case[name]["wav"] = make_wav(path)
        if len({case[name]["raw"]["sha256"] for name in ("original", "replay", "reconstruct")}) != 1:
            raise ValueError("actual unchanged PCM differs")
        if case["changed"]["raw"]["sha256"] != case["candidate_fresh"]["raw"]["sha256"]:
            raise ValueError("restored candidate differs from independent fresh decode")
        prefix_bytes = case["boundary_frames"] * 1920 * 4
        if (root / "original.f32").read_bytes()[:prefix_bytes] != (root / "candidate-fresh.f32").read_bytes()[:prefix_bytes]:
            raise ValueError("independent fresh candidate changed retained prefix")
        if case["schedule"] == "production-ramp":
            case["original_matches_retained_streaming"] = case["original"]["raw"]["sha256"] == streaming["sha256"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "probe", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900:
        parser.error("seconds must be1..900 (other900 reserved for AR comparison)")
    repo, output = args.repo.resolve(), args.output.resolve()
    output.relative_to(repo / ".tmp/base-acceptance")
    if output.exists():
        raise ValueError("new exclusive output directory required")
    probe, runtime = args.probe.resolve(), args.runtime.resolve()
    if probe.parent != runtime and list(probe.parent.glob("*.dll")):
        raise ValueError("local DLL overrides not allowed")
    required = 512 * 1024**2 + 128 * 1024**2
    free = shutil.disk_usage(repo).free
    if args.execute and free < required:
        raise ValueError("codec replay disk preflight failed")
    model = repo / MODELS["codec"][0]
    reference = repo / REFERENCE / "inputs/reference.i32"
    prior = json.loads((repo / PRIOR / "results.json").read_text(encoding="utf-8"))
    source_rows = {row["seed"]: row for row in prior["stages"]["recovery"]["rows"]
                   if row["research_recovery"]["policy"] == "rollback-1"}
    controls = {row["seed"]: row for row in prior["stages"]["controls"]["rows"]}
    before = {"probe": identity(probe), "model": identity(model), "reference": identity(reference),
              "runtime": [identity(path) for path in sorted(runtime.glob("*.dll"))]}
    if before["model"]["sha256"] != MODELS["codec"][1] or before["reference"]["sha256"] != PREPARED_HASHES["reference.i32"]:
        raise ValueError("codec/reference identity differs")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    planned = []
    for seed in SEEDS:
        row = source_rows[seed]
        item = {"seed": seed, "inputs": {}}
        for name, field in (("original", "original_codes"), ("candidate", "final_codes")):
            original = Path(row[field]["path"])
            if identity(original) != row[field]:
                raise ValueError("retained AR code identity differs")
            path = inputs / f"seed-{seed}-{name}.i32"
            with path.open("xb") as stream:
                stream.write(transpose(original.read_bytes()))
            validate_codes(path, original.stat().st_size // 64)
            item["inputs"][name] = {"source_frame_major": row[field], "codebook_major": identity(path)}
        planned.append(item)
    env = {key: value for key, value in environment(GPU, [runtime], False).items()
           if not key.startswith("QWEN_PREFILL_")}
    write_json(output / "plan.json", {"cases": planned, "assets_before": before,
        "source": source_identity(), "runner": identity(Path(__file__)),
        "gpu": gpu_identity(GPU) if args.execute else "not_queried_plan_only",
        "seconds": args.seconds, "disk_required_bytes": required, "disk_free_bytes": free,
        "environment": {k: v for k, v in env.items() if k.startswith(("CUDA_", "GGML_", "QWEN_"))},
        "scope": "teacher-forced private codec state; no live AR recovery/timing/listening qualification"})
    if not args.execute:
        return 0
    results = [{"seed": seed, "state": "not_run"} for seed in SEEDS]
    started = time.monotonic()
    try:
        for index, case in enumerate(planned):
            remaining = args.seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            seed = case["seed"]
            directory = output / f"seed-{seed}"
            command = [str(probe), str(model), str(reference),
                case["inputs"]["original"]["codebook_major"]["path"],
                case["inputs"]["candidate"]["codebook_major"]["path"], str(directory)]
            write_json(output / f"seed-{seed}-invocation.json", command)
            print(f"START codec seed{seed}", flush=True)
            try:
                with (output / f"seed-{seed}-stdout.log").open("xb") as out, (output / f"seed-{seed}-stderr.log").open("xb") as err:
                    code = launch_owned(command, runtime, env, out, err, min(600, remaining), output / f"seed-{seed}-watchdog.json")
                if code != 0:
                    raise ValueError(f"codec probe exited{code}")
                result = inspect(directory, controls[seed]["raw"])
                results[index] = {"seed": seed, "state": "mechanical_pass", "result": result}
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                results[index] = {"seed": seed, "state": "failed", "error": str(error)}
                break
            print(f"DONE codec seed{seed}", flush=True)
    finally:
        after = {"probe": identity(probe), "model": identity(model), "reference": identity(reference),
                 "runtime": [identity(path) for path in sorted(runtime.glob("*.dll"))]}
        write_json(output / "results.json", {"cases": results, "assets_after": after,
            "assets_unchanged": before == after, "elapsed_seconds": time.monotonic() - started,
            "human_listening": "not_run", "release": "blocked"})
    return int(before != after or any(row["state"] != "mechanical_pass" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
