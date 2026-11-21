"""Bound a matched-checkpoint speech.cpp run; retain all attempts, not only warm successes."""
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path

from run_eos_corpus import environment, gpu_identity, identity
from run_eos_research import quantiles, sha, wait_for_probe, write_json
from run_prefill_research import GPU


PLANNED_RUNS = 13
MAX_SAMPLES = 256 * 1920


def finite_number(value: object, *, positive: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def collect_results(output: Path, returncode: int) -> dict:
    """Retain every result and separate mechanical validity from timing eligibility."""
    rows, issues, seen_runs = [], [], set()
    try:
        lines = (output / "probe/results.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        lines = []
        issues.append(str(error))
    for position, line in enumerate(lines):
        errors = []
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("result must be a JSON object")
        except (ValueError, TypeError) as error:
            row = {"source_line": position + 1, "raw_line": line}
            errors.append(str(error))
        run = row.get("run")
        if (type(run) is int and 0 <= run < PLANNED_RUNS
                and type(row.get("case")) is int and row["case"] == 0):
            seen_runs.add(run)
        if (position >= PLANNED_RUNS or type(run) is not int or run != position
                or type(row.get("case")) is not int or row["case"] != 0):
            errors.append(f"expected ordered case 0 run {position}")
        if type(row.get("seed")) is not int or row["seed"] != 64:
            errors.append("expected explicit seed 64")
        for key, expected in (("natural_eos", True), ("invalid_pcm", False), ("capacity_exceeded", False)):
            if row.get(key) is not expected:
                errors.append(f"{key} must be {expected}")
        samples, frames = row.get("samples"), row.get("frames")
        if type(samples) is not int or not 0 < samples <= MAX_SAMPLES:
            errors.append("samples must be a positive bounded integer")
        if type(frames) is not int or not 0 < frames <= 256 or samples != frames * 1920:
            errors.append("frames and sample count must match the codec hop")
        if row.get("sample_rate") != 24000:
            errors.append("sample_rate must be 24000")
        if (not finite_number(row.get("first_pcm_ms"), positive=True)
                or not finite_number(row.get("total_ms"), positive=True)
                or (finite_number(row.get("first_pcm_ms")) and finite_number(row.get("total_ms"))
                    and row["total_ms"] < row["first_pcm_ms"])):
            errors.append("first PCM and total timings must be finite, positive and ordered")
        phases = row.get("phases")
        if (not isinstance(phases, dict) or "prefill_ms" not in phases
                or any(not finite_number(value) for value in phases.values())):
            errors.append("phase timings must be finite nonnegative numbers including prefill_ms")
        for key, identity_key in (("raw_f32", "raw_identity"), ("wav", "wav_identity")):
            try:
                artifact = Path(row[key])
                row[identity_key] = identity(artifact)
                if key == "raw_f32":
                    body = artifact.read_bytes()
                    if (not body or len(body) % 4 or type(samples) is not int or len(body) != samples * 4
                            or not all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", body))):
                        errors.append("invalid/nonfinite/empty PCM payload")
            except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
                errors.append(f"{key}: {error}")
        row.update(coverage="unassessed", mechanical_valid=not errors, validation_issues=errors)
        rows.append(row)
        issues.extend(f"line {position + 1}: {error}" for error in errors)
    if len(rows) != PLANNED_RUNS:
        issues.append(f"expected {PLANNED_RUNS} ordered rows; observed {len(rows)}")
    warm = [row for row in rows if row["mechanical_valid"] and row["run"] > 0]
    return {"returncode": returncode, "observed": len(rows), "planned": PLANNED_RUNS,
            "not_run": sorted(set(range(PLANNED_RUNS)) - seen_runs), "issues": issues, "rows": rows,
            "mechanical_valid": returncode == 0 and not issues,
            "valid_rows": sum(row["mechanical_valid"] for row in rows),
            "warm_first_pcm_ms": quantiles([row["first_pcm_ms"] for row in warm]),
            "warm_prefill_ms": quantiles([row["phases"]["prefill_ms"] for row in warm]),
            "natural_eos": sum(row.get("natural_eos") is True for row in rows), "listening": "not_run"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "checkout", "build", "probe", "model", "text", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--language", choices=("ru", "en"), default="ru")
    parser.add_argument("--dependency-directory", type=Path, action="append", default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(args.repo.resolve() / ".tmp" / "base-acceptance")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True, timeout=30).strip()
    if head != "fd747c11e50543f59f2bbe233a9c7228dc479a71":
        raise ValueError("comparator is not the frozen source")
    ggml = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout / "ggml", text=True, timeout=30).strip()
    if ggml != "79b50d239d0b88852814c873db0d85d43ed66d9b":
        raise ValueError("comparator GGML is not the frozen source")
    if sha(args.model) != "c1d5352f57bca5b8035f011305dc7bd4a1cb18ca3f93660e9e7a9a4120713fa8":
        raise ValueError("comparator model differs from tensor-parity receipt")
    gpu = gpu_identity(GPU)
    output.mkdir(parents=True, exist_ok=False)
    command = [str(args.probe.resolve()), str(args.model.resolve()), "gpu", str(output / "probe"),
               "ryan", args.language, "64", str(PLANNED_RUNS), str(args.text.resolve())]
    env = environment(GPU, [args.probe.parent, *args.dependency_directory], False)
    write_json(output / "plan.json", {"command": command, "gpu": gpu, "source_head": head, "ggml": ggml,
               "model": identity(args.model), "probe": identity(args.probe), "text": identity(args.text),
               "cmake_cache": identity(args.build / "CMakeCache.txt"),
               "source_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=args.checkout, text=True, timeout=30),
               "runtime": [identity(p) for d in args.dependency_directory for pattern in ("cublas*.dll", "cudart*.dll") for p in d.glob(pattern)],
               "sampling_scope": "model-file defaults retained per result; seed64 is not matched random draws across engines"})
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err, cwd=args.probe.parent)
        try:
            returncode = wait_for_probe(process, 180, output / "watchdog.json")
        except subprocess.TimeoutExpired:
            returncode = 124
    collected = collect_results(output, returncode)
    write_json(output / "collected.json", collected)
    return int(not collected["mechanical_valid"])


if __name__ == "__main__":
    raise SystemExit(main())
