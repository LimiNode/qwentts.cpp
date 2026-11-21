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
               "ryan", args.language, "64", "13", str(args.text.resolve())]
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
    rows, issues = [], []
    results = output / "probe/results.jsonl"
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                raw = Path(row["raw_f32"])
                body = raw.read_bytes()
                if len(body) != row["samples"] * 4 or not all(math.isfinite(v) for v in struct.unpack(f"<{len(body)//4}f", body)):
                    raise ValueError("invalid PCM payload")
                row.update(raw_identity=identity(raw), wav_identity=identity(Path(row["wav"])), coverage="unassessed")
                rows.append(row)
            except (OSError, ValueError, KeyError, struct.error) as error:
                issues.append(str(error))
                break
    warm = [r for r in rows if r["run"] > 0]
    write_json(output / "collected.json", {"returncode": returncode, "observed": len(rows), "planned": 13,
               "not_run": list(range(len(rows), 13)), "issues": issues, "rows": rows,
               "warm_first_pcm_ms": quantiles([r["first_pcm_ms"] for r in warm if r["first_pcm_ms"] is not None]),
               "warm_prefill_ms": quantiles([r["phases"]["prefill_ms"] for r in warm]),
               "natural_eos": sum(r["natural_eos"] for r in rows), "listening": "not_run"})
    return int(returncode != 0 or len(rows) != 13 or bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
