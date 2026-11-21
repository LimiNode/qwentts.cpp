"""Run a separate Nsight trace of three bounded prefill controls, not TTFA acceptance."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from run_eos_corpus import environment, gpu_identity, identity, source_identity
from run_eos_research import write_json
from run_prefill_research import GPU


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-directory", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root.parent.name == ".worktrees":
        root = root.parent.parent
    output = args.output.resolve()
    output.relative_to(root / ".tmp" / "base-acceptance")
    gpu = gpu_identity(GPU)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest.update(groups=manifest["groups"][:3], max_tokens=4, observe_decisions=False, observe_prefill=True)
    write_json(output / "manifest.json", manifest)
    env = environment(GPU, [args.probe.parent, *args.dependency_directory], False)
    env = {k: v for k, v in env.items() if not k.startswith("QWEN_PREFILL_")}
    env.update(QWEN_PREFILL_PROFILE="1", QWEN_PREFILL_NVTX="1")
    command = [str(args.nsys.resolve()), "profile", "--trace=cuda,nvtx", "--sample=none",
               "--cpuctxsw=none", "--cuda-graph-trace=graph", "--output=" + str(output / "trace"),
               str(args.probe.resolve()), str(output / "manifest.json"), str(output / "probe")]
    write_json(output / "plan.json", {"command": command, "gpu": gpu, "source": source_identity(),
               "nsys": identity(args.nsys), "probe": identity(args.probe),
               "source_manifest": identity(args.manifest), "bounded_frames": 4,
               "measurement_scope": "separate profiled prefill; max-token termination expected; not speech acceptance"})
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err, cwd=args.probe.parent)
        try:
            returncode = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            # Nsight owns an application child: terminate the exact owned tree,
            # unlike the ordinary childless probe watchdog.
            terminated = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                        capture_output=True, text=True, timeout=30)
            write_json(output / "watchdog.json", {"timeout": True, "nsys_pid": process.pid,
                       "tree_termination_returncode": terminated.returncode,
                       "stdout": terminated.stdout, "stderr": terminated.stderr})
            process.wait(timeout=15)
            returncode = 124
    write_json(output / "execution.json", {"returncode": returncode})
    if returncode:
        return returncode
    export = subprocess.run([str(args.nsys.resolve()), "export", "--type=sqlite",
                             "--output=" + str(output / "trace.sqlite"), str(output / "trace.nsys-rep")],
                            capture_output=True, text=True, timeout=90, env=env)
    write_json(output / "export.json", {"returncode": export.returncode, "stdout": export.stdout, "stderr": export.stderr})
    return export.returncode


if __name__ == "__main__":
    raise SystemExit(main())
