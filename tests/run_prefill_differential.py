"""Bounded prefill diagnostics; intentional four-frame outputs are not speech acceptance."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from run_eos_corpus import environment, gpu_identity, identity, source_identity
from run_eos_research import wait_for_probe, write_json
from run_prefill_research import GPU, collect, compare


def effective_issues(rows: list[dict], prefix: bool, diagnostic: bool,
                     manual: bool, prefix_rows: int) -> list[str]:
    """Reject ignored diagnostic options rather than mislabel a run."""
    issues = []
    for index, row in enumerate(rows):
        observation = row.get("prefill_observation")
        if not observation:
            issues.append(f"group {index}: missing effective prefill observation")
            continue
        if observation.get("diagnostic") != diagnostic:
            issues.append(f"group {index}: requested diagnostic setting was not effective")
        if observation.get("prefill_use_flash_attn") != (not manual):
            issues.append(f"group {index}: requested prefill attention mode was not effective")
        if prefix and (observation.get("prefix_source") != "full"
                       or observation.get("prefix_rows") != prefix_rows
                       or observation.get("prefix_hit") != (index > 0)):
            issues.append(f"group {index}: requested full-source cache mode was not effective")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Retained same-model prefix-study manifest with original and ru-yes groups")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-rows", type=int, required=True)
    parser.add_argument("--layer", default="0")
    parser.add_argument("--control", choices=("default", "no-fusion", "manual-attention", "forced-cublas", "forced-mmq"), default="default")
    parser.add_argument("--trace-matmul", action="store_true",
                        help="Requires the isolated tests/patches/prefill-matmul-dispatch.patch")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--dependency-directory", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.repo, args.output, args.probe = args.repo.resolve(), args.output.resolve(), args.probe.resolve()
    args.output.relative_to(args.repo / ".tmp" / "base-acceptance")
    if not 1 <= args.prefix_rows <= 4095 or not 1 <= args.seconds <= 1800:
        parser.error("prefix rows must be 1..4095 and seconds 1..1800")
    if args.layer != "all" and (not args.layer.isascii() or not args.layer.isdigit() or int(args.layer) > 127):
        parser.error("layer must be all or 0..127")
    if shutil.disk_usage(args.output.parent).free < 2 * 1024**3:
        raise ValueError("requires at least 2 GiB free")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    originals = [g for g in manifest["groups"] if Path(g["text_file"]).stem == "original"]
    short = next(g for g in manifest["groups"] if Path(g["text_file"]).stem == "ru-yes")
    manifest.update(groups=[originals[0], originals[0], short, originals[0]],
                    max_tokens=4, deadline_ms=30000, observe_decisions=False, observe_prefill=True)
    planned = [dict(text_label=label, seed=64, kind="diagnostic_not_latency")
               for label in ("original", "original", "ru-yes", "original")]
    if any(g["seeds"] != [64] for g in manifest["groups"]):
        raise ValueError("diagnostic requires explicit seed 64 in every group")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.output / "runtime"
    runtime.mkdir()
    source_probe = identity(args.probe)
    for payload in (args.probe, *sorted(args.probe.parent.glob("*.dll"))):
        shutil.copy2(payload, runtime / payload.name)
    args.probe = runtime / args.probe.name
    gpu = gpu_identity(GPU)
    payload_paths = {manifest["talker"], manifest["codec"],
                     *(g["text_file"] for g in manifest["groups"])}
    for name in ("speaker_f32", "reference_i32", "reference_text_file"):
        if name in manifest:
            payload_paths.add(manifest[name])
    write_json(args.output / "plan.json", {
        "kind": "prefill_differential_not_latency_or_speech_qualification",
        "gpu": gpu, "source": source_identity(), "source_manifest": identity(args.manifest),
        "source_probe": source_probe, "probe": identity(args.probe),
        "runtime": [identity(p) for p in sorted(args.probe.parent.glob("*.dll"))],
        "inputs": [identity(Path(p)) for p in sorted(payload_paths)],
        "cases": planned, "control": args.control, "layer": args.layer,
        "prefix_rows": args.prefix_rows, "seconds": args.seconds,
        "trace_matmul": args.trace_matmul,
        "dispatch_patch": identity(Path(__file__).parent / "patches" / "prefill-matmul-dispatch.patch")
                          if args.trace_matmul or args.control.startswith("forced-") else None,
        "diagnostic_perturbation": "Tensor outputs/readbacks may change allocation/fusion. Compare ON/OFF controls.",
        "expected_end": "max_tokens=4; not natural-EOS or text-coverage evidence",
    })
    deadline = time.monotonic() + args.seconds
    results = {}
    for name, prefix, diagnostic in (("baseline-off", False, False), ("prefix-off", True, False),
                                     ("baseline-taps", False, True), ("prefix-taps", True, True)):
        if time.monotonic() >= deadline:
            results[name] = {"state": "not_run", "rows": [], "reason": "total budget exhausted"}
            continue
        directory = args.output / name
        directory.mkdir()
        write_json(directory / "manifest.json", manifest)
        env = environment(GPU, args.dependency_directory, False)
        env = {k: v for k, v in env.items() if not k.startswith("QWEN_PREFILL_")}
        env["QWEN_PREFILL_PROFILE"] = "1"
        if prefix:
            env.update(QWEN_PREFILL_PREFIX_ROWS=str(args.prefix_rows), QWEN_PREFILL_PREFIX_SOURCE="full")
        if diagnostic:
            env.update(QWEN_PREFILL_DIAG_DIR=str(directory / "taps"), QWEN_PREFILL_DIAG_LAYER=args.layer)
        if args.control == "no-fusion":
            env["GGML_CUDA_DISABLE_FUSION"] = "1"
        elif args.control == "manual-attention":
            env["QWEN_PREFILL_ATTN"] = "manual"
        elif args.control == "forced-cublas":
            env.update(GGML_CUDA_DISABLE_FUSION="1", GGML_RESEARCH_FORCE_CUBLAS_Q8="1",
                       GGML_CUDA_CUBLAS_COMPUTE_TYPE="f32")
        elif args.control == "forced-mmq":
            env.update(GGML_CUDA_DISABLE_FUSION="1", GGML_RESEARCH_FORCE_MMQ_Q8="1")
        if args.trace_matmul:
            env.update(GGML_CUDA_DISABLE_FUSION="1", GGML_RESEARCH_MATMUL_TRACE="1")
        write_json(directory / "environment.json", {
            k: v for k, v in env.items() if k.startswith(("QWEN_", "GGML_", "CUDA_"))
        })
        print(f"START {directory.name}", flush=True)
        code, state = None, "completed"
        with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
            process = subprocess.Popen([str(args.probe), str(directory / "manifest.json"), str(directory / "probe")],
                                       cwd=args.probe.parent, env=env, stdout=out, stderr=err)
            try:
                code = wait_for_probe(process, max(1, min(180, deadline-time.monotonic())), directory / "watchdog.json")
            except subprocess.TimeoutExpired:
                state = "timeout"
        rows, issues = collect(directory, planned, False)
        dispatch = [line for line in (directory / "stderr.log").read_text(encoding="utf-8", errors="replace").splitlines()
                    if "RESEARCH_MATMUL" in line]
        if args.trace_matmul and not dispatch:
            issues.append("requested matmul route trace was not effective")
        if args.control.startswith("forced-") and args.trace_matmul and any(f"route={args.control}" not in line for line in dispatch):
            issues.append("requested forced matmul route was not effective")
        issues.extend(effective_issues(rows, prefix, diagnostic, args.control == "manual-attention", args.prefix_rows))
        if diagnostic and not (directory / "taps").is_dir():
            issues.append("requested diagnostics produced no tap directory")
        # The ordinary collector allows up to 256; this run deliberately uses four.
        if any(row["frames"] > 4 or row["samples"] > 7680 for row in rows):
            issues.append("output exceeds four-frame diagnostic bound")
        if code != 0 or len(rows) != len(planned) or issues:
            state = "incomplete" if state == "completed" else state
        results[name] = {"state": state, "returncode": code, "rows": rows, "issues": issues,
                         "matmul_dispatch": dispatch}
        if name == "prefix-off" and "baseline-off" in results:
            results[name]["comparison"] = compare(rows, results["baseline-off"]["rows"])
        elif name.endswith("-taps"):
            results[name]["instrumentation_comparison"] = compare(rows, results[name.replace("-taps", "-off")]["rows"])
            if name == "prefix-taps":
                results[name]["comparison"] = compare(rows, results["baseline-taps"]["rows"])
        write_json(directory / "collected.json", results[name])
        print(f"DONE {name}: {state}, {len(rows)}/4", flush=True)
    write_json(args.output / "results.json", results)
    return int(any(value["state"] != "completed" for value in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
