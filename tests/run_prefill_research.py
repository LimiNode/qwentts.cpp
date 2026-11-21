"""Bounded, paired CMP50HX prefill experiments. No release/listening verdict."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import time
from pathlib import Path

from run_eos_corpus import environment, gpu_identity, identity, source_identity
from run_eos_research import make_wav, parse_frames, quantiles, sha, wait_for_probe, write_json

GPU = "GPU-40361931-6cb5-ac58-a059-5ba3e70986fb"
VARIANTS = {
    "unobserved": {},
    "baseline": {},
    "last-only": {"QWEN_PREFILL_LAST_ONLY": "1"},
    "kv-f16": {"QWEN_PREFILL_KV": "f16"},
    "window64": {"QWEN_PREFILL_WINDOW": "64"},
    "retained": {"QWEN_PREFILL_RETAIN": "1"},
    "combined": {"QWEN_PREFILL_LAST_ONLY": "1", "QWEN_PREFILL_KV": "f16",
                 "QWEN_PREFILL_WINDOW": "64", "QWEN_PREFILL_RETAIN": "1"},
    "baseline-end": {},
    "prefix": {},
    "prefix-combined": {"QWEN_PREFILL_LAST_ONLY": "1", "QWEN_PREFILL_KV": "f16",
                        "QWEN_PREFILL_WINDOW": "64", "QWEN_PREFILL_RETAIN": "1"},
}
INPUT_HASHES = {
    "speaker.f32": "4119368fc3a673c29187bbef0c48d3c789e6f05390bae61f3303c4f10ea38698",
    "reference.i32": "069ac5c280be070c2e5bf006cb9b58a2d248f34e1bdd37aab62310f2ffdd9443",
    "reference.txt": "e1e6fe3cd4561d3c895e23834b44737e4bdc3b9fd92bf2f1f703bef8272e43f4",
    "original.txt": "d917e809733280d0fcc4603828e8253593839a22e29161216868f18ef4ba8a5e",
}


def cases(causal: bool) -> list[dict]:
    if causal:
        return [dict(text_label=label, seed=seed, kind="causal")
                for label, seed in (("original", 64), ("original", 2), ("original", 14),
                                    ("ru-yes", 64), ("ru-two-sentences", 64))]
    # A->A, A->B->A, longer shape, languages, and discovered EOS failures.
    return ([dict(text_label="original", seed=64, kind="first_use")]
            + [dict(text_label="original", seed=64, kind="warm_pair") for _ in range(12)]
            + [dict(text_label=label, seed=seed, kind="shape_or_eos_control")
               for label, seed in (("ru-yes", 64), ("original", 64),
                                   ("ru-two-sentences", 64), ("original", 64),
                                   ("en-unseen", 64), ("original", 2), ("original", 14),
                                   ("original", 64))])


def logits_difference(left: Path, right: Path) -> dict:
    a, b = left.read_bytes(), right.read_bytes()
    if len(a) != len(b) or len(a) % 4:
        return {"shape_equal": False, "byte_equal": False}
    av, bv = struct.unpack(f"<{len(a)//4}f", a), struct.unpack(f"<{len(b)//4}f", b)
    if not all(math.isfinite(v) for v in (*av, *bv)):
        raise ValueError("nonfinite logits in comparison")
    return {"shape_equal": True, "byte_equal": a == b,
            "max_abs": max((abs(x-y) for x, y in zip(av, bv)), default=0.0),
            "rms": math.sqrt(sum((x-y)**2 for x, y in zip(av, bv))/max(1, len(av))),
            "argmax_equal": max(range(len(av)), key=av.__getitem__)
                            == max(range(len(bv)), key=bv.__getitem__)}


def collect(directory: Path, planned: list[dict], trace: bool) -> tuple[list[dict], list[str]]:
    path = directory / "probe" / "results.jsonl"
    issues = []
    if not path.exists():
        return [], ["no flushed result file"]
    try:
        traces = parse_frames(directory / "stderr.log") if trace else []
    except (OSError, ValueError, KeyError) as error:
        traces = []
        issues.append(f"trace: {error}")
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
            if index >= len(planned) or (row["group"], row["lane"], row["seed"]) != (index, 0, planned[index]["seed"]):
                raise ValueError("result order differs from frozen plan")
            if not 0 <= row["frames"] <= 256 or not 0 <= row["samples"] <= 256 * 1920:
                raise ValueError("result exceeds frozen bounds")
        except (ValueError, KeyError, TypeError) as error:
            issues.append(f"group {index}: {error}")
            break
        raw = directory / "probe" / f"group-{index}-lane-0.f32"
        row.update(planned[index], coverage="unassessed")
        try:
            if raw.stat().st_size != row["samples"] * 4:
                raise ValueError("PCM size mismatch")
            row.update(raw=identity(raw), wav=make_wav(raw))
        except (OSError, ValueError) as error:
            issues.append(f"group {index} audio: {error}")
            row["artifact_error"] = str(error)
        observation = row.get("prefill_observation")
        if observation:
            logits = directory / "probe" / observation["logits_f32"]
            try:
                row["logits"] = identity(logits)
            except OSError as error:
                issues.append(f"group {index} logits: {error}")
        if trace:
            if index >= len(traces) or traces[index]["seed"] != row["seed"]:
                issues.append(f"group {index}: trace seed/order mismatch")
            else:
                row["codes"] = traces[index]["codes"]
                if len(row["codes"]) != min(128, row["frames"]):
                    issues.append(f"group {index}: missing bounded complete-frame trace")
        rows.append(row)
    return rows, issues


def summarize(rows: list[dict]) -> dict:
    warm = [r for r in rows if r["kind"] == "warm_pair"]
    fields = {"first_pcm_ms": [r["chunks"][0]["ms"] for r in warm if r["chunks"]],
              "prefill_ms": [r["prefill_ms"] for r in warm],
              "total_ms": [r["total_ms"] for r in warm]}
    for field in ("setup_ms", "graph_build_ms", "schedule_alloc_ms", "upload_ms", "compute_wait_ms", "readback_ms"):
        fields[field] = [r["prefill_observation"][field] for r in warm if r.get("prefill_observation")]
    return {"observed": len(rows), "warm": {k: quantiles(v) for k, v in fields.items()},
            "errors": sum(r["status"] != 0 or r["invalid_pcm"] or r["capacity_exceeded"] for r in rows),
            "natural_eos": sum(r["finish_reason"] == 1 for r in rows),
            "max_tokens": sum(r["finish_reason"] == 2 for r in rows),
            "coverage": "unassessed", "human_listening": "not_run"}


def compare(rows: list[dict], baseline: list[dict]) -> dict:
    result = []
    for row, control in zip(rows, baseline):
        if (row["text_label"], row["seed"]) != (control["text_label"], control["seed"]):
            raise ValueError("unpaired experiment")
        item = {"group": row["group"], "raw_equal": (row["raw"]["sha256"] == control["raw"]["sha256"])
                if "raw" in row and "raw" in control else None,
                "frames_equal": row["frames"] == control["frames"],
                "terminal_equal": (row["status"], row["finish_reason"]) == (control["status"], control["finish_reason"])}
        if "logits" in row and "logits" in control:
            item["logits"] = logits_difference(Path(row["logits"]["path"]), Path(control["logits"]["path"]))
        if "codes" in row and "codes" in control:
            item["bounded_codes_equal"] = row["codes"] == control["codes"]
        result.append(item)
    return {"candidate_observed": len(rows), "baseline_observed": len(baseline),
            "paired": len(result), "same_observed_denominator": len(rows) == len(baseline), "rows": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--customvoice-codec", type=Path)
    parser.add_argument("--models", nargs="+", choices=("ru-stress", "base", "customvoice"), default=["ru-stress", "base", "customvoice"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=[name for name in VARIANTS if not name.startswith("prefix")])
    parser.add_argument("--prefix-rows", type=int)
    parser.add_argument("--russian-only", action="store_true")
    parser.add_argument("--dependency-directory", type=Path, action="append", default=[])
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    args.output.relative_to(args.repo / ".tmp" / "base-acceptance")
    if (not 1 <= args.seconds <= 1800 or "baseline" not in args.variants
            or args.variants.index("baseline") != int(args.variants[0] == "unobserved")
            or len(args.variants) != len(set(args.variants))):
        parser.error("bounded budget and unique variants with baseline before candidates required")
    if any(v.startswith("prefix") for v in args.variants) and not (args.prefix_rows and 1 <= args.prefix_rows <= 4095):
        parser.error("prefix variants require --prefix-rows 1..4095 after byte-level discovery")
    model_paths = {
        "ru-stress": args.repo / "models/ru-stress/qwen-talker-1.7b-ru-stress-Q8_0.gguf",
        "base": args.repo / "models/qwen-talker-1.7b-base-Q8_0.gguf",
        "customvoice": args.repo / "models/prefill-customvoice-native-0c0e3051/qwen-talker-1.7b-customvoice-Q8_0.gguf",
    }
    expected = {"ru-stress": "f4672741608d145fae169075cb0ec775a5382cbe9183eccb59c7ab9cc4797427",
                "base": "4b9a33a236908dd9435a42f7a396e38038329d053b704342a6413c08544c4fda",
                "customvoice": "cab2cff67a0a557310febe558dc83076b28ed790e491867eb2751759f4cd89fa"}
    gpu = gpu_identity(GPU)
    models = {label: identity(model_paths[label]) for label in args.models}
    if any(models[label]["sha256"] != expected[label] for label in args.models):
        raise ValueError("model identity differs from frozen study")
    for name, digest in INPUT_HASHES.items():
        if sha(args.inputs / name) != digest:
            raise ValueError(f"conditioning/input identity mismatch: {name}")
    if shutil.disk_usage(args.output.parent).free < 2 * 1024**3:
        raise ValueError("requires 2 GiB free")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    for name in (*INPUT_HASHES, "ru-yes.txt", "ru-two-sentences.txt", "en-unseen.txt"):
        shutil.copyfile(args.inputs / name, inputs / name)
    planned = cases(args.causal)
    if args.russian_only:
        planned = [row for row in planned if not row["text_label"].startswith("en-")]
    runtime = [identity(p) for p in sorted(args.probe.parent.glob("*.dll"))]
    runtime += [identity(p) for directory in args.dependency_directory for p in sorted(directory.glob("cublas*.dll"))]
    runtime += [identity(p) for directory in args.dependency_directory for p in sorted(directory.glob("cudart*.dll"))]
    cache = args.probe.parent / "CMakeCache.txt"
    write_json(args.output / "plan.json", {"cases": planned, "variants": args.variants, "models": models,
               "codec": identity(args.codec), "probe": identity(args.probe), "source": source_identity(),
               "customvoice_codec": identity(args.customvoice_codec) if args.customvoice_codec else None,
               "runtime_files": runtime, "cmake_cache": identity(cache) if cache.exists() else None,
               "gpu": gpu, "seconds": args.seconds, "trace": args.causal,
               "prefix_rows": args.prefix_rows,
               "inputs": {p.name: identity(p) for p in inputs.iterdir()},
               "timing_scope": "host wall; CUDA kernel/event time requires independent profiler"})
    deadline = time.monotonic() + args.seconds
    all_results = {}
    for label in args.models:
        variants = {}
        for variant in args.variants:
            if time.monotonic() >= deadline:
                variants[variant] = {"state": "not_run", "reason": "overall deadline"}
                continue
            directory = args.output / f"{label}-{variant}"
            directory.mkdir()
            observed = variant != "unobserved"
            codec = args.customvoice_codec if label == "customvoice" and args.customvoice_codec else args.codec
            manifest = {"talker": str(model_paths[label]), "codec": str(codec.resolve()),
                        "max_batch": 1, "max_tokens": 256, "deadline_ms": 60000,
                        "stream_max_chunk_frames": 8, "observe_decisions": args.causal,
                        "observe_prefill": observed, "groups": [
                            {"text_file": str(inputs / f"{r['text_label']}.txt"),
                             "language": "english" if r["text_label"].startswith("en-") else "russian",
                             "seeds": [r["seed"]]} for r in planned]}
            if label == "customvoice":
                manifest["speaker"] = "ryan"
            else:
                manifest.update(speaker_f32=str(inputs / "speaker.f32"), reference_i32=str(inputs / "reference.i32"),
                                reference_text_file=str(inputs / "reference.txt"), reference_frames=226)
            write_json(directory / "manifest.json", manifest)
            env = environment(GPU, [args.probe.parent, *args.dependency_directory], args.causal)
            env = {k: v for k, v in env.items() if not k.startswith("QWEN_PREFILL_")}
            env.update(QWEN_PREFILL_PROFILE="1" if observed else "0", **VARIANTS[variant])
            if variant.startswith("prefix"):
                env["QWEN_PREFILL_PREFIX_ROWS"] = str(args.prefix_rows)
            write_json(directory / "environment.json", {k: v for k, v in env.items() if k.startswith(("GGML_", "CUDA_", "QWEN_"))})
            status = "completed"
            returncode = None
            print(f"START {label}/{variant}", flush=True)
            with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
                process = subprocess.Popen([str(args.probe.resolve()), str(directory / "manifest.json"), str(directory / "probe")],
                                           stdout=out, stderr=err, env=env, cwd=args.probe.parent)
                try:
                    returncode = wait_for_probe(process, max(1, min(240, deadline-time.monotonic())), directory / "watchdog.json")
                except subprocess.TimeoutExpired:
                    status = "timeout"
            if returncode not in (None, 0):
                status = "failed"
            rows, issues = collect(directory, planned, args.causal)
            if issues or len(rows) != len(planned):
                status = "incomplete" if status == "completed" else status
            variants[variant] = {"state": status, "returncode": returncode, "rows": rows,
                                 "issues": issues, "not_run_groups": list(range(len(rows), len(planned))),
                                 "summary": summarize(rows)}
            if variant != "baseline" and "baseline" in variants:
                variants[variant]["comparison"] = compare(rows, variants["baseline"].get("rows", []))
            write_json(directory / "collected.json", variants[variant])
            print(f"DONE {label}/{variant} {status} {len(rows)}/{len(planned)}", flush=True)
        if "unobserved" in variants and "baseline" in variants:
            variants["unobserved"]["comparison"] = compare(variants["unobserved"].get("rows", []), variants["baseline"].get("rows", []))
        all_results[label] = variants
        write_json(args.output / f"{label}-results.json", variants)
    write_json(args.output / "results.json", all_results)
    return int(any(v["state"] != "completed" or len(v.get("rows", [])) != len(planned)
                   for variants in all_results.values() for v in variants.values()))


if __name__ == "__main__":
    raise SystemExit(main())
