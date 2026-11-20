"""Prespecified native reseed/prefix experiment; no speech-acceptance verdict."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import time
import wave
from pathlib import Path

ORIGINAL = "МГУ находится в Москве. Инженер проверил USB-порт, GPS и новый процессор."
VOICE = "kraftwerk_robot_ru_bootstrap_fidelity"
TEXTS = {
    "original": (ORIGINAL, "russian"),
    "ru-short": ("Да!", "russian"),
    "ru-unseen": (
        "Дежурный инженер осмотрел оборудование "
        "и передал смену следующему специалисту.",
        "russian",
    ),
    "en-unseen": (
        "The engineer inspected the equipment and handed the next shift "
        "a detailed report about the completed work.",
        "english",
    ),
}
JOB = re.compile(r"\[ResearchJob\] serial=(\d+) seed=(\d+)")
FRAME = re.compile(
    r"\[ResearchFrame\] serial=(\d+) slot=(\d+) frame=(\d+) codes=([\d,]+)"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def dump_body(path: Path) -> tuple[tuple[int, ...], bytes]:
    if not 8 <= path.stat().st_size <= 1024 * 1024:
        raise ValueError("retained dump outside bounded size")
    data = path.read_bytes()
    ndim = struct.unpack_from("<i", data)[0]
    if not 1 <= ndim <= 4:
        raise ValueError("invalid retained dump dimensions")
    shape = struct.unpack_from("<" + "i" * ndim, data, 4)
    body = data[4 + 4 * ndim :]
    if any(size <= 0 for size in shape) or len(body) != math.prod(shape) * 4:
        raise ValueError("invalid retained dump size")
    return shape, body


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    values = sorted(values)
    return {
        "n": len(values),
        "p50": statistics.median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
    }


def group(text: str, seeds: list[int]) -> dict[str, object]:
    return {"text": text, "seeds": seeds}


def plan() -> list[dict[str, object]]:
    runs = [
        {
            "name": "causal-baseline",
            "policy": "baseline",
            "prefix": None,
            "max_batch": 1,
            "groups": [
                group("original", [seed])
                for seed in (2, 4, 14, 21, 31, 1, 42, 43, 44) * 2
            ],
        }
    ]
    for boundary in (1, 2, 3):
        runs.append(
            {
                "name": f"switch-{boundary}",
                "policy": f"switch-after:{boundary}",
                "prefix": None,
                "max_batch": 1,
                "groups": [group("original", [seed]) for seed in (2, 14) * 2],
            }
        )
    independent = [
        group(text, [seed]) for text in TEXTS for seed in (*range(64, 72), 64)
    ]
    for policy in ("baseline", "per-frame", "early-2", "early-4"):
        runs.append(
            {
                "name": f"independent-{policy}",
                "policy": policy,
                "prefix": None,
                "max_batch": 1,
                "groups": independent,
            }
        )
    for prefix_seed in (2, 14):
        for length in (1, 2):
            prefix = f"seed-{prefix_seed}-frames-{length}"
            runs.append(
                {
                    "name": f"single-{prefix}",
                    "policy": "baseline",
                    "prefix": prefix,
                    "max_batch": 1,
                    "groups": [
                        group("original", [seed])
                        for seed in (
                            prefix_seed,
                            1,
                            *range(64, 72),
                            64,
                            71,
                            prefix_seed,
                        )
                    ],
                }
            )
            runs.append(
                {
                    "name": f"dual-{prefix}",
                    "policy": "baseline",
                    "prefix": prefix,
                    "max_batch": 2,
                    "groups": [
                        group("original", list(pair))
                        for pair in (
                            (1, 1),
                            (64, 65),
                            (66, 67),
                            (68, 69),
                            (70, 71),
                            (64, 65),
                        )
                    ],
                }
            )
    runs.append(
        {
            "name": "dual-independent",
            "policy": "baseline",
            "prefix": None,
            "max_batch": 2,
            "groups": [
                group("original", list(pair))
                for pair in (
                    (2, 42),
                    (14, 43),
                    (64, 65),
                    (66, 67),
                    (68, 69),
                    (70, 71),
                    (64, 65),
                )
            ],
        }
    )
    return runs


def parse_frames(path: Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    by_serial: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if match := JOB.search(line):
            serial, seed = map(int, match.groups())
            if serial in by_serial:
                raise ValueError("duplicate research job")
            row = {"serial": serial, "seed": seed, "codes": []}
            jobs.append(row)
            by_serial[serial] = row
        elif match := FRAME.search(line):
            serial, _, frame, text = match.groups()
            codes = by_serial[int(serial)]["codes"]
            values = [int(value) for value in text.split(",")]
            if int(frame) != len(codes) or len(values) != 16:
                raise ValueError("malformed/interleaved research frames")
            codes.append(values)
    return jobs


def make_wav(source: Path) -> dict[str, object]:
    pcm = array.array("f")
    pcm.frombytes(source.read_bytes())
    if any(not math.isfinite(value) for value in pcm):
        raise ValueError("nonfinite PCM")
    samples = array.array(
        "h", (max(-32768, min(32767, round(value * 32768))) for value in pcm)
    )
    target = source.with_suffix(".wav")
    with target.open("xb") as stream, wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(samples.tobytes())
    return {
        "wav": target.name,
        "sha256": sha(target),
        "samples": len(pcm),
        "rms": math.sqrt(sum(value * value for value in pcm) / len(pcm)) if pcm else 0,
        "coverage": "unassessed",
        "human_listening": "unassessed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "probe",
        "output",
        "talker",
        "codec",
        "reference_dump",
        "registry",
        "discovery_analysis",
    ):
        parser.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--dependency-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--only",
        nargs="*",
        help="Prespecify a subset by run name, before seeing results",
    )
    args = parser.parse_args()
    started = time.monotonic()
    overall_seconds = 1800
    args.output.mkdir(parents=False, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    if shutil.disk_usage(args.output).free < 2 * 1024**3:
        raise ValueError("at least 2 GiB of free disk required")
    runtime_files = list(args.probe.parent.glob("*.dll"))
    for directory in args.dependency_dir:
        if not directory.is_dir():
            raise ValueError("dependency directory missing")
        runtime_files.extend(
            path
            for path in directory.glob("*.dll")
            if path.name.lower().startswith(("cublas", "cudart", "vcomp"))
        )
    source_root = Path(__file__).resolve().parent.parent
    source_identity = {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root, text=True
        ),
        "submodules": subprocess.check_output(
            ["git", "submodule", "status"], cwd=source_root, text=True
        ).strip(),
    }
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.free",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    gpu_line = next(
        (line for line in gpu.splitlines() if line.startswith(args.gpu_uuid + ",")),
        None,
    )
    if not gpu_line or int(gpu_line.rsplit(",", 1)[1].strip().split()[0]) < 8192:
        raise ValueError("expected GPU absent or insufficient free VRAM")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    voice = next(row for row in registry["voices"] if row["voice_id"] == VOICE)
    if voice.get("preserve_reference_text_whitespace") is not True:
        raise ValueError("reference whitespace contract mismatch")
    (inputs / "reference.txt").write_bytes(voice["reference_text"].encode("utf-8"))
    shape, speaker = dump_body(args.reference_dump / "spk-emb.bin")
    if shape != (2048,):
        raise ValueError("speaker shape mismatch")
    (inputs / "speaker.f32").write_bytes(speaker)
    shape, ref = dump_body(args.reference_dump / "ref-codes.bin")
    if shape != (16, 226):
        raise ValueError("reference codes shape mismatch")
    values = struct.unpack("<" + "f" * (len(ref) // 4), ref)
    if any(not value.is_integer() or not 0 <= value < 2048 for value in values):
        raise ValueError("reference code value mismatch")
    (inputs / "reference.i32").write_bytes(
        struct.pack("<" + "i" * len(values), *(int(value) for value in values))
    )
    discovery = json.loads(args.discovery_analysis.read_text(encoding="utf-8"))
    known = {row["seed"]: row for row in discovery["requests"]}
    for seed in (2, 14):
        if (
            known[seed]["text"] != ORIGINAL
            or known[seed]["frames"] != 3
            or len(known[seed]["codes"]) != 3
        ):
            raise ValueError("unexpected known bad discovery history")
        for length in (1, 2):
            text = (
                "\n".join(
                    " ".join(map(str, row)) for row in known[seed]["codes"][:length]
                )
                + "\n"
            )
            (inputs / f"seed-{seed}-frames-{length}.txt").write_bytes(text.encode())
    for label, (text, _) in TEXTS.items():
        (inputs / f"{label}.txt").write_bytes(text.encode("utf-8"))
    runs = plan()
    if args.only:
        if set(args.only) - {run["name"] for run in runs}:
            raise ValueError("unknown run name")
        runs = [run for run in runs if run["name"] in args.only]
    write_json(
        args.output / "plan.json",
        {
            "runs": runs,
            "max_tokens": 256,
            "frame_proxy_max": 12,
            "overall_deadline_seconds": overall_seconds,
            "per_process_seconds": 240,
            "scope": (
                "native engine callback research; no live selection, "
                "physical playback, ASR/listening or acceptance claim"
            ),
            "gpu": gpu_line,
            "voice": voice,
            "source": source_identity,
            "runtime_sha256": {
                str(path.resolve()): sha(path) for path in runtime_files
            },
            "cmake_cache_sha256": sha(args.probe.parent / "CMakeCache.txt"),
            "source_inputs": {
                str(path.resolve()): sha(path)
                for path in (
                    args.probe,
                    args.talker,
                    args.codec,
                    args.registry,
                    args.discovery_analysis,
                    args.reference_dump / "spk-emb.bin",
                    args.reference_dump / "ref-codes.bin",
                    Path(__file__),
                )
            },
            "prepared_inputs": {path.name: sha(path) for path in inputs.iterdir()},
        },
    )
    all_rows = []
    for run in runs:
        name = run["name"]
        manifest = {
            "talker": str(args.talker.resolve()),
            "codec": str(args.codec.resolve()),
            "speaker_f32": str((inputs / "speaker.f32").resolve()),
            "reference_i32": str((inputs / "reference.i32").resolve()),
            "reference_frames": 226,
            "reference_text_file": str((inputs / "reference.txt").resolve()),
            "max_batch": run["max_batch"],
            "max_tokens": 256,
            "deadline_ms": 60000,
            "stream_max_chunk_frames": 8,
            "groups": [
                {
                    "text_file": str((inputs / (row["text"] + ".txt")).resolve()),
                    "language": TEXTS[row["text"]][1],
                    "seeds": row["seeds"],
                }
                for row in run["groups"]
            ],
        }
        manifest_path = args.output / f"{name}-manifest.json"
        write_json(manifest_path, manifest)
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [str(path.resolve()) for path in args.dependency_dir] + [env["PATH"]]
        )
        for key in list(env):
            if key.startswith("QWEN_RESEARCH_") or key.startswith("QWEN_TTS_"):
                env.pop(key)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
                "GGML_BACKEND": "CUDA0",
                "QWEN_TTS_FULL_AR_TRACE": "1",
                "QWEN_RESEARCH_FRAME_SEED_POLICY": run["policy"],
            }
        )
        if run["prefix"]:
            env["QWEN_RESEARCH_PREFIX_FILE"] = str(
                (inputs / (run["prefix"] + ".txt")).resolve()
            )
        command = [
            str(args.probe.resolve()),
            str(manifest_path.resolve()),
            str((args.output / name).resolve()),
        ]
        write_json(
            args.output / f"{name}-invocation.json",
            {
                "command": command,
                "environment": {
                    key: env[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "GGML_BACKEND",
                        "QWEN_TTS_FULL_AR_TRACE",
                        "QWEN_RESEARCH_FRAME_SEED_POLICY",
                        "QWEN_RESEARCH_PREFIX_FILE",
                    )
                    if key in env
                },
                "manifest_sha256": sha(manifest_path),
            },
        )
        remaining = overall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("overall experiment deadline")
        print("START", name, flush=True)
        with (
            (args.output / f"{name}-stdout.log").open("xb") as stdout,
            (args.output / f"{name}-stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, env=env, cwd=args.probe.parent
            )
            try:
                status = process.wait(timeout=min(240, remaining))
            except subprocess.TimeoutExpired:
                write_json(
                    args.output / f"{name}-watchdog.json",
                    {"pid": process.pid, "timeout": True},
                )
                # This probe launches no children. Terminate the owned process
                # directly so the watchdog does not depend on a helper process.
                process.kill()
                process.wait(timeout=15)
                raise
        if status:
            write_json(
                args.output / f"{name}-process-failure.json",
                {"exit_code": status, "command": command, "evidence_preserved": True},
            )
            raise RuntimeError(f"probe failed for {name}: {status}; evidence preserved")
        rows = [
            json.loads(line)
            for line in (args.output / name / "results.jsonl").read_text().splitlines()
        ]
        frames = parse_frames(args.output / f"{name}-stderr.log")
        expected = [seed for row in run["groups"] for seed in row["seeds"]]
        if [row["seed"] for row in rows] != expected or [
            row["seed"] for row in frames
        ] != expected:
            raise ValueError("request/trace order mismatch")
        for row, trace in zip(rows, frames, strict=True):
            if row["status"] == 0 and len(trace["codes"]) != min(128, row["frames"]):
                raise ValueError("complete frame trace count mismatch")
            row.update(
                {
                    "run": name,
                    "policy": run["policy"],
                    "prefix": run["prefix"],
                    "text_label": run["groups"][row["group"]]["text"],
                    "first_use_group": row["group"] == 0,
                    "codes": trace["codes"],
                    "serial": trace["serial"],
                    "observed_trace_frames": len(trace["codes"]),
                    "coverage": "unassessed",
                }
            )
            row["wav"] = make_wav(
                args.output / name / f"group-{row['group']}-lane-{row['lane']}.f32"
            )
            row["pcm_sha256"] = sha(
                args.output / name / f"group-{row['group']}-lane-{row['lane']}.f32"
            )
            row["duration_proxy"] = (
                row["text_label"] != "ru-short"
                and row["status"] == 0
                and not row["invalid_pcm"]
                and not row["capacity_exceeded"]
                and row["finish_reason"] == 1
                and row["frames"] <= 12
            )
            if run["prefix"]:
                original_seed, length = map(
                    int,
                    re.fullmatch(r"seed-(\d+)-frames-(\d+)", run["prefix"]).groups(),
                )
                if row["codes"][:length] != known[original_seed]["codes"][:length]:
                    raise ValueError("forced prefix did not match exactly")
            if name == "causal-baseline" and row["seed"] in (2, 14):
                if (
                    row["frames"] != 3
                    or row["finish_reason"] != 1
                    or row["codes"] != known[row["seed"]]["codes"]
                ):
                    raise ValueError(
                        "baseline did not reproduce frozen failure; "
                        "stop rather than pool different controls"
                    )
        if run["prefix"] and run["max_batch"] == 1:
            original_seed = int(run["prefix"].split("-")[1])
            controls = [row for row in rows if row["seed"] == original_seed]
            for control in controls:
                if (
                    control["status"] != 0
                    or control["finish_reason"] != 1
                    or control["codes"] != known[original_seed]["codes"]
                ):
                    raise ValueError("self-seed prefix replay changed frozen failure")
                baselines = [
                    row
                    for row in all_rows
                    if row["run"] == "causal-baseline" and row["seed"] == original_seed
                ]
                if baselines and any(
                    row["pcm_sha256"] != control["pcm_sha256"] for row in baselines
                ):
                    raise ValueError("self-seed prefix replay changed baseline PCM")
        write_json(args.output / f"{name}-analysis.json", rows)
        all_rows.extend(rows)
        print(
            "DONE",
            name,
            "rows",
            len(rows),
            "short_proxy",
            sum(row["duration_proxy"] for row in rows),
            flush=True,
        )
    write_json(args.output / "all-observations.json", all_rows)
    summaries = []
    for run in runs:
        rows = [row for row in all_rows if row["run"] == run["name"]]
        warm = [row for row in rows if not row["first_use_group"]]
        summaries.append(
            {
                "run": run["name"],
                "n": len(rows),
                "short_duration_proxy": sum(row["duration_proxy"] for row in rows),
                "max_tokens": sum(row["finish_reason"] == 2 for row in rows),
                "errors": sum(row["status"] != 0 for row in rows),
                "warm_callback_ttfa_ms": quantiles(
                    [row["chunks"][0]["ms"] for row in warm if row["chunks"]]
                ),
                "warm_rtf": quantiles(
                    [
                        row["group_total_ms"] / (row["samples"] / 24)
                        for row in warm
                        if row["samples"]
                    ]
                ),
                "coverage": "unassessed",
                "physical_playback": "not_run",
            }
        )
    write_json(args.output / "summary.json", summaries)
    inventory = {
        str(path.relative_to(args.output)): {
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
        for path in args.output.rglob("*")
        if path.is_file()
    }
    write_json(args.output / "sha256-inventory.json", inventory)


if __name__ == "__main__":
    main()
