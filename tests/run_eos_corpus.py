"""Frozen CMP-only, single-slot corpus; duration is not speech coverage."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

from run_eos_research import (
    ORIGINAL,
    TEXTS,
    make_wav,
    parse_frames,
    quantiles,
    sha,
    wait_for_probe,
    write_json,
)

CORPUS = {
    "original": (ORIGINAL, "russian", False),
    "ru-yes": ("Да!", "russian", True),
    "ru-no": ("Нет.", "russian", True),
    "en-hi": ("Hi!", "english", True),
    "en-thanks": ("Thank you.", "english", True),
    "ru-unseen": (*TEXTS["ru-unseen"], False),
    "ru-two-sentences": (
        "В мастерско́й завершили проверку. Теперь можно закрыть дверь и отправить "
        "подробный отчёт о выполненной работе.",
        "russian",
        False,
    ),
    "en-unseen": (*TEXTS["en-unseen"], False),
}
PREPARED_HASHES = {
    "speaker.f32": "4119368fc3a673c29187bbef0c48d3c789e6f05390bae61f3303c4f10ea38698",
    "reference.i32": "069ac5c280be070c2e5bf006cb9b58a2d248f34e1bdd37aab62310f2ffdd9443",
    "reference.txt": "e1e6fe3cd4561d3c895e23834b44737e4bdc3b9fd92bf2f1f703bef8272e43f4",
}
OVERALL_SECONDS = 900
PROCESS_SECONDS = 600
MIN_FREE_DISK = 2 * 1024**3


def cases() -> list[dict]:
    """Keep discovery controls and repeats out of prospective denominators."""
    result = []
    for repeat in range(2):
        for seed in (2, 14, 42):
            result.append(
                {
                    "kind": "discovery_control",
                    "text_label": "original",
                    "seed": seed,
                    "repeat": repeat,
                    "short_valid": False,
                }
            )
    for label, (_, _, short) in CORPUS.items():
        for seed in range(64, 80):
            result.append(
                {
                    "kind": "prospective",
                    "text_label": label,
                    "seed": seed,
                    "repeat": 0,
                    "short_valid": short,
                }
            )
        result.append(
            {
                "kind": "repeat",
                "text_label": label,
                "seed": 64,
                "repeat": 1,
                "short_valid": short,
            }
        )
    return [
        {
            "group": index,
            **row,
            "prior_baseline_exposure": row["kind"] == "discovery_control"
            or (
                row["text_label"] in ("original", "ru-yes", "ru-unseen", "en-unseen")
                and row["seed"] < 72
            ),
        }
        for index, row in enumerate(result)
    ]


def identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def prepare_inputs(source: Path, target: Path) -> dict:
    """Copy exact retained conditioning, never reconstruct its transcript."""
    payloads = {}
    for name, expected in PREPARED_HASHES.items():
        path = source / name
        if not path.is_file() or path.stat().st_size > 65536 or sha(path) != expected:
            raise ValueError(f"retained conditioning hash/size mismatch: {name}")
        payloads[name] = path.read_bytes()
    speaker = payloads["speaker.f32"]
    reference = payloads["reference.i32"]
    transcript = payloads["reference.txt"].decode("utf-8", errors="strict")
    if len(speaker) != 2048 * 4 or not all(
        math.isfinite(v) for v in struct.unpack("<2048f", speaker)
    ):
        raise ValueError("expected finite 2048-dimensional speaker embedding")
    if len(reference) != 16 * 226 * 4 or any(
        not 0 <= v < 2048 for v in struct.unpack("<3616i", reference)
    ):
        raise ValueError("expected codebook-major 16x226 reference codes")
    if not transcript.strip() or "\x00" in transcript:
        raise ValueError("invalid reference transcript")
    target.mkdir(exist_ok=False)
    for name, payload in payloads.items():
        with (target / name).open("xb") as stream:
            stream.write(payload)
    for label, (text, _, _) in CORPUS.items():
        with (target / f"{label}.txt").open("xb") as stream:
            stream.write(text.encode("utf-8"))
    return {path.name: identity(path) for path in sorted(target.iterdir())}


def environment(gpu_uuid: str, dependencies: list[Path], trace: bool) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("QWEN_RESEARCH_", "QWEN_TTS_", "GGML_"))
    }
    env["PATH"] = os.pathsep.join(
        [str(path.resolve()) for path in dependencies] + [env.get("PATH", "")]
    )
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "GGML_BACKEND": "CUDA0",
            "QWEN_RESEARCH_FRAME_SEED_POLICY": "baseline",
        }
    )
    if trace:
        env["QWEN_TTS_FULL_AR_TRACE"] = "1"
    return env


def gpu_identity(gpu_uuid: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            gpu_uuid,
            "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    fields = [field.strip() for field in result.split(",")]
    if len(fields) != 5 or fields[0] != gpu_uuid or "CMP 50HX" not in fields[1]:
        raise ValueError("this experiment requires the explicitly selected CMP 50HX")
    if int(fields[-1].split()[0]) < 8192:
        raise ValueError("at least 8192 MiB of free device memory required")
    return result


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {
        name: subprocess.check_output(
            ["git", *command], cwd=root, text=True, timeout=30
        ).strip()
        for name, command in {
            "head": ["rev-parse", "HEAD"],
            "status": ["status", "--porcelain"],
            "submodules": ["submodule", "status"],
        }.items()
    }


def manifest(args: argparse.Namespace, planned: list[dict]) -> dict:
    inputs = args.output / "inputs"
    return {
        "talker": str(args.talker.resolve()),
        "codec": str(args.codec.resolve()),
        "speaker_f32": str((inputs / "speaker.f32").resolve()),
        "reference_i32": str((inputs / "reference.i32").resolve()),
        "reference_frames": 226,
        "reference_text_file": str((inputs / "reference.txt").resolve()),
        "max_batch": 1,
        "max_tokens": 256,
        "deadline_ms": 60000,
        "stream_max_chunk_frames": 8,
        "observe_decisions": not args.timing_off,
        "groups": [
            {
                "text_file": str((inputs / f"{row['text_label']}.txt").resolve()),
                "language": CORPUS[row["text_label"]][1],
                "seeds": [row["seed"]],
            }
            for row in planned
        ],
    }


def collect(
    output: Path,
    planned: list[dict],
    trace_required: bool,
    issues: list[str],
    observe_decisions: bool | None = None,
) -> list[dict]:
    """Retain already flushed outcomes even when a later group/process fails."""
    results = output / "corpus" / "results.jsonl"
    try:
        traces = parse_frames(output / "corpus-stderr.log")
    except (OSError, ValueError, KeyError) as error:
        issues.append(f"frame trace: {error}")
        traces = []
    rows = []
    if not results.exists():
        return rows
    for line in results.read_text(encoding="utf-8").splitlines():
        index = len(rows)
        try:
            row = json.loads(line)
            if index >= len(planned) or (row["group"], row["lane"], row["seed"]) != (
                index,
                0,
                planned[index]["seed"],
            ):
                raise ValueError("flushed result order differs from frozen cases")
            for field in (
                "status",
                "finish_reason",
                "frames",
                "samples",
                "invalid_pcm",
                "capacity_exceeded",
                "chunks",
            ):
                if field not in row:
                    raise ValueError(f"missing terminal field: {field}")
            for field in ("status", "finish_reason", "frames", "samples"):
                if type(row[field]) is not int:
                    raise ValueError(f"invalid terminal integer: {field}")
            if not 0 <= row["frames"] <= 256 or not 0 <= row["samples"] <= 256 * 1920:
                raise ValueError("terminal audio exceeds frozen generation bound")
            if not isinstance(row["chunks"], list):
                raise ValueError("invalid terminal chunk list")
        except (ValueError, KeyError, TypeError) as error:
            issues.append(f"result after {index} retained groups: {error}")
            break
        codes = traces[index]["codes"] if index < len(traces) else []
        if observe_decisions is not None:
            decisions = row.get("decisions")
            capture_valid = (
                row.get("decision_observation_enabled") is observe_decisions
                and row.get("decision_capture_invalid") is False
                and isinstance(decisions, list)
                and (observe_decisions or not decisions)
                and (
                    not observe_decisions
                    or row["status"] != 0
                    or row["frames"] == 0
                    or bool(decisions)
                )
            )
            row["decision_capture_evidence_valid"] = capture_valid
            if not capture_valid:
                issues.append(f"decision capture missing/invalid for group {index}")
        if index < len(traces) and traces[index]["seed"] != row["seed"]:
            issues.append(f"trace seed differs from frozen group {index}")
            codes = []
        if (
            trace_required
            and row["status"] == 0
            and len(codes) != min(128, row["frames"])
        ):
            issues.append(
                f"requested complete-frame trace missing/truncated for group {index}"
            )
        raw = output / "corpus" / f"group-{index}-lane-0.f32"
        row.update(
            {
                **planned[index],
                "policy": "baseline",
                "prefix": None,
                "first_use_group": index == 0,
                "codes": codes,
                "codes_available": bool(codes),
                "observed_trace_frames": len(codes),
                "coverage": "unassessed",
                "human_listening": "unassessed",
                "execution_state": "observed",
            }
        )
        try:
            if raw.stat().st_size != row["samples"] * 4:
                raise ValueError("raw PCM byte count does not match terminal outcome")
            row["raw"] = identity(raw)
            row["wav"] = make_wav(raw)
        except (OSError, ValueError) as error:
            row["artifact_error"] = str(error)
        row["duration_proxy"] = (
            not row["short_valid"]
            and row["status"] == 0
            and row["finish_reason"] == 1
            and not row["invalid_pcm"]
            and not row["capacity_exceeded"]
            and "artifact_error" not in row
            and row["frames"] <= 12
        )
        rows.append(row)
    return rows


def dispositions(planned: list[dict], rows: list[dict], reason: str) -> list[dict]:
    return [
        {**row, "execution_state": "observed", "coverage": "unassessed"} for row in rows
    ] + [
        {
            **row,
            "execution_state": "not_run",
            "reason": reason,
            "coverage": "unassessed",
        }
        for row in planned[len(rows) :]
    ]


def summary(rows: list[dict]) -> dict:
    prospective = [row for row in rows if row["kind"] == "prospective"]
    warm = [row for row in rows if not row["first_use_group"]]
    return {
        "observed": len(rows),
        "planned": 142,
        "prospective_observed": len(prospective),
        "prospective_planned": 128,
        "prospective_prior_exposed_observed": sum(
            row["prior_baseline_exposure"] for row in prospective
        ),
        "prospective_not_in_previous_baseline_observed": sum(
            not row["prior_baseline_exposure"] for row in prospective
        ),
        "short_duration_proxy_nonshort_prospective": sum(
            row["duration_proxy"] for row in prospective
        ),
        "proxy_eligible_prospective_observed": sum(
            not row["short_valid"] for row in prospective
        ),
        "engine_errors": sum(row["status"] != 0 for row in rows),
        "max_tokens": sum(row["finish_reason"] == 2 for row in rows),
        "artifact_errors": sum("artifact_error" in row for row in rows),
        "decision_capture_invalid": sum(
            bool(row.get("decision_capture_invalid")) for row in rows
        ),
        "warm_callback_ttfa_ms_all_observed": quantiles(
            [row["chunks"][0]["ms"] for row in warm if row["chunks"]]
        ),
        "warm_all_observed_outcomes": {
            "engine_errors": sum(row["status"] != 0 for row in warm),
            "max_tokens": sum(row["finish_reason"] == 2 for row in warm),
            "invalid_pcm": sum(bool(row["invalid_pcm"]) for row in warm),
        },
        "decision_capture_evidence_errors": sum(
            row.get("decision_capture_evidence_valid") is False for row in rows
        ),
        "coverage": "unassessed",
        "human_listening": "unassessed",
        "physical_playback": "not_run",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "output", "talker", "codec", "prepared-inputs"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--model-family", choices=("RU-stress", "Base"), default="RU-stress")
    parser.add_argument("--dependency-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch the frozen plan; otherwise only prepare it",
    )
    parser.add_argument(
        "--timing-off",
        action="store_true",
        help="Control run without internal timing events",
    )
    parser.add_argument(
        "--trace-frames",
        action="store_true",
        help="Enable bounded AR codes; changes timing instrumentation",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    args.output.mkdir(parents=False, exist_ok=False)
    planned = cases()
    rows = []
    outcome = {"status": "preflight_failed", "evidence_preserved": True}
    try:
        if shutil.disk_usage(args.output).free < MIN_FREE_DISK:
            raise ValueError("at least 2 GiB free disk required")
        if not all(path.is_dir() for path in args.dependency_dir):
            raise ValueError("dependency directory missing")
        prepared = prepare_inputs(args.prepared_inputs, args.output / "inputs")
        gpu = gpu_identity(args.gpu_uuid)
        runtime = set(args.probe.parent.glob("*.dll"))
        for directory in args.dependency_dir:
            runtime.update(
                path
                for path in directory.glob("*.dll")
                if path.name.lower().startswith(("cublas", "cudart", "vcomp"))
            )
        manifest_path = args.output / "corpus-manifest.json"
        write_json(manifest_path, manifest(args, planned))
        env = environment(args.gpu_uuid, args.dependency_dir, args.trace_frames)
        command = [
            str(args.probe.resolve()),
            str(manifest_path.resolve()),
            str((args.output / "corpus").resolve()),
        ]
        write_json(
            args.output / "plan.json",
            {
                "cases": planned,
                "scope": (
                    f"CMP 50HX / {args.model_family} / retained Kraftwerk / frontend OFF / "
                    "N=1 baseline, no prefix"
                ),
                "model_family": args.model_family,
                "coverage": "unassessed",
                "prior_exposure_caveat": (
                    "32/128 corpus seed/text pairs appeared in the earlier "
                    "baseline matrix; the 128 are not all held-out cases. "
                    "The remaining 96 were not in that matrix."
                ),
                "overall_seconds": OVERALL_SECONDS,
                "process_seconds": PROCESS_SECONDS,
                "group_deadline_ms": 60000,
                "duration_proxy_max_frames": 12,
                "observe_decisions": not args.timing_off,
                "trace_frames": args.trace_frames,
                "gpu": gpu,
                "source": source_identity(),
                "prepared_inputs": prepared,
                "source_prepared_inputs": {
                    name: identity(args.prepared_inputs / name)
                    for name in PREPARED_HASHES
                },
                "binaries_models_scripts": [
                    identity(path)
                    for path in (
                        args.probe,
                        args.talker,
                        args.codec,
                        Path(__file__),
                        Path(__file__).with_name("run_eos_research.py"),
                    )
                ],
                "runtime_dlls": [identity(path) for path in sorted(runtime)],
                "runtime_dll_scope": (
                    "all probe siblings; cublas/cudart/vcomp DLLs from explicit "
                    "dependency directories, not a loaded-module audit"
                ),
                "cmake_cache": identity(args.probe.parent / "CMakeCache.txt"),
                "argv": argv if argv is not None else sys.argv[1:],
                "binary_source_caveat": (
                    "checkout identity does not itself prove which source "
                    "produced a binary"
                ),
                "missing_case_caveat": (
                    "not_run means no retained terminal result; after process "
                    "failure the final in-flight group may have started"
                ),
            },
        )
        write_json(
            args.output / "invocation.json",
            {
                "command": command,
                "manifest_sha256": sha(manifest_path),
                "environment": {
                    key: value
                    for key, value in env.items()
                    if key.startswith(("QWEN_", "GGML_", "CUDA_")) or key == "PATH"
                },
            },
        )
        outcome["status"] = "plan_only"
        if args.execute:
            remaining = OVERALL_SECONDS - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("overall budget exhausted before launch")
            if shutil.disk_usage(args.output).free < MIN_FREE_DISK:
                raise ValueError("free disk fell below 2 GiB before launch")
            outcome["watchdog_seconds"] = min(PROCESS_SECONDS, remaining)
            with (
                (args.output / "corpus-stdout.log").open("xb") as stdout,
                (args.output / "corpus-stderr.log").open("xb") as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    cwd=args.probe.parent,
                )
                try:
                    code = wait_for_probe(
                        process,
                        outcome["watchdog_seconds"],
                        args.output / "watchdog.json",
                    )
                except KeyboardInterrupt:
                    wait_for_probe(process, 0, args.output / "interrupt-watchdog.json")
                    raise
            outcome.update(
                {
                    "status": "complete" if code == 0 else "process_failed",
                    "exit_code": code,
                }
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        TimeoutError,
    ) as error:
        outcome.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        if (args.output / "corpus-stderr.log").exists():
            try:
                issues = []
                rows = collect(
                    args.output, planned, args.trace_frames, issues, not args.timing_off
                )
                if issues:
                    outcome.update(
                        {"status": "analysis_failed", "analysis_errors": issues}
                    )
            except (OSError, ValueError, KeyError) as error:
                outcome.update(
                    {"status": "analysis_failed", "analysis_error": str(error)}
                )
        if outcome["status"] == "complete" and len(rows) != len(planned):
            outcome.update(
                {"status": "incomplete", "error": "not all frozen cases retired"}
            )
        outcome["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "run-status.json", outcome)
        write_json(args.output / "all-observations.json", rows)
        write_json(
            args.output / "case-dispositions.json",
            dispositions(planned, rows, outcome["status"]),
        )
        write_json(args.output / "summary.json", summary(rows))
        write_json(
            args.output / "sha256-inventory.json",
            {
                str(path.relative_to(args.output)): identity(path)
                for path in sorted(args.output.rglob("*"))
                if path.is_file()
            },
        )
    print(json.dumps({"status": outcome["status"], **summary(rows)}))
    return 0 if outcome["status"] in ("plan_only", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
