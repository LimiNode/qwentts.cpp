"""Report the first free predictor-token mismatch in paired dump directories.

The predictor graph uses one sampler pass per acoustic codebook.  This tool
keeps that mapping explicit so a downstream mismatch is not accidentally
reported as the causal boundary:

``frame * 16 + codebook`` is the Philox subsequence for predictor codebook
``codebook`` (where codebook 1 is graph step 0).  The corresponding graph
step is therefore ``codebook - 1``.

The input dumps use the repository's small rank/shape/f32 container format;
no torch or numpy dependency is required.  JSON output is intended to be
attached to hardware evidence and remains useful when optional logits dumps
are absent.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


def load_f32_dump(path: Path) -> tuple[tuple[int, ...], list[float]]:
    """Load a qwentts diagnostic dump and return its shape and flat values."""
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"{path} is truncated before its rank")
    ndim = struct.unpack_from("<i", raw, 0)[0]
    header_size = 4 + 4 * ndim
    if ndim < 0 or len(raw) < header_size:
        raise ValueError(f"{path} has an invalid shape header")
    shape = struct.unpack_from(f"<{ndim}i", raw, 4)
    payload = raw[header_size:]
    if len(payload) % 4:
        raise ValueError(f"{path} has a non-f32 payload")
    values = list(struct.unpack(f"<{len(payload) // 4}f", payload))
    expected = math.prod(shape)
    if expected != len(values):
        raise ValueError(f"{path} shape {shape} expects {expected} values, got {len(values)}")
    return tuple(shape), values


def _codes(path: Path) -> tuple[tuple[int, ...], list[list[int]]]:
    shape, values = load_f32_dump(path)
    if len(shape) != 2:
        raise ValueError(f"{path} must be a rank-2 code dump, got {shape}")
    rows, groups = shape
    return shape, [
        [int(round(values[row * groups + group])) for group in range(groups)]
        for row in range(rows)
    ]


def _summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def predictor_dump_name(kind: str, frame: int, step: int) -> str:
    """Return the native/Python predictor dump name for a frame and step."""
    if frame == 0:
        return f"predictor-{kind}-step{step}.bin"
    return f"predictor-{kind}-frame{frame}-step{step}.bin"


def _metric(native: Path, python: Path) -> dict[str, float] | None:
    if not native.is_file() or not python.is_file():
        return None
    shape_a, a = load_f32_dump(native)
    shape_b, b = load_f32_dump(python)
    if shape_a != shape_b:
        raise ValueError(
            f"paired metric dumps must have identical shapes, got {shape_a} and {shape_b}"
        )
    n = len(a)
    if not n:
        return {"cosine": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
    aa = a[:n]
    bb = b[:n]
    dot = sum(x * y for x, y in zip(aa, bb))
    norm_a = math.sqrt(sum(x * x for x in aa))
    norm_b = math.sqrt(sum(y * y for y in bb))
    cosine = dot / (norm_a * norm_b) if norm_a > 1e-12 and norm_b > 1e-12 else 0.0
    cosine = max(-1.0, min(1.0, cosine))
    del shape_a, shape_b
    deltas = [abs(x - y) for x, y in zip(aa, bb)]
    return {
        "cosine": cosine,
        "max_abs": max(deltas),
        "mean_abs": sum(deltas) / len(deltas),
    }


def analyze(native_dir: Path, python_dir: Path) -> dict[str, Any]:
    """Build a machine-readable first-mismatch report for paired runs."""
    native_shape, native = _codes(native_dir / "codes-full.bin")
    python_shape, python = _codes(python_dir / "codes-full.bin")
    if native_shape != python_shape:
        raise ValueError(
            f"paired code dumps must have identical shapes, got {native_shape} and {python_shape}"
        )
    common_frames = len(native)
    common_groups = native_shape[1]

    mismatch: dict[str, Any] | None = None
    for frame in range(common_frames):
        for codebook in range(common_groups):
            if native[frame][codebook] != python[frame][codebook]:
                mismatch = {
                    "frame": frame,
                    "codebook": codebook,
                    "native_token": native[frame][codebook],
                    "python_token": python[frame][codebook],
                }
                if codebook > 0:
                    mismatch.update(
                        graph_step=codebook - 1,
                        philox_subsequence=frame * 16 + codebook,
                    )
                break
        if mismatch is not None:
            break

    report: dict[str, Any] = {
        "native_shape": list(native_shape),
        "python_shape": list(python_shape),
        "common_frames": common_frames,
        "common_codebooks": common_groups,
        "first_mismatch": mismatch,
    }
    if mismatch is None or mismatch["codebook"] == 0:
        return report

    frame = mismatch["frame"]
    step = mismatch["graph_step"]
    native_prefix = predictor_dump_name("logits", frame, step)
    python_prefix = native_prefix
    report["predictor_logits"] = _metric(native_dir / native_prefix, python_dir / python_prefix)
    native_hidden = predictor_dump_name("hidden", frame, step)
    report["predictor_hidden"] = _metric(native_dir / native_hidden, python_dir / native_hidden)
    native_summary_path = native_dir / f"sampler-frame{frame}-step{step}.txt"
    python_summary_path = python_dir / "sampler-diagnostic.txt"
    native_summary = _summary(native_summary_path)
    python_summary = _summary(python_summary_path)
    report["sampler_target"] = {
        "frame": frame,
        "graph_step": step,
        "philox_subsequence": frame * 16 + 1 + step,
        "native_summary_path": native_summary_path.name,
        "native_summary_present": bool(native_summary),
        "python_summary_path": python_summary_path.name,
        "python_summary_present": bool(python_summary),
        "python_summary_matches_target": (
            python_summary.get("frame") == str(frame)
            and python_summary.get("predictor_step") == str(step)
        ),
    }
    report["native_sampler_summary"] = native_summary
    report["python_sampler_summary"] = python_summary
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--python-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="also write the report to this path")
    args = parser.parse_args()
    report = analyze(args.native_dir, args.python_dir)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.json:
        args.json.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
