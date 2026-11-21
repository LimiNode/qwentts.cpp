"""Compare bounded layer/KV captures; existing NumPy is used only for offline analysis."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from run_eos_corpus import identity
from run_eos_research import write_json

MAX_TENSOR_BYTES = 32 * 1024**2
STAGES = ("input", "attn-norm", "q-proj", "k-proj", "v-proj", "q-rope", "k-rope",
          "attn-output", "o-proj", "attn-residual", "mlp-norm", "mlp-residual")
VIEW_OPS = {"VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
BLOCKS = ("baseline-off", "prefix-off", "baseline-taps", "prefix-taps")
LABELS = ("original", "original", "ru-yes", "original")


def tensor_header(data: bytes, path: Path) -> tuple[tuple[int, ...], int]:
    """Validate bounded little-endian headers before unpacking or allocating."""
    if not 8 <= len(data) <= MAX_TENSOR_BYTES:
        raise ValueError(f"tensor size outside diagnostic bounds: {path}")
    dimensions = struct.unpack_from("<i", data)[0]
    if not 1 <= dimensions <= 4:
        raise ValueError(f"invalid dimension count: {path}")
    offset = 4 * (dimensions + 1)
    if len(data) < offset:
        raise ValueError(f"truncated tensor header: {path}")
    shape = struct.unpack_from("<" + "i" * dimensions, data, 4)
    if any(n <= 0 for n in shape) or len(data) - offset != math.prod(shape) * 4:
        raise ValueError(f"invalid shape/payload: {path}")
    return shape, offset


def require_numpy() -> None:
    if np is None:
        raise RuntimeError("offline tensor comparison requires an existing NumPy installation")


def tensor(path: Path) -> np.ndarray:
    if path.stat().st_size > MAX_TENSOR_BYTES:
        raise ValueError(f"tensor size outside diagnostic bounds: {path}")
    data = path.read_bytes()
    shape, offset = tensor_header(data, path)
    require_numpy()
    result = np.frombuffer(data, dtype="<f4", offset=offset).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite diagnostic tensor: {path}")
    return result


def difference(left: np.ndarray, right: np.ndarray) -> dict:
    require_numpy()
    if left.shape != right.shape:
        raise ValueError(f"unpaired tensor shapes {left.shape}/{right.shape}")
    if not left.size or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("empty or nonfinite diagnostic tensor comparison")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {"shape": list(left.shape), "byte_equal": left.tobytes() == right.tobytes(),
            "different_values": int(np.count_nonzero(left != right)),
            "max_abs": float(np.max(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta * delta)))}


def bounded_int(value: object, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def validate_results(results: dict) -> None:
    """Do not turn partial, failed, reordered or unmatched runs into evidence."""
    if not isinstance(results, dict) or set(results) != set(BLOCKS):
        raise ValueError("requires all four diagnostic blocks")
    for name in BLOCKS:
        block = results[name]
        if (not isinstance(block, dict) or block.get("state") != "completed"
                or type(block.get("returncode")) is not int or block["returncode"] != 0
                or block.get("issues") != []):
            raise ValueError(f"incomplete or failed diagnostic block: {name}")
        rows = block.get("rows")
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError(f"requires the complete four-case diagnostic: {name}")
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or row.get("group") != index
                    or not bounded_int(row.get("group"), 0, 3) or not bounded_int(row.get("lane"), 0, 0)
                    or not bounded_int(row.get("seed"), 64, 64) or row.get("text_label") != LABELS[index]
                    or not bounded_int(row.get("status"), 0, 0) or not bounded_int(row.get("finish_reason"), 1, 2)
                    or not bounded_int(row.get("frames"), 1, 4)
                    or not bounded_int(row.get("samples"), 1, 7680) or row["samples"] != row["frames"] * 1920
                    or any(row.get(key) is not False for key in
                           ("invalid_pcm", "capacity_exceeded", "decision_capture_invalid"))):
                raise ValueError(f"invalid or unpaired diagnostic row: {name}/{index}")
            obs = row.get("prefill_observation")
            if (not isinstance(obs, dict) or not bounded_int(obs.get("T"), 1, 512)
                    or not bounded_int(obs.get("n_past"), 0, 511)
                    or obs["T"] + obs["n_past"] > 512
                    or obs.get("diagnostic") is not name.endswith("-taps")):
                raise ValueError(f"invalid effective observation: {name}/{index}")
            if name.endswith("-taps") and not bounded_int(obs.get("diagnostic_serial"), index + 1, index + 1):
                raise ValueError(f"invalid diagnostic serial: {name}/{index}")
            if name.startswith("prefix"):
                if (obs.get("prefix_source") != "full" or obs.get("prefix_hit") is not (index > 0)
                        or not bounded_int(obs.get("prefix_rows"), 1, 511)
                        or obs["n_past"] != (obs["prefix_rows"] if index else 0)):
                    raise ValueError(f"invalid prefix observation: {name}/{index}")
            elif obs["n_past"] != 0:
                raise ValueError(f"baseline has cached context: {name}/{index}")
    for index in range(4):
        total = results["baseline-off"]["rows"][index]["prefill_observation"]["T"]
        for name in BLOCKS:
            obs = results[name]["rows"][index]["prefill_observation"]
            if obs["T"] + obs["n_past"] != total:
                raise ValueError(f"unpaired prompt shape: {name}/{index}")
            if name.startswith("prefix") and obs["prefix_rows"] != results["prefix-off"]["rows"][0]["prefill_observation"]["prefix_rows"]:
                raise ValueError(f"unpaired prefix boundary: {name}/{index}")


def contained_file(root: Path, path: Path) -> Path:
    """Reject symlink escapes as well as lexical traversal in a tap filename."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"capture escapes diagnostic root: {path}") from exc
    if not path.is_file():
        raise ValueError(f"missing diagnostic capture: {path}")
    return path


def load_taps(root: Path, directory: Path) -> list[dict]:
    path = contained_file(root, directory / "taps.jsonl")
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError(f"tap metadata exceeds diagnostic bounds: {path}")
    taps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            tap = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid tap metadata: {path}") from exc
        if (not isinstance(tap, dict) or not bounded_int(tap.get("layer"), 0, 127)
                or tap.get("stage") not in STAGES or not isinstance(tap.get("op"), str)
                or not tap["op"]):
            raise ValueError(f"invalid tap identity: {path}")
        name = tap.get("file")
        expected = f"layer-{tap['layer']:02d}-{tap['stage']}.bin"
        if (not isinstance(name, str) or PureWindowsPath(name).drive or "\\" in name
                or PurePosixPath(name).name != name or name != expected):
            raise ValueError(f"unsafe or mismatched tap filename: {name!r}")
        contained_file(root, directory / name)
        taps.append(tap)
    layers = sorted({tap["layer"] for tap in taps})
    if not layers or [(t["layer"], t["stage"]) for t in taps] != [
            (layer, stage) for layer in layers for stage in STAGES]:
        raise ValueError(f"incomplete, duplicate or reordered tap set: {path}")
    return taps


def tap_unusable_reason(left: dict, right: dict) -> str | None:
    for tap in (left, right):
        if tap["op"] in VIEW_OPS and tap.get("output_retention") != "view_and_backing_chain":
            return "legacy view capture has no retained backing allocation; values are unusable"
    return None


def native_kv(root: Path, path: Path) -> tuple[str, tuple[int, ...], bytes]:
    """A missing/truncated raw snapshot must not compare equal to another empty file."""
    metadata = json.loads(contained_file(root, path.with_suffix(".json")).read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or metadata.get("type") not in ("f16", "f32")
            or metadata.get("packed_head_major") is not True
            or any(not bounded_int(metadata.get(key), 1, 4096) for key in ("heads", "rows", "head_dim"))):
        raise ValueError(f"invalid native KV metadata: {path}")
    shape = tuple(metadata[key] for key in ("heads", "rows", "head_dim"))
    expected = math.prod(shape) * (2 if metadata["type"] == "f16" else 4)
    raw = contained_file(root, path.with_suffix(".raw"))
    if (expected > MAX_TENSOR_BYTES or not bounded_int(metadata.get("raw_bytes"), expected, expected)
            or raw.stat().st_size != expected):
        raise ValueError(f"invalid native KV payload size: {path}")
    tensor_bytes = 16 + math.prod(shape) * 4
    if tensor_bytes > MAX_TENSOR_BYTES or contained_file(root, path).stat().st_size != tensor_bytes:
        raise ValueError(f"invalid native KV tensor payload size: {path}")
    if tensor_header(path.read_bytes(), path)[0] != shape:
        raise ValueError(f"native KV metadata disagrees with tensor shape: {path}")
    return metadata["type"], shape, raw.read_bytes()


def analyze(root: Path) -> dict:
    root = root.resolve()
    results = json.loads(contained_file(root, root / "results.json").read_text(encoding="utf-8"))
    validate_results(results)
    require_numpy()
    output = {"source_results": identity(root / "results.json"), "numpy": np.__version__,
              "timing_eligible": False, "tap_comparison_complete": True,
              "first_difference_scope": "first usable captured tap; excludes unusable legacy view values",
              "cases": [], "captures": []}
    baseline = results["baseline-taps"]["rows"]
    candidate = results["prefix-taps"]["rows"]
    for index, (original, cached) in enumerate(zip(baseline, candidate)):
        obs = cached["prefill_observation"]
        baseline_dir = root / "baseline-taps" / "taps" / f"prefill-{original['prefill_observation']['diagnostic_serial']:04d}"
        candidate_dir = root / "prefix-taps" / "taps" / f"prefill-{obs['diagnostic_serial']:04d}"
        offset = obs["n_past"]
        taps = load_taps(root, candidate_dir)
        baseline_taps = load_taps(root, baseline_dir)
        if [(t["layer"], t["stage"]) for t in taps] != [(t["layer"], t["stage"]) for t in baseline_taps]:
            raise ValueError("unpaired baseline/candidate tap identities")
        case = {"group": index, "label": cached["text_label"], "seed": cached["seed"], "n_past": offset,
                "T": obs["T"], "layer_taps": [], "unusable_taps": [], "kv": [], "first_difference": None}
        for control, tap in zip(baseline_taps, taps):
            reason = tap_unusable_reason(control, tap)
            if reason:
                case["unusable_taps"].append({"layer": tap["layer"], "stage": tap["stage"], "reason": reason})
                output["tap_comparison_complete"] = False
                continue
            left_path, right_path = baseline_dir / tap["file"], candidate_dir / tap["file"]
            left, right = tensor(left_path), tensor(right_path)
            if left.ndim != 2 or right.ndim != 2 or left.shape[0] != obs["T"] + offset or right.shape[0] != obs["T"]:
                raise ValueError("tap tensor shape disagrees with effective prompt rows")
            measured = difference(left[offset:], right)
            item = {"layer": tap["layer"], "stage": tap["stage"], **measured}
            case["layer_taps"].append(item)
            if not measured["byte_equal"] and case["first_difference"] is None:
                case["first_difference"] = item
        for layer, part in ((layer, part) for layer in sorted({tap["layer"] for tap in taps}) for part in ("k", "v")):
            after = contained_file(root, candidate_dir / f"layer-{layer:02d}-live-after-{part}.bin")
            base_after = contained_file(root, baseline_dir / after.name)
            kv = {"layer": layer, "part": part,
                  "baseline_current_vs_candidate_after": difference(tensor(base_after), tensor(after))}
            if offset:
                live = contained_file(root, candidate_dir / f"layer-{layer:02d}-live-prefix-before-{part}.bin")
                saved = contained_file(root, candidate_dir / f"layer-{layer:02d}-saved-prefix-before-{part}.bin")
                source = contained_file(root, root / "prefix-taps" / "taps" / "prefill-0001" /
                                        f"layer-{layer:02d}-saved-after-snapshot-{part}.bin")
                source_live = contained_file(root, source.with_name(f"layer-{layer:02d}-live-after-{part}.bin"))
                kv.update({
                    "snapshot_vs_original_live": difference(tensor(source_live)[:, :offset], tensor(source)),
                    "saved_still_original": native_kv(root, source) == native_kv(root, saved),
                    "restore_native_byte_exact": native_kv(root, live) == native_kv(root, saved),
                    "prefix_unchanged_after_suffix": difference(tensor(live), tensor(after)[:, :offset]),
                    "baseline_current_prefix_vs_restored": difference(tensor(base_after)[:, :offset], tensor(live)),
                })
            case["kv"].append(kv)
        output["cases"].append(case)
    for directory in (root / "baseline-taps" / "taps", root / "prefix-taps" / "taps"):
        output["captures"].extend(identity(contained_file(root, p)) for p in sorted(directory.rglob("*")) if p.is_file())
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output = analyze(args.directory.resolve())
    write_json(args.directory / "layer-analysis.json", output)
    for case in output["cases"]:
        first = case["first_difference"]
        restores = [kv["restore_native_byte_exact"] for kv in case["kv"] if "restore_native_byte_exact" in kv]
        print(json.dumps({"group": case["group"], "label": case["label"], "first_difference": first,
                          "unusable_taps": len(case["unusable_taps"]),
                          "restores_exact": sum(restores), "restores_checked": len(restores)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
