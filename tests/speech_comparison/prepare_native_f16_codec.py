"""Make a native-layout codec with the comparator's F16/F32 storage policy."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import gguf
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def payload_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def half_weight(name, array):
    return (array.ndim >= 2 and name.endswith(".weight") and
            ".vq_" not in name and ".dwconv." not in name)


def canonical(name, data):
    """Map native decoder storage to the comparator's names and axis order."""
    if name.startswith("tok_dec.vq_"):
        match = re.fullmatch(r"tok_dec.vq_(first|rest)\.(\d+)\.codebook", name)
        if match:
            return f"codec.vq.{match[1]}.codebook.{match[2]}", data
        match = re.fullmatch(r"tok_dec.vq_(first|rest)\.output_proj.weight", name)
        if match:
            return f"codec.vq.{match[1]}.out_proj", data[:, :, 0]
    if name.startswith("tok_dec.pre_conv."):
        mapped = name.replace("tok_dec.", "codec.")
        return mapped, np.transpose(data, (2, 0, 1)) if name.endswith("weight") else data
    if name.startswith("tok_dec.pre_tfm."):
        mapped = name.replace("tok_dec.pre_tfm.", "codec.tf.")
        mapped = mapped.replace(".input_proj.", ".in_proj.").replace(".output_proj.", ".out_proj.")
        mapped = mapped.replace(".attn_output", ".attn_o")
        if ".blk." in mapped or mapped == "codec.tf.norm.weight":
            mapped = mapped.removesuffix(".weight")
        return mapped, data
    match = re.fullmatch(r"tok_dec.upsample\.(\d+)\.(.+)", name)
    if match:
        suffix = match[2]
        mapped = f"codec.up.{match[1]}." + suffix.replace("pwconv", "pw").replace("conv.", "tconv.") if suffix.startswith("conv.") else f"codec.up.{match[1]}." + suffix.replace("pwconv", "pw")
        if suffix == "conv.weight":
            data = np.transpose(data, (2, 1, 0))
        elif suffix == "dwconv.weight":
            data = np.transpose(data[:, 0, :])
        return mapped, data
    match = re.fullmatch(r"tok_dec.dec\.(\d+)\.(.+)", name)
    if match:
        index, suffix = int(match[1]), match[2]
        if index in (0, 6):
            mapped = "codec.dec." + ("in_conv." if index == 0 else "out_conv.") + suffix.removeprefix("conv.")
        elif index == 5:
            mapped = "codec.dec.out_snake." + suffix.removeprefix("snake.")
        else:
            mapped = f"codec.dec.blk.{index - 1}." + suffix
            mapped = mapped.replace(".conv_t.", ".tconv.").replace(".act1.", ".snake1.").replace(".act2.", ".snake2.")
        if mapped.endswith(".alpha"):
            data = np.exp(data).reshape(-1)
        elif mapped.endswith(".beta"):
            data = (1.0 / (np.exp(data) + 1e-9)).reshape(-1)
            mapped = mapped.removesuffix(".beta") + ".inv_beta"
        elif mapped.endswith(".weight"):
            data = np.transpose(data, (2, 1, 0) if ".tconv." in mapped else (2, 0, 1))
        return mapped, data
    raise ValueError(name)


parser = argparse.ArgumentParser()
parser.add_argument("checkout", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("speech_f32", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--verify-only", action="store_true")
parser.add_argument("--receipt", type=Path)
args = parser.parse_args()
assert args.verify_only or not args.output.exists(), args.output
manifest = json.loads((args.checkpoint / "model-runtime-manifest.json").read_text())
assert manifest["revision"] == "0c0e3051f131929182e2c023b9537f8b1c68adfe"
for artifact in manifest["runtime_files"]:
    if artifact["path"].startswith("speech_tokenizer/"):
        source = args.checkpoint / artifact["path"]
        assert source.stat().st_size == artifact["size_bytes"]
        assert digest(source) == artifact["sha256"]
original_writer = gguf.GGUFWriter


class StorageWriter(original_writer):
    def add_tensor(self, name, tensor, **kwargs):
        if half_weight(name, tensor):
            tensor = tensor.astype(np.float16)
        return super().add_tensor(name, tensor, **kwargs)


converter = args.checkout / "convert.py"
spec = importlib.util.spec_from_file_location("native_codec_converter", converter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not args.verify_only:
    gguf.GGUFWriter = StorageWriter
    try:
        assert module.convert_tokenizer_12hz(args.checkpoint / "speech_tokenizer", args.output) == 0
    finally:
        gguf.GGUFWriter = original_writer
native = gguf.GGUFReader(str(args.output))
speech = {t.name: t for t in gguf.GGUFReader(str(args.speech_f32)).tensors if t.name.startswith("codec.")}
rows = []
for tensor in native.tensors:
    if not tensor.name.startswith("tok_dec."):
        continue
    name, data = canonical(tensor.name, tensor.data)
    counterpart = speech[name]
    expected = counterpart.data.astype(data.dtype)
    row = {"native": tensor.name, "speech": name, "native_type": tensor.tensor_type.name,
           "canonical_dtype": str(data.dtype), "shape_match": data.shape == expected.shape,
           "native_canonical_sha256": payload_digest(data), "speech_cast_sha256": payload_digest(expected)}
    row["status"] = "exact" if row["shape_match"] and row["native_canonical_sha256"] == row["speech_cast_sha256"] else "different"
    rows.append(row)
assert len(rows) == len(speech), (len(rows), len(speech))
counts = dict(Counter(row["status"] for row in rows))
receipt = {"output": str(args.output), "size_bytes": args.output.stat().st_size, "sha256": digest(args.output),
           "compared_model": str(args.speech_f32), "compared_model_sha256": digest(args.speech_f32),
           "source_checkpoint_revision": manifest["revision"], "source_codec_sha256": digest(args.checkpoint / "speech_tokenizer" / "model.safetensors"),
           "converter": str(converter), "converter_sha256": digest(converter), "adapter_sha256": digest(Path(__file__)),
           "storage_policy": "matrices/convolutions F16 except codebooks, their projections and depthwise convolution; remaining vectors F32",
           "parity_scope": "all decoder weights, native axis/Snake parameter transformations canonicalized and compared with speech F32 cast to matched storage; encoder excluded",
           "counts": counts, "tensors": rows}
receipt_path = args.receipt or args.output.with_suffix(".provenance.json")
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"path": str(args.output), "sha256": receipt["sha256"], "counts": counts}), flush=True)
raise SystemExit(0 if counts == {"exact": len(rows)} else 1)
