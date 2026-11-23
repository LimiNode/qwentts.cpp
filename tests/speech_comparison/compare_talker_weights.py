"""Compare native and speech.cpp Talker/CP payloads after explicit renaming."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from gguf import GGUFReader


def counterpart(name):
    name = name.replace("code_pred.", "cp.").replace(".lm_head.", ".head.")
    name = name.replace(".mtp_proj.", ".in_proj.").replace(".output_norm", ".norm")
    name = name.replace(".attn_output", ".attn_o")
    if name.endswith(".weight") and not (".text_proj." in name or ".in_proj." in name):
        name = name.removesuffix(".weight")
    return name


def payload_hash(tensor):
    data = memoryview(tensor.data).cast("B")
    h = hashlib.sha256()
    for at in range(0, len(data), 8 * 1024 * 1024):
        h.update(data[at:at + 8 * 1024 * 1024])
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("native", type=Path)
parser.add_argument("speech", type=Path)
parser.add_argument("receipt", type=Path)
args = parser.parse_args()
native = GGUFReader(str(args.native))
speech = {tensor.name: tensor for tensor in GGUFReader(str(args.speech)).tensors}
rows = []
for tensor in native.tensors:
    name = counterpart(tensor.name)
    other = speech.get(name)
    row = {"native": tensor.name, "speech": name, "native_type": tensor.tensor_type.name}
    if other is None:
        row["status"] = "missing_counterpart"
    else:
        row.update(speech_type=other.tensor_type.name,
                   shape_match=tensor.shape.tolist() == other.shape.tolist(),
                   native_sha256=payload_hash(tensor), speech_sha256=payload_hash(other))
        row["status"] = ("exact" if row["shape_match"] and
                         tensor.tensor_type == other.tensor_type and
                         row["native_sha256"] == row["speech_sha256"] else "different")
    rows.append(row)
counts = dict(Counter(row["status"] for row in rows))
result = {"native": str(args.native), "speech": str(args.speech),
          "scope": "all native Talker and CP tensors; codec excluded because native keeps it in a separate file",
          "counts": counts, "tensors": rows}
args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(counts), flush=True)
raise SystemExit(0 if counts == {"exact": len(rows)} else 1)
