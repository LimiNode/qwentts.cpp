"""Record model-free structural checks and quantizer provenance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from gguf import GGUFReader


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("quantizer", type=Path)
parser.add_argument("receipt", type=Path)
args = parser.parse_args()
reader = GGUFReader(str(args.output))
counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)
assert counts["Q8_0"] > 0
assert len({tensor.name for tensor in reader.tensors}) == len(reader.tensors)
for tensor in reader.tensors:
    assert tensor.data.nbytes == tensor.n_bytes
result = {
    "output": str(args.output),
    "size_bytes": args.output.stat().st_size,
    "sha256": digest(args.output),
    "quantizer": str(args.quantizer),
    "quantizer_sha256": digest(args.quantizer),
    "tensor_count": len(reader.tensors),
    "tensor_types": dict(counts),
    "scope": "GGUF structure and file identity; no model execution or quality claim",
    "tensors": [{"name": t.name, "shape": t.shape.tolist(),
                 "type": t.tensor_type.name, "bytes": t.n_bytes} for t in reader.tensors],
}
args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "tensors"}, indent=2))
