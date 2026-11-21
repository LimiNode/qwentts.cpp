"""Convert the retained pinned checkpoint without downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["native", "speech"])
    parser.add_argument("checkout", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.checkpoint / "model-runtime-manifest.json").read_text())
    assert manifest["revision"] == "0c0e3051f131929182e2c023b9537f8b1c68adfe"
    for artifact in manifest["runtime_files"]:
        source = args.checkpoint / artifact["path"]
        assert source.stat().st_size == artifact["size_bytes"], source
        assert sha256(source) == artifact["sha256"], source
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "kind": args.kind,
        "source_checkpoint": str(args.checkpoint),
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "source_manifest_sha256": sha256(args.checkpoint / "model-runtime-manifest.json"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "safetensors", "gguf")},
        "weights_downloaded": False,
    }
    if args.kind == "native":
        converter = args.checkout / "convert.py"
        spec = importlib.util.spec_from_file_location("native_converter", converter)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = args.output / "qwen-talker-1.7b-customvoice-F32.gguf"
        assert not output.exists(), output
        assert module.convert_talker_base(args.checkpoint, output, "1.7B") == 0
    else:
        reference = args.checkout / "reference" / "qwen3-tts"
        sys.path.insert(0, str(reference))
        import pins

        assert pins.MODELS["1.7b"]["revision"] == manifest["revision"]
        pins.snapshot = lambda pin: str(args.checkpoint)
        converter = reference / "convert.py"
        sys.argv = [str(converter), "1.7b", str(args.output)]
        runpy.run_path(str(converter), run_name="__main__")
        provenance["adapter"] = "pins.snapshot returns already verified local checkpoint; upstream conversion logic unchanged"
        provenance["model_card_sha256"] = sha256(args.checkpoint / "README.md")
    provenance["converter_path"] = str(converter)
    provenance["converter_sha256"] = sha256(converter)
    provenance["adapter_sha256"] = sha256(Path(__file__))
    provenance["outputs"] = [{"path": str(path), "size_bytes": path.stat().st_size,
                              "sha256": sha256(path)} for path in args.output.glob("*.gguf")]
    receipt = args.output / "conversion-provenance.json"
    receipt.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
