"""Verify an expected prefix mismatch fails before synthesis; not a speech benchmark."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from run_eos_corpus import environment, gpu_identity, identity, source_identity
from run_eos_research import wait_for_probe, write_json
from run_prefill_research import GPU, collect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "probe", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prefix-rows", type=int, required=True)
    parser.add_argument("--dependency-directory", type=Path, action="append", default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(args.repo.resolve() / ".tmp" / "base-acceptance")
    if not 5 < args.prefix_rows < 4096:
        parser.error("requires a discovered same-language prefix longer than five rows")
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    original = Path(source["groups"][0]["text_file"])
    source.update(max_tokens=4, observe_decisions=False, observe_prefill=True, groups=[
        {"text_file": str(original), "language": "russian", "seeds": [64]},
        {"text_file": str(original.parent / "ru-yes.txt"), "language": "russian", "seeds": [64]},
        {"text_file": str(original.parent / "en-unseen.txt"), "language": "english", "seeds": [64]},
    ])
    gpu = gpu_identity(GPU)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", source)
    env = environment(GPU, [args.probe.parent, *args.dependency_directory], False)
    env = {k: v for k, v in env.items() if not k.startswith("QWEN_PREFILL_")}
    env.update(QWEN_PREFILL_PROFILE="1", QWEN_PREFILL_PREFIX_SOURCE="full",
               QWEN_PREFILL_PREFIX_ROWS=str(args.prefix_rows))
    write_json(output / "plan.json", {"gpu": gpu, "source": source_identity(),
               "probe": identity(args.probe), "source_manifest": identity(args.manifest),
               "prefix_rows": args.prefix_rows, "expected": "two bounded results, then language-prefix rejection",
               "scope": "max-token diagnostic; no natural-EOS or speech-quality assertion"})
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        process = subprocess.Popen([str(args.probe.resolve()), str(output / "manifest.json"), str(output / "probe")],
                                   stdout=out, stderr=err, env=env, cwd=args.probe.parent)
        try:
            returncode = wait_for_probe(process, 60, output / "watchdog.json")
        except subprocess.TimeoutExpired:
            returncode = 124
    planned = [dict(text_label=label, seed=64, kind="negative_control")
               for label in ("original", "ru-yes", "en-unseen")]
    rows, issues = collect(output, planned, False)
    diagnostic = (output / "stderr.log").read_text(encoding="utf-8", errors="replace")
    passed = (returncode == 1 and len(rows) == 2 and not issues
              and rows[0]["prefill_observation"]["prefix_hit"] is False
              and rows[1]["prefill_observation"]["prefix_hit"] is True
              and "research prefix embedding mismatch: cached rows are not invariant" in diagnostic
              and not (output / "probe/group-2-lane-0.f32").exists())
    write_json(output / "collected.json", {"expected_rejection_pass": passed, "returncode": returncode,
               "rows": rows, "issues": issues,
               "not_tested": "post-rejection recovery/poisoned-handle admission; the probe exits at first rejection"})
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
