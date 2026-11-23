"""Prespecified RU/EN and tiny-reply stress test of the live rollback probe."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from run_eos_corpus import PREPARED_HASHES, environment, gpu_identity, identity, source_identity
from run_eos_recovery import GPU, MODELS, REFERENCE, launch_owned, validate_pcm
from run_eos_research import make_wav, sha, write_json
from run_live_stream_recovery import (
    HOP, MAX_TOKENS, asset_identity, collect, disk_bound, make_manifest,
    positive_finite_float, require_clean_source,
)

CORPUS = {
    "ru-yes": ("Да.", "russian", True),
    "ru-no": ("Нет.", "russian", True),
    "ru-thanks": ("Спасибо.", "russian", True),
    "ru-ready": ("Готово.", "russian", True),
    "en-yes": ("Yes.", "english", True),
    "en-no": ("No.", "english", True),
    "en-hi": ("Hi!", "english", True),
    "en-thanks": ("Thank you.", "english", True),
    "ru-short": ("Я здесь.", "russian", True),
    "ru-new": ("После обеда инженер проверит оборудование и отправит результаты испытаний.", "russian", False),
    "ru-two": ("На улице закончился дождь. Мы открыли окно и продолжили обсуждение нового проекта.", "russian", False),
    "en-new": ("The technician checked the equipment and sent a detailed report before lunch.", "english", False),
}
SEEDS = (1009, 2003, 3001)
CONTINUATION_SEED = 64


def case(label: str, seed: int, mode: str) -> dict:
    return {"text_label": label, "language": CORPUS[label][1], "seed": seed,
            "control_key": f"{label}:{seed}", "mode": mode}


def schedule() -> list[dict]:
    controls = [{**case(label, seed, "baseline"), "repeat": repeat}
                for repeat in range(2) for label in CORPUS for seed in SEEDS]
    replay = [{**case(label, SEEDS[0], "replay"), "snapshot_frame": 1} for label in CORPUS]
    recovery = [{**case(label, seed, "rollback1"), "continuation_seed": CONTINUATION_SEED}
                for label in CORPUS for seed in SEEDS]
    return [{"name": name, "cases": cases} for name, cases in
            (("controls", controls), ("replay", replay), ("recovery", recovery))]


def cells(cases: list[dict], rows: list[dict], reason: str) -> list[dict]:
    return [{"group": index, "case": item,
             "state": ("mechanical_pass" if rows[index].get("mechanical_valid") else "failed")
                       if index < len(rows) else "not_run",
             "reason": ("flushed artifact checks" if rows[index].get("mechanical_valid")
                        else rows[index].get("error", "artifact validation failed"))
                       if index < len(rows) else reason}
            for index, item in enumerate(cases)]


def annotate(rows: list[dict]) -> None:
    for row in rows:
        if not row.get("mechanical_valid"):
            continue
        short = CORPUS[row["case"]["text_label"]][2]
        row["short_request"] = short
        row["potential_valid_short_intervention"] = bool(
            short and row["mode"] == "rollback1" and row["triggered"])
        row["semantic_error_inferred_from_duration"] = False


def failed_artifacts(directory: Path) -> tuple[list[dict], list[str]]:
    """Retain audible partial failures as unassessed evidence, not candidates."""
    artifacts, issues = [], []
    for path in sorted((directory / "probe").glob("group-*-failed-*.f32")):
        try:
            size = path.stat().st_size
            if size % (HOP * 4) or size > MAX_TOKENS * HOP * 4:
                raise ValueError("failed PCM size is not a bounded complete-frame multiple")
            validate_pcm(path, size // (HOP * 4))
            artifacts.append({"raw": identity(path), "wav": make_wav(path), "human_listening": "not_run"})
        except (OSError, ValueError) as error:
            issues.append(f"failed artifact {path.name}: {error}")
    return artifacts, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "probe", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--deadline-monotonic", type=positive_finite_float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error("seconds must be1..1800")
    repo, output = args.repo.resolve(), args.output.resolve()
    output.relative_to(repo / ".tmp/base-acceptance")
    if output.exists():
        raise ValueError("new exclusive output directory required")
    probe, runtime = args.probe.resolve(), args.runtime.resolve()
    if probe.parent != runtime and list(probe.parent.glob("*.dll")):
        raise ValueError("local DLL overrides not allowed")
    stages = schedule()
    required, free = disk_bound(stages), shutil.disk_usage(repo).free
    if args.execute and free < required:
        raise ValueError(f"disk preflight needs {required} bytes including512MiB reserve; have{free}")
    source = source_identity()
    if args.execute:
        require_clean_source(source)
    before = asset_identity(repo, probe, runtime)
    if [item["sha256"] for item in before["models"]] != [value[1] for value in MODELS.values()]:
        raise ValueError("original model identities differ")
    reference = repo / REFERENCE / "inputs"
    if any(sha(reference / name) != expected for name, expected in PREPARED_HASHES.items()):
        raise ValueError("original conditioning differs")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in PREPARED_HASHES:
        shutil.copyfile(reference / name, inputs / name)
    for label, (text, _, _) in CORPUS.items():
        with (inputs / f"{label}.txt").open("xb") as stream:
            stream.write(text.encode("utf-8"))
    inputs_before = [identity(path) for path in sorted(inputs.iterdir())]
    for stage in stages:
        write_json(output / (stage["name"] + "-manifest.json"), make_manifest(repo, inputs, stage["cases"]))
    env = {key: value for key, value in environment(GPU, [runtime], False).items()
           if not key.startswith("QWEN_PREFILL_")}
    write_json(output / "plan.json", {"schedule": stages, "assets_before": before,
        "source": source, "runner": identity(Path(__file__)), "collector": identity(Path(__file__).with_name("run_live_stream_recovery.py")),
        "inputs": inputs_before, "seconds": args.seconds, "deadline_monotonic": args.deadline_monotonic,
        "gpu": gpu_identity(GPU) if args.execute else "not_queried_plan_only",
        "disk_required_bytes": required, "disk_free_bytes": free,
        "environment": {key: value for key, value in env.items() if key.startswith(("QWEN_", "GGML_", "CUDA_"))},
        "prespecified_gate": "72baselines establish36exact repeat pairs;12unchanged replays;36conditional rollback1groups",
        "trigger": "one naturalEOS<=8frames only; this can disturb legitimate tiny replies and is not a detector",
        "scope": "RU-stress/Kraftwerk RU/EN stress corpus; no semantic/listening/detector/latency/release qualification"})
    results = {stage["name"]: {"state": "not_run", "rows": [],
               "cells": cells(stage["cases"], [], "stage not started")}
               for stage in stages}
    write_json(output / "planned-cells.json", results)
    if not args.execute:
        return 0
    controls: dict[str, dict] = {}
    started = time.monotonic()
    deadline = min(started + args.seconds, args.deadline_monotonic or float("inf"))
    interrupted = False
    try:
        for stage in stages:
            name, cases = stage["name"], stage["cases"]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            directory = output / name
            directory.mkdir()
            print(f"START {name}: {len(cases)} groups", flush=True)
            launch_error = None
            with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
                try:
                    code = launch_owned([str(probe), str(output / (name + "-manifest.json")), str(directory / "probe")],
                        runtime, env, out, err, min(600, remaining), directory / "watchdog.json")
                except (OSError, subprocess.SubprocessError) as error:
                    code, launch_error = None, str(error)
                except KeyboardInterrupt:
                    interrupted, code = True, None
            rows, issues = collect(directory, cases, controls, establish_controls=name == "controls")
            annotate(rows)
            failed, failed_issues = failed_artifacts(directory)
            issues.extend(failed_issues)
            if launch_error is not None:
                issues.append("owned probe launch/wait failed: " + launch_error)
            passed = code == 0 and len(rows) == len(cases) and not issues
            if name == "controls" and len(controls) != len(CORPUS) * len(SEEDS):
                issues.append("baseline stage did not establish all36text/seed controls")
                passed = False
            results[name] = {"state": "mechanical_pass" if passed else "failed", "returncode": code,
                "rows": rows, "issues": issues, "failed_artifacts": failed,
                "cells": cells(cases, rows, "no flushed result; watchdog/failure may have interrupted this cell"),
                "missing_groups": max(0, len(cases) - len(rows))}
            write_json(output / (name + "-collected.json"), results[name])
            print(f"DONE {name}: {results[name]['state']} {len(rows)}/{len(cases)}", flush=True)
            if not passed or interrupted:
                break
    finally:
        try:
            after = asset_identity(repo, probe, runtime)
            source_after = source_identity()
            inputs_after = [identity(path) for path in sorted(inputs.iterdir())]
        except (OSError, subprocess.SubprocessError) as error:
            after, source_after, inputs_after = {"observation_error": str(error)}, None, []
        reason = "budget exhausted" if time.monotonic() >= deadline else "earlier gate failed or interrupted"
        for stage in stages:
            if results[stage["name"]]["state"] == "not_run":
                results[stage["name"]]["cells"] = cells(stage["cases"], [], reason)
        write_json(output / "results.json", {"stages": results, "controls": controls,
            "assets_after": after, "assets_unchanged": before == after,
            "source_after": source_after, "source_unchanged": source == source_after,
            "inputs_after": inputs_after, "inputs_unchanged": inputs_before == inputs_after,
            "elapsed_seconds": time.monotonic() - started, "interrupted": interrupted,
            "word_coverage": "unassessed", "human_listening": "not_run", "release": "blocked"})
    return int(interrupted or before != after or source != source_after or inputs_before != inputs_after
               or any(row["state"] != "mechanical_pass" for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
