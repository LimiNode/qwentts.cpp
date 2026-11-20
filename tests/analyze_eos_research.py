"""Offline descriptive analysis; duration and timing are not speech acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Row = dict[str, Any]
PREFIX = re.compile(r"seed-(\d+)-frames-(\d+)")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: list[float]) -> Row:
    values = sorted(values)
    return {
        "n": len(values),
        "p50": statistics.median(values) if values else None,
        "p95_nearest_rank": values[math.ceil(len(values) * 0.95) - 1]
        if values
        else None,
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
    }


def outcome(row: Row) -> str:
    """Use terminal state only for retrospective scoring, never live selection."""
    if row["status"] != 0 or row["invalid_pcm"] or row["capacity_exceeded"]:
        return "error_or_cancelled"
    if row["finish_reason"] == 2:
        return "max_tokens"
    if row["finish_reason"] != 1:
        return "other_non_natural_terminal"
    if row["samples"] == 0:
        return "empty_pcm"
    if row["text_label"] == "ru-short":
        return "natural_eos_short_text"
    return (
        "natural_eos_short_duration_proxy"
        if row["frames"] <= 12
        else "natural_eos_above_duration_proxy"
    )


def identity(row: Row) -> Row:
    return {
        key: row[key]
        for key in ("run", "group", "lane", "text_label", "seed", "prefix")
    }


def first_rows(rows: list[Row]) -> dict[tuple[str, int], Row]:
    first = {}
    for row in rows:
        first.setdefault((row["text_label"], row["seed"]), row)
    return first


def compare(a: Row, b: Row) -> Row:
    """Compare all retained complete codebook frames and full raw-f32 PCM."""
    complete = (
        a["status"] == b["status"] == 0
        and len(a["codes"]) == a["frames"]
        and len(b["codes"]) == b["frames"]
    )
    shared = 0
    for left, right in zip(a["codes"], b["codes"], strict=False):
        if left != right:
            break
        shared += 1
    pcm_known = a.get("verified_pcm_sha256") and b.get("verified_pcm_sha256")
    return {
        "a": identity(a),
        "b": identity(b),
        "a_outcome": outcome(a),
        "b_outcome": outcome(b),
        "a_frames": a["frames"],
        "b_frames": b["frames"],
        "a_samples": a["samples"],
        "b_samples": b["samples"],
        "retained_complete_frames_equal": a["codes"] == b["codes"],
        "retained_trace_covers_both_generations": complete,
        "common_complete_frame_prefix_length": shared,
        "full_pcm_equal": bool(
            a["verified_pcm_sha256"] == b["verified_pcm_sha256"]
            and a["samples"] == b["samples"]
        )
        if pcm_known
        else None,
    }


def timing(row: Row) -> Row:
    """No terminal field influences this hypothetical step-four timing test."""
    chunks = row["chunks"]
    steps = row["steps"]
    first = chunks[0] if chunks else None
    fourth = steps[3]["ms"] if len(steps) >= 4 else None
    budget = first["samples"] / 24 if first else None
    deadline = first["ms"] + budget if first else None
    return {
        "text_eligible": row["text_label"] != "ru-short",
        "callback_ttfa_ms": first["ms"] if first else None,
        "first_chunk_audio_ms": budget,
        "step_four_completion_ms": fourth,
        "step_four_after_first_pcm_ms": fourth - first["ms"]
        if fourth is not None and first
        else None,
        "step_four_within_first_chunk_budget": fourth <= deadline
        if fourth is not None and deadline is not None
        else None,
        "step_four_budget_margin_ms": deadline - fourth
        if fourth is not None and deadline is not None
        else None,
        "first_to_second_callback_ms": chunks[1]["ms"] - first["ms"]
        if len(chunks) >= 2
        else None,
        "group_span_per_lane_audio_rtf": row["group_total_ms"] / (row["samples"] / 24)
        if row["samples"]
        else None,
    }


def asset_path(root: Path, row: Row) -> Path:
    path = (
        root / row["run"] / f"group-{row['group']}-lane-{row['lane']}.f32"
    ).resolve()
    if not path.is_relative_to(root):
        raise ValueError("PCM asset escaped matrix directory")
    return path


def load_rows(root: Path, plan: Row) -> tuple[list[Row], list[Row], dict[str, str]]:
    complete = root / "all-observations.json"
    hashes = {}
    if complete.exists():
        rows = read_json(complete)
        hashes[complete.name] = sha(complete)
    else:
        rows = []
        for run in plan["runs"]:
            path = root / (run["name"] + "-analysis.json")
            if path.exists():
                rows.extend(read_json(path))
                hashes[path.name] = sha(path)
    missing = []
    for run in plan["runs"]:
        observed = [row for row in rows if row["run"] == run["name"]]
        expected = sum(len(group["seeds"]) for group in run["groups"])
        if len(observed) != expected:
            raw = root / run["name"] / "results.jsonl"
            watchdog = root / (run["name"] + "-watchdog.json")
            missing.append(
                {
                    "run": run["name"],
                    "planned_lanes": expected,
                    "analyzed_lanes": len(observed),
                    "raw_results_exist": raw.exists(),
                    "raw_results_are_not_scored_as_success": True,
                    "watchdog": read_json(watchdog) if watchdog.exists() else None,
                }
            )
    return rows, missing, hashes


def paired_reseed(rows: list[Row]) -> list[Row]:
    baseline = first_rows([row for row in rows if row["run"] == "independent-baseline"])
    reports = []
    for name in ("per-frame", "early-2", "early-4"):
        changed = first_rows(
            [row for row in rows if row["run"] == "independent-" + name]
        )
        common = baseline.keys() & changed.keys()
        pairs = [compare(baseline[key], changed[key]) for key in common]
        reports.append(
            {
                "policy": name,
                "first_occurrence_only": True,
                "paired_n": len(pairs),
                "unpaired_baseline": sorted(set(baseline) - set(changed)),
                "unpaired_policy": sorted(set(changed) - set(baseline)),
                "by_text": [
                    {
                        "text": label,
                        "n": len(subset),
                        "transitions": dict(
                            Counter(
                                pair["a_outcome"] + " -> " + pair["b_outcome"]
                                for pair in subset
                            )
                        ),
                    }
                    for label in sorted({key[0] for key in common})
                    if (
                        subset := [
                            pair for pair in pairs if pair["a"]["text_label"] == label
                        ]
                    )
                ],
                "pairs": sorted(
                    pairs, key=lambda pair: (pair["a"]["text_label"], pair["a"]["seed"])
                ),
            }
        )
    return reports


def repeated_runs(rows: list[Row]) -> list[Row]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run"], row["text_label"], row["seed"])].append(row)
    return [
        {
            "run": key[0],
            "text": key[1],
            "seed": key[2],
            "observations": len(values),
            "comparisons_to_first": [
                {
                    **compare(values[0], other),
                    "same_group_different_lane": other["group"] == values[0]["group"],
                }
                for other in values[1:]
            ],
        }
        for key, values in sorted(grouped.items())
        if len(values) > 1
    ]


def prefix_outcomes(root: Path, rows: list[Row]) -> list[Row]:
    result = []
    for prefix in sorted({row["prefix"] for row in rows if row["prefix"]}):
        match = PREFIX.fullmatch(prefix)
        if not match:
            raise ValueError("unknown prefix format")
        prefix_seed, frames = map(int, match.groups())
        path = root / "inputs" / (prefix + ".txt")
        values = [int(value) for value in path.read_text().split()]
        if len(values) != frames * 16:
            raise ValueError("malformed complete-frame prefix")
        expected = [
            values[offset : offset + 16] for offset in range(0, len(values), 16)
        ]
        subset = [row for row in rows if row["prefix"] == prefix]
        modes = []
        for width in (1, 2):
            selected = [row for row in subset if row["admitted_together"] == width]
            for control in (True, False):
                selected_class = [
                    row for row in selected if (row["seed"] == prefix_seed) == control
                ]
                unique = list(first_rows(selected_class).values())
                modes.append(
                    {
                        "lanes": width,
                        "class": "self_seed_replay"
                        if control
                        else "other_seed_continuation",
                        "all_n": len(selected_class),
                        "all_outcomes": dict(Counter(map(outcome, selected_class))),
                        "first_seed_occurrences_n": len(unique),
                        "first_seed_occurrences_outcomes": dict(
                            Counter(map(outcome, unique))
                        ),
                        "prefix_mismatches": [
                            identity(row)
                            for row in selected_class
                            if row["codes"][:frames] != expected
                        ],
                    }
                )
        result.append(
            {
                "prefix": prefix,
                "frames": frames,
                "input_sha256": sha(path),
                "modes": modes,
            }
        )
    return result


def batch_comparisons(root: Path, rows: list[Row]) -> Row:
    same_seed = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run"], row["group"])].append(row)
    first_pcm = []
    transitions = []
    for key, lanes in sorted(groups.items()):
        if len(lanes) != 2:
            continue
        lanes.sort(key=lambda row: row["lane"])
        a, b = lanes
        if a["prefix"]:
            counts = [
                row["chunks"][0]["samples"] if row["chunks"] else 0 for row in lanes
            ]
            same_count = counts[0] == counts[1] and counts[0] > 0
            payloads = []
            for row, count in zip(lanes, counts, strict=True):
                path = asset_path(root, row)
                if path.exists():
                    with path.open("rb") as stream:
                        payloads.append(stream.read(count * 4))
                else:
                    payloads.append(None)
            first_pcm.append(
                {
                    "a": identity(a),
                    "b": identity(b),
                    "first_chunk_samples": counts,
                    "same_sample_count": same_count,
                    "raw_f32_first_chunk_identical": payloads[0] == payloads[1]
                    if same_count and all(value is not None for value in payloads)
                    else None,
                }
            )
        previous = 2
        for index, step in enumerate(a["steps"]):
            if previous == 2 and step["active"] == 1:
                natural = [
                    row["lane"]
                    for row in lanes
                    if row["status"] == 0
                    and row["finish_reason"] == 1
                    and row["frames"] + 1 == index + 1
                ]
                transitions.append(
                    {
                        "run": key[0],
                        "group": key[1],
                        "step_one_based": index + 1,
                        "ms": step["ms"],
                        "step_span_ms": step["ms"]
                        - (a["steps"][index - 1]["ms"] if index else a["admitted_ms"]),
                        "next_step_span_ms": a["steps"][index + 1]["ms"] - step["ms"]
                        if len(a["steps"]) > index + 1
                        else None,
                        "natural_eos_retiring_lanes_inferred_from_frame_count": natural,
                        "selected_loser_cancelled": False,
                    }
                )
            previous = step["active"]
    for prefix in sorted({row["prefix"] for row in rows if row["prefix"]}):
        single = first_rows(
            [
                row
                for row in rows
                if row["prefix"] == prefix and row["admitted_together"] == 1
            ]
        )
        dual = first_rows(
            [
                row
                for row in rows
                if row["prefix"] == prefix and row["admitted_together"] == 2
            ]
        )
        same_seed.extend(
            compare(single[key], dual[key])
            for key in sorted(single.keys() & dual.keys())
        )
    return {
        "single_vs_dual_first_same_seed": same_seed,
        "dual_forced_prefix_first_pcm": first_pcm,
        "natural_or_other_retirement_width_two_to_one": transitions,
        "no_branch_selection_or_state_fork_was_implemented": True,
    }


def timing_summaries(rows: list[Row]) -> list[Row]:
    result = []
    for name in sorted({row["run"] for row in rows}):
        selected = [row for row in rows if row["run"] == name]
        warm = [row for row in selected if not row["first_use_group"]]
        measured = [timing(row) for row in warm]
        group_rows = defaultdict(list)
        for row in warm:
            group_rows[row["group"]].append(row)
        result.append(
            {
                "run": name,
                "all_outcomes": dict(Counter(map(outcome, selected))),
                "warm_lane_count": len(warm),
                "first_use_lane_count": len(selected) - len(warm),
                "warm_callback_ttfa_ms": quantiles(
                    [
                        row["callback_ttfa_ms"]
                        for row in measured
                        if row["callback_ttfa_ms"] is not None
                    ]
                ),
                "warm_group_span_per_lane_audio_rtf": quantiles(
                    [
                        row["group_span_per_lane_audio_rtf"]
                        for row in measured
                        if row["group_span_per_lane_audio_rtf"] is not None
                    ]
                ),
                "warm_group_span_per_sum_lane_audio_rtf": quantiles(
                    [
                        group[0]["group_total_ms"]
                        / (sum(row["samples"] for row in group) / 24)
                        for group in group_rows.values()
                        if sum(row["samples"] for row in group)
                    ]
                ),
                "warm_first_to_second_callback_ms": quantiles(
                    [
                        row["first_to_second_callback_ms"]
                        for row in measured
                        if row["first_to_second_callback_ms"] is not None
                    ]
                ),
                "warm_text_eligible_step_four_budget_margin_ms": quantiles(
                    [
                        row["step_four_budget_margin_ms"]
                        for row in measured
                        if row["text_eligible"]
                        and row["step_four_budget_margin_ms"] is not None
                    ]
                ),
                "warm_text_eligible_step_four_within_budget_n": sum(
                    row["text_eligible"]
                    and row["step_four_within_first_chunk_budget"] is True
                    for row in measured
                ),
                "warm_text_eligible_step_four_measured_n": sum(
                    row["text_eligible"] and row["step_four_completion_ms"] is not None
                    for row in measured
                ),
                "per_lane": [{**identity(row), **timing(row)} for row in selected],
            }
        )
    return result


def analyze(root: Path) -> Row:
    root = root.resolve(strict=True)
    plan = read_json(root / "plan.json")
    rows, missing, hashes = load_rows(root, plan)
    for row in rows:
        path = asset_path(root, row)
        row["verified_pcm_sha256"] = sha(path) if path.exists() else None
        if path.exists() and path.stat().st_size != row["samples"] * 4:
            raise ValueError("raw PCM sample count mismatch")
        if row.get("pcm_sha256") and row["verified_pcm_sha256"] != row["pcm_sha256"]:
            raise ValueError("raw PCM hash does not match retained result")
    baseline = first_rows([row for row in rows if row["run"] == "causal-baseline"])
    switched = []
    for row in rows:
        if row["run"].startswith("switch-"):
            key = row["text_label"], row["seed"]
            if key in baseline:
                switched.append(compare(baseline[key], row))
    return {
        "schema_version": 1,
        "scope": (
            "offline descriptive engine research, not speech or release acceptance"
        ),
        "coverage": "unassessed",
        "human_listening": "unassessed",
        "physical_playback": "not_run",
        "limitations": [
            "Natural EOS above the duration proxy is not proof of "
            "text coverage or speech quality.",
            "Repeats are not independent seeds; paired policy tables "
            "use first occurrence only.",
            "Trace equality covers at most 128 complete 16-codebook frames, "
            "not all long outputs.",
            "Full PCM comparisons hash unquantized f32 bytes, "
            "not only quantized WAV samples.",
            "Step-four budget is a common-clock callback observation, "
            "not an implemented selector.",
            "Two pre-admitted lanes duplicate prefill; "
            "no zero-cost state fork is tested.",
            "RTF uses common group span; per-lane phase spans are not added together.",
            "Width-two-to-one events are natural/other retirements, "
            "not selected-loser cancellation.",
            "Errors, limits and watchdog timeouts are not successful recoveries.",
        ],
        "matrix_directory": str(root),
        "plan_sha256": sha(root / "plan.json"),
        "source_analysis_sha256": hashes,
        "analyzed_observations": len(rows),
        "incomplete_or_unanalyzed_runs": missing,
        "paired_reseed": paired_reseed(rows),
        "switch_after_preserved_frames": switched,
        "deterministic_repeats": repeated_runs(rows),
        "forced_prefix_outcomes": prefix_outcomes(root, rows),
        "batch": batch_comparisons(root, rows),
        "timing": timing_summaries(rows),
    }


class ContractTests(unittest.TestCase):
    def row(self, **changes: Any) -> Row:
        row = {
            "run": "sample",
            "group": 0,
            "lane": 0,
            "text_label": "original",
            "seed": 2,
            "prefix": None,
            "status": 0,
            "invalid_pcm": False,
            "capacity_exceeded": False,
            "finish_reason": 1,
            "frames": 3,
            "samples": 5760,
            "codes": [[1] * 16] * 3,
            "verified_pcm_sha256": "a",
            "group_total_ms": 400,
            "chunks": [{"ms": 200, "samples": 1920}, {"ms": 260, "samples": 3840}],
            "steps": [{"ms": 200}, {"ms": 230}, {"ms": 260}, {"ms": 270}],
        }
        return row | changes

    def test_failed_and_limit_not_natural(self) -> None:
        self.assertEqual(outcome(self.row(status=6)), "error_or_cancelled")
        self.assertEqual(outcome(self.row(finish_reason=2)), "max_tokens")
        self.assertEqual(outcome(self.row(samples=0)), "empty_pcm")
        self.assertEqual(
            outcome(self.row(text_label="ru-short")), "natural_eos_short_text"
        )

    def test_timing_independent_of_terminal(self) -> None:
        self.assertEqual(
            timing(self.row()), timing(self.row(status=6, finish_reason=2))
        )
        self.assertEqual(timing(self.row())["step_four_budget_margin_ms"], 10)
        self.assertFalse(timing(self.row(text_label="ru-short"))["text_eligible"])

    def test_first_occurrence_not_repeat(self) -> None:
        self.assertEqual(
            first_rows([self.row(), self.row(group=1, frames=99)])[("original", 2)][
                "frames"
            ],
            3,
        )

    def test_complete_frame_and_float_pcm(self) -> None:
        changed = self.row(
            codes=[[1] * 16, [1] * 15 + [2], [1] * 16], verified_pcm_sha256="b"
        )
        result = compare(self.row(), changed)
        self.assertEqual(result["common_complete_frame_prefix_length"], 1)
        self.assertFalse(result["full_pcm_equal"])
        self.assertFalse(result["retained_complete_frames_equal"])
        self.assertTrue(result["retained_trace_covers_both_generations"])
        self.assertFalse(
            compare(self.row(frames=200), self.row(frames=200))[
                "retained_trace_covers_both_generations"
            ]
        )

    def test_nearest_rank(self) -> None:
        self.assertEqual(quantiles(list(range(1, 21)))["p95_nearest_rank"], 19)
        self.assertIsNone(quantiles([])["p50"])

    def test_paired_policy_ignores_repeats(self) -> None:
        rows = [
            self.row(run="independent-baseline"),
            self.row(run="independent-baseline", group=1, frames=99),
            self.row(run="independent-per-frame", frames=99),
            self.row(run="independent-per-frame", group=1),
        ]
        result = paired_reseed(rows)[0]
        self.assertEqual(result["paired_n"], 1)
        self.assertEqual(result["pairs"][0]["a_frames"], 3)
        self.assertEqual(result["pairs"][0]["b_frames"], 99)

    def test_natural_retirement_not_selected_loser(self) -> None:
        steps = [
            {"ms": 200, "active": 2},
            {"ms": 230, "active": 2},
            {"ms": 260, "active": 2},
            {"ms": 270, "active": 1},
            {"ms": 310, "active": 1},
        ]
        rows = [
            self.row(steps=steps, admitted_ms=100, admitted_together=2),
            self.row(steps=steps, lane=1, frames=20, admitted_together=2),
        ]
        result = batch_comparisons(Path(__file__).parent, rows)
        transition = result["natural_or_other_retirement_width_two_to_one"][0]
        self.assertEqual(
            transition["natural_eos_retiring_lanes_inferred_from_frame_count"], [0]
        )
        self.assertEqual(transition["next_step_span_ms"], 40)
        self.assertFalse(transition["selected_loser_cancelled"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    if args.matrix is None or args.output is None:
        parser.error("matrix and NEW output JSON paths required")
    result = analyze(args.matrix)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
