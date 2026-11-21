"""Model-free checks for the paired prefill experiment's evidence helpers."""

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import run_prefill_research as research


class PrefillResearchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        if root.parent.name == ".worktrees":
            root = root.parent.parent
        cls.temp_root = root / ".tmp" / "prefill-validation"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_cases_keep_warm_pairs_shapes_and_discovered_failures(self):
        planned = research.cases(False)
        self.assertEqual(planned[0]["kind"], "first_use")
        warm = [row for row in planned if row["kind"] == "warm_pair"]
        self.assertEqual(len(warm), 12)
        self.assertTrue(all(row["text_label"] == "original" and row["seed"] == 64 for row in warm))
        self.assertEqual([row["text_label"] for row in planned[13:15]], ["ru-yes", "original"])
        self.assertEqual([row["text_label"] for row in planned[15:17]], ["ru-two-sentences", "original"])
        self.assertEqual([row["seed"] for row in planned[-3:-1]], [2, 14])
        self.assertEqual(planned, research.cases(False))
        self.assertEqual(len(research.cases(True)), 5)
        self.assertTrue(all(row["kind"] == "causal" for row in research.cases(True)))

    def test_variants_have_single_factor_controls(self):
        self.assertEqual(research.VARIANTS["baseline"], {})
        self.assertEqual(research.VARIANTS["baseline-end"], {})
        for label in ("last-only", "kv-f16", "window64", "retained"):
            self.assertEqual(len(research.VARIANTS[label]), 1)
        combined = {}
        for label in ("last-only", "kv-f16", "window64", "retained"):
            combined.update(research.VARIANTS[label])
        self.assertEqual(research.VARIANTS["combined"], combined)

    def test_retained_input_hashes_are_sha256(self):
        for value in research.INPUT_HASHES.values():
            self.assertEqual(len(value), 64)
            self.assertEqual(len(bytes.fromhex(value)), 32)

    def test_logits_report_identity_and_numerical_difference(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            first, second = Path(folder) / "a.f32", Path(folder) / "b.f32"
            first.write_bytes(struct.pack("<3f", 1, 2, 4))
            second.write_bytes(first.read_bytes())
            equal = research.logits_difference(first, second)
            self.assertTrue(equal["shape_equal"])
            self.assertTrue(equal["byte_equal"])
            self.assertTrue(equal["argmax_equal"])
            self.assertEqual(equal["max_abs"], 0)
            second.write_bytes(struct.pack("<3f", 1.5, 2, 3))
            changed = research.logits_difference(first, second)
            self.assertFalse(changed["byte_equal"])
            self.assertTrue(changed["argmax_equal"])
            self.assertEqual(changed["max_abs"], 1)
            self.assertAlmostEqual(changed["rms"], math.sqrt(1.25 / 3))
            second.write_bytes(struct.pack("<2f", 1, 2))
            self.assertFalse(research.logits_difference(first, second)["shape_equal"])
            second.write_bytes(struct.pack("<3f", 1, math.inf, 3))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                research.logits_difference(first, second)

    def fixture(self, root, *, trace=False):
        (root / "probe").mkdir()
        terminal = {
            "group": 0, "lane": 0, "seed": 64, "status": 0,
            "finish_reason": 1, "frames": 1, "samples": 3,
            "invalid_pcm": False, "capacity_exceeded": False,
            "chunks": [{"ms": 3, "samples": 3}],
            "prefill_ms": 2, "total_ms": 4, "prefill_observation": None,
        }
        (root / "probe/group-0-lane-0.f32").write_bytes(struct.pack("<3f", 0, 0.25, -0.25))
        (root / "probe/results.jsonl").write_text(json.dumps(terminal) + "\n", encoding="utf-8")
        if trace:
            codes = ",".join(str(value) for value in range(16))
            (root / "stderr.log").write_text(
                "[ResearchJob] serial=1 seed=64\n"
                f"[ResearchFrame] serial=1 slot=0 frame=0 codes={codes}\n",
                encoding="utf-8",
            )
        planned = [{"text_label": "original", "seed": 64, "kind": "warm_pair"}]
        return terminal, planned

    def test_collect_retains_partial_prefix_wav_and_unassessed_coverage(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            _, planned = self.fixture(root, trace=True)
            planned.append({"text_label": "original", "seed": 2, "kind": "control"})
            rows, issues = research.collect(root, planned, True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(issues, [])
            self.assertEqual(rows[0]["codes"], [list(range(16))])
            self.assertEqual(rows[0]["coverage"], "unassessed")
            self.assertEqual(rows[0]["wav"]["human_listening"], "unassessed")
            self.assertTrue((root / "probe/group-0-lane-0.wav").is_file())

    def test_collect_rejects_wrong_seed_before_claiming_a_pair(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            _, planned = self.fixture(root)
            planned[0]["seed"] = 2
            rows, issues = research.collect(root, planned, False)
            self.assertEqual(rows, [])
            self.assertTrue(any("order" in issue for issue in issues))

    def test_collect_keeps_earlier_outcome_when_later_json_is_incomplete(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            terminal, planned = self.fixture(root)
            planned.append({"text_label": "original", "seed": 2, "kind": "control"})
            (root / "probe/results.jsonl").write_text(
                json.dumps(terminal) + '\n{"group":', encoding="utf-8"
            )
            rows, issues = research.collect(root, planned, False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(issues), 1)
            self.assertTrue((root / "probe/group-0-lane-0.wav").is_file())

    def test_collect_retains_terminal_failure_when_artifact_is_missing(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            terminal, planned = self.fixture(root)
            terminal.update(status=-1, finish_reason=0)
            (root / "probe/results.jsonl").write_text(json.dumps(terminal) + "\n", encoding="utf-8")
            (root / "probe/group-0-lane-0.f32").unlink()
            rows, issues = research.collect(root, planned, False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], -1)
            self.assertEqual(len(issues), 1)
            self.assertIn("artifact_error", rows[0])

    def test_compare_reports_outcome_differences_and_rejects_mispair(self):
        row = {"group": 0, "text_label": "original", "seed": 64,
               "status": 0, "finish_reason": 1, "frames": 3,
               "raw": {"sha256": "a"}, "codes": [[1] * 16]}
        other = {**row, "frames": 4, "raw": {"sha256": "b"}, "codes": [[2] * 16]}
        comparison = research.compare([row], [other])
        self.assertEqual(comparison["paired"], 1)
        self.assertTrue(comparison["same_observed_denominator"])
        result = comparison["rows"][0]
        self.assertFalse(result["raw_equal"])
        self.assertFalse(result["frames_equal"])
        self.assertFalse(result["bounded_codes_equal"])
        self.assertTrue(result["terminal_equal"])
        with self.assertRaisesRegex(ValueError, "unpaired"):
            research.compare([row], [{**other, "seed": 2}])
        incomplete = research.compare([row], [])
        self.assertEqual(incomplete["paired"], 0)
        self.assertFalse(incomplete["same_observed_denominator"])

    def test_summary_does_not_mix_first_use_with_warm_measurements(self):
        common = {"status": 0, "finish_reason": 1,
                  "invalid_pcm": False, "capacity_exceeded": False,
                  "prefill_ms": 2, "total_ms": 4, "prefill_observation": None}
        rows = [dict(common, kind="first_use", chunks=[{"ms": 900}]),
                dict(common, kind="warm_pair", chunks=[{"ms": 3}]),
                dict(common, kind="warm_pair", chunks=[{"ms": 5}])]
        result = research.summarize(rows)
        self.assertEqual(result["observed"], 3)
        self.assertEqual(result["natural_eos"], 3)
        self.assertEqual(result["warm"]["first_pcm_ms"]["n"], 2)
        self.assertEqual(result["warm"]["first_pcm_ms"]["p50"], 4)
        self.assertEqual(result["human_listening"], "not_run")


if __name__ == "__main__":
    unittest.main()
