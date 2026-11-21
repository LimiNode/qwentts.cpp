"""Model-free checks for matched-comparator evidence and timing eligibility."""

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_speech_comparison as comparison


class SpeechComparisonContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        if root.parent.name == ".worktrees":
            root = root.parent.parent
        cls.temp_root = root / ".tmp" / "speech-comparison-validation"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def fixture(self, root):
        probe = root / "probe"
        probe.mkdir()
        rows = []
        for run in range(comparison.PLANNED_RUNS):
            raw, wav = probe / f"run-{run}.f32", probe / f"run-{run}.wav"
            raw.write_bytes(struct.pack("<3840f", *([0.25, -0.25] * 1920)))
            wav.write_bytes(b"retained WAV identity fixture")
            rows.append({"case": 0, "run": run, "seed": 64, "frames": 2, "samples": 3840,
                         "sample_rate": 24000, "natural_eos": True, "invalid_pcm": False,
                         "capacity_exceeded": False, "first_pcm_ms": run or 1000,
                         "total_ms": 2000, "phases": {"prefill_ms": 3},
                         "raw_f32": str(raw), "wav": str(wav)})
        self.write_rows(root, rows)
        return rows

    @staticmethod
    def write_rows(root, rows):
        (root / "probe/results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def test_complete_ordered_finite_evidence_is_valid_and_cold_is_not_warm(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            rows = self.fixture(root)
            original = (root / "probe/results.jsonl").read_bytes()
            collected = comparison.collect_results(root, 0)
            self.assertTrue(collected["mechanical_valid"])
            self.assertEqual(collected["issues"], [])
            self.assertEqual(collected["valid_rows"], 13)
            self.assertEqual(collected["not_run"], [])
            self.assertEqual(collected["warm_first_pcm_ms"], {"n": 12, "p50": 6.5, "p95": 12})
            self.assertEqual(collected["natural_eos"], 13)
            self.assertEqual(collected["listening"], "not_run")
            self.assertEqual(collected["rows"][4]["coverage"], "unassessed")
            self.assertIn("raw_identity", collected["rows"][4])
            self.assertIn("wav_identity", collected["rows"][4])
            self.assertEqual((root / "probe/results.jsonl").read_bytes(), original)
            self.assertTrue(all(Path(row["raw_f32"]).is_file() for row in rows))

    def test_terminal_and_integrity_failures_are_retained_but_excluded(self):
        for key, value in (("natural_eos", False), ("invalid_pcm", True),
                           ("capacity_exceeded", True), ("natural_eos", 1),
                           ("invalid_pcm", "false"), ("capacity_exceeded", None)):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
                root = Path(folder)
                rows = self.fixture(root)
                rows[4][key] = value
                rows[4]["first_pcm_ms"] = 1000
                self.write_rows(root, rows)
                collected = comparison.collect_results(root, 0)
                self.assertFalse(collected["mechanical_valid"])
                self.assertEqual(collected["observed"], 13)
                self.assertEqual(collected["valid_rows"], 12)
                self.assertFalse(collected["rows"][4]["mechanical_valid"])
                self.assertEqual(collected["rows"][4][key], value)
                self.assertEqual(collected["warm_first_pcm_ms"]["n"], 11)
                self.assertEqual(collected["warm_first_pcm_ms"]["p95"], 12)
                self.assertTrue(Path(rows[4]["raw_f32"]).is_file())

    def test_bad_payloads_and_missing_artifacts_do_not_discard_later_rows(self):
        for payload in (b"", b"bad", struct.pack("<3840f", *([math.nan] * 3840)),
                        struct.pack("<3840f", *([math.inf] * 3840)), None):
            with self.subTest(payload_size=None if payload is None else len(payload)), tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
                root = Path(folder)
                rows = self.fixture(root)
                if payload is None:
                    Path(rows[4]["raw_f32"]).unlink()
                else:
                    Path(rows[4]["raw_f32"]).write_bytes(payload)
                collected = comparison.collect_results(root, 0)
                self.assertFalse(collected["mechanical_valid"])
                self.assertEqual(len(collected["rows"]), 13)
                self.assertFalse(collected["rows"][4]["mechanical_valid"])
                self.assertTrue(collected["rows"][12]["mechanical_valid"])
                self.assertEqual(collected["warm_first_pcm_ms"]["n"], 11)
                self.assertIn("wav_identity", collected["rows"][4])
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            rows = self.fixture(root)
            Path(rows[4]["wav"]).unlink()
            self.assertFalse(comparison.collect_results(root, 0)["rows"][4]["mechanical_valid"])

    def test_nonfinite_empty_and_inconsistent_metadata_is_invalid(self):
        changes = ({"first_pcm_ms": math.nan}, {"first_pcm_ms": None}, {"first_pcm_ms": 0},
                   {"total_ms": math.inf}, {"total_ms": 1}, {"samples": 0}, {"samples": True},
                   {"frames": 3}, {"sample_rate": 48000}, {"phases": {"prefill_ms": -1}},
                   {"phases": {"prefill_ms": math.nan}}, {"phases": {}},
                   {"first_pcm_ms": 10**400})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
                root = Path(folder)
                rows = self.fixture(root)
                rows[4].update(change)
                self.write_rows(root, rows)
                collected = comparison.collect_results(root, 0)
                self.assertFalse(collected["mechanical_valid"])
                self.assertEqual(collected["warm_first_pcm_ms"]["n"], 11)
                self.assertEqual(collected["warm_prefill_ms"]["n"], 11)

    def test_duplicate_reordered_extra_and_wrong_seed_rows_fail(self):
        for change in ({"run": 3}, {"run": True}, {"case": 1}, {"seed": 65}):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
                root = Path(folder)
                rows = self.fixture(root)
                rows[4].update(change)
                self.write_rows(root, rows)
                collected = comparison.collect_results(root, 0)
                self.assertFalse(collected["mechanical_valid"])
                self.assertFalse(collected["rows"][4]["mechanical_valid"])
                self.assertEqual(collected["warm_first_pcm_ms"]["n"], 11)
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            rows = self.fixture(root)
            self.write_rows(root, rows + [rows[-1]])
            collected = comparison.collect_results(root, 0)
            self.assertFalse(collected["mechanical_valid"])
            self.assertEqual(collected["observed"], 14)
            self.assertFalse(collected["rows"][-1]["mechanical_valid"])
            self.assertEqual(collected["warm_first_pcm_ms"]["n"], 12)

    def test_truncated_json_is_retained_and_collection_continues(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            rows = self.fixture(root)
            lines = [json.dumps(row) for row in rows]
            lines[4] = '{"run":'
            (root / "probe/results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            collected = comparison.collect_results(root, 0)
            self.assertFalse(collected["mechanical_valid"])
            self.assertEqual(collected["observed"], 13)
            self.assertEqual(collected["not_run"], [4])
            self.assertEqual(collected["rows"][4]["raw_line"], lines[4])
            self.assertTrue(collected["rows"][12]["mechanical_valid"])
            self.assertEqual(collected["warm_first_pcm_ms"]["n"], 11)

    def test_partial_missing_and_process_failure_are_not_success(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            rows = self.fixture(root)
            collected = comparison.collect_results(root, 124)
            self.assertFalse(collected["mechanical_valid"])
            self.assertEqual(collected["valid_rows"], 13)
            self.assertEqual(collected["returncode"], 124)
            self.write_rows(root, rows[:2])
            collected = comparison.collect_results(root, 0)
            self.assertFalse(collected["mechanical_valid"])
            self.assertEqual(collected["not_run"], list(range(2, 13)))
            self.assertEqual(collected["warm_first_pcm_ms"]["n"], 1)
            (root / "probe/results.jsonl").unlink()
            collected = comparison.collect_results(root, 0)
            self.assertFalse(collected["mechanical_valid"])
            self.assertEqual(collected["warm_first_pcm_ms"]["n"], 0)

    def test_main_returns_failure_and_writes_failed_collection_without_gpu(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            output = root / ".tmp/base-acceptance/mock-run"
            args = SimpleNamespace(repo=root, checkout=root, build=root, probe=root / "probe.exe",
                                   model=root / "model.gguf", text=root / "text.txt", output=output,
                                   language="ru", dependency_directory=[])
            report = {"mechanical_valid": False, "rows": [{"natural_eos": False}]}
            with (patch.object(comparison.argparse.ArgumentParser, "parse_args", return_value=args),
                  patch.object(comparison.subprocess, "check_output", side_effect=[
                      "fd747c11e50543f59f2bbe233a9c7228dc479a71", "79b50d239d0b88852814c873db0d85d43ed66d9b", ""]),
                  patch.object(comparison, "sha", return_value="c1d5352f57bca5b8035f011305dc7bd4a1cb18ca3f93660e9e7a9a4120713fa8"),
                  patch.object(comparison, "gpu_identity", return_value="mock GPU"),
                  patch.object(comparison, "identity", return_value={}),
                  patch.object(comparison, "environment", return_value={}),
                  patch.object(comparison.subprocess, "Popen"),
                  patch.object(comparison, "wait_for_probe", return_value=0),
                  patch.object(comparison, "collect_results", return_value=report)):
                self.assertEqual(comparison.main(), 1)
            self.assertEqual(json.loads((output / "collected.json").read_text()), report)
            self.assertTrue((output / "stdout.log").is_file())


if __name__ == "__main__":
    unittest.main()
