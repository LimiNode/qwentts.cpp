"""Model-free collection/gate contracts for the buffered checkpoint experiment."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_eos_recovery as study


class RecoveryStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        cls.temporary_root.mkdir(exist_ok=True)

    def test_schedule_frozen_counts_boundaries_and_seeds(self):
        stages = study.schedule()
        self.assertEqual([len(stage["cases"]) for stage in stages], [14, 21, 21])
        self.assertEqual([row["seed"] for row in stages[0]["cases"][:7]], [2, 14, 72, 75, 78, 42, 64])
        self.assertEqual({row["recovery"]["snapshot_frame"] for row in stages[1]["cases"]}, {0, 1, 2})
        self.assertTrue(all(row["recovery"]["continuation_seed"] != row["seed"] for row in stages[2]["cases"]))

    def test_disk_bound_includes_both_attempts_and_reserve(self):
        stages = study.schedule()
        self.assertEqual(study.disk_bound(stages), study.RESERVE_BYTES + 98 * 256 * 1920 * 6 + 32 * 1024**2)

    def test_smoke_is_known_short_replay_not_recovery(self):
        stages = study.schedule(True)
        self.assertEqual([len(stage["cases"]) for stage in stages], [10, 15])
        self.assertTrue(all(case["seed"] in study.KNOWN for stage in stages for case in stage["cases"]))
        self.assertEqual(study.disk_bound(stages, 8, 8 * 1024**2),
                         study.RESERVE_BYTES + 40 * 8 * 1920 * 6 + 8 * 1024**2)

    def test_low_disk_blocks_before_hashing_models_or_launching(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            repo = Path(temporary)
            argv = ["study", "--repo", str(repo), "--probe", str(repo / "binary/probe.exe"),
                    "--runtime", str(repo / "runtime"), "--output", str(repo / ".tmp/base-acceptance/new"),
                    "--execute"]
            with (patch("sys.argv", argv),
                  patch.object(study.shutil, "disk_usage", return_value=SimpleNamespace(free=0)),
                  patch.object(study, "identity") as identity,
                  patch.object(study.subprocess, "Popen") as launch,
                  patch.object(study, "gpu_identity") as gpu):
                with self.assertRaisesRegex(ValueError, "disk preflight"):
                    study.main()
                identity.assert_not_called()
                launch.assert_not_called()
                gpu.assert_not_called()

    def test_manifest_fixed_predictor_and_original_text(self):
        manifest = study.make_manifest(Path("E:/repo"), Path("E:/repo/inputs"), study.schedule()[1]["cases"])
        self.assertEqual(manifest["max_batch"], 1)
        self.assertEqual(manifest["reference_frames"], 226)
        self.assertEqual(manifest["groups"][0]["recovery"], {"policy": "replay", "snapshot_frame": 0})
        self.assertNotIn("sampler_control", manifest["groups"][0])

    def test_replay_requires_full_codes_pcm_and_count(self):
        row = {"mechanical_valid": True, "research_recovery": {
            "triggered": True, "replay_codes_equal": True, "replay_pcm_equal": True}}
        self.assertTrue(study.replay_passed([row], 1))
        self.assertFalse(study.replay_passed([row], 2))
        for field in ("triggered", "replay_codes_equal", "replay_pcm_equal"):
            bad = json.loads(json.dumps(row))
            bad["research_recovery"][field] = False
            self.assertFalse(study.replay_passed([bad], 1))

    def test_controls_require_old_raw_bytes_not_only_duration(self):
        row = {"mechanical_valid": True, "seed": 2, "frames": 3, "finish_reason": 1,
               "raw": {"sha256": "a"}}
        old = [{"text_label": "original", **row}]
        self.assertTrue(study.controls_match([row], old))
        row["raw"] = {"sha256": "b"}
        self.assertFalse(study.controls_match([row], old))

    def test_complete_codes_reject_partial_and_oov(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            path = Path(temporary) / "codes.i32"
            path.write_bytes(struct.pack("<16i", *range(16)))
            study.validate_codes(path, 1)
            with self.assertRaises(ValueError):
                study.validate_codes(path, 2)
            path.write_bytes(struct.pack("<16i", *([2048] * 16)))
            with self.assertRaises(ValueError):
                study.validate_codes(path, 1)

    def test_pcm_rejects_size_and_nonfinite(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            path = Path(temporary) / "pcm.f32"
            path.write_bytes(struct.pack("<1920f", *([0.] * 1920)))
            study.validate_pcm(path, 1)
            with self.assertRaises(ValueError):
                study.validate_pcm(path, 2)
            path.write_bytes(struct.pack("<1920f", *([float("nan")] * 1920)))
            with self.assertRaises(ValueError):
                study.validate_pcm(path, 1)

    def test_empty_collection_preserves_missing_disposition(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            rows, issues = study.collect(Path(temporary), [{"seed": 2}])
            self.assertEqual(rows, [])
            self.assertEqual(issues, ["no flushed results"])

    def collect_replay_fixture(self, directory, *, lie=False, streaming=False):
        probe = directory / "probe"
        probe.mkdir()
        pcm = struct.pack("<1920f", *([0.01] * 1920))
        codes = struct.pack("<16i", *range(16))
        stem = probe / "group-0-lane-0"
        for suffix, data in ((".f32", pcm), ("-original.f32", pcm),
                             ("-original.i32", codes), ("-codes.i32", codes)):
            Path(str(stem) + suffix).write_bytes(data)
        row = {"group": 0, "lane": 0, "seed": 42, "status": 0, "frames": 1,
               "invalid_pcm": False, "capacity_exceeded": False, "decision_capture_invalid": False,
               "execution_mode": "streaming" if streaming else "buffered-uncommitted", "chunks": [],
               "research_recovery": {"policy": "replay", "original_frames": 1, "committed_samples": 0,
                    "prefill_count": 1, "streaming_qualified": False, "triggered": True,
                    "trigger_max_frames": 8, "attempt_count": 2, "original_status": 0,
                    "original_finish_reason": 1, "restore_frame": 0, "continuation_seed": 42,
                    "replay_codes_equal": not lie, "replay_pcm_equal": True}}
        (probe / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return study.collect(directory, [{"seed": 42, "recovery": {"policy": "replay", "snapshot_frame": 0}}])

    def test_watchdog_write_error_still_kills_and_reaps_owned_process(self):
        process = Mock()
        process.poll.return_value = None
        with (patch.object(study.subprocess, "Popen", return_value=process),
              patch.object(study, "wait_for_probe", side_effect=OSError("disk full"))):
            with self.assertRaisesRegex(OSError, "disk full"):
                study.launch_owned(["probe"], Path("runtime"), {}, None, None, 1, Path("watchdog.json"))
        process.kill.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)

    def test_effective_restore_contract_rejects_wrong_seed_boundary_or_original(self):
        case = {"seed": 2, "recovery": {"policy": "rollback-1", "continuation_seed": 64}}
        recovery = {"original_status": 0, "original_finish_reason": 1, "original_frames": 3,
                    "trigger_max_frames": 8, "triggered": True, "attempt_count": 2,
                    "restore_frame": 2, "continuation_seed": 64}
        study.validate_recovery({"research_recovery": recovery}, case, study.KNOWN)
        for key, value in (("original_status", 1), ("original_finish_reason", 3), ("original_frames", 4),
                           ("trigger_max_frames", 7), ("triggered", False), ("attempt_count", 1),
                           ("restore_frame", 1), ("continuation_seed", 65)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate_recovery({"research_recovery": {**recovery, key: value}}, case, study.KNOWN)

    def test_healthy_long_control_may_skip_intervention(self):
        case = {"seed": 42, "recovery": {"policy": "rollback-2", "continuation_seed": 64}}
        recovery = {"original_status": 0, "original_finish_reason": 1, "original_frames": 90,
                    "trigger_max_frames": 8, "triggered": False, "attempt_count": 1, "restore_frame": -1}
        study.validate_recovery({"research_recovery": recovery}, case, {42: 90})

    def test_collector_independently_checks_replay_bytes(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            rows, issues = self.collect_replay_fixture(Path(temporary))
            self.assertEqual(issues, [])
            self.assertTrue(study.replay_passed(rows, 1))
            self.assertEqual(rows[0]["word_coverage"], "unassessed")

    def test_collector_rejects_false_replay_equality(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            rows, issues = self.collect_replay_fixture(Path(temporary), lie=True)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("disagrees", issues[0])

    def test_collector_rejects_streaming_mislabel(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            rows, issues = self.collect_replay_fixture(Path(temporary), streaming=True)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("falsely labeled streaming", issues[0])


if __name__ == "__main__":
    unittest.main()
