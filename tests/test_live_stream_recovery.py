"""Model-free artifact and gate contracts for the live streaming study."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_live_stream_recovery as study


class LiveStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        cls.temporary_root.mkdir(exist_ok=True)

    def fixture(self, directory, mode="replay", frame=2, seed=64, frames=4):
        probe = directory / "probe"
        probe.mkdir()
        count = frames * study.HOP
        pcm = struct.pack(f"<{count}f", *([0.01] * count))
        codes = struct.pack(f"<{frames * 16}i", *range(frames * 16))
        original_frames = min(3, frames) if mode == "rollback1" and seed in study.KNOWN else frames
        original_pcm = pcm[:original_frames * study.HOP * 4]
        original_codes = codes[:original_frames * 16 * 4]
        historical_pcm, historical_codes = directory / "retained.f32", directory / "retained.i32"
        historical_pcm.write_bytes(original_pcm)
        historical_codes.write_bytes(original_codes)
        case = {"seed": seed, "mode": mode}
        if mode == "replay":
            case["snapshot_frame"] = frame
        elif mode == "rollback1":
            case["continuation_seed"] = 64
        triggered = mode == "replay" or mode == "rollback1" and seed in study.KNOWN
        committed = study.HOP if triggered and frame == 2 and seed != 75 else 0
        stem = probe / "group-0"
        for suffix, data in ((".f32", pcm), ("-codes.i32", codes), ("-fresh-codec.f32", pcm),
                             ("-committed.f32", pcm),
                             ("-replay-step-original.f32", pcm[:(1 if frame == 1 else min(frames, 3)) * 1920 * 4]),
                             ("-replay-step-restored.f32", pcm[:(1 if frame == 1 else min(frames, 3)) * 1920 * 4]),
                             ("-original.f32", original_pcm), ("-original.i32", original_codes),
                             ("-committed-prefix.f32", pcm[:committed * 4])):
            Path(str(stem) + suffix).write_bytes(data)
        row = {"group": 0, "seed": seed, "mode": mode, "status": 0, "finish_reason": 1,
               "frames": frames, "checks_passed": True, "held_peak_samples": study.HOP * min(frames, 3),
               "samples": count, "committed_samples": count, "prefill_count": 1,
               "physical_playback": False,
               "fresh_codec_pcm_equal": True, "triggered": triggered,
               "unchanged_step_pcm_equal": True,
               "restore_frame": frame if triggered else -1,
               "decoded_frames": 1 if triggered else 0,
               "pending_frames": frame - 1 if triggered else 0,
               "codec_target": 2, "committed_samples_at_restore": committed,
               "committed_prefix_equal": True, "original_frames": original_frames,
               "original_status": 0, "original_finish_reason": 1, "continuation_seed": 64}
        self.write_row(directory, row)
        control = {"frames": original_frames, "raw": study.identity(historical_pcm),
                   "codes": study.identity(historical_codes)}
        return row, case, {seed: control}

    @staticmethod
    def write_row(directory, row):
        (directory / "probe/results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_schedule_exact_discovery_seeds_and_boundaries(self):
        stages = study.schedule()
        self.assertEqual([len(stage["cases"]) for stage in stages], [14, 14, 6, 7])
        self.assertEqual([row["seed"] for row in stages[0]["cases"][:7]], list(study.SEEDS))
        self.assertEqual({row["snapshot_frame"] for row in stages[1]["cases"]}, {1, 2})
        self.assertEqual({row["mode"] for row in stages[2]["cases"]}, set(study.NEGATIVE_MODES))
        self.assertTrue(all(row["continuation_seed"] != row["seed"] for row in stages[3]["cases"]))

    def test_disk_bound_covers_original_fresh_final_and_reserve(self):
        count = sum(len(stage["cases"]) for stage in study.schedule())
        expected = study.RESERVE_BYTES + count * (256 * 1920 * 20 + 1024**2) + 64 * 1024**2
        self.assertEqual(study.disk_bound(study.schedule()), expected)

    def test_manifest_frozen_reference_deadline_and_no_sampler_changes(self):
        result = study.make_manifest(Path("E:/repo"), Path("E:/repo/inputs"), study.schedule()[1]["cases"])
        self.assertEqual(result["reference_frames"], 226)
        self.assertEqual(result["deadline_ms"], 60000)
        self.assertEqual(result["groups"][7]["snapshot_frame"], 2)
        self.assertTrue(all("sampler_control" not in group for group in result["groups"]))
        self.assertTrue(all("repeat" not in group for group in result["groups"]))

    def test_source_must_be_clean_with_pinned_submodules(self):
        source = {"head": "abc", "status": "", "submodules": " c044c6f external/ggml"}
        study.require_clean_source(source)
        for changed in ({**source, "status": " M src/pipeline-tts.cpp"},
                        {**source, "submodules": "+c044c6f external/ggml"},
                        {**source, "submodules": "-c044c6f external/ggml"},
                        {**source, "submodules": "U044c6f external/ggml"}):
            with self.assertRaisesRegex(ValueError, "clean source"):
                study.require_clean_source(changed)

    def test_low_disk_blocks_before_hashing_models_or_querying_gpu(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            argv = ["study", "--repo", str(root), "--probe", str(root / "bin/probe.exe"),
                    "--runtime", str(root / "runtime"), "--output", str(root / ".tmp/base-acceptance/new"),
                    "--execute"]
            with (patch("sys.argv", argv),
                  patch.object(study.shutil, "disk_usage", return_value=SimpleNamespace(free=0)),
                  patch.object(study, "asset_identity") as assets,
                  patch.object(study, "source_identity") as source,
                  patch.object(study, "gpu_identity") as gpu,
                  patch.object(study, "launch_owned") as launch):
                with self.assertRaisesRegex(ValueError, "disk preflight"):
                    study.main()
                assets.assert_not_called()
                source.assert_not_called()
                gpu.assert_not_called()
                launch.assert_not_called()

    def test_empty_collection_preserves_missing_state(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            self.assertEqual(study.collect(Path(temporary), [{"seed": 2}], {}),
                             ([], ["no flushed results"]))

    def test_replay_frame_two_preserves_pending_row_and_committed_prefix(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory)
            rows, issues = study.collect(directory, [case], controls)
            self.assertEqual(issues, [])
            self.assertTrue(rows[0]["mechanical_valid"])
            self.assertEqual(rows[0]["committed_samples_at_restore"], 1920)
            self.assertEqual(rows[0]["word_coverage"], "unassessed")
            self.assertEqual(rows[0]["human_listening"], "not_run")

    def test_replay_frame_one_has_zero_commit(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory, frame=1)
            rows, issues = study.collect(directory, [case], controls)
            self.assertEqual(issues, [])
            self.assertTrue(rows[0]["mechanical_valid"])
            self.assertEqual(rows[0]["committed_samples_at_restore"], 0)

    def test_replay_frame_two_terminal_drain_does_not_commit_seed75(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory, seed=75, frames=2)
            rows, issues = study.collect(directory, [case], controls)
            self.assertEqual(issues, [])
            self.assertTrue(rows[0]["mechanical_valid"])
            self.assertEqual(rows[0]["committed_samples_at_restore"], 0)
            self.assertEqual(rows[0]["replay_step_original"]["bytes"], 2 * 1920 * 4)

    def test_actual_prefix_bytes_overrule_positive_probe_flags(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory)
            (directory / "probe/group-0-committed-prefix.f32").write_bytes(b"x" * (1920 * 4))
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("actual committed prefix changed", issues[0])

    def test_actual_fresh_pcm_overrules_positive_probe_flags(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory)
            (directory / "probe/group-0-fresh-codec.f32").write_bytes(struct.pack("<7680f", *([0.] * 7680)))
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("independent full streaming-codec reconstruction", issues[0])

    def test_codes_and_pcm_both_required_for_unchanged_replay(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory)
            (directory / "probe/group-0-codes.i32").write_bytes(struct.pack("<64i", *([1] * 64)))
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("unchanged trajectory differs", issues[0])

    def test_restore_rejects_stale_watermark_wrong_pending_and_capacity(self):
        for change in ({"committed_samples_at_restore": 3840},
                       {"decoded_frames": 2, "pending_frames": 0},
                       {"held_peak_samples": 16 * 1920 + 1},
                       {"committed_samples_at_restore": True},
                       {"restore_frame": 1}):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
                directory = Path(temporary)
                row, case, controls = self.fixture(directory)
                self.write_row(directory, {**row, **change})
                rows, issues = study.collect(directory, [case], controls)
                self.assertFalse(rows[0]["mechanical_valid"])
                self.assertEqual(len(issues), 1)

    def test_rollback_checks_original_and_complete_code_prefix(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory, mode="rollback1", seed=2)
            rows, issues = study.collect(directory, [case], controls)
            self.assertEqual(issues, [])
            self.assertTrue(rows[0]["mechanical_valid"])
            directory = directory / "changed"
            directory.mkdir()
            _, case, controls = self.fixture(directory, mode="rollback1", seed=2)
            codes = directory / "probe/group-0-codes.i32"
            codes.write_bytes(struct.pack("<64i", *([1] * 64)))
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("complete-code prefix", issues[0])

    def test_extra_rows_and_wrong_case_identity_fail(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            row, case, controls = self.fixture(directory)
            self.write_row(directory, {**row, "seed": 42})
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("identity mismatch", issues[0])
            self.write_row(directory, row)
            with (directory / "probe/results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            rows, issues = study.collect(directory, [case], controls)
            self.assertEqual(len(rows), 2)
            self.assertFalse(rows[1]["mechanical_valid"])

    def negative_fixture(self, directory, mode="cancel"):
        row, _, controls = self.fixture(directory, mode="baseline")
        original_pcm = Path(controls[64]["raw"]["path"]).read_bytes()[:1920 * 4]
        original_codes = Path(controls[64]["codes"]["path"]).read_bytes()[:64]
        for suffix, payload in ((".f32", original_pcm), ("-codes.i32", original_codes),
                                ("-original.f32", original_pcm), ("-original.i32", original_codes),
                                ("-committed.f32", b"")):
            (directory / "probe" / ("group-0" + suffix)).write_bytes(payload)
        row.update({"mode": mode, "status": -5, "finish_reason": 0, "frames": 1,
                    "samples": 1920, "original_frames": 1, "original_status": -5,
                    "original_finish_reason": 0, "holdback_cap_samples": 16 * 1920,
                    "held_peak_samples": 1920, "committed_samples": 0,
                    "negative_guard_pass": True, "restore_rejected": True})
        self.write_row(directory, row)
        return row, {"mode": mode, "seed": 64}, controls

    def test_negative_cancel_preserves_partial_artifacts_without_speech_success(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.negative_fixture(directory)
            rows, issues = study.collect(directory, [case], controls)
            self.assertEqual(issues, [])
            self.assertTrue(rows[0]["mechanical_valid"])
            self.assertEqual(rows[0]["word_coverage"], "not_assessed_guard_case")

    def test_negative_rejects_false_guard_status_commits_and_changed_partial_audio(self):
        for change in ({"restore_rejected": False}, {"negative_guard_pass": False}, {"status": 0},
                       {"committed_samples": 1920}, {"triggered": True}, {"frames": 2}):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
                directory = Path(temporary)
                row, case, controls = self.negative_fixture(directory)
                self.write_row(directory, {**row, **change})
                rows, issues = study.collect(directory, [case], controls)
                self.assertFalse(rows[0]["mechanical_valid"])
                self.assertEqual(len(issues), 1)
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.negative_fixture(directory)
            (directory / "probe/group-0.f32").write_bytes(struct.pack("<1920f", *([0.] * 1920)))
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("already-observed PCM", issues[0])

    def test_negative_stale_requires_crossing_the_saved_codec_boundary(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.negative_fixture(directory, mode="stale")
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("did not pass its decoded boundary", issues[0])

    def test_actual_final_sink_bytes_overrule_positive_native_flags(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory)
            (directory / "probe/group-0-committed.f32").write_bytes(struct.pack("<7680f", *([0.] * 7680)))
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("actual final sink bytes", issues[0])

    def test_actual_one_step_replay_bytes_overrule_positive_native_flags(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            directory = Path(temporary)
            _, case, controls = self.fixture(directory)
            (directory / "probe/group-0-replay-step-restored.f32").write_bytes(
                struct.pack("<5760f", *([0.] * 5760)))
            rows, issues = study.collect(directory, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("actual one-step unchanged", issues[0])


if __name__ == "__main__":
    unittest.main()
