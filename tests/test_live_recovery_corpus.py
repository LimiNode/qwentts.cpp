"""Model-free contracts for the prespecified RU/EN live recovery corpus."""

import hashlib
import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_live_recovery_corpus as study
import run_live_stream_recovery as collector
import test_live_stream_recovery as fixtures


TEXT_HASHES = {
    "ru-yes": "06bfa6e648ed6023d697a1842ef5691f552d7aa5818edd705d7354a224cf0121",
    "ru-no": "ab3be5c500dddf8142266bbb28eb1b3fac6735768d98f7a892d55f20d6645374",
    "ru-thanks": "d34921bc56be01bba3a15fae2d5bc8d5a5e717c50b6682d2b25a52b704f3968b",
    "ru-ready": "12c711122847c8c8b458f6a8bfc3ea1c288645545b639c0b873f7818f8a61b8c",
    "en-yes": "5f9a2b795615ba6a3d5455fd5624d773fbca5bcd16249c421fd37411dc9837da",
    "en-no": "38178a20b470cfd18299fb1593dd4ba706f1a83321dac051d01086efd8b7a96f",
    "en-hi": "ca51ce1fb15acc6d69b8a5700256172fcc507e02073e6f19592e341bd6508ab8",
    "en-thanks": "8569efc4ed5a6273140c2b6ec0ab8d99f4398210808fbb7c0cbbae4fbcc21350",
    "ru-short": "c0ded82b6f75fcbfcc082519670227776ed6451dcef2f0262c271a9090f9e19a",
    "ru-new": "81d172f4216d1c823b90bc372c5aaba310040bb34d2cd2c7efee0226e53f4e52",
    "ru-two": "65626c5598e624958fc14bfdb7cae8cd5234deb7cca918e18dada784a1ac24e5",
    "en-new": "3018a89fbe5067e0d36270b182478bb4b506d00d1ceff136d99e9df6d9e85d52",
}


class CorpusTests(unittest.TestCase):
    write_row = staticmethod(fixtures.LiveStudyTests.write_row)
    fixture = fixtures.LiveStudyTests.fixture

    @classmethod
    def setUpClass(cls):
        cls.temporary_root = Path(__file__).resolve().parents[1] / ".tmp"
        cls.temporary_root.mkdir(exist_ok=True)

    def test_exact_frozen_text_order_bytes_languages_and_short_labels(self):
        actual = {label: hashlib.sha256(text.encode("utf-8")).hexdigest()
                  for label, (text, _, _) in study.CORPUS.items()}
        self.assertEqual(list(actual), list(TEXT_HASHES))
        self.assertEqual(actual, TEXT_HASHES)
        self.assertEqual(sum(language == "russian" for _, language, _ in study.CORPUS.values()), 7)
        self.assertEqual(sum(short for _, _, short in study.CORPUS.values()), 9)

    def test_prespecified_schedule_and_exact_repeat_pairs(self):
        controls, replay, recovery = study.schedule()
        self.assertEqual([len(stage["cases"]) for stage in (controls, replay, recovery)], [72, 12, 36])
        self.assertEqual(study.SEEDS, (1009, 2003, 3001))
        first, second = controls["cases"][:36], controls["cases"][36:]
        self.assertEqual([{key: value for key, value in row.items() if key != "repeat"} for row in first],
                         [{key: value for key, value in row.items() if key != "repeat"} for row in second])
        self.assertEqual([row["seed"] for row in first], list(study.SEEDS) * 12)
        self.assertEqual(len({row["control_key"] for row in first}), 36)
        self.assertEqual([row["text_label"] for row in replay["cases"]], list(study.CORPUS))
        self.assertTrue(all(row["seed"] == 1009 and row["snapshot_frame"] == 1 for row in replay["cases"]))
        self.assertTrue(all(row["continuation_seed"] == 64 for row in recovery["cases"]))

    def test_manifest_uses_text_language_but_excludes_collector_metadata(self):
        cases = study.schedule()[0]["cases"]
        inputs = Path("inputs")
        manifest = study.make_manifest(Path("repo"), inputs, cases)
        self.assertEqual(manifest["reference_frames"], 226)
        self.assertEqual(manifest["deadline_ms"], 60000)
        self.assertEqual(manifest["max_tokens"], 256)
        for case, group in zip(cases, manifest["groups"]):
            self.assertEqual(group["text_file"], str(inputs / (case["text_label"] + ".txt")))
            self.assertEqual(group["language"], study.CORPUS[case["text_label"]][1])
            self.assertEqual(set(group), {"text_file", "language", "seed", "mode"})

    def test_disk_reservation_for_all_120_groups(self):
        self.assertEqual(study.disk_bound(study.schedule()), 1909456896)

    def test_baseline_establishes_keyed_control_only_after_validation(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            _, case, _ = self.fixture(root, mode="baseline", seed=1009)
            case.update(study.case("ru-yes", 1009, "baseline"))
            controls = {}
            rows, issues = study.collect(root, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("missing matching", issues[0])
            self.assertEqual(controls, {})
            rows, issues = study.collect(root, [case], controls, establish_controls=True)
            self.assertEqual(issues, [])
            self.assertEqual(set(controls), {"ru-yes:1009"})
            self.assertTrue(rows[0]["mechanical_valid"])
            repeat = root / "repeat"
            repeat.mkdir()
            self.fixture(repeat, mode="baseline", seed=1009)
            rows, issues = study.collect(repeat, [case], controls, establish_controls=True)
            self.assertEqual(issues, [])
            self.assertTrue(rows[0]["mechanical_valid"])

    def test_invalid_first_baseline_does_not_create_control(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            row, case, _ = self.fixture(root, mode="baseline", seed=1009)
            self.write_row(root, {**row, "checks_passed": False})
            controls = {}
            rows, _ = study.collect(root, [case], controls, establish_controls=True)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertEqual(controls, {})

    def test_same_seed_on_distinct_texts_cannot_share_control(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            _, _, original = self.fixture(root, mode="baseline", seed=1009)
            controls = {"ru-yes:1009": original[1009]}
            rows, issues = study.collect(root, [study.case("en-yes", 1009, "baseline")], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("missing matching", issues[0])

    def test_changed_repeat_pcm_or_codes_fails_even_with_local_parity(self):
        for changed in ("pcm", "codes"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
                root = Path(temporary)
                _, case, controls = self.fixture(root, mode="baseline", seed=1009)
                if changed == "pcm":
                    data = struct.pack("<7680f", *([0.02] * 7680))
                    for suffix in (".f32", "-fresh-codec.f32", "-committed.f32"):
                        (root / "probe" / ("group-0" + suffix)).write_bytes(data)
                else:
                    (root / "probe/group-0-codes.i32").write_bytes(struct.pack("<64i", *([1] * 64)))
                rows, issues = study.collect(root, [case], controls, establish_controls=True)
                self.assertFalse(rows[0]["mechanical_valid"])
                self.assertIn("unchanged trajectory", issues[0])

    def test_unseen_seed_checkpoint_two_commit_depends_on_history_not_seed(self):
        for frames, expected in ((2, 0), (4, 1920)):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
                root = Path(temporary)
                _, case, controls = self.fixture(root, seed=1009, frames=frames)
                rows, issues = study.collect(root, [case], controls)
                self.assertEqual(issues, [])
                self.assertEqual(rows[0]["committed_samples_at_restore"], expected)

    def test_unseen_short_seed_triggers_conditional_rollback(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            _, case, controls = self.fixture(root, mode="rollback1", seed=1009, original_frames=3)
            rows, issues = study.collect(root, [case], controls)
            self.assertEqual(issues, [])
            self.assertTrue(rows[0]["triggered"])

    def test_long_original_stays_byte_exact_without_intervention(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            _, case, controls = self.fixture(root, mode="rollback1", seed=1009, frames=9)
            rows, issues = study.collect(root, [case], controls)
            self.assertEqual(issues, [])
            self.assertFalse(rows[0]["triggered"])
            changed = root / "changed"
            changed.mkdir()
            self.fixture(changed, mode="rollback1", seed=1009, frames=9)
            codes = changed / "probe/group-0-codes.i32"
            codes.write_bytes(struct.pack("<144i", *([1] * 144)))
            rows, issues = study.collect(changed, [case], controls)
            self.assertFalse(rows[0]["mechanical_valid"])
            self.assertIn("unchanged trajectory", issues[0])

    def test_short_intervention_annotation_makes_no_semantic_claim(self):
        rows = [{"case": study.case("ru-yes", 1009, "rollback1"), "mode": "rollback1",
                 "triggered": True, "mechanical_valid": True, "word_coverage": "unassessed",
                 "human_listening": "not_run"}, {"mechanical_valid": False}]
        study.annotate(rows)
        self.assertTrue(rows[0]["potential_valid_short_intervention"])
        self.assertFalse(rows[0]["semantic_error_inferred_from_duration"])
        self.assertEqual(rows[0]["word_coverage"], "unassessed")
        self.assertEqual(rows[0]["human_listening"], "not_run")
        self.assertNotIn("short_request", rows[1])

    def test_cells_preserve_failed_and_not_run_identities(self):
        cases = study.schedule()[1]["cases"]
        cells = study.cells(cases, [{"mechanical_valid": True}, {"mechanical_valid": False,
                              "error": "bad PCM"}], "budget exhausted")
        self.assertEqual([cell["state"] for cell in cells[:3]], ["mechanical_pass", "failed", "not_run"])
        self.assertEqual(cells[1]["reason"], "bad PCM")
        self.assertEqual(cells[2]["reason"], "budget exhausted")
        self.assertEqual([cell["case"] for cell in cells], cases)

    def test_failed_pcm_is_retained_as_unassessed_wav(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            (root / "probe").mkdir()
            raw = root / "probe/group-0-failed-observed.f32"
            raw.write_bytes(struct.pack("<1920f", *([0.01] * 1920)))
            before = raw.read_bytes()
            artifacts, issues = study.failed_artifacts(root)
            self.assertEqual(issues, [])
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]["human_listening"], "not_run")
            self.assertTrue(raw.with_suffix(".wav").is_file())
            self.assertEqual(raw.read_bytes(), before)

    def test_malformed_and_nonfinite_failed_pcm_remains_raw(self):
        for data in (b"x", struct.pack("<1920f", *([float("nan")] * 1920))):
            with self.subTest(size=len(data)), tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
                root = Path(temporary)
                (root / "probe").mkdir()
                raw = root / "probe/group-0-failed-observed.f32"
                raw.write_bytes(data)
                artifacts, issues = study.failed_artifacts(root)
                self.assertEqual(artifacts, [])
                self.assertEqual(len(issues), 1)
                self.assertEqual(raw.read_bytes(), data)
                self.assertFalse(raw.with_suffix(".wav").exists())

    def argv(self, root, execute=False):
        argv = ["study", "--repo", str(root), "--probe", str(root / "probe.exe"),
                "--runtime", str(root), "--output", str(root / ".tmp/base-acceptance/new")]
        return argv + (["--execute"] if execute else [])

    def test_low_disk_preflight_precedes_assets_source_and_gpu(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            with (patch("sys.argv", self.argv(root, True)),
                  patch.object(study.shutil, "disk_usage", return_value=SimpleNamespace(free=0)),
                  patch.object(study, "asset_identity") as assets,
                  patch.object(study, "source_identity") as source,
                  patch.object(study, "gpu_identity") as gpu,
                  patch.object(study, "launch_owned") as launch):
                with self.assertRaisesRegex(ValueError, "disk preflight"):
                    study.main()
                for mocked in (assets, source, gpu, launch):
                    mocked.assert_not_called()

    def test_dirty_execute_precedes_assets_and_gpu(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            with (patch("sys.argv", self.argv(root, True)),
                  patch.object(study.shutil, "disk_usage", return_value=SimpleNamespace(free=10**10)),
                  patch.object(study, "source_identity", return_value={"status": " M test", "submodules": ""}),
                  patch.object(study, "asset_identity") as assets,
                  patch.object(study, "gpu_identity") as gpu):
                with self.assertRaisesRegex(ValueError, "clean source"):
                    study.main()
                assets.assert_not_called()
                gpu.assert_not_called()

    def prepare_main(self, root):
        reference = root / study.REFERENCE / "inputs"
        reference.mkdir(parents=True)
        for name in study.PREPARED_HASHES:
            (reference / name).write_bytes(b"fixture")
        return {"models": [{"sha256": value[1]} for value in study.MODELS.values()]}

    def test_plan_only_writes_all_cells_and_exact_text_without_querying_gpu(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            assets = self.prepare_main(root)
            with (patch("sys.argv", self.argv(root)),
                  patch.object(study, "asset_identity", return_value=assets),
                  patch.object(study, "source_identity", return_value={"status": "dirty-plan-allowed"}),
                  patch.object(study, "sha", side_effect=lambda path: study.PREPARED_HASHES[path.name]),
                  patch.object(study, "gpu_identity") as gpu,
                  patch.object(study, "launch_owned") as launch):
                self.assertEqual(study.main(), 0)
                gpu.assert_not_called()
                launch.assert_not_called()
            output = root / ".tmp/base-acceptance/new"
            for label, (text, _, _) in study.CORPUS.items():
                self.assertEqual((output / "inputs" / (label + ".txt")).read_bytes(), text.encode("utf-8"))
            planned = json.loads((output / "planned-cells.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(len(stage["cells"]) for stage in planned.values()), 120)
            self.assertTrue(all(cell["state"] == "not_run" for stage in planned.values() for cell in stage["cells"]))

    def test_failed_first_stage_stops_later_launches_and_retains_not_run_cells(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            assets = self.prepare_main(root)
            source = {"status": "", "submodules": " c044c6f external/ggml"}
            with (patch("sys.argv", self.argv(root, True)),
                  patch.object(study.shutil, "disk_usage", return_value=SimpleNamespace(free=10**10)),
                  patch.object(study, "asset_identity", return_value=assets),
                  patch.object(study, "source_identity", return_value=source),
                  patch.object(study, "sha", side_effect=lambda path: study.PREPARED_HASHES[path.name]),
                  patch.object(study, "gpu_identity", return_value={"uuid": study.GPU}),
                  patch.object(study, "launch_owned", return_value=1) as launch):
                self.assertEqual(study.main(), 1)
                launch.assert_called_once()
            result = json.loads((root / ".tmp/base-acceptance/new/results.json").read_text(encoding="utf-8"))
            self.assertEqual([stage["state"] for stage in result["stages"].values()], ["failed", "not_run", "not_run"])
            self.assertEqual(len(result["stages"]["controls"]["cells"]), 72)
            self.assertTrue(all(cell["state"] == "not_run" for cell in result["stages"]["recovery"]["cells"]))
            self.assertEqual(result["human_listening"], "not_run")
            self.assertEqual(result["word_coverage"], "unassessed")
            self.assertEqual(result["release"], "blocked")

    def test_invalid_absolute_deadline_rejected_before_preflight_by_both_runners(self):
        for runner in (study, collector):
            for value in ("nan", "inf", "-inf", "0", "-1"):
                with self.subTest(runner=runner.__name__, value=value):
                    argv = self.argv(Path("unused")) + ["--deadline-monotonic=" + value]
                    with (patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()),
                          patch.object(runner, "asset_identity") as assets):
                        with self.assertRaises(SystemExit) as error:
                            runner.main()
                        self.assertEqual(error.exception.code, 2)
                        assets.assert_not_called()

    def test_expired_shared_deadline_leaves_all_cells_not_run_without_launch(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            assets = self.prepare_main(root)
            source = {"status": "", "submodules": " c044c6f external/ggml"}
            with (patch("sys.argv", self.argv(root, True) + ["--deadline-monotonic", "1"]),
                  patch.object(study.shutil, "disk_usage", return_value=SimpleNamespace(free=10**10)),
                  patch.object(study, "asset_identity", return_value=assets),
                  patch.object(study, "source_identity", return_value=source),
                  patch.object(study, "sha", side_effect=lambda path: study.PREPARED_HASHES[path.name]),
                  patch.object(study, "gpu_identity", return_value={"uuid": study.GPU}),
                  patch.object(study, "launch_owned") as launch):
                self.assertEqual(study.main(), 1)
                launch.assert_not_called()
            result = json.loads((root / ".tmp/base-acceptance/new/results.json").read_text(encoding="utf-8"))
            cells = [cell for stage in result["stages"].values() for cell in stage["cells"]]
            self.assertEqual(len(cells), 120)
            self.assertTrue(all(cell["state"] == "not_run" for cell in cells))
            self.assertTrue(all(cell["reason"] == "budget exhausted" for cell in cells))

    def test_expired_shared_deadline_also_prevents_discovery_runner_launch(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            root = Path(temporary)
            assets = self.prepare_main(root)
            (root / study.REFERENCE / "inputs/original.txt").write_bytes(collector.ORIGINAL.encode("utf-8"))
            retained_receipt = root / collector.PRIOR
            retained_receipt.parent.mkdir(parents=True)
            retained_receipt.write_bytes(b"{}")
            source = {"status": "", "submodules": " c044c6f external/ggml"}
            with (patch("sys.argv", self.argv(root, True) + ["--deadline-monotonic", "1"]),
                  patch.object(collector.shutil, "disk_usage", return_value=SimpleNamespace(free=10**10)),
                  patch.object(collector, "asset_identity", return_value=assets),
                  patch.object(collector, "source_identity", return_value=source),
                  patch.object(collector, "sha", side_effect=lambda path: study.PREPARED_HASHES[path.name]),
                  patch.object(collector, "retained_controls", return_value={}),
                  patch.object(collector, "gpu_identity", return_value={"uuid": study.GPU}),
                  patch.object(collector, "launch_owned") as launch):
                self.assertEqual(collector.main(), 1)
                launch.assert_not_called()
            result = json.loads((root / ".tmp/base-acceptance/new/results.json").read_text(encoding="utf-8"))
            self.assertTrue(all(stage["state"] == "not_run" and stage["rows"] == []
                                for stage in result["stages"].values()))


if __name__ == "__main__":
    unittest.main()
