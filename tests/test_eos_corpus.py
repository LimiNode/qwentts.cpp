"""Model-free checks for frozen corpus planning, preservation and failures."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_eos_corpus as corpus


class CorpusContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / ".tmp"
        cls.temp_root.mkdir(exist_ok=True)

    def test_frozen_cases_and_denominators(self):
        rows = corpus.cases()
        self.assertEqual(len(rows), 142)
        prospective = [row for row in rows if row["kind"] == "prospective"]
        self.assertEqual(len(prospective), 128)
        self.assertEqual(
            len({(row["text_label"], row["seed"]) for row in prospective}), 128
        )
        self.assertEqual(sum(row["short_valid"] for row in prospective), 64)
        self.assertEqual(sum(row["prior_baseline_exposure"] for row in prospective), 32)
        self.assertEqual([row["seed"] for row in rows[:6]], [2, 14, 42, 2, 14, 42])
        self.assertEqual(sum(row["kind"] == "repeat" for row in rows), 8)
        for label in corpus.CORPUS:
            self.assertEqual(
                [row["seed"] for row in prospective if row["text_label"] == label],
                list(range(64, 80)),
            )
        self.assertIn("мастерско\u0301й", corpus.CORPUS["ru-two-sentences"][0])
        self.assertEqual(
            corpus.sha(Path(__file__).with_name("run_eos_research.py")),
            corpus.sha(Path(corpus.sha.__code__.co_filename)),
        )

    def fixture_inputs(self, root):
        source = root / "prepared"
        source.mkdir()
        payloads = {
            "speaker.f32": struct.pack("<2048f", *([0.5] * 2048)),
            "reference.i32": struct.pack("<3616i", *([0] * 3616)),
            "reference.txt": "  Точный текст.\n".encode(),
        }
        for name, payload in payloads.items():
            (source / name).write_bytes(payload)
        return source, {name: corpus.sha(source / name) for name in payloads}

    def test_conditioning_exact_copy_hash_shape_and_exclusivity(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            source, hashes = self.fixture_inputs(root)
            with patch.object(corpus, "PREPARED_HASHES", hashes):
                corpus.prepare_inputs(source, root / "inputs")
                self.assertEqual(
                    (source / "reference.txt").read_bytes(),
                    (root / "inputs/reference.txt").read_bytes(),
                )
                with self.assertRaises(FileExistsError):
                    corpus.prepare_inputs(source, root / "inputs")
                (source / "speaker.f32").write_bytes(b"bad")
                with self.assertRaisesRegex(ValueError, "hash/size"):
                    corpus.prepare_inputs(source, root / "other")
                hashes["speaker.f32"] = corpus.sha(source / "speaker.f32")
                with self.assertRaisesRegex(ValueError, "2048"):
                    corpus.prepare_inputs(source, root / "other")

    def test_environment_is_baseline_and_trace_optional(self):
        with patch.dict(
            corpus.os.environ,
            {
                "PATH": "old",
                "QWEN_RESEARCH_PREFIX_FILE": "bad",
                "QWEN_TTS_FULL_AR_TRACE": "1",
                "GGML_HIDDEN": "x",
            },
            clear=True,
        ):
            env = corpus.environment("GPU-exact", [], False)
            self.assertNotIn("QWEN_RESEARCH_PREFIX_FILE", env)
            self.assertNotIn("QWEN_TTS_FULL_AR_TRACE", env)
            self.assertNotIn("GGML_HIDDEN", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-exact")
            self.assertEqual(env["QWEN_RESEARCH_FRAME_SEED_POLICY"], "baseline")

    def test_partial_results_without_frame_logs_and_short_valid_proxy(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            (root / "corpus").mkdir()
            (root / "corpus-stderr.log").write_text("No frame tracing enabled.\n")
            planned = [
                {
                    "group": 0,
                    "seed": 64,
                    "text_label": "ru-yes",
                    "kind": "prospective",
                    "repeat": 0,
                    "short_valid": True,
                    "prior_baseline_exposure": True,
                },
                {
                    "group": 1,
                    "seed": 65,
                    "text_label": "ru-unseen",
                    "kind": "prospective",
                    "repeat": 0,
                    "short_valid": False,
                    "prior_baseline_exposure": False,
                },
            ]
            raw = root / "corpus/group-0-lane-0.f32"
            raw.write_bytes(struct.pack("<3f", 0, 0.25, -0.25))
            terminal = {
                "group": 0,
                "lane": 0,
                "seed": 64,
                "status": 0,
                "finish_reason": 1,
                "frames": 2,
                "samples": 3,
                "invalid_pcm": False,
                "capacity_exceeded": False,
                "chunks": [{"ms": 1, "samples": 3}],
            }
            (root / "corpus/results.jsonl").write_text(json.dumps(terminal) + "\n")
            issues = []
            rows = corpus.collect(root, planned, False, issues)
            self.assertEqual(issues, [])
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["codes_available"])
            self.assertFalse(rows[0]["duration_proxy"])
            self.assertEqual(rows[0]["coverage"], "unassessed")
            states = corpus.dispositions(planned, rows, "watchdog_timeout")
            self.assertEqual(states[1]["execution_state"], "not_run")
            self.assertEqual(states[1]["reason"], "watchdog_timeout")
            self.assertTrue(raw.with_suffix(".wav").exists())

    def test_malformed_later_result_preserves_previous_outcome(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            (root / "corpus").mkdir()
            (root / "corpus-stderr.log").write_text("")
            planned = corpus.cases()
            terminal = {
                "group": 0,
                "lane": 0,
                "seed": 2,
                "status": 0,
                "finish_reason": 1,
                "frames": 3,
                "samples": 3,
                "invalid_pcm": False,
                "capacity_exceeded": False,
                "chunks": [],
            }
            (root / "corpus/group-0-lane-0.f32").write_bytes(
                struct.pack("<3f", 0, 0.25, -0.25)
            )
            (root / "corpus/results.jsonl").write_text(
                json.dumps(terminal) + '\n{"group":'
            )
            issues = []
            rows = corpus.collect(root, planned, False, issues)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(issues), 1)
            self.assertEqual(
                corpus.dispositions(planned, rows, "analysis_failed")[1][
                    "execution_state"
                ],
                "not_run",
            )

    def test_plan_only_never_launches_and_freezes_invocation(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            source, hashes = self.fixture_inputs(root)
            for name in ("probe.exe", "talker.gguf", "codec.gguf", "CMakeCache.txt"):
                (root / name).write_bytes(b"model-free fixture")
            argv = [
                "--probe",
                str(root / "probe.exe"),
                "--talker",
                str(root / "talker.gguf"),
                "--codec",
                str(root / "codec.gguf"),
                "--prepared-inputs",
                str(source),
                "--output",
                str(root / "output"),
                "--gpu-uuid",
                "GPU-fixture",
                "--timing-off",
            ]
            with (
                patch.object(corpus, "PREPARED_HASHES", hashes),
                patch.object(corpus, "gpu_identity", return_value="CMP fixture"),
                patch.object(
                    corpus, "source_identity", return_value={"head": "fixture"}
                ),
                patch.object(corpus.subprocess, "Popen") as launch,
                patch.object(corpus.shutil, "disk_usage", return_value=SimpleNamespace(free=corpus.MIN_FREE_DISK)),
            ):
                self.assertEqual(corpus.main(argv), 0)
                launch.assert_not_called()
            manifest = json.loads((root / "output/corpus-manifest.json").read_text())
            self.assertEqual(len(manifest["groups"]), 142)
            self.assertEqual(manifest["max_batch"], 1)
            self.assertFalse(manifest["observe_decisions"])
            self.assertEqual(manifest["deadline_ms"], 60000)
            states = json.loads((root / "output/case-dispositions.json").read_text())
            self.assertTrue(all(row["execution_state"] == "not_run" for row in states))
            self.assertTrue((root / "output/invocation.json").is_file())
            self.assertTrue((root / "output/sha256-inventory.json").is_file())

    def test_gpu_selection_rejects_another_device(self):
        with patch.object(corpus.subprocess, "run") as query:
            query.return_value.stdout = (
                "GPU-fixture, NVIDIA GeForce RTX 4090, driver, 24000 MiB, 23000 MiB"
            )
            with self.assertRaisesRegex(ValueError, "CMP 50HX"):
                corpus.gpu_identity("GPU-fixture")
            self.assertEqual(query.call_args.args[0][1:3], ["--id", "GPU-fixture"])

    def test_low_disk_guard_blocks_before_model_or_process(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            argv = ["--probe", str(root / "probe.exe"), "--talker", str(root / "talker.gguf"),
                    "--codec", str(root / "codec.gguf"), "--prepared-inputs", str(root / "inputs"),
                    "--output", str(root / "output"), "--gpu-uuid", "GPU-fixture", "--execute"]
            with (
                patch.object(corpus.shutil, "disk_usage", return_value=SimpleNamespace(free=corpus.MIN_FREE_DISK - 1)),
                patch.object(corpus, "prepare_inputs") as prepare,
                patch.object(corpus, "gpu_identity") as gpu,
                patch.object(corpus.subprocess, "Popen") as launch,
            ):
                self.assertEqual(corpus.main(argv), 1)
                prepare.assert_not_called()
                gpu.assert_not_called()
                launch.assert_not_called()
            status = json.loads((root / "output/run-status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_failed_child_retains_completed_group_and_bounds_watchdog(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            root = Path(folder)
            source, hashes = self.fixture_inputs(root)
            for name in ("probe.exe", "talker.gguf", "codec.gguf", "CMakeCache.txt"):
                (root / name).write_bytes(b"model-free fixture")
            output = root / "output"
            argv = [
                "--probe",
                str(root / "probe.exe"),
                "--talker",
                str(root / "talker.gguf"),
                "--codec",
                str(root / "codec.gguf"),
                "--prepared-inputs",
                str(source),
                "--output",
                str(output),
                "--gpu-uuid",
                "GPU-fixture",
                "--execute",
            ]

            def launch(*_args, **_kwargs):
                self.assertTrue((output / "plan.json").is_file())
                self.assertTrue((output / "invocation.json").is_file())
                (output / "corpus").mkdir()
                (output / "corpus/group-0-lane-0.f32").write_bytes(
                    struct.pack("<3f", 0, 0.25, -0.25)
                )
                terminal = {
                    "group": 0,
                    "lane": 0,
                    "seed": 2,
                    "status": 0,
                    "finish_reason": 1,
                    "frames": 3,
                    "samples": 3,
                    "invalid_pcm": False,
                    "capacity_exceeded": False,
                    "chunks": [],
                    "decision_observation_enabled": True,
                    "decision_capture_invalid": False,
                    "decisions": [{"frame": 0, "ms": 1}],
                }
                (output / "corpus/results.jsonl").write_text(
                    json.dumps(terminal) + "\n"
                )
                return object()

            with (
                patch.object(corpus, "PREPARED_HASHES", hashes),
                patch.object(corpus, "gpu_identity", return_value="CMP fixture"),
                patch.object(
                    corpus, "source_identity", return_value={"head": "fixture"}
                ),
                patch.object(corpus.subprocess, "Popen", side_effect=launch),
                patch.object(corpus, "wait_for_probe", return_value=7) as wait,
                patch.object(corpus.shutil, "disk_usage", return_value=SimpleNamespace(free=corpus.MIN_FREE_DISK)),
            ):
                self.assertEqual(corpus.main(argv), 1)
                self.assertGreater(wait.call_args.args[1], 0)
                self.assertLessEqual(wait.call_args.args[1], 600)
            status = json.loads((output / "run-status.json").read_text())
            self.assertEqual(status["status"], "process_failed")
            self.assertEqual(status["exit_code"], 7)
            rows = json.loads((output / "all-observations.json").read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["coverage"], "unassessed")
            states = json.loads((output / "case-dispositions.json").read_text())
            self.assertEqual(
                sum(row["execution_state"] == "not_run" for row in states), 141
            )


if __name__ == "__main__":
    unittest.main()
