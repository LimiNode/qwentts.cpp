"""Model-free contracts for bounded short-utterance and timing experiments."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_prefill_short as study


class ShortPrefillContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        if root.parent.name == ".worktrees":
            root = root.parent.parent
        cls.temporary = root / ".tmp" / "prefill-short-validation"
        cls.temporary.mkdir(parents=True, exist_ok=True)

    def row(self, index=0, prefix=False, phase="quality"):
        observation = {"diagnostic": False, "kv_f16": False, "last_only": False, "retain": False,
                       "alignment": 256, "nonfinite_logits": 0, "prefill_use_flash_attn": True,
                       "prefix_rows": 9 if prefix else 0, "prefix_source": "full",
                       "prefix_hit": prefix and index > 0}
        return {"group": index, "text_label": "original", "seed": 64, "kind": "warm_pair",
                "status": 0, "finish_reason": 1, "frames": 3, "samples": 5760,
                "invalid_pcm": False, "capacity_exceeded": False, "raw": {"sha256": "a"},
                "chunks": [{"ms": 123}], "prefill_ms": 100, "total_ms": 300,
                "decision_observation_enabled": False, "decisions": [],
                "prefill_observation_enabled": phase == "quality",
                "prefill_observation": observation if phase == "quality" else None}

    def test_frozen_quality_denominators_and_single_language_blocks(self):
        self.assertEqual(len(study.blocks("quality")), 6)
        self.assertEqual(sum(len(study.cases("quality", language)) for _, language, _ in study.blocks("quality")), 108)
        for language, phrases in study.SHORT.items():
            rows = study.cases("quality", language)
            self.assertEqual((rows[0]["kind"], rows[-1]["kind"]), ("primer_start", "primer_end"))
            self.assertEqual(rows[0]["text_label"], rows[-1]["text_label"])
            self.assertEqual({(r["text_label"], r["seed"]) for r in rows[1:-1]},
                             {(label, seed) for label in phrases for seed in study.SEEDS})

    def test_timing_counterbalance_and_first_use_are_separate(self):
        schedule = study.blocks("timing")
        self.assertEqual([r[2] for r in schedule], list(study.VARIANTS) + list(study.VARIANTS[::-1]))
        for warm in (6, 8):
            rows = study.cases("timing", "russian", warm)
            self.assertEqual(len(rows), warm + 1)
            self.assertEqual(rows[0]["kind"], "first_use")
            self.assertTrue(all(r["seed"] == 64 and r["text_label"] == "original" for r in rows))
            self.assertEqual(sum(r["kind"] == "warm_pair" for r in rows), warm)

    def test_environment_hygiene_and_manifest_forwarding(self):
        with patch.dict(os.environ, {"QWEN_PREFILL_DIAG_DIR": "bad", "QWEN_PREFILL_RETAIN": "1",
                                    "GGML_RESEARCH_FORCE_CUBLAS_Q8": "1", "QWEN_TTS_FULL_AR_TRACE": "1"}):
            for phase in ("quality", "timing"):
                for variant in study.VARIANTS:
                    env = study.configured_environment(phase, variant, [])
                    self.assertEqual(env["GGML_CUDA_DISABLE_FUSION"], "1")
                    self.assertEqual(env["QWEN_PREFILL_PROFILE"], "1" if phase == "quality" else "0")
                    self.assertEqual("GGML_RESEARCH_MATMUL_TRACE" in env, phase == "quality")
                    self.assertEqual("GGML_RESEARCH_FORCE_MMQ_Q8" in env, variant == "prefix-mmq")
                    for forbidden in ("QWEN_PREFILL_DIAG_DIR", "QWEN_PREFILL_RETAIN", "GGML_RESEARCH_FORCE_CUBLAS_Q8",
                                      "QWEN_TTS_FULL_AR_TRACE"):
                        self.assertNotIn(forbidden, env)
                    plan = study.cases(phase, "english")
                    manifest = study.manifest_for({"talker": "m", "codec": "c", "diagnostic_dump": True},
                                                   Path("inputs"), plan, "english", phase)
                    self.assertEqual(manifest["speaker"], "ryan")
                    self.assertEqual(manifest["max_batch"], 1)
                    self.assertEqual(manifest["max_tokens"], 256)
                    self.assertFalse(manifest["observe_decisions"])
                    self.assertEqual(manifest["observe_prefill"], phase == "quality")
                    self.assertTrue(all(g["language"] == "english" and len(g["seeds"]) == 1 for g in manifest["groups"]))
                    self.assertNotIn("diagnostic_dump", manifest)

    def test_quality_requires_effective_hits_and_actual_forced_route(self):
        rows = [self.row(i, prefix=True) for i in range(2)]
        route = "RESEARCH_MATMUL route=forced-mmq"
        self.assertEqual(study.evidence_issues(rows, "quality", "prefix-mmq", route), [])
        self.assertTrue(study.evidence_issues(rows, "quality", "prefix-mmq", ""))
        self.assertTrue(study.evidence_issues(rows, "quality", "prefix-mmq", "RESEARCH_MATMUL route=mmvq"))
        self.assertTrue(study.evidence_issues([self.row()], "quality", "baseline", route))
        self.assertTrue(study.evidence_issues(rows, "quality", "prefix-default", route))
        for field, value in (("prefix_hit", False), ("prefix_rows", 86), ("prefix_source", "split"),
                             ("diagnostic", True), ("kv_f16", True), ("alignment", 64)):
            changed = copy.deepcopy(rows)
            changed[1]["prefill_observation"][field] = value
            self.assertTrue(study.evidence_issues(changed, "quality", "prefix-mmq", route))

    def test_timing_rejects_observers_and_logs_but_checks_operational_prefix(self):
        rows = [self.row(i, prefix=True, phase="timing") for i in range(2)]
        logs = "\n".join("[ResearchPrefix] " + json.dumps({"prefix_rows": 9, "prefix_source": "full", "prefix_hit": i > 0})
                         for i in range(2))
        self.assertEqual(study.evidence_issues(rows, "timing", "prefix-mmq", logs), [])
        for extra in ("RESEARCH_MATMUL route=forced-mmq", "[ResearchPrefill] {}", "[ResearchFrame]"):
            self.assertTrue(study.evidence_issues(rows, "timing", "prefix-mmq", logs + "\n" + extra))
        self.assertTrue(study.evidence_issues(rows, "timing", "prefix-mmq", ""))
        rows[1]["prefill_observation_enabled"] = True
        self.assertTrue(study.evidence_issues(rows, "timing", "prefix-mmq", logs))

    def test_incomplete_and_failed_blocks_do_not_enter_timing(self):
        result = {"state": "completed", "returncode": 0, "issues": [], "rows": [self.row(phase="timing")]}
        valid = study.timing_summary(result, 1)
        self.assertTrue(valid["eligible"])
        self.assertEqual(valid["warm"]["first_pcm_ms"]["n"], 1)
        for change in ({"state": "timeout"}, {"returncode": 1}, {"issues": ["missing route proof"]}, {"rows": []}):
            summary = study.timing_summary({**result, **change}, 1)
            self.assertFalse(summary["eligible"])
            self.assertEqual(summary["warm"]["first_pcm_ms"]["n"], 0)
        for field, value in (("finish_reason", 2), ("invalid_pcm", True), ("capacity_exceeded", True), ("status", -1)):
            row = {**self.row(phase="timing"), field: value}
            self.assertFalse(study.timing_summary({**result, "rows": [row]}, 1)["eligible"])

    def test_complete_comparison_does_not_hide_equal_partial_denominators(self):
        result = {"state": "completed", "returncode": 0, "issues": [], "rows": [self.row()]}
        self.assertTrue(study.complete_comparison(result, result, 1)["complete"])
        self.assertFalse(study.complete_comparison(result, result, 2)["complete"])
        self.assertFalse(study.complete_comparison({**result, "state": "incomplete"}, result, 1)["complete"])

    def test_freeze_runtime_never_overwrites_and_checks_conflicting_dlls(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            (source / "probe.exe").write_bytes(b"exe")
            (source / "qwen.dll").write_bytes(b"dll")
            probe, identities = study.freeze_runtime(source / "probe.exe", [], root / "frozen")
            self.assertEqual(probe.read_bytes(), b"exe")
            self.assertEqual(len(identities), 2)
            with self.assertRaises(FileExistsError):
                study.freeze_runtime(source / "probe.exe", [], root / "frozen")
            dependencies = root / "dependencies"
            dependencies.mkdir()
            (source / "cublas.dll").write_bytes(b"one")
            (dependencies / "cublas.dll").write_bytes(b"two")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                study.freeze_runtime(source / "probe.exe", [dependencies], root / "conflict")

    def test_timing_reference_requires_complete_quality_and_identical_payloads(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            def payload(name, data):
                (root / name).write_bytes(data)
                return study.identity(root / name)
            runtime = [payload("probe.exe", b"exe")]
            models = [payload("model.gguf", b"model"), payload("codec.gguf", b"codec")]
            source_original = payload("source-original.txt", b"retained original")
            source_manifest = payload("source-manifest.json", b"{}")
            (root / "inputs").mkdir()
            texts = {"original": "retained original", "en-primer": study.EN_PRIMER,
                     **study.SHORT["russian"], **study.SHORT["english"]}
            for name, text in texts.items():
                (root / "inputs" / (name + ".txt")).write_text(text, encoding="utf-8")
            plan = {"phase": "quality", "runtime": runtime, "model_inputs": models,
                    "source_original": source_original, "source_manifest": source_manifest,
                    "inputs": [study.identity(p) for p in (root / "inputs").iterdir()],
                    "scripts": [study.identity(p) for p in study.script_paths()]}
            audio = {"raw": payload("audio.raw", b"\0" * (5760 * 4)), "wav": payload("audio.wav", b"wav"),
                     "logits": payload("logits.f32", b"logits")}
            results = {}
            for _, language, variant in study.blocks("quality"):
                name = f"r0-{language}-{variant}"
                rows = [{**self.row(i, prefix=variant != "baseline"), **case, **audio}
                        for i, case in enumerate(study.cases("quality", language))]
                results[name] = {"state": "completed", "returncode": 0, "issues": [], "rows": rows}
                (root / name).mkdir()
                (root / name / "stderr.log").write_text(
                    "RESEARCH_MATMUL route=" + ("forced-mmq" if variant == "prefix-mmq" else "mmq"),
                    encoding="utf-8")
                manifest = study.manifest_for({"talker": models[0]["path"], "codec": models[1]["path"]},
                                              root / "inputs", study.cases("quality", language), language, "quality")
                (root / name / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (root / name / "environment.json").write_text(
                    json.dumps(study.configured_environment("quality", variant, [])), encoding="utf-8")
                for key in ("manifest", "environment", "stderr"):
                    results[name][key] = study.identity(root / name / (key + (".log" if key == "stderr" else ".json")))
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
            self.assertIn("plan", study.validate_reference(root, runtime, models))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                study.validate_reference(root, [{**runtime[0], "sha256": "changed"}], models)
            first = next(iter(results))
            results[first]["returncode"] = 1
            (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study.validate_reference(root, runtime, models)
            results[first]["returncode"] = 0
            (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
            for changed in (root / "audio.raw", root / "audio.wav", root / "logits.f32",
                            root / "inputs" / "ru-yes.txt", root / first / "manifest.json",
                            root / first / "environment.json", root / first / "stderr.log"):
                retained = changed.read_bytes()
                changed.write_bytes(retained + b"changed")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    study.validate_reference(root, runtime, models)
                changed.write_bytes(retained)
            plan["scripts"][0]["sha256"] = "changed"
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                study.validate_reference(root, runtime, models)
            plan["scripts"][0] = study.identity(study.script_paths()[0])
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            results[first]["rows"] = results[first]["rows"][:-1]
            (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study.validate_reference(root, runtime, models)


if __name__ == "__main__":
    unittest.main()
