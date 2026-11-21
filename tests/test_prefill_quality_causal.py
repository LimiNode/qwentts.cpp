"""Model-free contracts for immutable-runtime causal quality controls."""

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_prefill_quality_causal as study


class CausalQualityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        if root.parent.name == ".worktrees":
            root = root.parent.parent
        cls.temporary = root / ".tmp/prefill-quality-causal-validation"
        cls.temporary.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def row(index=0, prefix=False):
        return {"group": index, "lane": 0, "text_label": "ru-no", "seed": 64, "kind": "short",
                "status": 0, "finish_reason": 1, "frames": 3, "samples": 5760,
                "invalid_pcm": False, "capacity_exceeded": False,
                "raw": {"sha256": "same"}, "chunks": [{"ms": 123}],
                "codes": [[100 + frame] * 16 for frame in range(3)],
                "decision_observation_enabled": False, "decisions": [],
                "prefill_observation_enabled": True,
                "prefill_observation": {"diagnostic": False, "kv_f16": False, "last_only": False,
                    "retain": False, "alignment": 256, "nonfinite_logits": 0,
                    "prefill_use_flash_attn": True, "prefix_rows": 9 if prefix else 0,
                    "prefix_hit": prefix and index > 0, "prefix_source": "full" if prefix else "split"}}

    @staticmethod
    def trace(rows, prefix=False):
        lines = ["RESEARCH_MATMUL route=" + ("forced-mmq" if prefix else "mmq")]
        for index, row in enumerate(rows):
            lines.append(f"[ResearchJob] serial={index} seed={row['seed']}")
            for frame, codes in enumerate(row["codes"]):
                vector = ",".join(map(str, codes))
                lines += [f"[Sample] step={frame} c0={codes[0]} u=0.25 subseq={16 * frame}",
                          f"[ARTrace] predictor step={frame} codes={vector}",
                          f"[ResearchFrame] serial={index} slot=0 frame={frame} codes={vector}"]
            if row["finish_reason"] == 1 and row["frames"] < 128:
                lines.append(f"[Sample] step={row['frames']} c0=2150 u=0.5 subseq={16 * row['frames']}")
        return "\n".join(lines)

    def test_frozen_schedule_is_14_fresh_processes_then_six_full18_blocks(self):
        jobs = study.schedule()
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({job["name"] for job in jobs}), 20)
        self.assertEqual(sum(len(job["cases"]) for job in jobs), 122)
        self.assertEqual([job["phase"] for job in jobs[:14]], ["cold"] * 14)
        self.assertTrue(all(len(job["cases"]) == 1 and job["variant"] == "baseline" for job in jobs[:14]))
        for position, (language, label, seed) in enumerate(study.COLD):
            pair = jobs[position * 2:position * 2 + 2]
            self.assertEqual([job["repeat"] for job in pair], [0, 1])
            self.assertTrue(all((job["language"], job["cases"][0]["text_label"], job["cases"][0]["seed"])
                                == (language, label, seed) for job in pair))
        self.assertEqual([(job["language"], job["variant"]) for job in jobs[14:18]],
                         [(language, variant) for language in ("russian", "english")
                          for variant in ("baseline", "prefix-mmq")])
        self.assertEqual([job["variant"] for job in jobs[18:]], ["prefix-default", "forced-mmq-no-prefix"])
        for job in jobs[14:]:
            self.assertEqual(job["cases"], study.short.cases("quality", job["language"]))

    def test_environment_has_only_prescribed_controls_and_observers(self):
        with patch.dict(os.environ, {"QWEN_PREFILL_KV": "f16", "QWEN_PREFILL_DIAG_DIR": "bad",
                                    "GGML_RESEARCH_FORCE_CUBLAS_Q8": "1", "QWEN_TTS_DUMP": "bad"}):
            for variant in ("baseline", "prefix-mmq", "prefix-default", "forced-mmq-no-prefix"):
                env = study.configured_environment(variant, Path("runtime"))
                self.assertEqual(env["QWEN_TTS_FULL_AR_TRACE"], "1")
                self.assertEqual(env["GGML_RESEARCH_MATMUL_TRACE"], "1")
                self.assertEqual(env["GGML_CUDA_DISABLE_FUSION"], "1")
                self.assertEqual(env["QWEN_PREFILL_PROFILE"], "1")
                self.assertEqual(env["QWEN_RESEARCH_FRAME_SEED_POLICY"], "baseline")
                self.assertEqual("GGML_RESEARCH_FORCE_MMQ_Q8" in env,
                                 variant in ("prefix-mmq", "forced-mmq-no-prefix"))
                self.assertEqual("QWEN_PREFILL_PREFIX_ROWS" in env, variant.startswith("prefix"))
                for forbidden in ("QWEN_PREFILL_KV", "QWEN_PREFILL_DIAG_DIR",
                                  "GGML_RESEARCH_FORCE_CUBLAS_Q8", "QWEN_TTS_DUMP"):
                    self.assertNotIn(forbidden, env)
                manifest = study.short.manifest_for({"talker": "m", "codec": "c"}, Path("inputs"),
                                                    study.schedule()[0]["cases"], "russian", "quality")
                self.assertEqual(manifest["max_tokens"], 256)
                self.assertTrue(manifest["observe_prefill"])
                self.assertFalse(manifest["observe_decisions"])

    def test_sampler_schedule_preserves_seeds_repeats_and_default_primers(self):
        jobs = study.sampler_schedule()
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(len(job["cases"]) for job in jobs), 54)
        for job in jobs:
            self.assertEqual(job["variant"], "baseline")
            rows = job["cases"]
            self.assertEqual((rows[0]["kind"], rows[-1]["kind"]), ("primer_start", "primer_end"))
            for left, right in zip(rows[1:-1:2], rows[2:-1:2]):
                self.assertEqual((left["text_label"], left["seed"]), (right["text_label"], right["seed"]))
                self.assertEqual((left["repeat"], right["repeat"]), (0, 1))

    def test_sampler_echo_rejects_ignored_controls_and_changed_eos_policy(self):
        values = dict(do_sample=True, temperature=0.9, top_k=1, top_p=1.0, repetition_penalty=1.05,
                      subtalker_do_sample=True, subtalker_temperature=0.9, subtalker_top_k=50,
                      subtalker_top_p=1.0, min_new_tokens=2, eos_guard_enabled=False)
        row = dict(sampler_control="talker-top1", effective_sampler=values)
        planned = [dict(kind="short")]
        self.assertEqual(study.sampler_issues([row], planned, "talker-top1"), [])
        for key, value in (("top_k", 50), ("eos_guard_enabled", True), ("min_new_tokens", 4),
                           ("temperature", float("nan")), ("temperature", True), ("temperature", None)):
            changed = copy.deepcopy(row)
            changed["effective_sampler"][key] = value
            self.assertTrue(study.sampler_issues([changed], planned, "talker-top1"))
        self.assertTrue(study.sampler_issues([row], [dict(kind="primer_start")], "talker-top1"))

    def test_expected_trace_is_allowed_but_unknown_dump_and_bad_sample_are_not_hidden(self):
        rows = [self.row()]
        trace = self.trace(rows)
        self.assertEqual(study.evidence_issues(rows, "baseline", trace), [])
        for extra in ("[ResearchDecision] unexpected", "[ARTrace] prompt seed=64",
                      "[ARTrace] sample step=0", "[ARTrace] forced Talker frame step=0",
                      "[FrameSeed] policy=per-frame", "[ResearchPrefillDiagnostic] dump",
                      "[ResearchFrame] serial=0 slot=0 frame=3 codes=1,2", "[ARTrace] mystery"):
            self.assertTrue(study.evidence_issues(rows, "baseline", trace + "\n" + extra), extra)
        for changed in (trace.replace("subseq=32", "subseq=33"),
                        trace.replace("step=2 c0=102", "step=1 c0=102"),
                        trace.replace("c0=102 u=", "c0=999 u="),
                        trace.replace("[Sample] step=3 c0=2150 u=0.5 subseq=48", "")):
            self.assertTrue(study.evidence_issues(rows, "baseline", changed))

    def test_actual_prefix_hit_rows_and_route_are_checked_even_with_trace(self):
        rows = [self.row(index, True) for index in range(2)]
        trace = self.trace(rows, True)
        self.assertEqual(study.evidence_issues(rows, "prefix-mmq", trace), [])
        self.assertTrue(study.evidence_issues(rows, "prefix-mmq", trace.replace("forced-mmq", "mmq")))
        for key, value in (("prefix_rows", 0), ("prefix_hit", False), ("prefix_source", "split"),
                           ("alignment", 64), ("diagnostic", True)):
            changed = copy.deepcopy(rows)
            changed[1]["prefill_observation"][key] = value
            self.assertTrue(study.evidence_issues(changed, "prefix-mmq", trace))
        no_prefix = [self.row()]
        forced = self.trace(no_prefix, True)
        self.assertEqual(study.evidence_issues(no_prefix, "forced-mmq-no-prefix", forced), [])
        self.assertTrue(study.evidence_issues(no_prefix, "forced-mmq-no-prefix", forced.replace("forced-mmq", "mmq")))
        self.assertTrue(study.evidence_issues(rows, "forced-mmq-no-prefix", trace))

    def test_first_full16_difference_reports_frame_and_codebook_not_just_talker(self):
        baseline = self.row()
        candidate = copy.deepcopy(baseline)
        candidate["codes"][1][11] = 22
        self.assertEqual(study.first_code_difference(candidate, baseline)["first_difference"],
                         {"frame": 1, "codebook": 11, "candidate": 22, "baseline": 101})
        self.assertEqual(study.first_code_difference(baseline, baseline)["state"], "equal_bounded_prefix")
        candidate["codes"] = candidate["codes"][:1]
        self.assertEqual(study.first_code_difference(candidate, baseline)["state"], "different_trace_lengths")
        self.assertFalse(study.compare_rows([], [], 18)["all_raw_frames_terminal_equal"])
        self.assertTrue(study.compare_rows([baseline], [baseline], 1)["all_raw_frames_terminal_equal"])
        candidate["raw"]["sha256"] = "different"
        self.assertFalse(study.compare_rows([candidate], [baseline], 1)["all_raw_frames_terminal_equal"])

    def reference_fixture(self, root):
        reference = root / study.REFERENCE
        runtime = reference / "runtime"
        runtime.mkdir(parents=True)
        for name in ("eos-research-probe.exe", "ggml-cuda.dll", "nested/other.dll"):
            file = runtime / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(name.encode())
        models = [root / path for path in study.MODEL_PATHS]
        for path in models:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        before = study.snapshot(runtime, models)
        plan = {"runtime": before["runtime"], "model_inputs": before["models"],
                "source_probe": study.identity(runtime / "eos-research-probe.exe"), "inputs": []}
        # Production's reference is a flat, completely enumerated frozen runtime.
        nested = runtime / "nested/other.dll"
        nested.replace(runtime / "other.dll")
        nested.parent.rmdir()
        plan["runtime"] = study.snapshot(runtime, models)["runtime"]
        study.write_json(reference / "plan.json", plan)
        study.write_json(reference / "results.json", {})
        return reference, runtime, models, plan

    def test_snapshot_hashes_all_runtime_files_and_models_in_place(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            _, runtime, models, plan = self.reference_fixture(root)
            before = study.snapshot(runtime, models)
            self.assertEqual(len(before["runtime"]), 3)
            self.assertEqual(before["models"], plan["model_inputs"])
            (runtime / "other.dll").write_bytes(b"changed")
            self.assertNotEqual(before, study.snapshot(runtime, models))
            (runtime / "new-file.txt").write_bytes(b"unexpected")
            self.assertEqual(len(study.snapshot(runtime, models)["runtime"]), 4)

    def test_reference_paths_and_pinned_runtime_and_models_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            reference, runtime, models, plan = self.reference_fixture(root)
            probe_hash = study.identity(runtime / "eos-research-probe.exe")["sha256"]
            cuda_hash = study.identity(runtime / "ggml-cuda.dll")["sha256"]
            model_hashes = tuple(study.identity(path)["sha256"] for path in models)
            with (patch.object(study, "PROBE_SHA", probe_hash), patch.object(study, "CUDA_SHA", cuda_hash),
                  patch.object(study.short, "CUSTOM_HASHES", model_hashes),
                  patch.object(study.short, "validate_reference", return_value={"retained": True}) as old_check):
                self.assertEqual(study.validate_reference(root)[3], {"retained": True})
                old_check.assert_called_once()
                for field in ("runtime", "model_inputs"):
                    changed = copy.deepcopy(plan)
                    changed[field][0]["path"] = str(root / Path(changed[field][0]["path"]).name)
                    (reference / "plan.json").write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "paths or identities"):
                        study.validate_reference(root)
                (reference / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
                (runtime / "ggml-cuda.dll").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "probe/CUDA identity"):
                    study.validate_reference(root)

    def test_deadline_prevents_launch_and_process_timeout_preserves_partial_evidence(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            job = study.schedule()[0]
            with patch.object(study.time, "monotonic", return_value=100), patch.object(study.subprocess, "Popen") as launch:
                result = study.execute(job, root, {}, root, root, 99)
                self.assertEqual(result["state"], "not_run")
                launch.assert_not_called()
            row = self.row()
            with (patch.object(study.time, "monotonic", return_value=100),
                  patch.object(study.subprocess, "Popen") as launch,
                  patch.object(study, "wait_for_probe", side_effect=subprocess.TimeoutExpired("probe", 7)) as wait,
                  patch.object(study, "collect", return_value=([row], [])),
                  patch.object(study, "evidence_issues", return_value=[])):
                result = study.execute(job, root, {"talker": "m", "codec": "c"}, root, root, 107)
                self.assertEqual(result["state"], "timeout")
                self.assertEqual(len(result["rows"]), 1)
                self.assertEqual(wait.call_args.args[1], 7)
                self.assertEqual(launch.call_args.kwargs["cwd"], root)
                self.assertEqual(launch.call_args.args[0][0], str(root / "eos-research-probe.exe"))
                self.assertEqual(study.collect.call_args.args[-2:], (True, "custom_voice"))
                self.assertTrue((root / job["name"] / "collected.json").exists())

    def test_interruption_stops_owned_process_before_returning_partial_outputs(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            process = Mock()
            process.wait.return_value = -1
            with (patch.object(study.time, "monotonic", return_value=100),
                  patch.object(study.subprocess, "Popen", return_value=process),
                  patch.object(study, "wait_for_probe", side_effect=KeyboardInterrupt),
                  patch.object(study, "collect", return_value=([], ["partial"])),
                  patch.object(study, "evidence_issues", return_value=[])):
                result = study.execute(study.schedule()[0], root, {"talker": "m", "codec": "c"}, root, root, 1000)
            self.assertEqual(result["state"], "interrupted")
            process.kill.assert_called_once()
            process.wait.assert_called_once_with(timeout=15)
            self.assertEqual(result["not_run_groups"], [0])

    def test_main_hashes_assets_after_interruption_and_lists_every_not_run_case(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            output = root / ".tmp/base-acceptance/new-causal-run"
            before = {"runtime": [], "models": [{"path": "m"}, {"path": "c"}]}
            args = SimpleNamespace(repo=root, output=output, seconds=1, sampler_probe=None)
            with (patch.object(study.argparse.ArgumentParser, "parse_args", return_value=args),
                  patch.object(study, "validate_reference", return_value=({"inputs": []}, {}, before, {})),
                  patch.object(study, "gpu_identity", return_value="mock GPU"),
                  patch.object(study, "source_identity", return_value={}),
                  patch.object(study, "execute", side_effect=KeyboardInterrupt),
                  patch.object(study, "snapshot", return_value={"runtime": ["changed"]}) as after,
                  patch.object(study, "comparisons", return_value={"trace_neutral": {"x": {"all_raw_frames_terminal_equal": False}}})):
                self.assertEqual(study.main(), 1)
            after.assert_called_once()
            report = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertTrue(report["interrupted"])
            self.assertTrue(report["integrity_issues"])
            self.assertEqual(len(report["results"]), 20)
            self.assertTrue(all(result["state"] == "not_run" for result in report["results"].values()))
            self.assertFalse(report["latency_eligible"])
            self.assertEqual(report["word_coverage"], "unassessed")

    def test_integrity_failure_alone_returns_one_and_retains_result(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as folder:
            root = Path(folder)
            output = root / ".tmp/base-acceptance/integrity-only"
            args = SimpleNamespace(repo=root, output=output, seconds=1, sampler_probe=None)
            before = {"runtime": [], "models": [{"path": "m"}, {"path": "c"}]}
            with (patch.object(study.argparse.ArgumentParser, "parse_args", return_value=args),
                  patch.object(study, "validate_reference", return_value=({"inputs": []}, {}, before, {})),
                  patch.object(study, "gpu_identity", return_value="mock GPU"),
                  patch.object(study, "source_identity", return_value={}),
                  patch.object(study, "execute", return_value={"state": "completed", "rows": []}),
                  patch.object(study, "snapshot", return_value={"runtime": ["changed"]}),
                  patch.object(study, "comparisons", return_value={"trace_neutral": {"x": {"all_raw_frames_terminal_equal": True}}})):
                self.assertEqual(study.main(), 1)
            report = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertFalse(report["interrupted"])
            self.assertTrue(report["integrity_issues"])


if __name__ == "__main__":
    unittest.main()
