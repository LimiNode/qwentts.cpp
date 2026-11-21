"""Model-free diagnostics validation; NumPy comparisons use only an existing install."""

from __future__ import annotations

import copy
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_prefill_differential as analysis


def write_tensor(path: Path, shape: tuple[int, ...], values=None) -> None:
    count = math.prod(shape)
    if values is None:
        values = list(range(1, count + 1))
    path.write_bytes(struct.pack("<" + "i" * (len(shape) + 1), len(shape), *shape)
                     + struct.pack("<" + "f" * count, *values))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def results_fixture() -> dict:
    results = {}
    for name in analysis.BLOCKS:
        rows = []
        for group, label in enumerate(analysis.LABELS):
            total = 2 if group == 2 else 4
            past = 1 if name.startswith("prefix") and group else 0
            observation = {"diagnostic": name.endswith("-taps"), "diagnostic_serial": group + 1,
                           "T": total - past, "n_past": past}
            if name.startswith("prefix"):
                observation.update(prefix_source="full", prefix_rows=1, prefix_hit=group > 0)
            rows.append({"group": group, "lane": 0, "seed": 64, "text_label": label,
                         "status": 0, "finish_reason": 2, "frames": 4, "samples": 7680,
                         "invalid_pcm": False, "capacity_exceeded": False,
                         "decision_capture_invalid": False, "prefill_observation": observation})
        results[name] = {"state": "completed", "returncode": 0, "issues": [], "rows": rows}
    return results


class DiagnosticValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        if root.parent.name == ".worktrees":
            root = root.parent.parent
        cls.temp_root = root / ".tmp" / "prefill-differential-validation"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_taps(self, directory, *, view=False, pinned=True):
        directory.mkdir(parents=True, exist_ok=True)
        taps = []
        for stage in analysis.STAGES:
            tap = {"layer": 0, "stage": stage, "file": f"layer-00-{stage}.bin",
                   "op": "RESHAPE" if view and stage == "attn-output" else "MUL_MAT"}
            if pinned:
                tap["output_retention"] = "view_and_backing_chain"
            (directory / tap["file"]).write_bytes(b"placeholder")
            taps.append(tap)
        self.save_taps(directory, taps)
        return taps

    @staticmethod
    def save_taps(directory, taps):
        (directory / "taps.jsonl").write_text(
            "".join(json.dumps(tap) + "\n" for tap in taps), encoding="utf-8")

    def test_valid_little_endian_header(self):
        path = self.root / "tensor.bin"
        write_tensor(path, (2, 3))
        self.assertEqual(analysis.tensor_header(path.read_bytes(), path), ((2, 3), 12))

    def test_truncated_headers_raise_value_error_not_struct_error(self):
        for dimensions in range(1, 5):
            header = struct.pack("<i", dimensions) + b"\0" * (dimensions * 4 - 1)
            for end in range(len(header) + 1):
                with self.subTest(dimensions=dimensions, end=end):
                    with self.assertRaises(ValueError):
                        analysis.tensor_header(header[:end], self.root / "bad.bin")

    def test_invalid_dimensions_shapes_payloads_and_oversize(self):
        invalid = [struct.pack("<2i", d, 1) for d in (-1, 0, 5, 2**31 - 1)]
        invalid += [struct.pack("<2i", 1, n) + b"\0" * 4 for n in (0, -1, 2**31 - 1)]
        invalid += [struct.pack("<2i", 1, 1), struct.pack("<2if", 1, 1, 1) + b"x"]
        with patch.object(analysis, "MAX_TENSOR_BYTES", 64):
            invalid.append(b"\0" * 65)
            for payload in invalid:
                with self.subTest(payload=payload[:16]):
                    with self.assertRaises(ValueError):
                        analysis.tensor_header(payload, self.root / "bad.bin")

    def test_numpy_is_optional_until_numerical_comparison(self):
        path = self.root / "tensor.bin"
        write_tensor(path, (1,))
        with patch.object(analysis, "np", None):
            self.assertEqual(analysis.tensor_header(path.read_bytes(), path), ((1,), 8))
            with self.assertRaisesRegex(RuntimeError, "existing NumPy"):
                analysis.tensor(path)

    def test_complete_results_are_valid(self):
        analysis.validate_results(results_fixture())

    def test_incomplete_blocks_are_rejected_even_with_four_rows(self):
        for name in analysis.BLOCKS:
            for field, value in (("state", "timeout"), ("returncode", 1),
                                 ("issues", ["fault"]), ("rows", results_fixture()[name]["rows"][:3])):
                results = results_fixture()
                results[name][field] = value
                with self.subTest(name=name, field=field):
                    with self.assertRaises(ValueError):
                        analysis.validate_results(results)
        results = results_fixture()
        del results["baseline-off"]
        with self.assertRaisesRegex(ValueError, "all four"):
            analysis.validate_results(results)

    def test_invalid_rows_and_unpaired_shapes_are_rejected(self):
        changes = (("group", 2), ("group", False), ("seed", 65), ("lane", 1),
                   ("text_label", "different"), ("status", -1), ("status", False), ("finish_reason", 0),
                   ("frames", 5), ("samples", 0), ("invalid_pcm", True),
                   ("capacity_exceeded", True), ("decision_capture_invalid", True))
        for field, value in changes:
            results = results_fixture()
            results["prefix-taps"]["rows"][1][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    analysis.validate_results(results)
        for field, value in (("T", 2), ("T", True), ("n_past", -1), ("prefix_hit", False),
                             ("prefix_source", "split"), ("diagnostic", False), ("diagnostic_serial", 9)):
            results = results_fixture()
            results["prefix-taps"]["rows"][1]["prefill_observation"][field] = value
            with self.subTest(observation=field):
                with self.assertRaises(ValueError):
                    analysis.validate_results(results)

    def test_analyze_rejects_incomplete_runs_before_numpy_or_capture_reads(self):
        results = results_fixture()
        results["prefix-off"]["state"] = "incomplete"
        write_json(self.root / "results.json", results)
        with patch.object(analysis, "np", None):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analysis.analyze(self.root)

    def test_tap_set_requires_complete_ordered_unique_stages(self):
        directory = self.root / "taps"
        taps = self.write_taps(directory)
        self.assertEqual(len(analysis.load_taps(self.root, directory)), len(analysis.STAGES))
        for bad in ([], taps[:-1], taps + taps[:1], list(reversed(taps))):
            self.save_taps(directory, bad)
            with self.subTest(count=len(bad)):
                with self.assertRaises(ValueError):
                    analysis.load_taps(self.root, directory)

    def test_unsafe_tap_paths_and_mismatched_identities_are_rejected(self):
        directory = self.root / "taps"
        taps = self.write_taps(directory)
        names = ("../outside.bin", "/outside.bin", "C:/outside.bin", "C:outside.bin",
                 "\\\\server\\share\\outside.bin", "..\\outside.bin", "sub/../layer-00-input.bin",
                 "layer-00-input.bin:stream", "layer-01-input.bin", "layer-00-input.bin.", None)
        for name in names:
            candidate = copy.deepcopy(taps)
            candidate[0]["file"] = name
            self.save_taps(directory, candidate)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "filename"):
                    analysis.load_taps(self.root, directory)

    def test_malformed_or_missing_tap_metadata_is_rejected(self):
        directory = self.root / "taps"
        taps = self.write_taps(directory)
        for item in (None, [], {}, {**taps[0], "layer": True}, {**taps[0], "stage": "unknown"},
                     {**taps[0], "op": None}):
            self.save_taps(directory, [item])
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    analysis.load_taps(self.root, directory)
        (directory / "taps.jsonl").write_text('{"layer":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata"):
            analysis.load_taps(self.root, directory)

    def test_symlink_escape_is_rejected(self):
        directory = self.root / "taps"
        self.write_taps(directory)
        outside = self.root.parent / (self.root.name + "-outside.bin")
        outside.write_bytes(b"not a diagnostic")
        self.addCleanup(outside.unlink, missing_ok=True)
        target = directory / "layer-00-input.bin"
        target.unlink()
        try:
            target.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation not available: {exc}")
        with self.assertRaisesRegex(ValueError, "escapes"):
            analysis.load_taps(self.root, directory)

    def test_unpinned_view_is_unusable_if_either_side_is_legacy(self):
        safe = {"op": "RESHAPE", "output_retention": "view_and_backing_chain"}
        for operation in analysis.VIEW_OPS:
            legacy = {"op": operation}
            self.assertIn("unusable", analysis.tap_unusable_reason(legacy, safe))
            self.assertIn("unusable", analysis.tap_unusable_reason(safe, legacy))
        self.assertIsNone(analysis.tap_unusable_reason(safe, safe))
        self.assertIsNone(analysis.tap_unusable_reason({"op": "MUL_MAT"}, {"op": "MUL_MAT"}))

    def write_native(self, path, shape=(1, 1, 2), *, kind="f32"):
        write_tensor(path, shape)
        raw = struct.pack("<" + ("e" if kind == "f16" else "f") * math.prod(shape),
                          *range(1, math.prod(shape) + 1))
        path.with_suffix(".raw").write_bytes(raw)
        write_json(path.with_suffix(".json"), {"type": kind, "heads": shape[0], "rows": shape[1],
                   "head_dim": shape[2], "raw_bytes": len(raw), "packed_head_major": True})
        return raw

    def test_native_kv_requires_nonempty_size_consistent_metadata(self):
        path = self.root / "kv.bin"
        for kind in ("f16", "f32"):
            expected = self.write_native(path, kind=kind)
            self.assertEqual(analysis.native_kv(self.root, path), (kind, (1, 1, 2), expected))
            path.with_suffix(".raw").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "payload size"):
                analysis.native_kv(self.root, path)
        self.write_native(path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        for field, value in (("type", "q8_0"), ("rows", 0), ("rows", True),
                             ("raw_bytes", 1), ("packed_head_major", False)):
            write_json(path.with_suffix(".json"), {**metadata, field: value})
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    analysis.native_kv(self.root, path)

    def complete_fixture(self, *, legacy_view=False):
        results = results_fixture()
        write_json(self.root / "results.json", results)
        for name in ("baseline-taps", "prefix-taps"):
            for row in results[name]["rows"]:
                obs = row["prefill_observation"]
                directory = self.root / name / "taps" / f"prefill-{row['group'] + 1:04d}"
                taps = self.write_taps(directory, view=legacy_view, pinned=not legacy_view)
                count, offset = obs["T"], obs["n_past"]
                values = list(range(1 + offset * 2, 1 + (offset + count) * 2))
                for tap in taps:
                    write_tensor(directory / tap["file"], (count, 2), values)
                total = count + offset
                for part in ("k", "v"):
                    self.write_native(directory / f"layer-00-live-after-{part}.bin", (1, total, 2))
                    if name == "prefix-taps":
                        phases = ("saved-after-snapshot",) if not row["group"] else (
                            "live-prefix-before", "saved-prefix-before")
                        for phase in phases:
                            self.write_native(directory / f"layer-00-{phase}-{part}.bin")

    @unittest.skipIf(analysis.np is None, "offline NumPy not installed")
    def test_complete_synthetic_analysis_proves_paired_taps_and_restores(self):
        self.complete_fixture()
        result = analysis.analyze(self.root)
        self.assertFalse(result["timing_eligible"])
        self.assertTrue(result["tap_comparison_complete"])
        self.assertEqual(len(result["cases"]), 4)
        self.assertTrue(result["captures"])
        self.assertTrue(all(case["first_difference"] is None for case in result["cases"]))
        restores = [kv for case in result["cases"] for kv in case["kv"] if "restore_native_byte_exact" in kv]
        self.assertEqual(len(restores), 6)
        self.assertTrue(all(kv["restore_native_byte_exact"] for kv in restores))

    @unittest.skipIf(analysis.np is None, "offline NumPy not installed")
    def test_legacy_view_garbage_is_not_a_first_difference(self):
        self.complete_fixture(legacy_view=True)
        for name in ("baseline-taps", "prefix-taps"):
            for directory in (self.root / name / "taps").iterdir():
                # Invalid/nonfinite recycled bytes must not even be parsed as an observation.
                (directory / "layer-00-attn-output.bin").write_bytes(b"unusable view backing")
        result = analysis.analyze(self.root)
        self.assertFalse(result["tap_comparison_complete"])
        for case in result["cases"]:
            self.assertIsNone(case["first_difference"])
            self.assertEqual(case["unusable_taps"][0]["stage"], "attn-output")
            self.assertNotIn("attn-output", [tap["stage"] for tap in case["layer_taps"]])

    @unittest.skipIf(analysis.np is None, "offline NumPy not installed")
    def test_real_nonview_difference_is_still_reported(self):
        self.complete_fixture(legacy_view=True)
        path = self.root / "prefix-taps/taps/prefill-0003/layer-00-q-proj.bin"
        write_tensor(path, (1, 2), (3.5, 4))
        result = analysis.analyze(self.root)
        first = result["cases"][2]["first_difference"]
        self.assertEqual((first["layer"], first["stage"], first["max_abs"]), (0, "q-proj", 0.5))

    @unittest.skipIf(analysis.np is None, "offline NumPy not installed")
    def test_pinned_view_difference_is_usable(self):
        self.complete_fixture()
        for name in ("baseline-taps", "prefix-taps"):
            for directory in (self.root / name / "taps").iterdir():
                path = directory / "taps.jsonl"
                taps = [json.loads(line) for line in path.read_text().splitlines()]
                next(tap for tap in taps if tap["stage"] == "attn-output")["op"] = "RESHAPE"
                self.save_taps(directory, taps)
        write_tensor(self.root / "prefix-taps/taps/prefill-0003/layer-00-attn-output.bin", (1, 2), (3.5, 4))
        result = analysis.analyze(self.root)
        self.assertTrue(result["tap_comparison_complete"])
        self.assertEqual(result["cases"][2]["first_difference"]["stage"], "attn-output")

    @unittest.skipIf(analysis.np is None, "offline NumPy not installed")
    def test_missing_kv_pair_and_wrong_tap_shape_are_rejected(self):
        self.complete_fixture()
        missing = self.root / "prefix-taps/taps/prefill-0004/layer-00-live-after-v.bin"
        saved = missing.read_bytes()
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "missing diagnostic"):
            analysis.analyze(self.root)
        missing.write_bytes(saved)
        write_tensor(self.root / "prefix-taps/taps/prefill-0004/layer-00-input.bin", (1, 2))
        with self.assertRaisesRegex(ValueError, "prompt rows"):
            analysis.analyze(self.root)

    @unittest.skipIf(analysis.np is None, "offline NumPy not installed")
    def test_tensor_rejects_nonfinite_and_unmatched_shapes(self):
        path = self.root / "bad.bin"
        for value in (math.nan, math.inf, -math.inf):
            write_tensor(path, (1,), [value])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                analysis.tensor(path)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            analysis.difference(analysis.np.zeros((1,)), analysis.np.zeros((2,)))


if __name__ == "__main__":
    unittest.main()
