"""Regression tests for the first-predictor-mismatch diagnostic mapping."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_first_mismatch as afm


def _write_dump(path: Path, rows: list[list[int]]) -> None:
    shape = (len(rows), len(rows[0]))
    values = [float(value) for row in rows for value in row]
    path.write_bytes(
        struct.pack("<i2i", 2, *shape) + struct.pack(f"<{len(values)}f", *values)
    )


def _write_vector(path: Path, values: list[float]) -> None:
    path.write_bytes(
        struct.pack("<i i", 1, len(values))
        + struct.pack(f"<{len(values)}f", *values)
    )


class FirstMismatchTest(unittest.TestCase):
    """The first mismatch must identify the producing graph pass."""

    def test_predictor_mapping_uses_codebook_minus_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            native_dir = Path(temp_dir) / "native"
            python_dir = Path(temp_dir) / "python"
            native_dir.mkdir()
            python_dir.mkdir()
            native = [[10, 20, 30, 40], [50, 60, 70, 80]]
            python = [[10, 20, 30, 40], [50, 60, 71, 80]]
            _write_dump(native_dir / "codes-full.bin", native)
            _write_dump(python_dir / "codes-full.bin", python)

            report = afm.analyze(native_dir, python_dir)

            self.assertEqual(
                report["first_mismatch"],
                {
                    "frame": 1,
                    "codebook": 2,
                    "graph_step": 1,
                    "philox_subsequence": 18,
                    "native_token": 70,
                    "python_token": 71,
                },
            )

    def test_sampler_summary_mismatch_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            native_dir = Path(temp_dir) / "native"
            python_dir = Path(temp_dir) / "python"
            native_dir.mkdir()
            python_dir.mkdir()
            _write_dump(native_dir / "codes-full.bin", [[10, 20], [30, 40]])
            _write_dump(python_dir / "codes-full.bin", [[10, 20], [30, 41]])
            (python_dir / "sampler-diagnostic.txt").write_text(
                "frame=1\npredictor_step=2\n", encoding="ascii"
            )

            report = afm.analyze(native_dir, python_dir)

            self.assertFalse(report["sampler_target"]["python_summary_matches_target"])

    def test_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            native_dir = Path(temp_dir) / "native"
            python_dir = Path(temp_dir) / "python"
            native_dir.mkdir()
            python_dir.mkdir()
            _write_dump(native_dir / "codes-full.bin", [[10, 20], [30, 40]])
            _write_dump(python_dir / "codes-full.bin", [[10, 20], [30, 40], [50, 60]])

            with self.assertRaisesRegex(ValueError, "identical shapes"):
                afm.analyze(native_dir, python_dir)

    def test_frame_zero_predictor_dumps_are_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            native_dir = Path(temp_dir) / "native"
            python_dir = Path(temp_dir) / "python"
            native_dir.mkdir()
            python_dir.mkdir()
            _write_dump(native_dir / "codes-full.bin", [[10, 20, 30]])
            _write_dump(python_dir / "codes-full.bin", [[10, 20, 31]])
            for directory in (native_dir, python_dir):
                _write_vector(directory / "predictor-logits-step1.bin", [1.0, 2.0])
                _write_vector(directory / "predictor-hidden-step1.bin", [3.0, 4.0])

            report = afm.analyze(native_dir, python_dir)

            self.assertEqual(report["first_mismatch"]["frame"], 0)
            self.assertIsNotNone(report["predictor_logits"])
            self.assertIsNotNone(report["predictor_hidden"])


if __name__ == "__main__":
    unittest.main()
