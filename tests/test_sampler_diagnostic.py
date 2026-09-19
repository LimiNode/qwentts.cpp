"""Model-free regression for the Python sampler diagnostic seam."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional research deps
    raise unittest.SkipTest(f"diagnostic harness dependencies unavailable: {exc}")

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cossim_common as cc
except ModuleNotFoundError as exc:  # pragma: no cover - optional research deps
    cc = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _load_f64(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    ndim = struct.unpack_from("i", raw, 0)[0]
    offset = 4 + 4 * ndim
    shape = struct.unpack_from(f"{ndim}i", raw, 4)
    return np.frombuffer(raw, dtype=np.float64, offset=offset).reshape(shape)


@unittest.skipIf(cc is None, f"diagnostic harness dependencies unavailable: {_IMPORT_ERROR}")
class SamplerDiagnosticTest(unittest.TestCase):
    """The dump must describe the accumulator that selected the token."""

    def test_patched_multinomial_dumps_selection_accumulator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cc.enable_philox_sampling(1006)
            cc.enable_sampler_diagnostic(temp_dir, subseq=0)
            try:
                probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
                selected = cc.patched_multinomial(probabilities, 1)
            finally:
                torch.multinomial = cc._original_multinomial

            self.assertEqual(selected.item(), 1)
            dump_dir = Path(temp_dir)
            candidate_ids, _ = cc.load_dump(str(dump_dir / "sampler-candidate-ids.bin"))
            self.assertEqual(candidate_ids.astype(np.int64).tolist(), [3, 2, 1, 0])

            selection_cdf = _load_f64(dump_dir / "sampler-selection-cdf-f64.bin")
            self.assertTrue(np.allclose(selection_cdf, [0.1, 0.3, 0.6, 1.0], rtol=0, atol=1e-7))
            cdf_f32, _ = cc.load_dump(str(dump_dir / "sampler-cdf-f32.bin"))
            self.assertTrue(np.allclose(cdf_f32, [0.1, 0.3, 0.6, 1.0], rtol=0, atol=1e-7))
            selection_sum = _load_f64(dump_dir / "sampler-sum-f64.bin")
            selection_target = _load_f64(dump_dir / "sampler-target-f64.bin")
            self.assertEqual(selection_sum[0], 1.0)
            self.assertLess(abs(selection_target[0] - float(cc.philox_uniform(1006, 0))), 1e-12)
