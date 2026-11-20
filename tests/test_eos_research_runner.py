"""Model-free checks for the bounded research runner."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "runner", Path(__file__).with_name("run_eos_research.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / ".tmp"
        cls.temp_root.mkdir(exist_ok=True)

    def test_plan_is_fixed_and_matched(self):
        runs = runner.plan()
        self.assertEqual(len(runs), 17)
        self.assertEqual(len({run["name"] for run in runs}), len(runs))
        policies = [run for run in runs if run["name"].startswith("independent-")]
        self.assertEqual(len(policies), 4)
        self.assertTrue(all(run["groups"] == policies[0]["groups"] for run in policies))
        self.assertEqual(len(policies[0]["groups"]), 36)
        controls = runs[0]["groups"]
        self.assertEqual(controls[:9], controls[9:])
        for run in runs:
            for group in run["groups"]:
                self.assertLessEqual(len(group["seeds"]), run["max_batch"])

    def test_exact_original_input(self):
        self.assertEqual(
            runner.hashlib.sha256(runner.ORIGINAL.encode()).hexdigest(),
            "d917e809733280d0fcc4603828e8253593839a22e29161216868f18ef4ba8a5e",
        )

    def test_trace_separates_interleaved_lanes(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            path = Path(folder) / "trace.log"
            codes = ",".join(str(value) for value in range(16))
            path.write_text(
                "[ResearchJob] serial=1 seed=64\n[ResearchJob] serial=2 seed=65\n"
                f"[ResearchFrame] serial=2 slot=1 frame=0 codes={codes}\n"
                f"[ResearchFrame] serial=1 slot=0 frame=0 codes={codes}\n"
            )
            rows = runner.parse_frames(path)
            self.assertEqual([row["seed"] for row in rows], [64, 65])
            self.assertEqual(rows[0]["codes"], [list(range(16))])
            self.assertEqual(rows[1]["codes"], [list(range(16))])

    def test_outputs_are_exclusive_and_not_coverage(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            path = Path(folder) / "capture.f32"
            path.write_bytes(struct.pack("<fff", 0, 0.25, -0.25))
            result = runner.make_wav(path)
            self.assertEqual(result["coverage"], "unassessed")
            self.assertEqual(result["human_listening"], "unassessed")
            with self.assertRaises(FileExistsError):
                runner.make_wav(path)
            output = Path(folder) / "result.json"
            runner.write_json(output, result)
            self.assertEqual(json.loads(output.read_text()), result)
            with self.assertRaises(FileExistsError):
                runner.write_json(output, result)


if __name__ == "__main__":
    unittest.main()
