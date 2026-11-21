"""Model-free checks for effective prefill differential settings."""

import copy
import unittest

from run_prefill_differential import dispatch_issues, effective_issues


class EffectiveDifferentialSettings(unittest.TestCase):
    def fixture(self):
        return [{"prefill_observation": {"diagnostic": True, "prefill_use_flash_attn": True,
                  "prefix_source": "full", "prefix_rows": 9, "prefix_hit": hit}}
                for hit in (False, True, True, True)]

    def test_full_source_miss_then_hits_are_required(self):
        rows = self.fixture()
        self.assertEqual(effective_issues(rows, True, True, False, 9), [])
        for field, value in (("prefix_hit", False), ("prefix_rows", 0), ("prefix_source", "split")):
            candidate = copy.deepcopy(rows)
            candidate[1]["prefill_observation"][field] = value
            self.assertTrue(effective_issues(candidate, True, True, False, 9))

    def test_ignored_diagnostic_and_attention_options_are_rejected(self):
        self.assertTrue(effective_issues(self.fixture(), True, False, False, 9))
        self.assertTrue(effective_issues(self.fixture(), True, True, True, 9))

    def test_missing_observation_is_not_a_control(self):
        self.assertTrue(effective_issues([{}], False, False, False, 0))

    def test_manual_nonprefix_control(self):
        rows = [{"prefill_observation": {"diagnostic": False, "prefill_use_flash_attn": False}}]
        self.assertEqual(effective_issues(rows, False, False, True, 0), [])

    def test_requested_dispatch_must_be_observed(self):
        self.assertTrue(dispatch_issues([], "default", True))
        self.assertTrue(dispatch_issues(["RESEARCH_MATMUL route=mmvq"], "forced-mmq", True))
        self.assertTrue(dispatch_issues([], "forced-cublas", True))
        self.assertEqual(dispatch_issues(["RESEARCH_MATMUL route=forced-mmq"], "forced-mmq", True), [])
        self.assertEqual(dispatch_issues([], "default", False), [])


if __name__ == "__main__":
    unittest.main()
