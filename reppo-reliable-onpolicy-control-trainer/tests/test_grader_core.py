"""Scoring regressions; no candidate training or third-party dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import grader_core as grader


def synthetic_panel():
    # These anchors give a passing raw return of exactly -100.
    return {
        "envs": [{"name": f"env{i}", "r_rand": -1000.0, "r_ref": 0.0} for i in range(6)],
        "run_seeds": list(range(12)),
    }


def results_for(panel, raw_return):
    return [
        {"env": env["name"], "seed": seed, "status": "ok",
         "checkpoint_returns": [raw_return] * 40}
        for env in panel["envs"] for seed in panel["run_seeds"]
    ]


class ScoreTests(unittest.TestCase):
    def test_full_panel_examples(self):
        panel = synthetic_panel()
        for raw_return, expected in [(-1000, 0), (-101, 0), (-100, 0.5), (-50, 0.75), (0, 1)]:
            with self.subTest(raw_return=raw_return):
                scored = grader.score_panel(panel, results_for(panel, raw_return))
                self.assertAlmostEqual(scored["metric"], expected)
                self.assertEqual(grader.graded_reward(scored["metric"], True), scored["metric"])

    def test_improvement_past_reference_on_both_real_panels(self):
        root = Path(__file__).parent / "hidden_data"
        for split in ("intermediate", "final"):
            panel = json.loads((root / split / "panel.json").read_text())
            with self.subTest(split=split):
                reference = results_for(panel, 0)
                better = copy.deepcopy(reference)
                by_name = {e["name"]: e for e in panel["envs"]}
                for old, new in zip(reference, better):
                    ref = by_name[old["env"]]["r_ref"]
                    old["checkpoint_returns"] = [ref] * 40
                    new["checkpoint_returns"] = [ref / 2] * 40
                a = grader.score_panel(panel, reference)
                b = grader.score_panel(panel, better)
                self.assertEqual(a["n_reliable"], 72)
                self.assertEqual(b["n_reliable"], 72)
                self.assertEqual(a["mean_final_s"], b["mean_final_s"])
                self.assertLess(a["metric"], b["metric"])
                self.assertLess(b["metric"], 1)

    def test_missing_checkpoint_retains_denominator_and_other_performance(self):
        panel = synthetic_panel()
        results = results_for(panel, -50)
        results[0]["checkpoint_returns"][-1] = None
        results[0]["status"] = "budget exhausted"
        scored = grader.score_panel(panel, results)
        self.assertEqual(scored["n_reliable"], 71)
        self.assertAlmostEqual(scored["runs"][0]["performance_improvement"], 0.45)
        self.assertAlmostEqual(scored["metric"], 0.5 * 71 / 72 + 0.5 * (719 * 0.5 / 720))

    def test_failed_runs_stay_in_denominator(self):
        panel = synthetic_panel()
        results = results_for(panel, -50)
        results[0]["checkpoint_returns"] = [None] * 40
        results[0]["status"] = "crashed"
        scored = grader.score_panel(panel, results)
        self.assertAlmostEqual(scored["metric"], 0.75 * 71 / 72)
        self.assertEqual(scored["n_runs"], 72)

    def test_only_final_ten_checkpoints_count(self):
        panel = synthetic_panel()
        results = results_for(panel, -50)
        for r in results:
            r["checkpoint_returns"][:30] = [None] * 30
        self.assertAlmostEqual(grader.score_panel(panel, results)["metric"], 0.75)

    def test_invalid_or_missing_returns_cannot_earn_credit(self):
        for value in (None, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                panel = synthetic_panel()
                scored = grader.score_panel(panel, results_for(panel, value))
                self.assertEqual(scored["metric"], 0)
                json.dumps(scored, allow_nan=False)
        self.assertEqual(grader.graded_reward(1, False), 0)

    def test_reject_incomplete_duplicate_or_short_panel(self):
        panel = synthetic_panel()
        complete = results_for(panel, 0)
        duplicate = copy.deepcopy(complete)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        short = copy.deepcopy(complete)
        short[0]["checkpoint_returns"].pop()
        for results in ([], complete[:-1], duplicate, short):
            with self.subTest(length=len(results)):
                with self.assertRaises(ValueError):
                    grader.score_panel(panel, results)

    def test_reject_invalid_anchors(self):
        for rand, ref in ((-1, -2), (0, 0), (0, 1), (float("nan"), 0)):
            with self.subTest(rand=rand, ref=ref):
                with self.assertRaises(ValueError):
                    grader.performance_scores([0], rand, ref)

    def test_performance_is_bounded(self):
        self.assertEqual(grader.performance_scores([-1000, -100, -50, 0, 100], -1000, 0),
                         [0, 0, 0.5, 1, 1])

    def test_component_and_version_diagnostics(self):
        panel = synthetic_panel()
        scored = grader.score_panel(panel, results_for(panel, -50))
        self.assertEqual(scored["metric_name"], "reliability_performance_score")
        self.assertEqual(scored["scoring_version"], "1.1.0")
        self.assertEqual(scored["reliable_success_fraction"], 1)
        self.assertEqual(scored["mean_performance_improvement"], 0.5)

    def test_verifier_writes_small_metric_and_separate_raw_results(self):
        panel = synthetic_panel()
        results = results_for(panel, -50.123456789012345)
        runner = types.SimpleNamespace(
            harden_shared_writable_paths=lambda: [],
            execute_panel=lambda *a, **kw: results,
            DeliverableError=type("DeliverableError", (Exception,), {}),
        )
        spec = importlib.util.spec_from_file_location(
            "reppo_verifier_test", Path(__file__).with_name("test_state.py")
        )
        verifier = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"run_panel": runner}):
            spec.loader.exec_module(verifier)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verifier.METRIC_PATH = root / "metric.json"
            verifier.PANEL_RESULTS_PATH = root / "panel_results.json"
            with mock.patch.object(verifier, "_privilege_drop", return_value=()), \
                 mock.patch.object(grader, "load_panel", return_value=panel):
                scored = verifier.result()
            self.assertTrue(scored["valid"])
            self.assertEqual(scored["metric"], scored["reward"])
            self.assertLess(verifier.METRIC_PATH.stat().st_size, 64 * 1024)
            self.assertEqual(json.loads(verifier.PANEL_RESULTS_PATH.read_text()), results)
            self.assertNotIn("checkpoint_returns", scored["runs"][0])


if __name__ == "__main__":
    unittest.main()
