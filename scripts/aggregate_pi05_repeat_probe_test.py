from __future__ import annotations

import unittest

import aggregate_pi05_repeat_probe as aggregate
import build_pi05_repeat_probe as plan


def rows():
    result = []
    for suite, task, init in {row for values in plan.ASSIGNMENTS.values() for row in values}:
        plugin_id = {"libero_spatial": "spatial", "libero_object": "object",
                     "libero_goal": "goal", "libero_10": "long"}[suite]
        for rep in plan.REPLICATES:
            for policy in ("base", plugin_id):
                result.append({"policy_id": policy, "suite": suite, "task_id": task,
                               "init_id": init, "replicate_id": rep,
                               "status": "success", "success": True})
    return result


class AggregateRepeatProbeTest(unittest.TestCase):
    def test_complete_exact_grid_and_statistics(self):
        result = aggregate.aggregate(rows())
        self.assertTrue(result["complete"])
        self.assertEqual(result["coverage"], {"expected": 280, "observed_unique": 280,
                         "missing": 0, "extra": 0, "errors": 0,
                         "errors_retained_in_planned_denominator": True,
                         "errors_counted_as_policy_failures": False})
        self.assertEqual(len(result["cases"]), 14)
        for case in result["cases"]:
            self.assertEqual(case["marginal_wilson_95"]["base"]["trials"], 10)
            self.assertEqual(case["paired"]["both_succeeded"], 10)
            self.assertEqual(case["paired"]["exact_mcnemar_two_sided_p"], 1.0)

    def test_error_is_not_policy_failure_and_blocks_complete(self):
        evidence = rows()
        evidence[0].update(status="error", success=False)
        result = aggregate.aggregate(evidence)
        self.assertFalse(result["complete"])
        self.assertEqual(result["coverage"]["errors"], 1)
        case = next(item for item in result["cases"]
                    if item["case_id"] == f"{evidence[0]['suite']}/{evidence[0]['task_id']}/{evidence[0]['init_id']}")
        arm = "base" if evidence[0]["policy_id"] == "base" else "plugin"
        self.assertEqual(case["marginal_wilson_95"][arm]["trials"], 9)
        self.assertEqual(case["marginal_wilson_95"][arm]["errors"], 1)
        self.assertEqual(case["paired"]["error_pairs"], 1)

    def test_duplicate_rejected(self):
        evidence = rows()
        with self.assertRaises(ValueError):
            aggregate.aggregate(evidence + [dict(evidence[0])])

    def test_exact_paired_binomial(self):
        self.assertEqual(aggregate.exact_paired_pvalue(0, 0), 1.0)
        self.assertAlmostEqual(aggregate.exact_paired_pvalue(10, 0), 2 / 1024)


if __name__ == "__main__":
    unittest.main()
