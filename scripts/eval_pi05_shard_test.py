import unittest

import eval_pi05_shard as evaluator
import pi05_shard_spec as spec


class EvalShardPlanTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {"seed": 7, "workers": {"0": {"cases": spec.worker_cases(0)}}}

    def test_arm_and_policy_are_derived_not_read_from_plan(self):
        base = evaluator.fixed_batch_plan(self.manifest, 0, "base", None)
        self.assertEqual(len(base), 70)
        self.assertEqual({row["policy_id"] for row in base}, {"base"})
        plugin = evaluator.fixed_batch_plan(self.manifest, 0, "plugin", "libero_goal")
        self.assertTrue(plugin)
        self.assertEqual({row["policy_id"] for row in plugin}, {"goal"})
        self.assertEqual({row["suite"] for row in plugin}, {"libero_goal"})

    def test_fixed_service_batch_rejects_ambiguous_filter(self):
        with self.assertRaises(ValueError):
            evaluator.fixed_batch_plan(self.manifest, 0, "plugin", None)
        with self.assertRaises(ValueError):
            evaluator.fixed_batch_plan(self.manifest, 0, "base", "libero_goal")


if __name__ == "__main__":
    unittest.main()
