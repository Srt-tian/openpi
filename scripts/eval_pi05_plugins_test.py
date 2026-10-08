import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PATH = Path(__file__).with_name("eval_pi05_plugins.py")
SPEC = importlib.util.spec_from_file_location("eval_pi05_plugins", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvalPi05PluginsTest(unittest.TestCase):
    def test_plan_is_16_cases_per_arm_and_suite_base_then_plugin(self):
        plan = MODULE.execution_plan(7)
        self.assertEqual(len(plan), 32)
        self.assertEqual(sum(row["arm"] == "base" for row in plan), 16)
        self.assertEqual(sum(row["arm"] == "plugin" for row in plan), 16)
        for suite in MODULE.SUITES:
            rows = [row for row in plan if row["suite"] == suite]
            self.assertEqual([row["arm"] for row in rows], ["base"] * 4 + ["plugin"] * 4)
            base = [(row["task_id"], row["init_id"], row["policy_seed"]) for row in rows[:4]]
            plugin = [(row["task_id"], row["init_id"], row["policy_seed"]) for row in rows[4:]]
            self.assertEqual(base, plugin)
            self.assertEqual({row["policy_id"] for row in rows[:4]}, {"base"})
            self.assertEqual({row["policy_id"] for row in rows[4:]}, {MODULE.PLUGIN_IDS[suite]})
            self.assertEqual({row["task_id"] for row in rows}, set(MODULE.TASK_IDS_BY_SUITE[suite]))

    def test_quaternion_conversion_is_non_mutating_and_eight_state_contract(self):
        quat = np.asarray([0.0, 0.0, 0.0, 1.0])
        before = quat.copy()
        np.testing.assert_array_equal(MODULE.quat_to_axis_angle(quat), np.zeros(3))
        np.testing.assert_array_equal(quat, before)

    def test_summary_counts_errors_in_denominator(self):
        rows = []
        for item in MODULE.execution_plan(7):
            rows.append({**item, "success": item["arm"] == "base", "status": "error" if item["arm"] == "plugin" else "success"})
        summary = MODULE.summarize(rows, "a" * 64, 32)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["episodes"], 32)
        self.assertEqual(summary["successes"], 16)
        self.assertEqual(summary["errors"], 16)

    def test_done_on_last_allowed_action_counts_as_success(self):
        class Environment:
            def __init__(self, cap):
                self.cap = cap
                self.steps = 0

            def step(self, action):
                self.steps += 1
                return {}, 0.0, self.steps == self.cap, {}

        class Skill:
            def __init__(self):
                self.seeds = []

            def act(self, observation, instruction, memory):
                self.seeds.append(observation.policy["policy_seed"])
                return np.zeros((MODULE.REPLAN_STEPS, 7), dtype=float)

        def payload(raw, instruction, policy_id, policy_seed):
            return {
                "observation/image": np.zeros((1, 1, 3), dtype=np.uint8),
                "observation/state": np.zeros(8, dtype=float),
                "policy_seed": policy_seed,
            }

        cap = 7
        skill = Skill()
        trace = []
        done, executed, calls = MODULE.execute_policy_steps(
            Environment(cap), {}, skill, "prompt", "base", 7, cap, [], trace, payload_builder=payload
        )
        self.assertTrue(done)
        self.assertEqual(executed, cap)
        self.assertEqual(calls, 2)
        self.assertEqual(skill.seeds, [7, 7 + MODULE.CALL_SEED_STRIDE])
        self.assertEqual(len(trace), cap)
        self.assertEqual(trace[-1]["done"], True)
        self.assertEqual(trace[-1]["inference_call"], 1)

    def test_filtered_batch_has_one_fixed_policy_id(self):
        plan = MODULE.execution_plan(7, ("libero_goal",), "plugin")
        self.assertEqual(len(plan), 4)
        self.assertEqual(MODULE.expected_policy_ids(plan), {"goal"})

    def test_single_arm_summary_has_no_paired_delta(self):
        rows = [
            {**item, "case_id": item["id"], "success": True, "status": "success"}
            for item in MODULE.execution_plan(7, ("libero_goal",), "plugin")
        ]
        summary = MODULE.summarize(rows, "a" * 64, 4)
        self.assertFalse(summary["suites"]["libero_goal"]["pairing_complete"])
        self.assertIsNone(summary["suites"]["libero_goal"]["paired_success_delta"])

    def test_episode_transport_validates_and_records_actual_connection_metadata(self):
        metadata = {
            "policy_id": "goal",
            "checkpoint_sha256": "a" * 64,
            "policy_seed_protocol": MODULE.SEED_PROTOCOL,
            "adapter_sha256": "b" * 64,
        }

        class Inner:
            def __init__(self):
                self.metadata = metadata

            def infer(self, payload):
                return {"actions": "fixture"}

            def close(self):
                pass

        transport = MODULE.EpisodeIdentityTransport(Inner(), metadata)
        self.assertEqual(transport.infer({}), {"actions": "fixture"})
        self.assertEqual(transport.verified_metadata, metadata)

        wrong = dict(metadata, policy_id="base")
        inner = Inner()
        inner.metadata = wrong
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.EpisodeIdentityTransport(inner, metadata).infer({})

    def test_base_requires_explicit_null_adapter(self):
        metadata = {
            "policy_id": "base",
            "checkpoint_sha256": "a" * 64,
            "policy_seed_protocol": MODULE.SEED_PROTOCOL,
            "adapter_sha256": None,
        }
        self.assertEqual(MODULE.validated_metadata_subset(metadata, "base", "a" * 64), metadata)
        with self.assertRaisesRegex(ValueError, "must attest"):
            MODULE.validated_metadata_subset(dict(metadata, adapter_sha256="b" * 64), "base", "a" * 64)


if __name__ == "__main__":
    unittest.main()
