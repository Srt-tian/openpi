from __future__ import annotations

import unittest

import aggregate_pi05_goal3_phase_screen as target


IDENTITY = {"policy_id": "base", "checkpoint_sha256": "a" * 64,
            "base_graph": "original_pi05_libero", "adapter_sha256": None}


def evidence(which: str, steps: int = 110):
    stages = [{"instruction": instruction, "max_steps": budget, "skill": "pi05",
               "until": None, "on_timeout": "abort" if index == len(target.expected_stages(which)) - 1 else "next"}
              for index, (instruction, budget) in enumerate(target.expected_stages(which))]
    expected = target.expected_stages(which)
    boundary = expected[0][1] if len(expected) == 2 else None
    trace = [{"step": step, "stage": int(boundary is not None and step >= boundary),
              "skill": "pi05", "state": [0.0] * 8, "action": [0.0] * 7}
             for step in range(steps)]
    calls, receipts = [], []
    for index in range((steps + 4) // 5):
        seed, step = 1157 + index * 1000003, index * 5
        stage = int(boundary is not None and step >= boundary)
        calls.append({"inference_call": index, "policy_seed": seed, "metadata": IDENTITY,
                      "response_valid": True})
        receipts.append({"inference_call": index, "policy_seed": seed,
                         "prompt": stages[stage]["instruction"], "status": "ok",
                         "observation_image": {"sha256": "b" * 64, "shape": [224, 224, 3], "dtype": "uint8"},
                         "observation_wrist_image": {"sha256": "c" * 64, "shape": [224, 224, 3], "dtype": "uint8"},
                         "observation_state": {"sha256": "d" * 64, "shape": [8], "dtype": "float64"}})
    return {"case": {"policy_seed": 1157}, "harness": {"remember": [], "stages": stages},
            "runner": {"report": {"steps": steps, "trace": trace},
                       "policy_calls": calls, "payload_hashes": receipts}}


class Goal3PhaseScreenTest(unittest.TestCase):
    def test_prefix100_stage_and_prompt_pass(self):
        self.assertEqual(target.audit_episode(evidence("prefix100"), "prefix100", IDENTITY), [])

    def test_new_prefix_boundaries_and_prompts_pass(self):
        for which, boundary in (("prefix25", 25), ("prefix50", 50), ("prefix75", 75)):
            with self.subTest(which=which):
                value = evidence(which)
                self.assertEqual(value["runner"]["report"]["trace"][boundary - 1]["stage"], 0)
                self.assertEqual(value["runner"]["report"]["trace"][boundary]["stage"], 1)
                self.assertEqual(target.audit_episode(value, which, IDENTITY), [])

    def test_invalid_arm_sets_fail_closed(self):
        for arms in ([], ["prefix25"], ["control", "control"], ["control", "prefix42"]):
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                target.validate_arms(arms)
        with self.assertRaises(ValueError):
            target.expected_stages("prefix42")

    def test_early_done_before_switch_is_allowed(self):
        self.assertEqual(target.audit_episode(evidence("prefix150", 80), "prefix150", IDENTITY), [])

    def test_wrong_boundary_fails(self):
        value = evidence("prefix100")
        value["runner"]["report"]["trace"][99]["stage"] = 1
        self.assertIn("stage switch is not at the fixed boundary",
                      target.audit_episode(value, "prefix100", IDENTITY))

    def test_prompt_and_call_seed_fail_closed(self):
        value = evidence("prefix100")
        value["runner"]["payload_hashes"][20]["prompt"] = target.OPEN
        errors = target.audit_episode(value, "prefix100", IDENTITY)
        self.assertIn("payload prompt does not match stage instruction", errors)
        value = evidence("prefix100")
        value["runner"]["policy_calls"][1]["policy_seed"] += 1
        errors = target.audit_episode(value, "prefix100", IDENTITY)
        self.assertIn("noncontinuous inference seed/call identity", errors)

    def test_initial_receipt_excludes_prompt(self):
        left, right = evidence("control", 5), evidence("sequential", 5)
        self.assertEqual(target.initial_receipt(left), target.initial_receipt(right))
        self.assertTrue(target.valid_initial(target.initial_receipt(left)))

    def test_bad_call_count_status_and_receipt_fail(self):
        value = evidence("control")
        value["runner"]["policy_calls"].pop()
        errors = target.audit_episode(value, "control", IDENTITY)
        self.assertIn("policy call/hash recording incomplete", errors)
        self.assertIn("non-probe call count is not ceil(steps/5)", errors)
        value = evidence("control")
        value["runner"]["payload_hashes"][0]["status"] = "error"
        self.assertIn("noncontinuous inference seed/call identity",
                      target.audit_episode(value, "control", IDENTITY))
        broken = target.initial_receipt(evidence("control"))
        broken["observation_state"]["sha256"] = None
        self.assertFalse(target.valid_initial(broken))


if __name__ == "__main__":
    unittest.main()
