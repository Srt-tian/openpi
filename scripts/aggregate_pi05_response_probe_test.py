from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import aggregate_pi05_response_probe as target


def episode(manual=False):
    trace = [{"step": i, "state": [float(i)] * 8, "action": [float(i)] * 7,
              "skill": "pi05"} for i in range(8)]
    manual_rows = ([
        {"emission_index": 5, "kind": "close", "expected_actual_step": 5, "executed": True,
         "action": [0.0] * 7, "post_state8": [6.0] * 8},
        {"emission_index": 6, "kind": "reopen", "expected_actual_step": 6, "executed": True,
         "action": [0.0] * 7, "post_state8": [7.0] * 8},
    ] if manual else [])
    if manual:
        trace[5]["action"] = [0.0] * 7
        trace[6]["action"] = [0.0] * 7
    provenance = {"kind": "pi05_response_probe_v1", "execution_reconciled": True,
                  "manual_actions_emitted": 2 if manual else 0,
                  "events": [{"trigger": 5}] if manual else [],
                  "emitted_rows": copy.deepcopy(manual_rows), "executed_rows": manual_rows}
    return {"runner": {"report": {"trace": trace, "success": False, "status": "failure",
                                    "skills": {"pi05": {"provenance": provenance}}},
                       "policy_calls": [{"policy_seed": 7}, {"policy_seed": 1000010}],
                       "payload_hashes": [{"sha256": "a" * 64}, {"sha256": "b" * 64}]}}


class ResponseProbeAggregateTest(unittest.TestCase):
    def validation_plan(self):
        return {"schema": "pi05_response_probe_case_plan.v1",
                "task": {"suite": "libero_goal", "task_id": 3},
                "cases": [{"suite": "libero_goal", "task_id": 3,
                            "init_id": init_id, "replicate_id": replicate_id,
                            "cohort": (target.KNOWN_COHORT
                                       if (init_id, replicate_id) in target.EXPECTED
                                       else target.EXPANSION_COHORT)}
                           for init_id in range(10) for replicate_id in range(10)]}

    def load_plan(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(value))
            return target.load_case_plan(path)

    def test_explicit_case_plan_is_exact_100_with_19_plus_81(self):
        expected, cohorts = self.load_plan(self.validation_plan())
        self.assertEqual(expected, target.FULL_VALIDATION)
        self.assertEqual(sum(value == target.KNOWN_COHORT for value in cohorts.values()), 19)
        self.assertEqual(sum(value == target.EXPANSION_COHORT for value in cohorts.values()), 81)

    def test_explicit_case_plan_rejects_duplicate_and_wild_cases(self):
        duplicate = self.validation_plan()
        duplicate["cases"][-1] = copy.deepcopy(duplicate["cases"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load_plan(duplicate)
        for field, value in (("init_id", 10), ("replicate_id", -1), ("init_id", True)):
            wild = self.validation_plan()
            wild["cases"][0][field] = value
            with self.assertRaises(ValueError):
                self.load_plan(wild)
        unknown = self.validation_plan()
        unknown["cases"][0]["anything"] = "wildcard"
        with self.assertRaisesRegex(ValueError, "schema"):
            self.load_plan(unknown)

    def test_validation_breakdown_reports_all_known_expansion_and_each_init(self):
        rows = []
        for init_id in range(10):
            for replicate_id in range(10):
                rows.append({"init_id": init_id, "replicate_id": replicate_id,
                             "control_success": replicate_id % 2 == 0,
                             "probe_success": replicate_id % 3 == 0,
                             "outcome": "both" if replicate_id == 0 else "neither"})
        self.assertEqual(target.summarize(rows)["pairs"], 100)
        known = [row for row in rows
                 if (row["init_id"], row["replicate_id"]) in target.EXPECTED]
        expansion = [row for row in rows
                     if (row["init_id"], row["replicate_id"]) not in target.EXPECTED]
        self.assertEqual(target.summarize(known)["pairs"], 19)
        self.assertEqual(target.summarize(expansion)["pairs"], 81)
        self.assertTrue(all(target.summarize(
            [row for row in rows if row["init_id"] == init_id])["pairs"] == 10
                            for init_id in range(10)))

    def test_strong_parameters_raise_manual_bound_and_reserve_strictly(self):
        control, probe = episode(False), episode(True)
        provenance = probe["runner"]["report"]["skills"]["pi05"]["provenance"]
        provenance["parameters"] = {"lift_z_command": .2, "max_lift_steps": 20,
            "lift_target_m": .025, "native_reserve_steps": 80,
            "minimum_actual": 100, "max_manual_actions": 26,
            "trigger_remaining_steps": 111}
        result = target.causal_pair(control, probe)
        self.assertTrue(result["causal_gate_pass"])
        self.assertEqual(result["response_probe_parameters"], provenance["parameters"])

    def test_invalid_parameter_provenance_is_confounded(self):
        control, probe = episode(False), episode(True)
        probe["runner"]["report"]["skills"]["pi05"]["provenance"]["parameters"] = {
            "lift_z_command": .2, "max_lift_steps": 20, "lift_target_m": .025,
            "native_reserve_steps": 20, "minimum_actual": 100,
            "max_manual_actions": 14,
            "trigger_remaining_steps": 39}
        result = target.causal_pair(control, probe)
        self.assertFalse(result["causal_gate_pass"])
        self.assertIn("invalid response-probe provenance parameters", result["confounds"])

    def test_legacy_structured_parameters_default_minimum_actual_120(self):
        control, probe = episode(False), episode(True)
        provenance = probe["runner"]["report"]["skills"]["pi05"]["provenance"]
        provenance["parameters"] = {"lift_z_command": .2, "max_lift_steps": 20,
            "lift_target_m": .025, "native_reserve_steps": 80,
            "max_manual_actions": 26, "trigger_remaining_steps": 111}
        result = target.causal_pair(control, probe)
        self.assertTrue(result["causal_gate_pass"])
        self.assertEqual(result["response_probe_parameters"]["minimum_actual"], 120)

    def test_triggered_exact_prefix_passes(self):
        control, probe = episode(False), episode(True)
        result = target.causal_pair(control, probe)
        self.assertTrue(result["causal_gate_pass"])
        self.assertEqual(result["first_manual_actual_step"], 5)

    def test_pre_intervention_difference_is_confounded(self):
        control, probe = episode(False), episode(True)
        probe["runner"]["report"]["trace"][4]["state"][0] += 1e-15
        result = target.causal_pair(control, probe)
        self.assertFalse(result["causal_gate_pass"])
        self.assertIn("pre-intervention action/state trace differs", result["confounds"])

    def test_untriggered_result_difference_is_confounded(self):
        control, probe = episode(False), episode(False)
        probe["runner"]["report"]["success"] = True
        result = target.causal_pair(control, probe)
        self.assertFalse(result["causal_gate_pass"])
        self.assertIn("untriggered outcome differs", result["confounds"])

    def test_policy_payload_prefix_difference_is_confounded(self):
        control, probe = episode(False), episode(True)
        probe["runner"]["payload_hashes"][0] = copy.deepcopy(probe["runner"]["payload_hashes"][0])
        probe["runner"]["payload_hashes"][0]["sha256"] = "c" * 64
        result = target.causal_pair(control, probe)
        self.assertFalse(result["causal_gate_pass"])
        self.assertIn("pre-intervention payload_hashes differs", result["confounds"])

    def test_manual_trace_mismatch_is_confounded(self):
        control, probe = episode(False), episode(True)
        probe["runner"]["report"]["trace"][5]["action"][0] = 1.0
        result = target.causal_pair(control, probe)
        self.assertFalse(result["causal_gate_pass"])
        self.assertTrue(any("manual row does not match" in value for value in result["confounds"]))

    def test_explicit_done_truncation_is_allowed(self):
        control, probe = episode(False), episode(True)
        provenance = probe["runner"]["report"]["skills"]["pi05"]["provenance"]
        truncated = {"emission_index": 7, "kind": "settle", "expected_actual_step": 7,
                     "executed": False, "truncated_before_execution": True,
                     "action": [0.0] * 7}
        provenance["emitted_rows"].append(truncated)
        provenance["manual_actions_emitted"] = 3
        result = target.causal_pair(control, probe)
        self.assertTrue(result["causal_gate_pass"])


if __name__ == "__main__":
    unittest.main()
