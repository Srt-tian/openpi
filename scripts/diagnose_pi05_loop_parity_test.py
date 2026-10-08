#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest

import numpy as np

import diagnose_pi05_loop_parity as diagnostic


def payload(seed=1462):
    return {
        "observation/image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "observation/wrist_image": np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3),
        "observation/state": np.arange(8, dtype=np.float64),
        "prompt": "put the yellow and white mug in the microwave and close it",
        "policy_id": "goal",
        "policy_seed": seed,
    }


class FakeTransport:
    def __init__(self, outputs=None):
        self.metadata = {"policy_id": "goal"}
        self.outputs = list(outputs or [np.arange(35, dtype=np.float32).reshape(5, 7)])
        self.calls = 0
        self.closed = False

    def infer(self, request):
        del request
        value = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return {"actions": value.copy()}

    def close(self):
        self.closed = True


def hash_record(tag, response="same", seed=1462):
    return {
        "observation_image": {"sha256": f"image-{tag}"},
        "observation_wrist_image": {"sha256": f"wrist-{tag}"},
        "observation_state": {"sha256": f"state-{tag}"},
        "prompt": "prompt",
        "policy_id": "goal",
        "policy_seed": seed,
        "response_actions": {"sha256": response},
    }


class CaptureTest(unittest.TestCase):
    def test_capture_hashes_without_mutating_payload_or_adding_inference(self):
        request = payload()
        original = copy.deepcopy(request)
        inner = FakeTransport()
        records = []
        transport = diagnostic.ReplayCaptureTransport(inner, records, retain_payloads=True)
        response = transport.infer(request)
        self.assertEqual(inner.calls, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(transport.payloads), 1)
        self.assertEqual(request["policy_seed"], original["policy_seed"])
        for key in ("observation/image", "observation/wrist_image", "observation/state"):
            np.testing.assert_array_equal(request[key], original[key])
            np.testing.assert_array_equal(transport.payloads[0][key], original[key])
            self.assertIsNot(transport.payloads[0][key], request[key])
        self.assertEqual(
            records[0]["response_actions"]["sha256"],
            diagnostic.hashlib.sha256(response["actions"].tobytes()).hexdigest(),
        )

    def test_replay_calls_exactly_five_with_unchanged_seed(self):
        request = payload(seed=1462)
        inner = FakeTransport()
        result = diagnostic.replay_payload(lambda: inner, request)
        self.assertEqual(inner.calls, 5)
        self.assertTrue(inner.closed)
        self.assertEqual(result["policy_seed"], 1462)
        self.assertEqual(request["policy_seed"], 1462)
        self.assertTrue(result["stable"])


class AnalysisTest(unittest.TestCase):
    def test_same_input_different_action_selects_first_response_discordance(self):
        groups = {
            "A1": [hash_record("x", "a"), hash_record("y", "same")],
            "B": [hash_record("x", "b"), hash_record("y", "same")],
            "A2": [hash_record("x", "a"), hash_record("y", "same")],
        }
        result = diagnostic.analyze_hash_records(groups)
        self.assertEqual(result["selected_replay_call"], 0)
        self.assertEqual(result["selected_replay_reason"], "same_input_different_response")
        self.assertIsNone(result["first_input_hash_difference_call"])

    def test_input_difference_precedes_fallback(self):
        groups = {
            "A1": [hash_record("x"), hash_record("a")],
            "B": [hash_record("x"), hash_record("b")],
            "A2": [hash_record("x"), hash_record("a")],
        }
        result = diagnostic.analyze_hash_records(groups)
        self.assertEqual(result["selected_replay_call"], 1)
        self.assertEqual(result["selected_replay_reason"], "first_discordant_input")

    def test_no_difference_selects_zero_based_call37(self):
        rows = [hash_record(str(index), str(index), seed=1462 + index) for index in range(50)]
        result = diagnostic.analyze_hash_records({"A1": rows, "B": copy.deepcopy(rows), "A2": copy.deepcopy(rows)})
        self.assertEqual(result["selected_replay_call"], 37)
        self.assertEqual(result["selected_replay_reason"], "no_input_discordance_fallback_call37")


class SequenceTest(unittest.TestCase):
    def test_fixed_case_seed_and_aba_execution_order(self):
        case = diagnostic.fixed_case("goal")
        self.assertEqual(case["ambient_seed"], 1462)
        self.assertEqual(case["policy_seed"], 1462)
        observed = []

        def old(name):
            observed.append((name, "old"))
            return name

        def new(name):
            observed.append((name, "runner"))
            return name

        order, results = diagnostic.run_aba(old, new)
        self.assertEqual(order, ["A1", "B", "A2"])
        self.assertEqual(observed, [("A1", "old"), ("B", "runner"), ("A2", "old")])
        self.assertEqual(results, {"A1": "A1", "B": "B", "A2": "A2"})


if __name__ == "__main__":
    unittest.main()
