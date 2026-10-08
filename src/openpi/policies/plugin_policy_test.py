import tempfile
import unittest
from unittest import mock

import numpy as np

from openpi.policies import plugin_policy


class _FakePolicy:
    def __init__(self, offset):
        self.offset = offset

    def infer(self, observation, *, noise=None):
        del observation
        return {"actions": np.asarray(noise) + self.offset}


class PluginPolicyTest(unittest.TestCase):
    def test_policy_ids_are_explicit_and_do_not_offer_automatic_routing(self):
        self.assertEqual(plugin_policy.POLICY_IDS, ("base", "spatial", "object", "goal", "long"))
        for invalid in ("auto", "call", "grasp", "", "SPATIAL"):
            with self.assertRaisesRegex(ValueError, "automatic routing is not available"):
                plugin_policy._require_policy_id(invalid)

    def test_base_selection_keeps_zero_b_template_and_only_verifies_bundle(self):
        initial = {suite: {"suite": suite, "zero_b": True} for suite in plugin_policy.plugin_bank.DEFAULT_SUITES}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                plugin_policy.plugin_bank,
                "verify_adapter_bank",
                return_value={"global_update_count": 4000},
            ) as verify, mock.patch.object(
                plugin_policy.plugin_bank,
                "load_adapter",
                side_effect=AssertionError("base selection loaded a trained adapter"),
            ):
                selected, manifest = plugin_policy._select_adapter(
                    "base",
                    initial,
                    plugin_policy.Path(directory),
                    params_path=plugin_policy.Path(directory) / "params",
                    norm_hash="norm",
                    base_hash="base",
                    allow_verified_base_relocation=False,
                )
        self.assertEqual(selected, initial["spatial"])
        self.assertIsNot(selected, initial["spatial"])
        self.assertEqual(manifest["global_update_count"], 4000)
        verify.assert_called_once()

    def test_suite_selection_loads_exactly_the_explicit_adapter(self):
        initial = {suite: {"suite": suite} for suite in plugin_policy.plugin_bank.DEFAULT_SUITES}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                plugin_policy.plugin_bank,
                "load_adapter",
                return_value=({"trained": "goal"}, {"global_update_count": 4000}),
            ) as load:
                selected, _ = plugin_policy._select_adapter(
                    "goal",
                    initial,
                    plugin_policy.Path(directory),
                    params_path=plugin_policy.Path(directory) / "params",
                    norm_hash="norm",
                    base_hash="base",
                    allow_verified_base_relocation=False,
                )
        self.assertEqual(selected, {"trained": "goal"})
        self.assertEqual(load.call_args.args[1], "goal")
        self.assertIs(load.call_args.args[2], initial["goal"])

    def test_same_noise_alignment_controls_official_baseline_label(self):
        observations = [{"case": 1}, {"case": 2}]
        noises = [np.zeros((2, 7), np.float32), np.ones((2, 7), np.float32)]
        aligned = plugin_policy.measure_same_noise_base_alignment(
            _FakePolicy(0.0), _FakePolicy(0.0), observations, noises
        )
        self.assertTrue(aligned["official_baseline_label_allowed"])
        different = plugin_policy.measure_same_noise_base_alignment(
            _FakePolicy(1e-2), _FakePolicy(0.0), observations, noises
        )
        self.assertFalse(different["official_baseline_label_allowed"])

    def test_same_noise_alignment_rejects_unpaired_cases(self):
        with self.assertRaisesRegex(ValueError, "same nonzero length"):
            plugin_policy.measure_same_noise_base_alignment(_FakePolicy(0), _FakePolicy(0), [], [])

    def test_fixed_service_checks_identity_pops_controls_and_pairs_noise(self):
        inner = mock.Mock()
        inner.metadata = {"checkpoint_sha256": "a" * 64}
        inner.infer.return_value = {"actions": np.zeros((10, 7), np.float32)}
        service = plugin_policy.FixedPolicyService(inner, "spatial")
        observation = {"policy_id": "spatial", "policy_seed": 123, "state": np.zeros(8)}
        service.infer(observation)
        payload = inner.infer.call_args.args[0]
        noise = inner.infer.call_args.kwargs["noise"]
        self.assertNotIn("policy_id", payload)
        self.assertNotIn("policy_seed", payload)
        self.assertEqual(noise.shape, (10, 32))
        self.assertEqual(noise.dtype, np.float32)
        np.testing.assert_array_equal(
            noise,
            np.random.default_rng(123).standard_normal((10, 32), dtype=np.float32),
        )
        self.assertEqual(service.metadata["policy_id"], "spatial")
        self.assertEqual(service.metadata["policy_seed_protocol"], plugin_policy.POLICY_SEED_PROTOCOL)

    def test_fixed_service_rejects_wrong_id_and_invalid_seed_before_infer(self):
        inner = mock.Mock()
        inner.metadata = {"checkpoint_sha256": "b" * 64}
        service = plugin_policy.FixedPolicyService(inner, "goal")
        for payload in (
            {"policy_id": "base", "policy_seed": 1},
            {"policy_id": "goal", "policy_seed": True},
            {"policy_id": "goal", "policy_seed": -1},
            {"policy_id": "goal", "policy_seed": "1"},
        ):
            with self.assertRaises(ValueError):
                service.infer(payload)
        inner.infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
