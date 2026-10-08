"""Unit tests for plugin_objectives; no external data or pytest fixtures."""

from __future__ import annotations

import copy
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from openpi.training import plugin_bank, plugin_objectives


def _mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))


def _assert_tree_equal(test: unittest.TestCase, left, right) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    test.assertEqual(left_tree, right_tree)
    test.assertEqual(len(left_leaves), len(right_leaves))
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_leaf, right_leaf)


class TinyLossModel(nnx.Module):
    def __init__(self):
        self.base = nnx.Param(jnp.array([[1.5], [-0.5]], dtype=jnp.float32))
        self.adapter_lora = nnx.Param(jnp.array([[0.1], [0.2]], dtype=jnp.float32))

    def train(self):
        pass

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation @ (self.base.value + self.adapter_lora.value)
        return jnp.square(prediction - actions)


def _adapter_filter(path, value):
    del value
    return any("adapter_lora" in str(part) for part in path)


class TinyPrefixLlm:
    def __call__(self, embedded, *, mask, positions):
        del mask, positions
        prefix, suffix = embedded
        # The final hidden state is deterministic and deliberately differs by
        # token so the test verifies masked mean pooling, not merely shape.
        increments = jnp.arange(prefix.shape[1], dtype=jnp.float32)[None, :, None]
        output = jnp.broadcast_to(
            prefix[..., :1] + increments, (*prefix.shape[:2], 2048)
        )
        return (output, suffix), None


class TinyPrefixModel:
    def __init__(self):
        self.PaliGemma = types.SimpleNamespace(llm=TinyPrefixLlm())

    def embed_prefix(self, observation):
        batch = observation.state.shape[0]
        tokens = jnp.broadcast_to(
            jnp.asarray([[[1.0], [3.0], [9.0]]], dtype=jnp.float32),
            (batch, 3, 1),
        )
        mask = jnp.broadcast_to(jnp.asarray([[True, True, False]]), (batch, 3))
        ar_mask = jnp.zeros((3,), dtype=jnp.bool_)
        return tokens, mask, ar_mask


class PluginObjectivesTest(unittest.TestCase):
    def test_real_pi05_lora_abstract_feature_shape(self):
        config = plugin_bank.pi05_lora_config()

        def abstract_extract():
            return plugin_objectives.extract_call_features(
                config.create(jax.random.key(11)),
                config.fake_obs(batch_size=2),
                jnp.asarray([0.0, 1000.0], dtype=jnp.float32),
                jax.random.key(12),
            )

        # eval_shape exercises the real PI0.5-LoRA prefix graph without allocating
        # its multi-billion-parameter tensors or loading a checkpoint.
        features = jax.eval_shape(abstract_extract)
        self.assertEqual(features.shape, (2, plugin_objectives.CALL_FEATURE_DIM))
        self.assertEqual(features.dtype, jnp.float32)

    def test_loss_config_rejects_invalid_weights(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                plugin_objectives.LossConfig(handoff_weight=value)
            with self.assertRaises(ValueError):
                plugin_objectives.LossConfig(call_weight=value)

    def test_zero_handoff_weight_matches_original_and_ignores_nan_handoff(self):
        graphdef, frozen, adapter = plugin_bank.partition_model(
            TinyLossModel(), _adapter_filter
        )
        tx = optax.adamw(1e-2, weight_decay=0.1)
        original_opt = tx.init(adapter)
        demo_observation = jnp.asarray([[1.0, 2.0]], dtype=jnp.float32)
        demo_actions = jnp.asarray([[0.25]], dtype=jnp.float32)
        rng = jax.random.key(7)

        expected_adapter, expected_opt, expected_metrics = plugin_bank.make_step(
            graphdef, tx, _mesh()
        )(frozen, adapter, original_opt, demo_observation, demo_actions, rng)
        actual_adapter, actual_opt, actual_metrics = (
            plugin_objectives.make_handoff_step(graphdef, tx, _mesh(), 0.0)(
                frozen,
                adapter,
                original_opt,
                demo_observation,
                demo_actions,
                jnp.asarray([[float("nan"), float("nan")]]),
                jnp.asarray([[float("nan")]]),
                rng,
            )
        )

        _assert_tree_equal(self, actual_adapter, expected_adapter)
        _assert_tree_equal(self, actual_opt, expected_opt)
        np.testing.assert_array_equal(actual_metrics["loss"], expected_metrics["loss"])
        np.testing.assert_array_equal(
            actual_metrics["fm_loss"], expected_metrics["loss"]
        )
        self.assertEqual(int(actual_metrics["count"]), 0)

    def test_handoff_label_changes_selected_adapter_gradient(self):
        graphdef, frozen, adapter = plugin_bank.partition_model(
            TinyLossModel(), _adapter_filter
        )
        frozen_before = copy.deepcopy(frozen.to_pure_dict())
        tx = optax.sgd(0.05)
        step = plugin_objectives.make_handoff_step(graphdef, tx, _mesh(), 1.0)
        demo_obs = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)
        demo_actions = jnp.asarray([[0.0]], dtype=jnp.float32)
        handoff_obs = jnp.asarray([[1.0, -1.0]], dtype=jnp.float32)
        rng = jax.random.key(3)
        low, _, low_metrics = step(
            frozen,
            adapter,
            tx.init(adapter),
            demo_obs,
            demo_actions,
            handoff_obs,
            jnp.asarray([[0.0]], dtype=jnp.float32),
            rng,
        )
        high, _, high_metrics = step(
            frozen,
            adapter,
            tx.init(adapter),
            demo_obs,
            demo_actions,
            handoff_obs,
            jnp.asarray([[4.0]], dtype=jnp.float32),
            rng,
        )
        with self.assertRaises(AssertionError):
            _assert_tree_equal(self, low, high)
        _assert_tree_equal(self, frozen.to_pure_dict(), frozen_before)
        self.assertNotEqual(
            float(low_metrics["handoff_loss"]), float(high_metrics["handoff_loss"])
        )
        self.assertEqual(int(low_metrics["count"]), 1)

    def test_masked_loss_ignores_unobserved_nan(self):
        logits = jnp.asarray([[2.0, -1.0, 0.5]], dtype=jnp.float32)
        mask = jnp.asarray([[True, False, True]])
        with_nan, count = plugin_objectives.masked_call_loss(
            logits, jnp.asarray([[1.0, float("nan"), 0.0]]), mask
        )
        replaced, _ = plugin_objectives.masked_call_loss(
            logits, jnp.asarray([[1.0, 999.0, 0.0]]), mask
        )
        np.testing.assert_allclose(with_nan, replaced)
        self.assertTrue(bool(jnp.isfinite(with_nan)))
        self.assertEqual(float(count), 2.0)

    def test_all_missing_call_labels_do_not_advance_optimizer(self):
        head = plugin_objectives.init_call_head(
            jax.random.key(1), input_dim=4, hidden_dim=3
        )
        tx = optax.adamw(1e-2, weight_decay=0.2)
        opt_state = tx.init(head)
        head_before = copy.deepcopy(head)
        opt_before = copy.deepcopy(opt_state)
        new_head, new_opt, metrics = plugin_objectives.make_call_step(tx, 0.1)(
            head,
            opt_state,
            jnp.ones((2, 4), dtype=jnp.float32),
            jnp.full((2, len(plugin_objectives.POLICY_IDS)), jnp.nan),
            jnp.zeros((2, len(plugin_objectives.POLICY_IDS)), dtype=jnp.bool_),
        )
        _assert_tree_equal(self, new_head, head_before)
        _assert_tree_equal(self, new_opt, opt_before)
        self.assertEqual(float(metrics["call_loss"]), 0.0)
        self.assertEqual(float(metrics["observed_count"]), 0.0)
        self.assertEqual(float(metrics["head_grad_norm"]), 0.0)

    def test_observed_call_updates_head_but_not_features_and_zero_weight_is_exact_off(
        self,
    ):
        head = plugin_objectives.init_call_head(
            jax.random.key(21), input_dim=4, hidden_dim=3
        )
        tx = optax.adamw(1e-2, weight_decay=0.2)
        opt_state = tx.init(head)
        features = jnp.asarray(
            [[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]],
            dtype=jnp.float32,
        )
        targets = jnp.asarray(
            [
                [1.0, jnp.nan, jnp.nan, jnp.nan, jnp.nan],
                [jnp.nan, 0.0, jnp.nan, jnp.nan, jnp.nan],
            ],
            dtype=jnp.float32,
        )
        observed = jnp.asarray(
            [[True, False, False, False, False], [False, True, False, False, False]],
            dtype=jnp.bool_,
        )

        active_step = plugin_objectives.make_call_step(tx, 0.1)
        new_head, new_opt, metrics = active_step(
            head, opt_state, features, targets, observed
        )
        with self.assertRaises(AssertionError):
            _assert_tree_equal(self, new_head, head)
        with self.assertRaises(AssertionError):
            _assert_tree_equal(self, new_opt, opt_state)
        self.assertEqual(float(metrics["observed_count"]), 2.0)
        self.assertGreater(float(metrics["head_grad_norm"]), 0.0)

        feature_gradient = jax.grad(
            lambda candidate_features: active_step(
                head, opt_state, candidate_features, targets, observed
            )[2]["call_weighted_loss"]
        )(features)
        np.testing.assert_array_equal(feature_gradient, jnp.zeros_like(features))

        disabled_step = plugin_objectives.make_call_step(tx, 0.0)
        disabled_head, disabled_opt, disabled_metrics = disabled_step(
            head, opt_state, features, targets, observed
        )
        _assert_tree_equal(self, disabled_head, head)
        _assert_tree_equal(self, disabled_opt, opt_state)
        self.assertEqual(float(disabled_metrics["call_weighted_loss"]), 0.0)
        self.assertEqual(float(disabled_metrics["head_grad_norm"]), 0.0)
        self.assertEqual(float(disabled_metrics["observed_count"]), 2.0)

    def test_features_have_exact_shape_and_use_masked_prefix_mean(self):
        state = jnp.arange(64, dtype=jnp.float32).reshape(2, 32) / 64.0
        observation = types.SimpleNamespace(state=state)
        with mock.patch.object(
            plugin_objectives._model,
            "preprocess_observation",
            side_effect=lambda rng, obs, train: obs,
        ):
            features = plugin_objectives.extract_call_features(
                TinyPrefixModel(),
                observation,
                jnp.asarray([0.0, 1000.0]),
                jax.random.key(0),
            )
            state_gradient = jax.grad(
                lambda x: plugin_objectives.extract_call_features(
                    TinyPrefixModel(),
                    types.SimpleNamespace(state=x),
                    jnp.asarray([0.0, 1000.0]),
                    jax.random.key(0),
                ).sum()
            )(state)
        self.assertEqual(features.shape, (2, 2081))
        # Valid prefix outputs are 1 and 4, hence their masked mean is 2.5.
        np.testing.assert_allclose(features[:, :2048], 2.5)
        np.testing.assert_allclose(features[:, 2048:2080], state)
        np.testing.assert_allclose(features[:, -1], jnp.asarray([0.0, 1.0]), rtol=1e-6)
        self.assertTrue(bool(jnp.all(state_gradient == 0)))


if __name__ == "__main__":
    unittest.main()
