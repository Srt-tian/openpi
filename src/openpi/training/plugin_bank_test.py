import copy

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import plugin_bank


class TinyModel(nnx.Module):
    def __init__(self):
        self.base = nnx.Param(jnp.array([[2.0], [-1.0]]))
        self.expert_lora_a = nnx.Param(jnp.array([[0.2], [0.3]]))
        self.expert_lora_b = nnx.Param(jnp.zeros((1, 1)))

    def train(self):
        pass

    def __call__(self, x):
        return x @ self.base + (x @ self.expert_lora_a) @ self.expert_lora_b

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        return jnp.square(self(observation) - actions)


def _is_tiny_lora(path, value):
    del value
    return any("lora" in str(part) for part in path)


def _mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))


def _assert_tree_equal(left, right):
    jax.tree.map(np.testing.assert_array_equal, left, right)


def test_zero_b_has_zero_residual():
    model = TinyModel()
    x = jnp.array([[3.0, 4.0]])
    np.testing.assert_allclose(model(x), x @ model.base.value)


def test_step_changes_adapter_but_not_frozen_and_has_finite_gradients():
    graphdef, frozen, adapter = plugin_bank.partition_model(TinyModel(), _is_tiny_lora)
    frozen_before = copy.deepcopy(frozen.to_pure_dict())
    tx = optax.adam(1e-2)
    run_step = plugin_bank.make_step(graphdef, tx, _mesh())
    new_adapter, _, metrics = run_step(
        frozen,
        adapter,
        tx.init(adapter),
        jnp.array([[1.0, 2.0]]),
        jnp.array([[1.0]]),
        jax.random.key(0),
    )
    _assert_tree_equal(frozen.to_pure_dict(), frozen_before)
    assert all(bool(jnp.isfinite(value)) for value in metrics.values())
    assert not np.array_equal(
        new_adapter.to_pure_dict()["expert_lora_b"],
        adapter.to_pure_dict()["expert_lora_b"],
    )


def test_eval_step_is_finite_and_does_not_mutate_state():
    graphdef, frozen, adapter = plugin_bank.partition_model(TinyModel(), _is_tiny_lora)
    frozen_before = copy.deepcopy(frozen.to_pure_dict())
    adapter_before = copy.deepcopy(adapter.to_pure_dict())
    metrics = plugin_bank.make_eval_step(graphdef, _mesh())(
        frozen,
        adapter,
        jnp.array([[1.0, 2.0]]),
        jnp.array([[0.0]]),
        jax.random.key(0),
    )
    assert all(bool(jnp.isfinite(value)) for value in metrics.values())
    _assert_tree_equal(frozen.to_pure_dict(), frozen_before)
    _assert_tree_equal(adapter.to_pure_dict(), adapter_before)


def test_round_robin_does_not_touch_inactive_adapter_or_optimizer():
    graphdef, frozen, adapter = plugin_bank.partition_model(TinyModel(), _is_tiny_lora)
    adapters = {"first": adapter, "second": copy.deepcopy(adapter)}
    tx = optax.adamw(1e-2, weight_decay=0.1)
    opt_states = plugin_bank.initialize_optimizer_states(tx, adapters)
    inactive_adapter = copy.deepcopy(adapters["second"].to_pure_dict())
    inactive_opt = copy.deepcopy(opt_states["second"])

    new_adapters, new_opts, selected, _ = plugin_bank.update_selected_bank(
        0,
        plugin_bank.make_step(graphdef, tx, _mesh()),
        frozen,
        adapters,
        opt_states,
        jnp.array([[1.0, 2.0]]),
        jnp.array([[0.0]]),
        jax.random.key(1),
    )
    assert selected == "first"
    _assert_tree_equal(new_adapters["second"].to_pure_dict(), inactive_adapter)
    _assert_tree_equal(new_opts["second"], inactive_opt)


def test_adapter_pure_dict_round_trip():
    graphdef, frozen, adapter = plugin_bank.partition_model(TinyModel(), _is_tiny_lora)
    del graphdef, frozen
    serialized = jax.tree.map(np.asarray, adapter.to_pure_dict())
    restored = copy.deepcopy(adapter)
    restored.replace_by_pure_dict(serialized)
    _assert_tree_equal(restored.to_pure_dict(), adapter.to_pure_dict())


def test_adapter_bank_checkpoint_round_trip(tmp_path):
    _, _, adapter = plugin_bank.partition_model(TinyModel(), _is_tiny_lora)
    adapters = {suite: copy.deepcopy(adapter) for suite in plugin_bank.SUITES}
    tx = optax.adam(1e-2)
    opt_states = plugin_bank.initialize_optimizer_states(tx, adapters)
    steps = {suite: index for index, suite in enumerate(plugin_bank.SUITES)}
    path = plugin_bank.save_bank(
        tmp_path / "bank",
        adapters,
        opt_states,
        steps,
        base_checkpoint_path="/checkpoint/params",
        norm_stats_hash="norm-hash",
        base_manifest_hash="source-manifest-hash",
        metadata_extra={"commit": "abc123"},
    )
    restored_adapters, restored_opts, restored_steps, manifest = plugin_bank.load_bank(
        path,
        adapters,
        opt_states,
        expected_base_checkpoint_path="/checkpoint/params",
        expected_norm_stats_hash="norm-hash",
        expected_base_manifest_hash="source-manifest-hash",
    )
    for suite in plugin_bank.SUITES:
        _assert_tree_equal(restored_adapters[suite].to_pure_dict(), adapters[suite].to_pure_dict())
        _assert_tree_equal(restored_opts[suite], opt_states[suite])
    assert restored_steps == steps
    assert manifest["global_update_count"] == sum(steps.values())
    assert manifest["metadata_extra"]["commit"] == "abc123"


def test_materialize_requires_every_base_leaf_and_zeros_all_b_factors():
    template = {
        "base": jax.ShapeDtypeStruct((2, 2), jnp.bfloat16),
        "llm": {
            "expert_1": {
                "x_lora_a": jax.ShapeDtypeStruct((2, 1), jnp.float32),
                "x_lora_b": jax.ShapeDtypeStruct((1, 2), jnp.float32),
            }
        },
    }
    restored = {"base": np.ones((2, 2), dtype=np.float32)}
    params = plugin_bank._materialize_params(template, restored, seed=7)
    np.testing.assert_array_equal(params["llm"]["expert_1"]["x_lora_b"], 0)
    assert np.any(np.asarray(params["llm"]["expert_1"]["x_lora_a"]) != 0)

    try:
        plugin_bank._materialize_params(template, {}, seed=7)
    except ValueError as error:
        assert "missing base" in str(error)
    else:
        raise AssertionError("missing base parameter was accepted")
