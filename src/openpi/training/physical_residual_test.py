from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.model as model_api
import openpi.models.pi0 as pi0
from openpi.training.physical_residual import (
    PhysicalResidualHead, balanced_task_batch, paired_flow_inputs, physical_residual_loss,
    residual_head_value_and_grad, residual_training_loss, sample_actions_with_physical_residual,
)


def inputs():
    return (jnp.ones((2, 3, 5)), jnp.ones((2, 32)), jnp.ones((2, 3, 32)), jnp.array([.2, .8]))


def test_zero_init_padding_and_gate():
    head = PhysicalResidualHead(5, rngs=nnx.Rngs(0))
    out = head(*inputs())
    np.testing.assert_array_equal(out["residual32"], np.zeros((2, 3, 32)))
    np.testing.assert_allclose(out["corrected_velocity"], inputs()[2])
    np.testing.assert_allclose(out["surrogate_gain_gate"], .25, atol=1e-6)


def test_physical7_loss_ignores_padded_target_and_horizon_mask():
    base = jnp.zeros((2, 2, 32)); target = jnp.zeros((2, 2, 32)).at[..., 7:].set(99)
    loss, metrics = physical_residual_loss(base_velocity=base, residual7=jnp.zeros((2,2,7)),
        gate=jnp.full((2,2,1),.25), target_velocity=target, task_ids=jnp.array([0,1]),
        num_tasks=2, valid_horizon=jnp.array([[1,0],[1,1]]), gate_weight=0)
    assert float(loss) == 0 and float(metrics["physical_fm"]) == 0


def test_base_hidden_velocity_and_target_are_stop_gradient():
    head = PhysicalResidualHead(5, rngs=nnx.Rngs(1))
    hidden, state, velocity, time = inputs()
    assert float(jnp.linalg.norm(jax.grad(lambda h: head(h,state,velocity,time)["corrected_velocity"].sum())(hidden))) == 0
    assert float(jnp.linalg.norm(jax.grad(lambda v: head(hidden,state,v,time)["residual7"].sum())(velocity))) == 0
    residual=jnp.ones((2,3,7))*.1; gate=jnp.ones((2,3,1))*.25; target=jnp.zeros((2,3,32))
    fn=lambda y: physical_residual_loss(base_velocity=velocity,residual7=residual,gate=gate,
        target_velocity=y,task_ids=jnp.array([0,1]),num_tasks=2)[0]
    assert float(jnp.linalg.norm(jax.grad(fn)(target))) == 0


def test_paired_inputs_are_caller_determined_and_validate_shapes():
    actions=jnp.ones((2,3,32));noise=jnp.zeros_like(actions);out=paired_flow_inputs(actions,noise,jnp.array([.25,.5]))
    np.testing.assert_allclose(out["noisy_actions"][0],.75);np.testing.assert_allclose(out["target_velocity"],-1)
    with pytest.raises(ValueError): paired_flow_inputs(actions[...,:7],noise[...,:7],jnp.array([.25,.5]))


def test_balanced_sampler_is_deterministic_and_exact():
    intervals={task:[(task*100,50),(task*100+50,50)] for task in range(10)}
    a,ta=balanced_task_batch(intervals,seed=7,step=3);b,tb=balanced_task_batch(intervals,seed=7,step=3)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ta,tb)
    assert len(a)==40 and dict(zip(*np.unique(ta,return_counts=True),strict=True))=={i:4 for i in range(10)}
    assert all(task*100 <= index < task*100+100 for index,task in zip(a,ta,strict=True))


def test_invalid_shapes_and_sampler_fail_closed():
    head=PhysicalResidualHead(5,rngs=nnx.Rngs(2))
    with pytest.raises(ValueError): head(jnp.ones((2,3,4)),*inputs()[1:])
    with pytest.raises(ValueError): balanced_task_batch({i:[(0,1)] for i in range(9)},seed=0,step=0)


class _Observation:
    def __init__(self, state): self.state = state


class _CachedLlm:
    def __call__(self, inputs, **kwargs):
        del kwargs
        prefix, suffix = inputs
        if suffix is None: return (prefix, None), "fixed-cache"
        return (None, suffix + .125), "fixed-cache"


class _TinyFrozenBase:
    action_horizon, action_dim = 2, 32
    sample_actions = pi0.Pi0.sample_actions

    def __init__(self):
        self.PaliGemma = type("PaliGemma", (), {"llm": _CachedLlm()})()
        self.scale = jnp.asarray(1.25)
        self.flow_calls = 0

    def embed_prefix(self, observation):
        del observation
        return jnp.ones((2,1,4)), jnp.ones((2,1),bool), jnp.array([False])

    def embed_suffix(self, observation, x_t, time):
        del observation
        return x_t[...,:4] + time[:,None,None], jnp.ones((2,2),bool), jnp.array([True,False]), jnp.zeros((2,4))

    def action_out_proj(self, hidden):
        return jnp.tile(hidden, (1,1,8)) * self.scale

    def flow_features(self, observation, x_t, time):
        self.flow_calls += 1
        hidden = self.embed_suffix(observation, x_t, time)[0] + .125
        return self.action_out_proj(hidden), hidden


def test_training_wrapper_one_flow_call_and_head_only_grad(monkeypatch):
    monkeypatch.setattr(model_api, "preprocess_observation", lambda key, obs, train: obs)
    base = _TinyFrozenBase(); head = PhysicalResidualHead(4, rngs=nnx.Rngs(3))
    obs = _Observation(jnp.ones((2,32))); actions = jnp.ones((2,2,32))
    before = np.asarray(base.scale).copy()
    (loss, metrics), grads = residual_head_value_and_grad(
        base, head, jax.random.key(4), obs, actions, jnp.array([0,1]), num_tasks=2
    )
    assert jnp.isfinite(loss) and jnp.isfinite(metrics["physical_fm"])
    assert base.flow_calls == 1 and np.array_equal(np.asarray(base.scale), before)
    assert jax.tree.leaves(grads)


def test_opt_in_solver_zero_head_and_gate_off_match_native(monkeypatch):
    monkeypatch.setattr(model_api, "preprocess_observation", lambda key, obs, train: obs)
    base = _TinyFrozenBase(); head = PhysicalResidualHead(4, rngs=nnx.Rngs(5))
    obs = _Observation(jnp.ones((2,32))); noise = jnp.arange(128,dtype=jnp.float32).reshape(2,2,32)/100
    native = base.sample_actions(jax.random.key(0), obs, num_steps=3, noise=noise)
    no_head = sample_actions_with_physical_residual(
        base, None, jax.random.key(0), obs, num_steps=3, noise=noise
    )
    zero = sample_actions_with_physical_residual(base, head, jax.random.key(0), obs, num_steps=3, noise=noise)
    gate_off = sample_actions_with_physical_residual(
        base, head, jax.random.key(0), obs, num_steps=3, noise=noise, gate_override=0.0
    )
    np.testing.assert_array_equal(no_head, native)
    np.testing.assert_allclose(zero, native, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(gate_off, native)
    out = head(jnp.ones((2,2,4)), obs.state, jnp.ones((2,2,32)), jnp.ones((2,)))
    np.testing.assert_array_equal(out["residual32"][...,7:], 0)
