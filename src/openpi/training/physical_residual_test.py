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
    head = PhysicalResidualHead(5, width=8, horizon=3, ffn_dim=16, rngs=nnx.Rngs(0))
    out = head(*inputs())
    np.testing.assert_array_equal(out["residual32"], np.zeros((2, 3, 32)))
    np.testing.assert_allclose(out["corrected_velocity"], inputs()[2])
    np.testing.assert_allclose(out["surrogate_gain_gate"], .25, atol=1e-6)
    np.testing.assert_array_equal(out["surrogate_gain_gate"][:, :1], out["surrogate_gain_gate"][:, 1:2])


def test_physical7_loss_ignores_padded_target_and_horizon_mask():
    base = jnp.zeros((2, 2, 32)); target = jnp.zeros((2, 2, 32)).at[..., 7:].set(99)
    loss, metrics = physical_residual_loss(base_velocity=base, residual7=jnp.zeros((2,2,7)),
        gate=jnp.full((2,2,1),.25), target_velocity=target, task_ids=jnp.array([0,1]),
        num_tasks=2, valid_horizon=jnp.array([[1,0],[1,1]]), gate_weight=0)
    assert float(loss) == 0 and float(metrics["physical_fm"]) == 0


def test_base_hidden_velocity_and_target_are_stop_gradient():
    head = PhysicalResidualHead(5, width=8, horizon=3, ffn_dim=16, rngs=nnx.Rngs(1))
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
    head=PhysicalResidualHead(5,width=8,horizon=3,ffn_dim=16,rngs=nnx.Rngs(2))
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
    base = _TinyFrozenBase(); head = PhysicalResidualHead(4,width=8,horizon=2,ffn_dim=16,rngs=nnx.Rngs(3))
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
    base = _TinyFrozenBase(); head = PhysicalResidualHead(4,width=8,horizon=2,ffn_dim=16,rngs=nnx.Rngs(5))
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


def test_bfloat16_base_zero_residual_preserves_cached_native_update(monkeypatch):
    monkeypatch.setattr(model_api, "preprocess_observation", lambda key, obs, train: obs)
    base=_TinyFrozenBase();original=base.action_out_proj
    base.action_out_proj=lambda hidden: original(hidden).astype(jnp.bfloat16)
    head=PhysicalResidualHead(4,width=8,horizon=2,ffn_dim=16,rngs=nnx.Rngs(15))
    obs=_Observation(jnp.ones((2,32)));noise=jnp.arange(128,dtype=jnp.float32).reshape(2,2,32)/100
    native=base.sample_actions(jax.random.key(0),obs,num_steps=3,noise=noise)
    zero=sample_actions_with_physical_residual(base,head,jax.random.key(0),obs,num_steps=3,noise=noise)
    off=sample_actions_with_physical_residual(base,head,jax.random.key(0),obs,num_steps=3,noise=noise,gate_override=0.0)
    np.testing.assert_array_equal(zero,native)
    np.testing.assert_array_equal(off,native)


def test_gate_warmup_disables_only_auxiliary_bce_not_fm_gate_gradient():
    base=jnp.zeros((2,2,32));target=jnp.ones((2,2,32));residual=jnp.full((2,2,7),-.1);tasks=jnp.array([0,1])
    def loss(gate,suite_update):
        return physical_residual_loss(base_velocity=base,residual7=residual,gate=gate,
            target_velocity=target,task_ids=tasks,num_tasks=2,suite_update=suite_update)[0]
    gate=jnp.full((2,2,1),.25)
    assert float(jnp.linalg.norm(jax.grad(loss)(gate,0)))>0
    assert float(jnp.linalg.norm(jax.grad(loss)(gate,500)))>0


def test_raw_gate_training_matches_deployment_and_conflict_repro_is_removed():
    base=jnp.zeros((1,10,32)).at[...,:7].set(.25);target=jnp.zeros((1,10,32))
    residual=jnp.full((1,10,7),-.99);tasks=jnp.array([0]);mask=jnp.ones((1,10))
    def loss(g):
        return physical_residual_loss(base_velocity=base,residual7=residual,
            gate=jnp.full((1,10,1),g),target_velocity=target,task_ids=tasks,
            num_tasks=1,valid_horizon=mask,suite_update=500)[0]
    _,metrics=physical_residual_loss(base_velocity=base,residual7=residual,
        gate=jnp.full((1,10,1),.01),target_velocity=target,task_ids=tasks,
        num_tasks=1,valid_horizon=mask,suite_update=500)
    deployed=jnp.square(.25+.01*-.99)
    np.testing.assert_allclose(metrics["physical_fm"],deployed,rtol=1e-6)
    np.testing.assert_allclose(metrics["deployed_physical_fm"],deployed,rtol=1e-6)
    np.testing.assert_allclose(metrics["optimal_gate_target"],.25/.99,rtol=1e-6)
    assert float(jax.grad(loss)(jnp.array(.01)))<0


def test_zero_delta_has_zero_calibrated_blend_target_and_no_warmup_bce():
    base=jnp.ones((1,2,32));target=jnp.zeros((1,2,32));residual=jnp.zeros((1,2,7))
    _,warm=physical_residual_loss(base_velocity=base,residual7=residual,
        gate=jnp.full((1,2,1),.25),target_velocity=target,task_ids=jnp.array([0]),num_tasks=1,suite_update=499)
    _,active=physical_residual_loss(base_velocity=base,residual7=residual,
        gate=jnp.full((1,2,1),.25),target_velocity=target,task_ids=jnp.array([0]),num_tasks=1,suite_update=500)
    assert float(warm["effective_gate_weight"])==0
    assert float(active["optimal_gate_target"])==0


def test_masked_attention_pooling_and_invalid_residual_zero():
    head=PhysicalResidualHead(5,width=8,horizon=3,ffn_dim=16,rngs=nnx.Rngs(12))
    hidden,state,velocity,time=inputs();mask=jnp.array([[1,1,0],[1,0,0]],dtype=jnp.float32)
    out=head(hidden,state,velocity,time,mask)
    np.testing.assert_array_equal(np.asarray(out["residual7"])[np.asarray(mask)==0],0)
    with pytest.raises(ValueError,match="at least one valid"):
        head(hidden,state,velocity,time,jnp.zeros((2,3)))


def test_grouped_physical_loss_equal_weights_xyz_rotation_grip():
    base=jnp.zeros((3,1,32));gate=jnp.ones((3,1,1));tasks=jnp.arange(3);residual=jnp.zeros((3,1,7))
    target=jnp.zeros((3,1,32)).at[0,0,:3].set(1).at[1,0,3:6].set(1).at[2,0,6].set(1)
    _,metrics=physical_residual_loss(base_velocity=base,residual7=residual,gate=gate,
        target_velocity=target,task_ids=tasks,num_tasks=3,suite_update=0,correction_weight=0,gate_weight=0)
    np.testing.assert_allclose(metrics["base_physical_error"],1/3)
