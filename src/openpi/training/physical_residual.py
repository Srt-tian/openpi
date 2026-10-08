"""CPU-reviewable PI0.5 physical-residual prototype; not a training entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from flax import nnx
import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0

PHYSICAL_DIM = 7
PADDED_ACTION_DIM = 32


class PhysicalResidualHead(nnx.Module):
    """Small 7-D residual and surrogate-gain gate over frozen base features."""

    def __init__(self, feature_dim: int, state_dim: int = 8, bottleneck: int = 256, *, rngs: nnx.Rngs):
        if min(feature_dim, state_dim, bottleneck) <= 0 or state_dim > PADDED_ACTION_DIM:
            raise ValueError("feature_dim/state_dim/bottleneck must be positive and state_dim <= 32")
        self.feature_dim, self.state_dim = feature_dim, state_dim
        input_dim = feature_dim + state_dim + PHYSICAL_DIM + 3
        self.in_proj = nnx.Linear(input_dim, bottleneck, rngs=rngs)
        self.residual_out = nnx.Linear(
            bottleneck, PHYSICAL_DIM, kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros, rngs=rngs,
        )
        self.gate_out = nnx.Linear(
            bottleneck, 1, kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.constant(math.log(0.25 / 0.75)), rngs=rngs,
        )

    def __call__(self, base_action_hidden, normalized_state, base_velocity, flow_time):
        hidden = jax.lax.stop_gradient(jnp.asarray(base_action_hidden))
        state = jnp.asarray(normalized_state)
        velocity = jax.lax.stop_gradient(jnp.asarray(base_velocity))
        time = jnp.asarray(flow_time)
        if hidden.ndim != 3 or hidden.shape[-1] != self.feature_dim:
            raise ValueError("base_action_hidden must have shape [B,T,feature_dim]")
        batch, horizon = hidden.shape[:2]
        if state.shape != (batch, PADDED_ACTION_DIM) or velocity.shape != (batch, horizon, PADDED_ACTION_DIM):
            raise ValueError("normalized_state/base_velocity shape mismatch")
        if time.shape == (batch,):
            time = jnp.broadcast_to(time[:, None], (batch, horizon))
        if time.shape != (batch, horizon):
            raise ValueError("flow_time must have shape [B] or [B,T]")
        state = jnp.broadcast_to(state[:, None, : self.state_dim], (batch, horizon, self.state_dim))
        time_features = jnp.stack((time, jnp.sin(jnp.pi * time), jnp.cos(jnp.pi * time)), axis=-1)
        features = jnp.concatenate((hidden, state, velocity[..., :PHYSICAL_DIM], time_features), axis=-1)
        trunk = jax.nn.gelu(self.in_proj(features))
        residual7 = self.residual_out(trunk)
        gate = jax.nn.sigmoid(self.gate_out(trunk))
        residual32 = jnp.pad(residual7, ((0, 0), (0, 0), (0, PADDED_ACTION_DIM - PHYSICAL_DIM)))
        corrected = velocity + gate * residual32
        return {"corrected_velocity": corrected, "residual7": residual7,
                "residual32": residual32, "surrogate_gain_gate": gate}


def paired_flow_inputs(actions, noise, flow_time):
    """Construct one shared flow point/target; this helper never samples randomness."""
    actions, noise, time = map(jnp.asarray, (actions, noise, flow_time))
    if actions.ndim != 3 or actions.shape[-1] != PADDED_ACTION_DIM or noise.shape != actions.shape:
        raise ValueError("actions/noise must share shape [B,T,32]")
    if time.shape == (actions.shape[0],):
        canonical_time = time
        time = time[:, None, None]
    elif time.shape == actions.shape[:2]:
        canonical_time = time
        time = time[..., None]
    else:
        raise ValueError("flow_time must have shape [B] or [B,T]")
    actions, noise, time = (jax.lax.stop_gradient(value) for value in (actions, noise, time))
    return {"noisy_actions": time * noise + (1.0 - time) * actions,
            "target_velocity": noise - actions,
            "flow_time": jax.lax.stop_gradient(canonical_time)}


def _task_macro(values, task_ids, valid_examples, num_tasks: int):
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    task_ids = jnp.asarray(task_ids)
    if task_ids.ndim != 1 or values.shape != task_ids.shape or valid_examples.shape != task_ids.shape:
        raise ValueError("task macro inputs must all have shape [B]")
    weights = valid_examples.astype(values.dtype)
    assignment = jax.nn.one_hot(task_ids, num_tasks, dtype=values.dtype)
    sums = jnp.sum(assignment * (values * weights)[:, None], axis=0)
    counts = jnp.sum(assignment * weights[:, None], axis=0)
    present = counts > 0
    return jnp.sum(jnp.where(present, sums / jnp.maximum(counts, 1), 0)) / jnp.maximum(jnp.sum(present), 1)


def physical_residual_loss(
    *, base_velocity, residual7, gate, target_velocity, task_ids, num_tasks: int,
    valid_horizon=None, relative_margin: float = 0.05, correction_weight: float = 1e-3,
    gate_weight: float = 0.1,
):
    """Physical-7 task-macro FM plus paired teacher regret and surrogate gate loss."""
    base = jax.lax.stop_gradient(jnp.asarray(base_velocity))[..., :PHYSICAL_DIM]
    target = jax.lax.stop_gradient(jnp.asarray(target_velocity))[..., :PHYSICAL_DIM]
    residual, gate = jnp.asarray(residual7), jnp.asarray(gate)
    if base.ndim != 3 or target.shape != base.shape or residual.shape != base.shape:
        raise ValueError("base/target/residual must share shape [B,T,7]")
    if gate.shape != (*base.shape[:2], 1):
        raise ValueError("gate must have shape [B,T,1]")
    mask = jnp.ones(base.shape[:2], dtype=base.dtype) if valid_horizon is None else jnp.asarray(valid_horizon, base.dtype)
    if mask.shape != base.shape[:2]:
        raise ValueError("valid_horizon must have shape [B,T]")
    denom = jnp.maximum(mask.sum(axis=1), 1)
    reduce_horizon = lambda x: (x * mask).sum(axis=1) / denom
    e0 = reduce_horizon(jnp.mean(jnp.square(base - target), axis=-1))
    corrected = base + gate * residual
    ephi = reduce_horizon(jnp.mean(jnp.square(corrected - target), axis=-1))
    ungated_error = reduce_horizon(jnp.mean(jnp.square(base + residual - target), axis=-1))
    correction = reduce_horizon(jnp.mean(jnp.square(residual), axis=-1))
    valid_examples = mask.sum(axis=1) > 0
    fm = _task_macro(ephi, task_ids, valid_examples, num_tasks)
    regret = _task_macro(jax.nn.relu(ephi - e0 + relative_margin * e0), task_ids, valid_examples, num_tasks)
    correction_norm = _task_macro(correction, task_ids, valid_examples, num_tasks)
    # This detached teacher-relative label is a surrogate, not rollout-success supervision.
    label = jax.lax.stop_gradient((ungated_error < e0).astype(base.dtype))
    gate_probability = jnp.clip(reduce_horizon(gate[..., 0]), 1e-6, 1 - 1e-6)
    gate_bce = _task_macro(-(label * jnp.log(gate_probability) + (1-label) * jnp.log(1-gate_probability)),
                           task_ids, valid_examples, num_tasks)
    total = fm + regret + correction_weight * correction_norm + gate_weight * gate_bce
    return total, {"loss": total, "physical_fm": fm, "paired_relative_regret": regret,
                   "correction_norm": correction_norm, "surrogate_gate_bce": gate_bce,
                   "base_physical_error": _task_macro(e0, task_ids, valid_examples, num_tasks)}


def residual_training_loss(
    frozen_base_model, head: PhysicalResidualHead, rng, observation, actions, task_ids, *,
    num_tasks: int, valid_horizon=None, train: bool = True, relative_margin: float = 0.05,
    correction_weight: float = 1e-3, gate_weight: float = 0.1,
):
    """Preprocess once and evaluate one shared flow point for head-only training.

    The caller passes an explicit frozen base model. Base velocity/hidden and the
    flow target are stopped before the residual objective; this function does not
    update or return base parameters.
    """
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    actions = jnp.asarray(actions)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    flow = paired_flow_inputs(actions, noise, time)
    base_velocity, base_hidden = frozen_base_model.flow_features(
        observation, flow["noisy_actions"], flow["flow_time"]
    )
    base_velocity = jax.lax.stop_gradient(base_velocity)
    base_hidden = jax.lax.stop_gradient(base_hidden)
    outputs = head(base_hidden, observation.state, base_velocity, flow["flow_time"])
    loss, metrics = physical_residual_loss(
        base_velocity=base_velocity, residual7=outputs["residual7"],
        gate=outputs["surrogate_gain_gate"], target_velocity=flow["target_velocity"],
        task_ids=task_ids, num_tasks=num_tasks, valid_horizon=valid_horizon,
        relative_margin=relative_margin, correction_weight=correction_weight,
        gate_weight=gate_weight,
    )
    return loss, {**metrics, "mean_surrogate_gate": jnp.mean(outputs["surrogate_gain_gate"])}


def residual_head_value_and_grad(frozen_base_model, head: PhysicalResidualHead, *args, **kwargs):
    """Differentiate only ``head``; the explicit frozen base is a closed-over input."""
    def loss_fn(active_head):
        return residual_training_loss(frozen_base_model, active_head, *args, **kwargs)

    return nnx.value_and_grad(loss_fn, has_aux=True)(head)


def sample_actions_with_physical_residual(
    frozen_base_model, head: PhysicalResidualHead | None, rng, observation, *, num_steps: int = 10,
    noise=None, gate_override: float | None = None,
):
    """Opt-in PI0.5 cached-prefix solver; the model's native method is untouched."""
    if head is None:
        return frozen_base_model.sample_actions(
            rng, observation, num_steps=num_steps, noise=noise
        )
    observation = _model.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(
            rng, (batch_size, frozen_base_model.action_horizon, frozen_base_model.action_dim)
        )
    dt = -1.0 / num_steps
    prefix_tokens, prefix_mask, prefix_ar_mask = frozen_base_model.embed_prefix(observation)
    prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = frozen_base_model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    def step(carry):
        x_t, time = carry
        time_batch = jnp.broadcast_to(time, batch_size)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = frozen_base_model.embed_suffix(
            observation, x_t, time_batch
        )
        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_for_suffix, suffix_attn_mask], axis=-1)
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        )
        (_, suffix_out), _ = frozen_base_model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=suffix_positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
        )
        hidden = suffix_out[:, -frozen_base_model.action_horizon :]
        base_velocity = frozen_base_model.action_out_proj(hidden)
        outputs = head(
            jax.lax.stop_gradient(hidden), observation.state,
            jax.lax.stop_gradient(base_velocity), time_batch,
        )
        if gate_override == 0:
            velocity = base_velocity
        elif gate_override is not None:
            velocity = base_velocity + jnp.asarray(gate_override) * outputs["residual32"]
        else:
            velocity = outputs["corrected_velocity"]
        return x_t + dt * velocity, time + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    return jax.lax.while_loop(cond, step, (jnp.asarray(noise), 1.0))[0]


def balanced_task_batch(
    task_intervals: Mapping[int, Sequence[tuple[int, int]]], *, seed: int, step: int,
    samples_per_task: int = 4,
):
    """Return deterministic frame indices and task IDs, equally sampled by task."""
    tasks = sorted(task_intervals)
    if len(tasks) != 10 or samples_per_task != 4:
        raise ValueError("V2 LIBERO batches require exactly 10 tasks x 4 samples = global batch 40")
    indices, task_ids = [], []
    for task in tasks:
        intervals = list(task_intervals[task])
        if not intervals or any(type(start) is not int or type(length) is not int or start < 0 or length <= 0
                                for start, length in intervals):
            raise ValueError(f"invalid intervals for task {task}")
        total = sum(length for _, length in intervals)
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, task]))
        for offset in rng.integers(0, total, size=samples_per_task).tolist():
            for start, length in intervals:
                if offset < length:
                    indices.append(start + offset); task_ids.append(task); break
                offset -= length
    order = np.random.default_rng(np.random.SeedSequence([seed, step, 0xB40])).permutation(len(indices))
    return np.asarray(indices, np.int64)[order], np.asarray(task_ids, np.int32)[order]
