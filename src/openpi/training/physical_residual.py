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


class _TemporalBlock(nnx.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, *, rngs: nnx.Rngs):
        if width % heads:
            raise ValueError("temporal width must be divisible by attention heads")
        self.width, self.heads = width, heads
        self.pre_attn = nnx.LayerNorm(width, rngs=rngs)
        self.qkv = nnx.Linear(width, 3 * width, rngs=rngs)
        self.attn_out = nnx.Linear(width, width, rngs=rngs)
        self.pre_ffn = nnx.LayerNorm(width, rngs=rngs)
        self.ffn_in = nnx.Linear(width, ffn_dim, rngs=rngs)
        self.ffn_out = nnx.Linear(ffn_dim, width, rngs=rngs)

    def __call__(self, x, valid_horizon=None):
        batch, horizon = x.shape[:2]
        qkv = self.qkv(self.pre_attn(x)).reshape(batch, horizon, 3, self.heads, self.width // self.heads)
        q, k, v = (qkv[:, :, index] for index in range(3))
        logits = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
        if valid_horizon is not None:
            valid_horizon = jnp.asarray(valid_horizon, dtype=jnp.bool_)
            logits = jnp.where(valid_horizon[:, None, None, :], logits, jnp.finfo(logits.dtype).min)
        weights = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bhts,bshd->bthd", weights, v).reshape(batch, horizon, self.width)
        x = x + self.attn_out(attended)
        return x + self.ffn_out(jax.nn.gelu(self.ffn_in(self.pre_ffn(x))))


class PhysicalResidualHead(nnx.Module):
    """Two-block temporal physical residual with one calibrated chunk blend."""

    def __init__(self, feature_dim: int, state_dim: int = 8, width: int = 256, *,
                 horizon: int = 10, heads: int = 4, ffn_dim: int = 1024,
                 residual_bound: float = 1.0, rngs: nnx.Rngs):
        if min(feature_dim, state_dim, width, horizon, heads, ffn_dim) <= 0 or state_dim > PADDED_ACTION_DIM:
            raise ValueError("head dimensions must be positive and state_dim <= 32")
        if not math.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError("residual_bound must be finite and positive")
        self.feature_dim, self.state_dim = feature_dim, state_dim
        self.horizon, self.residual_bound = horizon, residual_bound
        input_dim = feature_dim + state_dim + PHYSICAL_DIM + 3
        self.in_proj = nnx.Linear(input_dim, width, rngs=rngs)
        self.position_embedding = nnx.Param(jax.random.normal(rngs.params(), (horizon, width)) * .02)
        self.block0 = _TemporalBlock(width, heads, ffn_dim, rngs=rngs)
        self.block1 = _TemporalBlock(width, heads, ffn_dim, rngs=rngs)
        self.final_norm = nnx.LayerNorm(width, rngs=rngs)
        self.residual_out = nnx.Linear(
            width, PHYSICAL_DIM, kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros, rngs=rngs,
        )
        self.gate_out = nnx.Linear(
            width, 1, kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.constant(math.log(0.25 / 0.75)), rngs=rngs,
        )

    def __call__(self, base_action_hidden, normalized_state, base_velocity, flow_time, valid_horizon=None):
        hidden = jax.lax.stop_gradient(jnp.asarray(base_action_hidden))
        state = jnp.asarray(normalized_state)
        velocity = jax.lax.stop_gradient(jnp.asarray(base_velocity))
        time = jnp.asarray(flow_time)
        if hidden.ndim != 3 or hidden.shape[-1] != self.feature_dim:
            raise ValueError("base_action_hidden must have shape [B,T,feature_dim]")
        batch, horizon = hidden.shape[:2]
        if horizon != self.horizon:
            raise ValueError(f"head requires configured horizon {self.horizon}")
        if state.shape != (batch, PADDED_ACTION_DIM) or velocity.shape != (batch, horizon, PADDED_ACTION_DIM):
            raise ValueError("normalized_state/base_velocity shape mismatch")
        if time.shape == (batch,):
            time = jnp.broadcast_to(time[:, None], (batch, horizon))
        if time.shape != (batch, horizon):
            raise ValueError("flow_time must have shape [B] or [B,T]")
        mask = jnp.ones((batch, horizon), dtype=jnp.bool_) if valid_horizon is None else jnp.asarray(valid_horizon, dtype=jnp.bool_)
        if mask.shape != (batch, horizon) or not isinstance(mask, jax.core.Tracer) and not np.all(np.asarray(mask).any(axis=1)):
            raise ValueError("valid_horizon must have shape [B,T] with at least one valid step per example")
        state = jnp.broadcast_to(state[:, None, : self.state_dim], (batch, horizon, self.state_dim))
        time_features = jnp.stack((time, jnp.sin(jnp.pi * time), jnp.cos(jnp.pi * time)), axis=-1)
        features = jnp.concatenate((hidden, state, velocity[..., :PHYSICAL_DIM], time_features), axis=-1)
        features = jnp.where(mask[..., None], features, 0)
        trunk = self.in_proj(features) + self.position_embedding[None]
        trunk = self.block0(trunk, mask)
        trunk = self.final_norm(self.block1(trunk, mask))
        residual7 = jnp.where(mask[..., None], self.residual_bound * jnp.tanh(self.residual_out(trunk)), 0)
        pooled = jnp.sum(jnp.where(mask[..., None], trunk, 0), axis=1) / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1)
        chunk_gate = jax.nn.sigmoid(self.gate_out(pooled))[:, None, :]
        gate = jnp.broadcast_to(chunk_gate, (batch, horizon, 1))
        residual32 = jnp.pad(residual7, ((0, 0), (0, 0), (0, PADDED_ACTION_DIM - PHYSICAL_DIM)))
        corrected = velocity + gate * residual32
        return {"corrected_velocity": corrected, "residual7": residual7,
                "residual32": residual32, "surrogate_gain_gate": gate}


def paired_flow_inputs(actions, noise, flow_time):
    """Construct the native IID-noise flow point/target without sampling randomness."""
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


def experimental_repeat_last_flow_inputs(actions, noise, flow_time, valid_horizon):
    """Experimental non-default padding mode that repeats the last valid action/noise row."""
    actions, noise = map(jnp.asarray, (actions, noise))
    if actions.ndim != 3 or actions.shape[-1] != PADDED_ACTION_DIM or noise.shape != actions.shape:
        raise ValueError("actions/noise must share shape [B,T,32]")
    mask = jnp.asarray(valid_horizon, dtype=jnp.bool_)
    if mask.shape != actions.shape[:2]:
        raise ValueError("valid_horizon must have shape [B,T]")
    clean_actions = jnp.where(mask[..., None], actions, 0)
    clean_noise = jnp.where(mask[..., None], noise, 0)
    positions = jnp.arange(actions.shape[1])[None, :]
    last_index = jnp.max(jnp.where(mask, positions, -1), axis=1)
    safe_index = jnp.maximum(last_index, 0)[:, None, None]
    last_actions = jnp.take_along_axis(clean_actions, safe_index, axis=1)
    last_noise = jnp.take_along_axis(clean_noise, safe_index, axis=1)
    has_valid = (last_index >= 0)[:, None, None]
    actions = jnp.where(mask[..., None], clean_actions, jnp.where(has_valid, last_actions, 0))
    noise = jnp.where(mask[..., None], clean_noise, jnp.where(has_valid, last_noise, 0))
    return paired_flow_inputs(actions, noise, flow_time)


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
    gate_weight: float = 0.1, suite_update: int | jax.Array = 0, gate_warmup_updates: int = 500,
):
    """Physical-7 task-macro FM, paired regret, and calibrated blend loss."""
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
    def grouped_error(prediction):
        squared = jnp.square(prediction - target)
        return (jnp.mean(squared[..., :3], axis=-1) + jnp.mean(squared[..., 3:6], axis=-1)
                + squared[..., 6]) / 3
    warmup = jnp.asarray(suite_update) < gate_warmup_updates
    e0 = reduce_horizon(grouped_error(base))
    corrected = base + gate * residual
    ephi = reduce_horizon(grouped_error(corrected))
    correction = reduce_horizon(jnp.mean(jnp.square(residual), axis=-1))
    valid_examples = mask.sum(axis=1) > 0
    fm = _task_macro(ephi, task_ids, valid_examples, num_tasks)
    regret = _task_macro(jax.nn.relu(ephi - e0 + relative_margin * e0), task_ids, valid_examples, num_tasks)
    correction_norm = _task_macro(correction, task_ids, valid_examples, num_tasks)
    # Analytic physical-error-minimizing chunk blend strength, not success supervision.
    weights7 = jnp.asarray([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/3], dtype=base.dtype)
    error = base - target
    weighted_mask = mask[..., None] * weights7
    numerator = -jnp.sum(weighted_mask * error * residual, axis=(1, 2))
    denominator = jnp.sum(weighted_mask * jnp.square(residual), axis=(1, 2))
    optimal_gate = jnp.where(denominator > 1e-8,
                             jnp.clip(numerator / (denominator + 1e-8), 0, 1), 0)
    label = jax.lax.stop_gradient(optimal_gate)
    gate_probability = jnp.clip(reduce_horizon(gate[..., 0]), 1e-6, 1 - 1e-6)
    gate_bce = _task_macro(-(label * jnp.log(gate_probability) + (1-label) * jnp.log(1-gate_probability)),
                           task_ids, valid_examples, num_tasks)
    effective_gate_weight = jnp.where(warmup, 0.0, gate_weight)
    total = fm + regret + correction_weight * correction_norm + effective_gate_weight * gate_bce
    deployed_gain = _task_macro(e0 - ephi, task_ids, valid_examples, num_tasks)
    optimal_gate_metric = _task_macro(label, task_ids, valid_examples, num_tasks)
    return total, {"loss": total, "physical_fm": fm, "deployed_physical_fm": fm,
                   "deployed_physical_gain": deployed_gain, "paired_relative_regret": regret,
                   "correction_norm": correction_norm, "surrogate_gate_bce": gate_bce,
                   "base_physical_error": _task_macro(e0, task_ids, valid_examples, num_tasks),
                   "optimal_gate_target": optimal_gate_metric,
                   "gate_warmup": warmup, "effective_gate_weight": effective_gate_weight}


def residual_training_loss(
    frozen_base_model, head: PhysicalResidualHead, rng, observation, actions, task_ids, *,
    num_tasks: int, valid_horizon=None, train: bool = True, relative_margin: float = 0.05,
    correction_weight: float = 1e-3, gate_weight: float = 0.1,
    suite_update: int | jax.Array = 0, gate_warmup_updates: int = 500,
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
    outputs = head(base_hidden, observation.state, base_velocity, flow["flow_time"],
                   valid_horizon if train else None)
    loss, metrics = physical_residual_loss(
        base_velocity=base_velocity, residual7=outputs["residual7"],
        gate=outputs["surrogate_gain_gate"], target_velocity=flow["target_velocity"],
        task_ids=task_ids, num_tasks=num_tasks, valid_horizon=valid_horizon,
        relative_margin=relative_margin, correction_weight=correction_weight,
        gate_weight=gate_weight, suite_update=suite_update, gate_warmup_updates=gate_warmup_updates,
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
            correction = jnp.zeros_like(outputs["residual32"])
        elif gate_override is not None:
            correction = jnp.asarray(gate_override) * outputs["residual32"]
        else:
            correction = outputs["surrogate_gain_gate"] * outputs["residual32"]
        # Preserve the cached native solver's arithmetic/dtype path exactly when
        # the correction is zero; add the residual as a separate delta.
        native_next = x_t + dt * base_velocity
        return native_next + dt * correction, time + dt

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
