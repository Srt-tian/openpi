"""Real-data objectives shared by the PI0.5 plugin-bank runner.

The handoff objective supervises one selected adapter with demonstrations and
successful real continuation trajectories.  The call objective trains a small,
shared head from the *current* observation only.  Neither objective differentiates
through a simulator, and the handoff loss does not imply direct parameter coupling
between different adapters: cross-policy information comes from the real handoff
data and the shared call head.

This module intentionally does not select banks or manufacture missing data.  The
runner owns routing and must use its original step when no handoff batch exists.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from openpi.models import model as _model
from openpi.models import pi0
from openpi.training import plugin_bank, sharding

POLICY_IDS = ("base", "spatial", "object", "goal", "long")
CALL_FEATURE_DIM = 2081
VLM_FEATURE_DIM = 2048
STATE_FEATURE_DIM = 32

CallHead: TypeAlias = dict[str, jax.Array]
HandoffStepFn: TypeAlias = Callable[
    [nnx.State, nnx.State, optax.OptState, Any, Any, Any, Any, jax.Array],
    tuple[nnx.State, optax.OptState, dict[str, jax.Array]],
]
FeatureExtractorFn: TypeAlias = Callable[
    [nnx.State, nnx.State, Any, Any, jax.Array], jax.Array
]
CallStepFn: TypeAlias = Callable[
    [CallHead, optax.OptState, jax.Array, jax.Array, jax.Array],
    tuple[CallHead, optax.OptState, dict[str, jax.Array]],
]


def _validate_weight(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


@dataclasses.dataclass(frozen=True)
class LossConfig:
    """Weights for optional real-data plugin objectives."""

    handoff_weight: float = 0.3
    call_weight: float = 0.1

    def __post_init__(self) -> None:
        _validate_weight("handoff_weight", self.handoff_weight)
        _validate_weight("call_weight", self.call_weight)


def _batch_count(tree: Any) -> jax.Array:
    leaves = jax.tree.leaves(tree)
    if not leaves or getattr(leaves[0], "ndim", 0) == 0:
        return jnp.asarray(0, dtype=jnp.int32)
    return jnp.asarray(leaves[0].shape[0], dtype=jnp.int32)


def make_handoff_step(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
    mesh: jax.sharding.Mesh,
    handoff_weight: float,
) -> HandoffStepFn:
    """Build a selected-adapter step with real successful-continuation loss.

    Only ``adapter`` is differentiated and optimizer-owned.  ``frozen`` is never
    donated.  When ``handoff_weight`` is zero, the original bank step is called
    with the original, unsplit RNG and handoff inputs are never inspected.  This
    makes disabling the objective numerically identical to the original update.
    """

    _validate_weight("handoff_weight", handoff_weight)
    if handoff_weight == 0:
        original_step = plugin_bank.make_step(graphdef, tx, mesh)

        def run_original(
            frozen: nnx.State,
            adapter: nnx.State,
            opt_state: optax.OptState,
            demo_observation: Any,
            demo_actions: Any,
            handoff_observation: Any,
            handoff_actions: Any,
            rng: jax.Array,
        ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
            del handoff_observation, handoff_actions
            new_adapter, new_opt_state, original_metrics = original_step(
                frozen, adapter, opt_state, demo_observation, demo_actions, rng
            )
            loss = original_metrics["loss"]
            metrics = {
                "loss": loss,
                "fm_loss": loss,
                "handoff_loss": jnp.zeros_like(loss),
                "grad_norm": original_metrics["grad_norm"],
                "adapter_norm": original_metrics["adapter_norm"],
                "count": jnp.asarray(0, dtype=jnp.int32),
            }
            return new_adapter, new_opt_state, metrics

        return run_original

    weight = jnp.asarray(handoff_weight, dtype=jnp.float32)

    def step(
        frozen: nnx.State,
        adapter: nnx.State,
        opt_state: optax.OptState,
        demo_observation: Any,
        demo_actions: Any,
        handoff_observation: Any,
        handoff_actions: Any,
        rng: jax.Array,
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        demo_rng, handoff_rng = jax.random.split(rng)

        def loss_fn(
            active_adapter: nnx.State,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            model = nnx.merge(graphdef, frozen, active_adapter)
            model.train()
            fm_loss = jnp.mean(
                model.compute_loss(demo_rng, demo_observation, demo_actions, train=True)
            )
            handoff_loss = jnp.mean(
                model.compute_loss(
                    handoff_rng,
                    handoff_observation,
                    handoff_actions,
                    train=True,
                )
            )
            return fm_loss + weight * handoff_loss, (fm_loss, handoff_loss)

        (loss, (fm_loss, handoff_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(adapter)
        updates, new_opt_state = tx.update(grads, opt_state, adapter)
        new_adapter = optax.apply_updates(adapter, updates)
        metrics = {
            "loss": loss,
            "fm_loss": fm_loss,
            "handoff_loss": handoff_loss,
            "grad_norm": optax.global_norm(grads),
            "adapter_norm": optax.global_norm(new_adapter),
            "count": _batch_count(handoff_actions),
        }
        return new_adapter, new_opt_state, metrics

    compiled = jax.jit(step)

    def run(*args: Any, **kwargs: Any):
        with sharding.set_mesh(mesh):
            return compiled(*args, **kwargs)

    return run


def init_call_head(
    rng: jax.Array,
    input_dim: int = CALL_FEATURE_DIM,
    hidden_dim: int = 256,
    num_policies: int = len(POLICY_IDS),
) -> CallHead:
    """Initialize the shared lightweight MLP as a pure parameter dictionary."""

    if input_dim <= 0 or hidden_dim <= 0 or num_policies <= 0:
        raise ValueError("call-head dimensions must all be positive")
    key1, key2 = jax.random.split(rng)
    init = jax.nn.initializers.glorot_uniform()
    return {
        "w1": init(key1, (input_dim, hidden_dim), jnp.float32),
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "w2": init(key2, (hidden_dim, num_policies), jnp.float32),
        "b2": jnp.zeros((num_policies,), dtype=jnp.float32),
    }


def logits_call_head(params: CallHead, features: jax.Array) -> jax.Array:
    """Return independent per-policy call logits."""

    hidden = jax.nn.gelu(
        jnp.asarray(features, dtype=jnp.float32) @ params["w1"] + params["b1"]
    )
    return hidden @ params["w2"] + params["b2"]


def extract_call_features(
    model: nnx.Module,
    observation: Any,
    budgets: Any,
    rng: jax.Array,
) -> jax.Array:
    """Extract 2081-D call features from the current observation only.

    The feature is ``[masked_mean(final VLM prefix hidden), normalized_state,
    normalized_budget]``.  It contains no terminal/future observation or success
    ground truth, and it is stopped before the call-head boundary.
    """

    observation = _model.preprocess_observation(rng, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    if prefix_out.shape[-1] != VLM_FEATURE_DIM:
        raise ValueError(
            f"expected {VLM_FEATURE_DIM}-D VLM prefix states, got {prefix_out.shape[-1]}"
        )
    state = jnp.asarray(observation.state, dtype=jnp.float32)
    if state.ndim != 2 or state.shape[-1] != STATE_FEATURE_DIM:
        raise ValueError(
            f"expected normalized state shape [batch, 32], got {state.shape}"
        )

    mask = jnp.asarray(prefix_mask, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    pooled = jnp.sum(
        jnp.asarray(prefix_out, dtype=jnp.float32) * mask[..., None], axis=1
    )
    pooled = pooled / denominator

    budget_steps = jnp.asarray(budgets, dtype=jnp.float32)
    if budget_steps.ndim == 0:
        budget_steps = jnp.broadcast_to(budget_steps, (state.shape[0], 1))
    elif budget_steps.ndim == 1:
        if budget_steps.shape[0] != state.shape[0]:
            raise ValueError("budget batch size does not match observation batch size")
        budget_steps = budget_steps[:, None]
    elif budget_steps.ndim != 2 or budget_steps.shape != (state.shape[0], 1):
        raise ValueError(
            f"expected budgets shape [batch] or [batch, 1], got {budget_steps.shape}"
        )
    budget_feature = jnp.log1p(budget_steps) / jnp.log(
        jnp.asarray(1001.0, dtype=jnp.float32)
    )

    features = jnp.concatenate([pooled, state, budget_feature], axis=-1)
    if features.shape[-1] != CALL_FEATURE_DIM:
        raise AssertionError(f"call feature contract changed: got {features.shape[-1]}")
    return jax.lax.stop_gradient(features)


def make_feature_extractor(
    graphdef: nnx.GraphDef,
    mesh: jax.sharding.Mesh,
) -> FeatureExtractorFn:
    """Build a pure JIT feature extractor over the frozen PI0 prefix path."""

    def feature_step(
        frozen: nnx.State,
        adapter: nnx.State,
        observation: Any,
        budgets: Any,
        rng: jax.Array,
    ) -> jax.Array:
        # PI0.5 action-expert adapters belong to expert 1.  Prefix inference only
        # executes expert 0, so the adapter completes the graph but cannot affect
        # the extracted feature (and a base/fallback graph definition still works).
        model = nnx.merge(graphdef, frozen, adapter)
        return extract_call_features(model, observation, budgets, rng)

    compiled = jax.jit(feature_step)

    def run(*args: Any, **kwargs: Any):
        with sharding.set_mesh(mesh):
            return compiled(*args, **kwargs)

    return run


def masked_call_loss(
    logits: jax.Array,
    targets: jax.Array,
    observed_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Stable sigmoid BCE normalized over labels that were actually observed."""

    logits = jnp.asarray(logits, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    observed = jnp.asarray(observed_mask, dtype=jnp.bool_)
    if logits.shape != targets.shape or logits.shape != observed.shape:
        raise ValueError(
            "logits, targets, and observed_mask must have the same shape; "
            f"got {logits.shape}, {targets.shape}, {observed.shape}"
        )

    # Select safe values before evaluating BCE: an untried policy may carry NaN
    # as its target and must never be interpreted as failure or poison the loss.
    safe_logits = jnp.where(observed, logits, 0.0)
    safe_targets = jnp.where(observed, targets, 0.0)
    elementwise = jax.nn.softplus(safe_logits) - safe_targets * safe_logits
    count = jnp.sum(observed, dtype=jnp.float32)
    loss = jnp.sum(jnp.where(observed, elementwise, 0.0)) / jnp.maximum(count, 1.0)
    return loss, count


def _masked_brier(
    logits: jax.Array,
    targets: jax.Array,
    observed_mask: jax.Array,
) -> jax.Array:
    observed = jnp.asarray(observed_mask, dtype=jnp.bool_)
    safe_logits = jnp.where(observed, jnp.asarray(logits, dtype=jnp.float32), 0.0)
    safe_targets = jnp.where(observed, jnp.asarray(targets, dtype=jnp.float32), 0.0)
    count = jnp.sum(observed, dtype=jnp.float32)
    squared_error = jnp.square(jax.nn.sigmoid(safe_logits) - safe_targets)
    return jnp.sum(jnp.where(observed, squared_error, 0.0)) / jnp.maximum(count, 1.0)


def make_call_step(
    tx: optax.GradientTransformation,
    call_weight: float,
) -> CallStepFn:
    """Build a JIT step which updates only call-head parameters.

    Features are always stop-gradient.  With no observed labels (or zero objective
    weight), the optimizer update branch is not executed, so momentum, schedules,
    and decoupled weight decay remain exactly unchanged.  ``brier`` is an
    uncalibrated diagnostic only; this API makes no calibration guarantee.
    """

    _validate_weight("call_weight", call_weight)
    weight = jnp.asarray(call_weight, dtype=jnp.float32)

    def step(
        head: CallHead,
        opt_state: optax.OptState,
        features: jax.Array,
        targets: jax.Array,
        observed_mask: jax.Array,
    ) -> tuple[CallHead, optax.OptState, dict[str, jax.Array]]:
        features = jax.lax.stop_gradient(features)
        diagnostic_logits = logits_call_head(head, features)
        diagnostic_loss, observed_count = masked_call_loss(
            diagnostic_logits, targets, observed_mask
        )
        brier = _masked_brier(diagnostic_logits, targets, observed_mask)

        def update(_: None):
            def loss_fn(active_head: CallHead) -> jax.Array:
                logits = logits_call_head(active_head, features)
                loss, _ = masked_call_loss(logits, targets, observed_mask)
                return weight * loss

            weighted_loss, grads = jax.value_and_grad(loss_fn)(head)
            updates, new_opt_state = tx.update(grads, opt_state, head)
            new_head = optax.apply_updates(head, updates)
            return new_head, new_opt_state, weighted_loss, optax.global_norm(grads)

        def skip(_: None):
            zero = jnp.zeros((), dtype=jnp.float32)
            return head, opt_state, weight * diagnostic_loss, zero

        should_update = jnp.logical_and(observed_count > 0, weight > 0)
        new_head, new_opt_state, weighted_loss, grad_norm = jax.lax.cond(
            should_update, update, skip, operand=None
        )
        metrics = {
            "call_loss": diagnostic_loss,
            "call_weighted_loss": weighted_loss,
            "observed_count": observed_count,
            "brier": brier,
            "head_grad_norm": grad_norm,
        }
        return new_head, new_opt_state, metrics

    return jax.jit(step)
