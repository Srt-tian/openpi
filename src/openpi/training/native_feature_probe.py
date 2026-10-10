"""Read-only native PI0.5 hidden probe for the proposed demo-support CFN."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

SCHEMA = "pi05-demo-support-feature-probe.v1"
FLOW_TIME = 0.1
EXECUTED_HORIZON = 5


def feature_probe_seed(base_seed: int, call_index: int) -> int:
    if type(base_seed) is not int or base_seed < 0 or type(call_index) is not int or call_index < 0:
        raise ValueError("invalid feature-probe seed inputs")
    raw = f"{SCHEMA}|noise|{base_seed}|{call_index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def common_noise(seed: int, horizon: int = 10, action_dim: int = 32) -> np.ndarray:
    if type(seed) is not int or seed < 0 or horizon != 10 or action_dim != 32:
        raise ValueError("feature probe requires an explicit seed and [10,32] noise")
    return np.random.default_rng(seed).standard_normal((horizon, action_dim), dtype=np.float32)


def canonical_physical_chunks(chunks7, action_stats) -> jnp.ndarray:
    chunks = np.asarray(chunks7, dtype=np.float32)
    if chunks.ndim != 3 or chunks.shape[1:] != (10, 7) or not np.isfinite(chunks).all():
        raise ValueError("provided chunks must be finite [batch,10,7]")
    q01, q99 = (np.asarray(getattr(action_stats, name), dtype=np.float32)[:7] for name in ("q01", "q99"))
    if q01.shape != (7,) or q99.shape != (7,) or not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError("base action quantiles must be finite physical7")
    normalized = (chunks - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    return jnp.pad(jnp.asarray(normalized), ((0, 0), (0, 0), (0, 25)))


def probe_features(model, preprocessed_observation, chunks7, action_stats, noise32, *, flow_time=FLOW_TIME):
    """Return mean final-layer action hidden over the five actions actually executed."""
    if not isinstance(flow_time, (float, int)) or float(flow_time) != FLOW_TIME:
        raise ValueError("feature probe flow_time must remain exactly 0.1")
    actions = canonical_physical_chunks(chunks7, action_stats)
    noise = np.asarray(noise32, dtype=np.float32)
    if noise.shape != (10, 32) or not np.isfinite(noise).all():
        raise ValueError("common noise must be finite [10,32]")
    shared = jnp.broadcast_to(jnp.asarray(noise), actions.shape)
    x_t = FLOW_TIME * shared + (1.0 - FLOW_TIME) * actions
    times = jnp.full((actions.shape[0],), FLOW_TIME, dtype=jnp.float32)
    _, hidden = model.flow_features(preprocessed_observation, x_t, times)
    hidden = jnp.asarray(hidden)
    if hidden.ndim != 3 or hidden.shape[:2] != actions.shape[:2] or hidden.shape[-1] != 1024:
        raise ValueError("native final action hidden must be [batch,10,1024]")
    result = jnp.mean(hidden[:, :EXECUTED_HORIZON], axis=1)
    if result.shape != (actions.shape[0], 1024) or not bool(jnp.isfinite(result).all()):
        raise ValueError("feature probe returned invalid features")
    return result


def schema_record(*, base_sha256: str, norm_sha256: str) -> Mapping[str, Any]:
    for value in (base_sha256, norm_sha256):
        if not isinstance(value, str) or len(value) != 64: raise ValueError("digest required")
    return {"schema":SCHEMA,"source":"native_pi05.flow_features","base_sha256":base_sha256,
      "norm_sha256":norm_sha256,"layer":"action_expert_final_hidden","pooling":"mean_first5_executed",
      "feature_width":1024,"flow_time":FLOW_TIME,"noise":"common_iid32_domain_separated_explicit",
      "action_representation":"physical7_base_quantile_no_clip_then_pad32",
      "candidate_horizon":10,"executed_horizon":EXECUTED_HORIZON,
      "observables":"same_preprocessed_rgb_wrist_state_fullprompt","claim":"support_feature_not_success_probability"}
