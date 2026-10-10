"""demo_support_cfn_v1: a feature-in/support-proxy-out CPU prototype.

This borrows TACO's fixed-random-prior residual-MSE idea, but does not implement
TACO feature search, feature extraction, or a success/Q estimator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

SCHEMA = "demo_support_cfn_v1"
FEATURE_FIELDS = {"base_sha256", "input_dim", "width", "output_dim", "pooling",
                  "layer", "time", "noise", "action_normalization", "executed_horizon"}


def validate_feature_schema(value: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != FEATURE_FIELDS or value != expected:
        raise ValueError("feature schema mismatch")
    if not isinstance(value["base_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["base_sha256"]) is None:
        raise ValueError("invalid base digest")
    for key in ("input_dim", "width", "output_dim"):
        if type(value[key]) is not int or value[key] < 1: raise ValueError(f"invalid {key}")
    for key in ("pooling", "layer", "time", "noise", "action_normalization"):
        if not isinstance(value[key], str) or not value[key]: raise ValueError(f"invalid {key}")
    if value["executed_horizon"] != 5:
        raise ValueError("feature schema must describe the five executed actions")
    return copy.deepcopy(value)


def coin_label(sample_key: str, output_dim: int = 64) -> torch.Tensor:
    """Fixed Rademacher label keyed only by the logical training sample."""
    if not isinstance(sample_key, str) or not sample_key or type(output_dim) is not int or output_dim < 1:
        raise ValueError("invalid logical sample key/output dimension")
    bits = bytearray()
    block = 0
    while len(bits) * 8 < output_dim:
        bits.extend(hashlib.sha256(f"{SCHEMA}|{sample_key}|{block}".encode()).digest())
        block += 1
    values = [1.0 if (bits[i // 8] >> (i % 8)) & 1 else -1.0 for i in range(output_dim)]
    return torch.tensor(values, dtype=torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        return x + self.ff(self.norm(x))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=1024, width=256, output_dim=64):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(3)))
        self.output = nn.Linear(width, output_dim)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.input_dim or not torch.isfinite(features).all():
            raise ValueError("features must be finite [batch,input_dim]")
        return self.output(self.blocks(torch.nn.functional.gelu(self.input(features))))


class Calibration(nn.Module):
    """Explicit frozen-at-serving Welford statistics over prior outputs."""
    def __init__(self, output_dim=64, epsilon=1e-6):
        super().__init__()
        if epsilon <= 0: raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("m2", torch.zeros(output_dim))

    @torch.no_grad()
    def update(self, values):
        for row in values.detach().to(self.mean):
            self.count.add_(1); delta = row - self.mean
            self.mean.add_(delta / self.count); self.m2.add_(delta * (row - self.mean))

    def variance(self):
        return self.m2 / torch.clamp(self.count - 1, min=1) + self.epsilon


class DemoSupportCFN(nn.Module):
    def __init__(self, input_dim=1024, width=256, output_dim=64):
        super().__init__()
        # Fork RNG state so construction never advances process-global RNG.
        # CPU generator only: do not initialize or mutate CUDA generators.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1); self.learned = ResidualMLP(input_dim, width, output_dim)
            torch.random.default_generator.manual_seed(0); self.prior = ResidualMLP(input_dim, width, output_dim)
        self.prior.requires_grad_(False); self.prior.eval()
        self.calibration = Calibration(output_dim)

    def forward(self, features):
        if self.calibration.count.item() < 2: raise RuntimeError("prior calibration requires at least two samples")
        prior = (self.prior(features) - self.calibration.mean) / torch.sqrt(self.calibration.variance())
        return prior + self.learned(features)

    @torch.no_grad()
    def calibrate(self, features):
        if features.ndim != 2 or len(features) < 1 or not torch.isfinite(features).all():
            raise ValueError("calibration features must be a nonempty finite matrix")
        self.calibration.update(self.prior(features))

    def loss(self, features, sample_keys: Sequence[str]):
        if len(sample_keys) != len(features): raise ValueError("one logical key per feature")
        target = torch.stack([coin_label(key, self.learned.output_dim) for key in sample_keys]).to(features)
        return torch.mean((self(features) - target) ** 2)

    @torch.no_grad()
    def support_proxy(self, features):
        """Higher means closer to calibrated demo support; not success probability."""
        was_training = self.training; self.eval()
        try:
            return -torch.mean(self(features) ** 2, dim=-1)
        finally:
            self.train(was_training)


def _chunk_digest(chunk) -> str:
    # select_candidate presents scorer inputs as validated float64 arrays.
    value = np.asarray(chunk, dtype=np.float64)
    return hashlib.sha256(json.dumps([list(value.shape), value.dtype.str]).encode() + value.tobytes()).hexdigest()


def provided_feature_scorer(model: DemoSupportCFN, features, candidate_chunks):
    """Build a harness callback from already-provided features; no extraction occurs."""
    value = torch.as_tensor(features, dtype=torch.float32)
    bound = tuple(_chunk_digest(chunk) for chunk in candidate_chunks)
    if value.ndim != 2 or len(value) != len(bound) or not bound:
        raise ValueError("one provided feature per predetermined candidate required")
    scores = model.support_proxy(value).cpu().numpy()
    def score(chunks, observables):
        if tuple(_chunk_digest(chunk) for chunk in chunks) != bound:
            raise ValueError("provided features are not bound to these ordered candidate chunks")
        return scores.copy()
    return score


def load_safe_state_dict(model: DemoSupportCFN, state: Mapping[str, torch.Tensor], schema, expected_schema):
    """Accept tensors already loaded by a safe format; never invokes pickle/torch.load."""
    validate_feature_schema(schema, expected_schema)
    template = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(template): raise ValueError("checkpoint keys mismatch")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != template[key].shape or value.dtype != template[key].dtype:
            raise ValueError(f"checkpoint tensor mismatch: {key}")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor nonfinite: {key}")
    model.load_state_dict(state, strict=True)
    if model.calibration.count.item() < 2 or not torch.isfinite(model.calibration.variance()).all() or not torch.all(model.calibration.variance() > 0):
        raise ValueError("checkpoint prior calibration invalid")


def unavailable_bypass(enabled: bool, checkpoint_state=None):
    if not enabled:
        if checkpoint_state is not None: raise ValueError("disabled CFN cannot consume checkpoint")
        return None
    if checkpoint_state is None: raise RuntimeError("demo_support_cfn_v1 is unavailable without verified checkpoint")
    raise RuntimeError("use explicit schema-checked construction")


def parameter_count(model: DemoSupportCFN):
    return {"learned": sum(p.numel() for p in model.learned.parameters()),
            "fixed_prior": sum(p.numel() for p in model.prior.parameters())}
