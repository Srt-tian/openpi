"""Inference-only loader for the PI0.5 LIBERO suite adapter bank."""

from __future__ import annotations

import copy
import hashlib
import json
import numbers
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np

from openpi import transforms
from openpi.policies import policy as policy_api
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import plugin_bank
from openpi.training import sharding


POLICY_IDS = ("base", *plugin_bank.DEFAULT_SUITES)
POLICY_SEED_PROTOCOL = "paired_episode_plus_call_1000003_numpy_pcg64_noise_10x32_f32_v1"


class FixedPolicyService(policy_api.BasePolicy):
    """Enforce one process/one policy and paired deterministic inference noise."""

    def __init__(self, policy: policy_api.Policy, policy_id: str):
        self._policy = policy
        self._policy_id = _require_policy_id(policy_id)
        self._metadata = dict(policy.metadata)
        checkpoint_sha256 = self._metadata.get("checkpoint_sha256")
        if not (
            isinstance(checkpoint_sha256, str)
            and len(checkpoint_sha256) == 64
            and checkpoint_sha256 == checkpoint_sha256.lower()
            and all(character in "0123456789abcdef" for character in checkpoint_sha256)
        ):
            raise ValueError("policy metadata requires a 64-character lowercase checkpoint_sha256")
        self._metadata.update({
            "policy_id": self._policy_id,
            "policy_seed_protocol": POLICY_SEED_PROTOCOL,
        })

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        payload = dict(obs)
        requested_policy = payload.pop("policy_id", None)
        if not isinstance(requested_policy, str) or requested_policy != self._policy_id:
            raise ValueError(
                f"request policy_id {requested_policy!r} does not match fixed service {self._policy_id!r}"
            )
        policy_seed = payload.pop("policy_seed", None)
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, numbers.Integral):
            raise ValueError("policy_seed must be a non-negative integer (bool is rejected)")
        policy_seed = int(policy_seed)
        if policy_seed < 0:
            raise ValueError("policy_seed must be a non-negative integer")
        noise = np.random.default_rng(policy_seed).standard_normal((10, 32), dtype=np.float32)
        return self._policy.infer(payload, noise=noise)


def measure_same_noise_base_alignment(
    zero_b_policy: policy_api.BasePolicy,
    original_base_policy: policy_api.BasePolicy,
    observations: list[dict[str, Any]],
    noises: list[np.ndarray],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Measure, but never assume, zero-B equivalence to the original base graph."""
    if not observations or len(observations) != len(noises):
        raise ValueError("observations and noises must have the same nonzero length")
    maximum = 0.0
    all_close = True
    for observation, noise in zip(observations, noises, strict=True):
        candidate = np.asarray(zero_b_policy.infer(observation, noise=noise)["actions"])
        reference = np.asarray(original_base_policy.infer(observation, noise=noise)["actions"])
        if candidate.shape != reference.shape:
            raise ValueError(f"action shape mismatch: zero-B={candidate.shape}, original={reference.shape}")
        maximum = max(maximum, float(np.max(np.abs(candidate - reference))))
        all_close = all_close and bool(np.allclose(candidate, reference, atol=atol, rtol=rtol))
    return {
        "cases": len(observations),
        "same_noise": True,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": maximum,
        "all_close": all_close,
        "official_baseline_label_allowed": all_close,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_inventory_hash(root: Path) -> str:
    rows = [(str(path.relative_to(root)), path.stat().st_size) for path in root.rglob("*") if path.is_file()]
    return hashlib.sha256(json.dumps(sorted(rows), separators=(",", ":")).encode()).hexdigest()


def _require_policy_id(policy_id: str) -> str:
    if policy_id not in POLICY_IDS:
        raise ValueError(f"policy_id must be one of {POLICY_IDS}; automatic routing is not available")
    return policy_id


def _select_adapter(
    policy_id: str,
    initial_adapters: dict[str, nnx.State],
    checkpoint: Path,
    *,
    params_path: Path,
    norm_hash: str,
    base_hash: str,
    allow_verified_base_relocation: bool,
) -> tuple[nnx.State, dict[str, Any]]:
    policy_id = _require_policy_id(policy_id)
    # All initial suite states share immutable zero-B leaves.  Keep one shallow
    # copy for a base control in the exact LoRA graph without duplicating base.
    zero_b_adapter = copy.copy(initial_adapters[plugin_bank.DEFAULT_SUITES[0]])
    bindings = {
        # A copied base may live at a different absolute path.  Relocation is
        # allowed only while both content bindings below remain mandatory.
        "expected_base_checkpoint_path": None if allow_verified_base_relocation else str(params_path),
        "expected_norm_stats_hash": norm_hash,
        "expected_base_manifest_hash": base_hash,
    }
    if policy_id == "base":
        manifest = plugin_bank.verify_adapter_bank(checkpoint, **bindings)
        return zero_b_adapter, manifest
    return plugin_bank.load_adapter(
        checkpoint,
        policy_id,
        initial_adapters[policy_id],
        **bindings,
    )


def create_plugin_policy(
    base_checkpoint: str | Path,
    plugin_checkpoint: str | Path,
    policy_id: str,
    *,
    default_prompt: str | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    num_fsdp_devices: int | None = None,
    allow_verified_base_relocation: bool = False,
) -> policy_api.Policy:
    """Create one explicitly selected base or suite policy for inference.

    This does not load optimizer state, training data, or an auxiliary routing
    head.  A process owns one fixed policy ID; callers must not infer atomic
    skills or automatic routing from the four LIBERO suite names.
    """
    policy_id = _require_policy_id(policy_id)
    base_checkpoint = Path(base_checkpoint).expanduser().resolve()
    plugin_checkpoint = Path(plugin_checkpoint).expanduser().resolve()
    params_path = base_checkpoint / "params"
    norm_path = base_checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    for required in (params_path, norm_path, plugin_checkpoint / "manifest.json"):
        if not required.exists():
            raise FileNotFoundError(required)

    norm_hash = _sha256_file(norm_path)
    base_hash = _checkpoint_inventory_hash(base_checkpoint)
    fsdp_devices = jax.device_count() if num_fsdp_devices is None else int(num_fsdp_devices)
    if fsdp_devices <= 0:
        raise ValueError("num_fsdp_devices must be positive")
    mesh = sharding.make_mesh(fsdp_devices)
    graphdef, frozen, initial_adapters = plugin_bank.initialize_bank(str(params_path), 42, mesh)
    adapter, manifest = _select_adapter(
        policy_id,
        initial_adapters,
        plugin_checkpoint,
        params_path=params_path,
        norm_hash=norm_hash,
        base_hash=base_hash,
        allow_verified_base_relocation=allow_verified_base_relocation,
    )
    del initial_adapters
    model = nnx.merge(graphdef, frozen, adapter)
    model.eval()

    model_config = plugin_bank.pi05_lora_config()
    config = training_config.get_config("pi05_libero")
    data_config = config.data.create(config.assets_dirs, model_config)
    norm_stats = normalize.load(norm_path.parent)
    metadata = {
        "policy_id": policy_id,
        "policy_bundle": str(plugin_checkpoint),
        "plugin_global_update_count": manifest["global_update_count"],
        "plugin_git_sha": manifest.get("metadata_extra", {}).get("git_sha"),
        "checkpoint_sha256": _sha256_file(plugin_checkpoint / "manifest.json"),
        "adapter_sha256": (
            None if policy_id == "base" else manifest["banks"][policy_id]["adapter_sha256"]
        ),
        "original_base_checkpoint_path": manifest["base"]["checkpoint_path"],
        "runtime_base_checkpoint_path": str(params_path),
        "verified_base_relocation": bool(
            allow_verified_base_relocation
            and manifest["base"]["checkpoint_path"] != str(params_path)
        ),
        "routing": "explicit_fixed_policy_id",
        "call_head_loaded": False,
        "suite_semantics": "LIBERO suite adapter; not an atomic robot skill",
        "base_semantics": (
            "zero-B adapter in the plugin LoRA graph; not labeled official baseline until same-noise "
            "numeric alignment with the original base graph is measured"
        ),
    }
    return policy_api.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=metadata,
    )
