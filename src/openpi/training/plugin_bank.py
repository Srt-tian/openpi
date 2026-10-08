"""Frozen-base, multi-adapter training primitives for PI0.5.

This module deliberately owns neither input pipelines nor checkpoint scheduling.  A
caller keeps one ``frozen_state`` and one adapter/optimizer pair per suite, then
uses :func:`select_suite` to update exactly one pair at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any, TypeAlias

import flax.nnx as nnx
from flax import serialization
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.training.sharding as sharding


SUITES = ("spatial", "object", "goal", "long")
# Kept as a descriptive alias for callers that do not want to depend on the
# benchmark-specific constant name.
DEFAULT_SUITES = SUITES

StateFilter: TypeAlias = Callable[[nnx.filterlib.PathParts, Any], bool]
StepFn: TypeAlias = Callable[
    [nnx.State, nnx.State, optax.OptState, Any, Any, jax.Array],
    tuple[nnx.State, optax.OptState, dict[str, jax.Array]],
]
EvalStepFn: TypeAlias = Callable[
    [nnx.State, nnx.State, Any, Any, jax.Array],
    dict[str, jax.Array],
]


def pi05_lora_config() -> pi0_config.Pi0Config:
    """Return the one supported model layout for a plugin bank."""
    return pi0_config.Pi0Config(
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_lora",
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
    )


def _path_strings(path: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(part) for part in path)


def is_action_expert_lora(path: nnx.filterlib.PathParts, value: Any) -> bool:
    """Match paths belonging to Gemma's action-expert LoRA variables."""
    del value
    parts = _path_strings(path)
    return (
        any("lora" in part for part in parts)
        and any("_1" in part for part in parts)
        and "llm" in "/".join(parts)
    )


ACTION_EXPERT_LORA_FILTER = nnx.All(nnx.Param, is_action_expert_lora)


def partition_model(
    model: nnx.Module, adapter_filter: nnx.filterlib.Filter = ACTION_EXPERT_LORA_FILTER
) -> tuple[nnx.GraphDef, nnx.State, nnx.State]:
    """Split a model into exhaustive frozen and adapter states.

    The complement is intentionally first: it contains parameters, statistics,
    and every other variable that must never become optimizer-owned.
    """
    graphdef, frozen, adapter = nnx.split(model, nnx.Not(adapter_filter), adapter_filter)
    if len(adapter) == 0:
        raise ValueError("adapter filter matched no variables")
    return graphdef, frozen, adapter


def _is_lora_path(path: tuple[Any, ...]) -> bool:
    return any("lora" in str(part) for part in path)


def _is_lora_b_path(path: tuple[Any, ...]) -> bool:
    name = str(path[-1])
    return name == "lora_b" or name.endswith("_lora_b")


def _assert_action_expert_lora(path: tuple[Any, ...]) -> None:
    parts = _path_strings(path)
    if not (any("_1" in part for part in parts) and "llm" in "/".join(parts)):
        raise ValueError(f"unexpected non-action-expert LoRA leaf: {'/'.join(parts)}")


def _materialize_params(
    template: Mapping[str, Any], restored: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    """Combine a complete restored base with freshly initialized LoRA leaves."""
    flat_template = traverse_util.flatten_dict(template)
    flat_restored = traverse_util.flatten_dict(restored)
    result: dict[tuple[Any, ...], Any] = {}
    lora_paths = sorted((path for path in flat_template if _is_lora_path(path)), key=str)
    keys = jax.random.split(jax.random.key(seed), len(lora_paths)) if lora_paths else ()
    key_by_path = dict(zip(lora_paths, keys, strict=True))

    errors: list[str] = []
    for path, expected in flat_template.items():
        if _is_lora_path(path):
            _assert_action_expert_lora(path)
            dtype = expected.dtype
            if _is_lora_b_path(path):
                value = jnp.zeros(expected.shape, dtype=dtype)
            else:
                value = jax.random.normal(key_by_path[path], expected.shape, dtype=jnp.float32)
                value = (value * 0.01).astype(dtype)
            result[path] = value
            continue

        if path not in flat_restored:
            errors.append(f"missing {'/'.join(map(str, path))}")
            continue
        value = flat_restored[path]
        if tuple(value.shape) != tuple(expected.shape):
            errors.append(
                f"shape {'/'.join(map(str, path))}: checkpoint={tuple(value.shape)}, "
                f"model={tuple(expected.shape)}"
            )
            continue
        result[path] = value

    if errors:
        preview = "\n  ".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise ValueError(f"checkpoint is not a complete PI0.5 base:\n  {preview}{suffix}")
    return traverse_util.unflatten_dict(result)


def _place_state(state: nnx.State, mesh: jax.sharding.Mesh) -> nnx.State:
    state_sharding = sharding.fsdp_sharding(state, mesh, log=False)
    return jax.device_put(state, state_sharding)


def initialize_bank(
    checkpoint_path: str,
    seed: int,
    mesh: jax.sharding.Mesh,
) -> tuple[nnx.GraphDef, nnx.State, dict[str, nnx.State]]:
    """Load one frozen PI0.5 base and initialize four independent LoRA banks.

    ``checkpoint_path`` is the official checkpoint's ``params`` directory.
    Restored tensors are first loaded as NumPy bfloat16 arrays.  No missing base
    tensor is filled from model initialization; missing paths or shape mismatches
    fail before any state is returned.
    """
    config = pi05_lora_config()
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    graphdef, template_state = nnx.split(abstract_model)
    restored = _model.restore_params(
        checkpoint_path,
        restore_type=np.ndarray,
        dtype=jnp.bfloat16,
    )
    params = _materialize_params(template_state.to_pure_dict(), restored, seed)
    template_state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, template_state)
    graphdef, frozen, adapter = partition_model(model)

    # Validate the partition itself, rather than relying solely on path naming in
    # the pure checkpoint tree.
    frozen = _place_state(frozen, mesh)
    adapter = _place_state(adapter, mesh)
    # JAX arrays are immutable, so initial leaves can be shared safely.  Each
    # selected update returns a new state; the base is never duplicated.
    banks = {suite: copy.copy(adapter) for suite in DEFAULT_SUITES}
    return graphdef, frozen, banks


def initialize_optimizer_states(
    tx: optax.GradientTransformation, adapters: Mapping[str, nnx.State]
) -> dict[str, optax.OptState]:
    """Create one optimizer state per bank (never a shared momentum tree)."""
    return {suite: tx.init(adapter) for suite, adapter in adapters.items()}


def select_suite(step: int, suites: Sequence[str] = DEFAULT_SUITES) -> str:
    """Select one suite by deterministic round-robin order."""
    if not suites:
        raise ValueError("at least one suite is required")
    return suites[int(step) % len(suites)]


def make_step(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
    mesh: jax.sharding.Mesh,
) -> StepFn:
    """Build a JIT step whose only differentiated argument is adapter state."""

    def step(
        frozen: nnx.State,
        adapter: nnx.State,
        opt_state: optax.OptState,
        observation: Any,
        actions: Any,
        rng: jax.Array,
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        def loss_fn(active_adapter: nnx.State) -> jax.Array:
            model = nnx.merge(graphdef, frozen, active_adapter)
            model.train()
            return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

        loss, grads = jax.value_and_grad(loss_fn)(adapter)
        updates, new_opt_state = tx.update(grads, opt_state, adapter)
        new_adapter = optax.apply_updates(adapter, updates)
        metrics = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "adapter_norm": optax.global_norm(new_adapter),
        }
        return new_adapter, new_opt_state, metrics

    # In particular, frozen buffers are not donated.  Keeping all arguments
    # non-donated also permits the initially identical banks to share buffers.
    compiled = jax.jit(step)

    def run(*args: Any, **kwargs: Any):
        with sharding.set_mesh(mesh):
            return compiled(*args, **kwargs)

    return run


def make_eval_step(graphdef: nnx.GraphDef, mesh: jax.sharding.Mesh) -> EvalStepFn:
    """Build a mutation-free holdout loss step for one selected adapter."""

    def eval_step(
        frozen: nnx.State,
        adapter: nnx.State,
        observation: Any,
        actions: Any,
        rng: jax.Array,
    ) -> dict[str, jax.Array]:
        model = nnx.merge(graphdef, frozen, adapter)
        model.eval()
        loss = jnp.mean(model.compute_loss(rng, observation, actions, train=False))
        return {"loss": loss, "adapter_norm": optax.global_norm(adapter)}

    compiled = jax.jit(eval_step)

    def run(*args: Any, **kwargs: Any):
        with sharding.set_mesh(mesh):
            return compiled(*args, **kwargs)

    return run


def update_selected_bank(
    step: int,
    step_fn: StepFn,
    frozen: nnx.State,
    adapters: Mapping[str, nnx.State],
    opt_states: Mapping[str, optax.OptState],
    observation: Any,
    actions: Any,
    rng: jax.Array,
) -> tuple[dict[str, nnx.State], dict[str, optax.OptState], str, dict[str, jax.Array]]:
    """Update exactly one adapter and its optimizer state.

    The returned mappings retain the exact inactive objects, preventing Adam
    momentum or decoupled weight decay from advancing for inactive suites.
    """
    if set(adapters) != set(opt_states):
        raise ValueError("adapter and optimizer suite keys differ")
    suite = select_suite(step, tuple(adapters))
    new_adapter, new_opt_state, metrics = step_fn(
        frozen,
        adapters[suite],
        opt_states[suite],
        observation,
        actions,
        rng,
    )
    new_adapters = dict(adapters)
    new_opt_states = dict(opt_states)
    new_adapters[suite] = new_adapter
    new_opt_states[suite] = new_opt_state
    return new_adapters, new_opt_states, suite, metrics


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: pathlib.Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _base_path_string(path: str) -> str:
    return path if path.startswith("gs://") else str(pathlib.Path(path).expanduser().resolve())


def _optimizer_to_bytes(opt_state: optax.OptState) -> tuple[bytes, int]:
    """Serialize optimizer array leaves; the caller-owned template holds structure."""
    leaves, _ = jax.tree.flatten(opt_state)
    payload = {str(index): leaf for index, leaf in enumerate(leaves)}
    return serialization.to_bytes(payload), len(leaves)


def _optimizer_from_bytes(template: optax.OptState, data: bytes, leaf_count: int) -> optax.OptState:
    template_leaves, treedef = jax.tree.flatten(template)
    if len(template_leaves) != leaf_count:
        raise ValueError(
            f"optimizer template has {len(template_leaves)} leaves, checkpoint expects {leaf_count}"
        )
    payload_template = {str(index): leaf for index, leaf in enumerate(template_leaves)}
    payload = serialization.from_bytes(payload_template, data)
    return jax.tree.unflatten(treedef, [payload[str(index)] for index in range(leaf_count)])


def save_bank(
    path: str | pathlib.Path,
    adapters: Mapping[str, nnx.State],
    opt_states: Mapping[str, optax.OptState],
    steps: Mapping[str, int],
    *,
    base_checkpoint_path: str,
    norm_stats_hash: str,
    base_manifest_hash: str,
    metadata_extra: Mapping[str, Any] | None = None,
    auxiliary_state: Any | None = None,
) -> pathlib.Path:
    """Atomically create an adapter-only bank checkpoint.

    The target is create-only so a failed or repeated save cannot replace a good
    checkpoint.  Callers should use a unique step directory for each commit.
    """
    target = pathlib.Path(path).expanduser().resolve()
    suites = tuple(adapters)
    if set(suites) != set(DEFAULT_SUITES):
        raise ValueError(f"bank must contain exactly {DEFAULT_SUITES}, got {suites}")
    if set(opt_states) != set(suites) or set(steps) != set(suites):
        raise ValueError("adapter, optimizer, and step suite keys differ")
    if any(int(count) < 0 for count in steps.values()):
        raise ValueError("suite optimizer update counts must be non-negative")
    if target.exists():
        raise FileExistsError(f"refusing to replace bank checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        banks: dict[str, dict[str, Any]] = {}
        for index, suite in enumerate(DEFAULT_SUITES):
            adapter_name = f"bank_{index:02d}.adapter.msgpack"
            optimizer_name = f"bank_{index:02d}.optimizer.msgpack"
            adapter_data = serialization.to_bytes(adapters[suite].to_pure_dict())
            optimizer_data, optimizer_leaf_count = _optimizer_to_bytes(opt_states[suite])
            _write_bytes(staging / adapter_name, adapter_data)
            _write_bytes(staging / optimizer_name, optimizer_data)
            banks[suite] = {
                "step": int(steps[suite]),
                "adapter_file": adapter_name,
                "adapter_sha256": _sha256(adapter_data),
                "optimizer_file": optimizer_name,
                "optimizer_sha256": _sha256(optimizer_data),
                "optimizer_leaf_count": optimizer_leaf_count,
            }

        manifest = {
            "schema_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "format": "flax-msgpack-no-pickle",
            "base": {
                "checkpoint_path": _base_path_string(base_checkpoint_path),
                "norm_stats_sha256": norm_stats_hash,
                "source_manifest_sha256": base_manifest_hash,
                "inference_dtype": "bfloat16",
            },
            "config": dataclasses.asdict(pi05_lora_config()),
            "suite_order": list(DEFAULT_SUITES),
            "global_update_count": sum(int(steps[suite]) for suite in DEFAULT_SUITES),
            "banks": banks,
            "metadata_extra": dict(metadata_extra or {}),
        }
        if auxiliary_state is not None:
            auxiliary_name = "auxiliary.msgpack"
            auxiliary_data, auxiliary_leaf_count = _optimizer_to_bytes(auxiliary_state)
            _write_bytes(staging / auxiliary_name, auxiliary_data)
            manifest.update({
                "auxiliary_file": auxiliary_name,
                "auxiliary_sha256": _sha256(auxiliary_data),
                "auxiliary_leaf_count": auxiliary_leaf_count,
            })
        # Fail before publishing if caller metadata is not portable JSON.
        manifest_data = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        _write_bytes(staging / "manifest.json", manifest_data)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, target)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def load_auxiliary_state(path: str | pathlib.Path, template: Any) -> Any:
    """Verify and restore an optional pure-tree auxiliary checkpoint payload.

    ``template`` supplies the PyTree definition and leaf shapes/dtypes, just as
    optimizer templates do for :func:`load_bank`.  Keeping this separate makes
    stage-A checkpoints (which have no auxiliary payload) fully readable.
    """
    root = pathlib.Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    auxiliary_file = manifest.get("auxiliary_file")
    if auxiliary_file is None:
        raise ValueError("plugin-bank checkpoint has no auxiliary state")
    if auxiliary_file != "auxiliary.msgpack":
        raise ValueError("unsafe or unexpected auxiliary payload name")
    data = (root / auxiliary_file).read_bytes()
    if _sha256(data) != manifest.get("auxiliary_sha256"):
        raise ValueError("auxiliary checksum mismatch")
    try:
        leaf_count = int(manifest["auxiliary_leaf_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid auxiliary leaf count") from error
    return _optimizer_from_bytes(template, data, leaf_count)


def _check_binding(actual: str, expected: str | None, label: str) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} mismatch: checkpoint={actual!r}, expected={expected!r}")


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_adapter_bank(
    path: str | pathlib.Path,
    *,
    expected_base_checkpoint_path: str | None = None,
    expected_norm_stats_hash: str | None = None,
    expected_base_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Validate an adapter bank for inference without reading optimizer payloads.

    Every adapter is streamed through SHA-256 so a process serving one suite does
    not silently accept corruption elsewhere in the declared policy bundle.  The
    optimizer files are intentionally neither opened nor required by this path.
    """
    root = pathlib.Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("format") != "flax-msgpack-no-pickle":
        raise ValueError("unsupported plugin-bank checkpoint format")
    if manifest.get("config") != dataclasses.asdict(pi05_lora_config()):
        raise ValueError("plugin-bank model configuration mismatch")
    if tuple(manifest.get("suite_order", ())) != DEFAULT_SUITES:
        raise ValueError("plugin-bank suite order mismatch")

    base = manifest["base"]
    expected_path = None if expected_base_checkpoint_path is None else _base_path_string(expected_base_checkpoint_path)
    _check_binding(base["checkpoint_path"], expected_path, "base checkpoint path")
    _check_binding(base["norm_stats_sha256"], expected_norm_stats_hash, "norm stats hash")
    _check_binding(base["source_manifest_sha256"], expected_base_manifest_hash, "base manifest hash")
    if base.get("inference_dtype") != "bfloat16":
        raise ValueError("plugin bank is not bound to a bfloat16 base")

    steps: dict[str, int] = {}
    for index, suite in enumerate(DEFAULT_SUITES):
        entry = manifest["banks"][suite]
        expected_file = f"bank_{index:02d}.adapter.msgpack"
        if entry.get("adapter_file") != expected_file:
            raise ValueError(f"unsafe or unexpected adapter payload name for {suite}")
        try:
            step = int(entry["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid suite update count for {suite}") from error
        if step < 0:
            raise ValueError(f"negative suite update count for {suite}")
        steps[suite] = step
        if _sha256_path(root / expected_file) != entry.get("adapter_sha256"):
            raise ValueError(f"adapter checksum mismatch for {suite}")
    if manifest.get("global_update_count") != sum(steps.values()):
        raise ValueError("global update count does not equal the suite update counts")
    return manifest


def load_adapter(
    path: str | pathlib.Path,
    suite: str,
    adapter_template: nnx.State,
    *,
    expected_base_checkpoint_path: str | None = None,
    expected_norm_stats_hash: str | None = None,
    expected_base_manifest_hash: str | None = None,
) -> tuple[nnx.State, dict[str, Any]]:
    """Restore one suite adapter after validating the complete adapter bundle."""
    if suite not in DEFAULT_SUITES:
        raise ValueError(f"unknown plugin suite {suite!r}; expected one of {DEFAULT_SUITES}")
    root = pathlib.Path(path).expanduser().resolve()
    manifest = verify_adapter_bank(
        root,
        expected_base_checkpoint_path=expected_base_checkpoint_path,
        expected_norm_stats_hash=expected_norm_stats_hash,
        expected_base_manifest_hash=expected_base_manifest_hash,
    )
    data = (root / manifest["banks"][suite]["adapter_file"]).read_bytes()
    pure = serialization.from_bytes(adapter_template.to_pure_dict(), data)
    restored = copy.deepcopy(adapter_template)
    restored.replace_by_pure_dict(pure)
    return restored, manifest


def load_adapters(
    path: str | pathlib.Path,
    adapter_templates: Mapping[str, nnx.State],
    *,
    expected_base_checkpoint_path: str | None = None,
    expected_norm_stats_hash: str | None = None,
    expected_base_manifest_hash: str | None = None,
) -> tuple[dict[str, nnx.State], dict[str, Any]]:
    """Restore all four adapters for inference without optimizer state."""
    if set(adapter_templates) != set(DEFAULT_SUITES):
        raise ValueError("adapter templates must contain all four suites")
    root = pathlib.Path(path).expanduser().resolve()
    manifest = verify_adapter_bank(
        root,
        expected_base_checkpoint_path=expected_base_checkpoint_path,
        expected_norm_stats_hash=expected_norm_stats_hash,
        expected_base_manifest_hash=expected_base_manifest_hash,
    )
    adapters = {}
    for suite in DEFAULT_SUITES:
        data = (root / manifest["banks"][suite]["adapter_file"]).read_bytes()
        pure = serialization.from_bytes(adapter_templates[suite].to_pure_dict(), data)
        restored = copy.deepcopy(adapter_templates[suite])
        restored.replace_by_pure_dict(pure)
        adapters[suite] = restored
    return adapters, manifest


def load_bank(
    path: str | pathlib.Path,
    adapter_templates: Mapping[str, nnx.State],
    opt_state_templates: Mapping[str, optax.OptState],
    *,
    expected_base_checkpoint_path: str | None = None,
    expected_norm_stats_hash: str | None = None,
    expected_base_manifest_hash: str | None = None,
) -> tuple[dict[str, nnx.State], dict[str, optax.OptState], dict[str, int], dict[str, Any]]:
    """Verify and restore an adapter-only bank using caller-owned templates."""
    root = pathlib.Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("format") != "flax-msgpack-no-pickle":
        raise ValueError("unsupported plugin-bank checkpoint format")
    if manifest.get("config") != dataclasses.asdict(pi05_lora_config()):
        raise ValueError("plugin-bank model configuration mismatch")
    if tuple(manifest.get("suite_order", ())) != DEFAULT_SUITES:
        raise ValueError("plugin-bank suite order mismatch")
    if set(adapter_templates) != set(DEFAULT_SUITES) or set(opt_state_templates) != set(DEFAULT_SUITES):
        raise ValueError("restore templates must contain all four suites")

    base = manifest["base"]
    expected_path = None if expected_base_checkpoint_path is None else _base_path_string(expected_base_checkpoint_path)
    _check_binding(base["checkpoint_path"], expected_path, "base checkpoint path")
    _check_binding(base["norm_stats_sha256"], expected_norm_stats_hash, "norm stats hash")
    _check_binding(base["source_manifest_sha256"], expected_base_manifest_hash, "base manifest hash")
    if base.get("inference_dtype") != "bfloat16":
        raise ValueError("plugin bank is not bound to a bfloat16 base")

    adapters: dict[str, nnx.State] = {}
    opt_states: dict[str, optax.OptState] = {}
    steps: dict[str, int] = {}
    for index, suite in enumerate(DEFAULT_SUITES):
        entry = manifest["banks"][suite]
        expected_adapter_file = f"bank_{index:02d}.adapter.msgpack"
        expected_optimizer_file = f"bank_{index:02d}.optimizer.msgpack"
        if (
            entry["adapter_file"] != expected_adapter_file
            or entry["optimizer_file"] != expected_optimizer_file
        ):
            raise ValueError(f"unsafe or unexpected payload name for {suite}")
        adapter_data = (root / entry["adapter_file"]).read_bytes()
        optimizer_data = (root / entry["optimizer_file"]).read_bytes()
        if _sha256(adapter_data) != entry["adapter_sha256"]:
            raise ValueError(f"adapter checksum mismatch for {suite}")
        if _sha256(optimizer_data) != entry["optimizer_sha256"]:
            raise ValueError(f"optimizer checksum mismatch for {suite}")

        template = adapter_templates[suite]
        pure = serialization.from_bytes(template.to_pure_dict(), adapter_data)
        restored = copy.deepcopy(template)
        restored.replace_by_pure_dict(pure)
        adapters[suite] = restored
        opt_states[suite] = _optimizer_from_bytes(
            opt_state_templates[suite], optimizer_data, int(entry["optimizer_leaf_count"])
        )
        steps[suite] = int(entry["step"])
    if manifest.get("global_update_count") != sum(steps.values()):
        raise ValueError("global update count does not equal the suite update counts")
    return adapters, opt_states, steps, manifest
