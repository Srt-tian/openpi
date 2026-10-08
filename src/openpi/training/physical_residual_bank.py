"""Four-bank optimizer and bound checkpoint primitives for PI0.5 residual heads."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from flax import nnx, serialization
import jax
import optax

from openpi.training.physical_residual import PhysicalResidualHead, residual_training_loss
from openpi.training.plugin_bank import _optimizer_from_bytes, _optimizer_to_bytes

SUITES = ("spatial", "object", "goal", "long")
FORMAT = "pi05-physical-residual-head-bank-v1"


def initialize_head_bank(feature_dim: int, seed: int, *, state_dim: int = 8, width: int = 256,
                         horizon: int = 10, heads: int = 4, ffn_dim: int = 1024,
                         residual_bound: float = 1.0):
    graphdef = None
    states = {}
    for index, suite in enumerate(SUITES):
        head = PhysicalResidualHead(
            feature_dim, state_dim, width, horizon=horizon, heads=heads, ffn_dim=ffn_dim,
            residual_bound=residual_bound,
            rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(seed), index)),
        )
        suite_graphdef, state = nnx.split(head)
        graphdef = suite_graphdef if graphdef is None else graphdef
        states[suite] = state
    return graphdef, states


def initialize_optimizer_states(tx: optax.GradientTransformation, states: Mapping[str, nnx.State]):
    if tuple(states) != SUITES:
        raise ValueError(f"head bank order must be {SUITES}")
    return {suite: tx.init(states[suite]) for suite in SUITES}


def update_selected_head(
    global_step: int, graphdef, frozen_base_model, states, optimizer_states, tx,
    rng, observation, actions, task_ids, *, num_tasks: int = 10, valid_horizon=None,
    loss_kwargs: Mapping[str, Any] | None = None,
):
    """Update exactly one suite head; inactive state and optimizer objects are retained."""
    if tuple(states) != SUITES or tuple(optimizer_states) != SUITES:
        raise ValueError("head/optimizer banks must contain the ordered four suites")
    suite = SUITES[int(global_step) % len(SUITES)]
    active_state = states[suite]

    def loss_fn(candidate_state):
        head = nnx.merge(graphdef, candidate_state)
        return residual_training_loss(
            frozen_base_model, head, rng, observation, actions, task_ids,
            num_tasks=num_tasks, valid_horizon=valid_horizon, **dict(loss_kwargs or {}),
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(active_state)
    updates, new_optimizer = tx.update(grads, optimizer_states[suite], active_state)
    new_states, new_optimizers = dict(states), dict(optimizer_states)
    new_states[suite] = optax.apply_updates(active_state, updates)
    new_optimizers[suite] = new_optimizer
    return new_states, new_optimizers, suite, {**metrics, "loss": loss}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes):
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def save_head_bank(
    path, states, optimizer_states, steps, *, base_checkpoint_path: str,
    norm_stats_sha256: str, base_manifest_sha256: str, head_config: Mapping[str, int],
    metadata: Mapping[str, Any] | None = None,
):
    if tuple(states) != SUITES or tuple(optimizer_states) != SUITES or tuple(steps) != SUITES:
        raise ValueError("checkpoint requires ordered four-suite head/optimizer/step banks")
    if any(type(steps[suite]) is not int or steps[suite] < 0 for suite in SUITES):
        raise ValueError("suite steps must be nonnegative integers")
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to replace head bank: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        banks = {}
        for index, suite in enumerate(SUITES):
            head_name, opt_name = f"bank_{index:02d}.head.msgpack", f"bank_{index:02d}.optimizer.msgpack"
            head_data = serialization.to_bytes(states[suite].to_pure_dict())
            opt_data, leaf_count = _optimizer_to_bytes(optimizer_states[suite])
            _write(staging / head_name, head_data); _write(staging / opt_name, opt_data)
            banks[suite] = {"step": steps[suite], "head_file": head_name,
                "head_sha256": _sha256(head_data), "optimizer_file": opt_name,
                "optimizer_sha256": _sha256(opt_data), "optimizer_leaf_count": leaf_count}
        manifest = {"schema_version": 1, "format": FORMAT,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "suite_order": list(SUITES), "global_update_count": sum(steps.values()),
            "base": {"checkpoint_path": str(Path(base_checkpoint_path).expanduser().resolve()),
                "norm_stats_sha256": norm_stats_sha256, "source_manifest_sha256": base_manifest_sha256},
            "head_config": dict(head_config), "banks": banks, "metadata": dict(metadata or {})}
        _write(staging / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        os.replace(staging, target)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try: os.fsync(parent_fd)
        finally: os.close(parent_fd)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    return target


def load_head_bank(
    path, state_templates, optimizer_templates, *, expected_base_checkpoint_path: str,
    expected_norm_stats_sha256: str, expected_base_manifest_sha256: str,
    expected_head_config: Mapping[str, int],
):
    root = Path(path).expanduser().resolve(); manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema_version"), manifest.get("format"), tuple(manifest.get("suite_order", ()))) != (1, FORMAT, SUITES):
        raise ValueError("unsupported residual head bank manifest")
    expected_base = {"checkpoint_path": str(Path(expected_base_checkpoint_path).expanduser().resolve()),
        "norm_stats_sha256": expected_norm_stats_sha256, "source_manifest_sha256": expected_base_manifest_sha256}
    if manifest.get("base") != expected_base or manifest.get("head_config") != dict(expected_head_config):
        raise ValueError("head bank base/config binding mismatch")
    states, optimizers, steps = {}, {}, {}
    for index, suite in enumerate(SUITES):
        entry = manifest["banks"][suite]
        head_name, opt_name = f"bank_{index:02d}.head.msgpack", f"bank_{index:02d}.optimizer.msgpack"
        if (entry.get("head_file"), entry.get("optimizer_file")) != (head_name, opt_name):
            raise ValueError(f"unsafe payload name for {suite}")
        head_data, opt_data = (root / head_name).read_bytes(), (root / opt_name).read_bytes()
        if _sha256(head_data) != entry.get("head_sha256") or _sha256(opt_data) != entry.get("optimizer_sha256"):
            raise ValueError(f"checksum mismatch for {suite}")
        template = state_templates[suite]
        pure = serialization.from_bytes(template.to_pure_dict(), head_data)
        restored = copy.deepcopy(template); restored.replace_by_pure_dict(pure); states[suite] = restored
        optimizers[suite] = _optimizer_from_bytes(
            optimizer_templates[suite], opt_data, int(entry["optimizer_leaf_count"])
        )
        steps[suite] = int(entry["step"])
    if manifest.get("global_update_count") != sum(steps.values()):
        raise ValueError("global update count mismatch")
    return states, optimizers, steps, manifest


def restore_suite_head(graphdef, states: Mapping[str, nnx.State], suite: str):
    """Serving smoke interface: select one restored head without routing claims."""
    if suite not in SUITES or set(states) != set(SUITES):
        raise ValueError("unknown suite or incomplete head bank")
    return nnx.merge(graphdef, states[suite])
