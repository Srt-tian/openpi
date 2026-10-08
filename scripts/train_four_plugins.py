#!/usr/bin/env python3
"""Train four independent pi0.5 LoRA adapters in one round-robin job.

This entrypoint deliberately does not import openpi.training.config or data_loader:
the pinned OpenPI checkout expects an older LeRobot API.  Dataset access is through
openpi.training.plugin_data, which reads the local v3 dataset directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SUITES = ("spatial", "object", "goal", "long")
POLICY_IDS = ("base", *SUITES)
DATA_SUITE = {
    "spatial": "libero_spatial", "object": "libero_object",
    "goal": "libero_goal", "long": "libero_10",
}
DEFAULT_BASE = Path("/pfs/user/data/physicalrsi_pi05/pi05_libero")
DEFAULT_DATA = Path("/pfs/user/data/libero/libero")
DEFAULT_OUT = Path("/pfs/user/data/physicalrsi_joint_plugins/PI05-libero-test")
TOTAL_UPDATES = 4_000
UPDATES_PER_SUITE = 1_000
SAVE_EVERY = 400
EVAL_EVERY = 400
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--from-checkpoint", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--rollout-manifest", type=Path)
    p.add_argument("--handoff-weight", type=float, default=0.3)
    p.add_argument("--call-weight", type=float, default=0.1)
    p.add_argument("--require-joint-losses", action="store_true")
    p.add_argument(
        "--fm-only",
        action="store_true",
        help="explicitly approved Stage A: disable handoff and call losses and require no rollout labels",
    )
    return p.parse_args(argv)


def validate_loss_args(args: argparse.Namespace) -> None:
    fm_only = bool(getattr(args, "fm_only", False))
    if fm_only and args.rollout_manifest is not None:
        raise ValueError("--fm-only cannot be combined with --rollout-manifest")
    if fm_only and args.require_joint_losses:
        raise ValueError("--fm-only cannot be combined with --require-joint-losses")
    for name in ("handoff_weight", "call_weight"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if fm_only:
        # Make the resolved configuration and saved run manifest explicit rather
        # than relying on absence of labels to deactivate auxiliary objectives.
        args.handoff_weight = 0.0
        args.call_weight = 0.0


def resolve_loss_status(args: argparse.Namespace, rollout_store) -> tuple[dict[str, Any], dict[str, list[Any]], list[Any]]:
    """Resolve label availability without importing JAX or loading model parameters."""
    fm_only = bool(getattr(args, "fm_only", False))
    handoff_records: dict[str, list[Any]] = {suite: [] for suite in SUITES}
    call_records: list[Any] = []
    if rollout_store is not None:
        handoff_records = {
            suite: list(rollout_store.handoff_records(target_policy_id=suite, split="train"))
            for suite in SUITES
        }
        call_records = list(rollout_store.call_records(split="train"))
    call_counts_by_policy = {
        policy_id: sum(record.attempted_policy_id == policy_id for record in call_records)
        for policy_id in POLICY_IDS
    }

    handoff: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        if fm_only:
            reason = "user_explicit_fm_only"
        elif rollout_store is None:
            reason = "no_rollout_manifest"
        elif args.handoff_weight == 0:
            reason = "zero_weight"
        elif not handoff_records[suite]:
            reason = "no_real_train_handoff_labels_for_target"
        else:
            reason = None
        handoff[suite] = {
            "active": reason is None,
            "count": len(handoff_records[suite]),
            "disabled_reason": reason,
        }
    if fm_only:
        call_reason = "user_explicit_fm_only"
    elif rollout_store is None:
        call_reason = "no_rollout_manifest"
    elif args.call_weight == 0:
        call_reason = "zero_weight"
    elif not call_records:
        call_reason = "no_uncensored_real_train_call_labels"
    else:
        call_reason = None
    status = {
        "weights": {"handoff": args.handoff_weight, "call": args.call_weight},
        "handoff": handoff,
        "call": {"active": call_reason is None, "count": len(call_records),
                 "counts_by_policy": call_counts_by_policy,
                 "full_policy_coverage": all(call_counts_by_policy.values()),
                 "missing_policy_coverage": [
                     policy_id for policy_id, count in call_counts_by_policy.items() if count == 0
                 ],
                 "disabled_reason": call_reason},
        "routing_head_status": "experimental_uncalibrated",
        "stage_selection": (
            "user_explicit_stage_a_fm_only" if fm_only else "resolved_from_joint_loss_inputs"
        ),
    }
    active_handoff_targets = [suite for suite, item in handoff.items() if item["active"]]
    status["active_handoff_targets"] = active_handoff_targets
    if (
        len(active_handoff_targets) == len(SUITES)
        and status["call"]["active"]
        and status["call"]["full_policy_coverage"]
    ):
        status["mode"] = "full_experimental_joint_losses"
    elif active_handoff_targets or status["call"]["active"]:
        status["mode"] = "partial_experimental_joint_losses"
    else:
        status["mode"] = "stage_a_fm_only"
    if args.require_joint_losses:
        missing = [f"handoff:{suite}" for suite, item in handoff.items() if not item["active"]]
        if not status["call"]["active"]:
            missing.append("call")
        missing.extend(
            f"call:{policy_id}"
            for policy_id, count in call_counts_by_policy.items()
            if count == 0
        )
        if missing:
            raise RuntimeError("--require-joint-losses cannot be satisfied: " + ", ".join(missing))
    return status, handoff_records, call_records


def validation_loss_records(rollout_store) -> tuple[dict[str, list[Any]], list[Any]]:
    """Query held-out real labels independently of training-loss availability."""
    if rollout_store is None:
        return {suite: [] for suite in SUITES}, []
    handoff = {
        suite: list(rollout_store.handoff_records(target_policy_id=suite, split="validation"))
        for suite in SUITES
    }
    return handoff, list(rollout_store.call_records(split="validation"))


def check_rollout_binding(summary: dict[str, Any], norm_hash: str, base_hash: str) -> None:
    if summary.get("source_base_inventory_sha256") != base_hash:
        raise ValueError("rollout manifest base checkpoint inventory binding mismatch")
    if summary.get("norm_stats_sha256") != norm_hash:
        raise ValueError("rollout manifest norm-stats binding mismatch")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_inventory_hash(root: Path) -> str:
    """Hash checkpoint names and sizes, without re-reading 12 GB parameter blobs."""
    rows = [(str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*") if p.is_file()]
    return hashlib.sha256(json.dumps(sorted(rows), separators=(",", ":")).encode()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_inputs(args: argparse.Namespace) -> tuple[Path, Path, str, str]:
    params = args.base_checkpoint / "params"
    norm = args.base_checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    data_info = args.data_root / "meta/info.json"
    data_home_raw = os.environ.get("OPENPI_DATA_HOME")
    if not data_home_raw:
        raise RuntimeError("OPENPI_DATA_HOME must be set; tokenizer download is forbidden")
    tokenizer = Path(data_home_raw) / "big_vision/paligemma_tokenizer.model"
    missing = [str(p) for p in (params, norm, data_info, tokenizer) if not p.exists()]
    if missing:
        raise FileNotFoundError("required local inputs missing (no download attempted): " + ", ".join(missing))
    return norm, tokenizer, sha256_file(norm), checkpoint_inventory_hash(args.base_checkpoint)


def numpy_collate(items: list[Any]) -> Any:
    """NumPy-only collation: workers never initialize JAX/GPU."""
    x = items[0]
    if isinstance(x, dict):
        return {k: numpy_collate([v[k] for v in items]) for k in x}
    if isinstance(x, np.ndarray):
        return np.stack(items)
    if isinstance(x, (np.generic, int, float, bool)):
        return np.asarray(items)
    if isinstance(x, str):
        return list(items)
    raise TypeError(f"unsupported batch leaf: {type(x)!r}")


def worker_init_cpu_only(worker_id: int) -> None:
    """Runs in spawned workers before dataset access/import can initialize a backend."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"


class DeterministicBatchSampler:
    """Infinite, resumable epoch permutations with a defined dropped tail."""
    def __init__(self, size: int, batch_size: int, seed: int, start_batch: int = 0):
        if size < batch_size:
            raise ValueError(f"dataset size {size} is smaller than batch {batch_size}")
        self.size, self.batch_size, self.seed, self.start_batch = size, batch_size, seed, start_batch
        self.batches_per_epoch = size // batch_size

    def __iter__(self):
        absolute = self.start_batch
        while True:
            epoch, within = divmod(absolute, self.batches_per_epoch)
            order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.size)
            while within < self.batches_per_epoch:
                lo = within * self.batch_size
                yield order[lo : lo + self.batch_size].tolist()
                absolute += 1; within += 1

    def __len__(self):
        return sys.maxsize


class FixedIndexBatchSampler:
    def __init__(self, indices: list[int]): self.indices = indices
    def __iter__(self): yield self.indices
    def __len__(self): return 1


def load_datasets(args: argparse.Namespace):
    from openpi.training.plugin_data import build_suite_datasets

    train_raw, val_raw, manifest = build_suite_datasets(
        args.data_root, horizon=10, seed=args.seed, holdout_per_task=2
    )
    train = {suite: train_raw[DATA_SUITE[suite]] for suite in SUITES}
    val = {suite: val_raw[DATA_SUITE[suite]] for suite in SUITES}
    return train, val, manifest


def sample_summary(x: Any) -> Any:
    def summarize(v):
        if isinstance(v, dict): return {k: summarize(x) for k, x in v.items()}
        if isinstance(v, np.ndarray): return {"shape": list(v.shape), "dtype": str(v.dtype)}
        if isinstance(v, list): return {"length": len(v), "first": str(v[0])[:160] if v else None}
        return str(v)[:160]
    return summarize(x)


def transformed_rollout_batch(store, records: list[Any], transform) -> dict[str, Any]:
    return numpy_collate([transform(dict(store.raw_sample(record))) for record in records])


def selected_rollout_batch(store, records: list[Any], transform, *, batch_size: int,
                           seed: int, step: int) -> tuple[list[Any], dict[str, Any]]:
    selected = list(store.select_batch(records, batch_size=batch_size, seed=seed, step=step))
    if not selected:
        raise RuntimeError("rollout batch selection unexpectedly returned no records")
    return selected, transformed_rollout_batch(store, selected, transform)


def run_preflight(args: argparse.Namespace, train, val, manifest, norm_hash: str, base_hash: str,
                  norm_path: Path, git_sha: str, *, rollout_store=None,
                  rollout_summary: dict[str, Any] | None = None,
                  loss_status: dict[str, Any] | None = None,
                  handoff_records: dict[str, list[Any]] | None = None,
                  call_records: list[Any] | None = None) -> None:
    report: dict[str, Any] = {"dataset_manifest": manifest, "norm_stats_sha256": norm_hash,
                              "base_inventory_sha256": base_hash, "git_sha": git_sha, "suites": {},
                              "resolved_loss_config": loss_status,
                              "rollout_manifest": rollout_summary}
    for suite in SUITES:
        if len(train[suite]) == 0 or len(val[suite]) == 0:
            raise RuntimeError(f"empty split for {suite}")
        # Real decoded first/last samples validate parquet, both video streams, horizon, and prompt.
        report["suites"][suite] = {
            "train_len": len(train[suite]), "val_len": len(val[suite]),
            "train_first": sample_summary(train[suite][0]),
            "train_last": sample_summary(train[suite][len(train[suite]) - 1]),
            "holdout_first": sample_summary(val[suite][0]),
            "holdout_last": sample_summary(val[suite][len(val[suite]) - 1]),
        }
    # Exercise the exact official transform chain and one spawned-worker batch per suite.
    transform = make_transform(norm_path)
    call_transform = make_transform(norm_path, require_actions=False)
    loaders = make_loaders(args, train, transform, steps={s: 0 for s in SUITES},
                           train=True, persistent_workers=False)
    for suite in SUITES:
        batch = next(iter(loaders[suite]))
        report["suites"][suite]["transformed_batch"] = sample_summary(batch)
    if rollout_store is not None:
        assert handoff_records is not None and call_records is not None
        report["rollout_branch_batches"] = {"handoff": {}, "call": None}
        for index, suite in enumerate(SUITES):
            if loss_status["handoff"][suite]["active"]:
                _, batch = selected_rollout_batch(
                    rollout_store, handoff_records[suite], transform,
                    batch_size=args.batch_size, seed=args.seed + index, step=0,
                )
                # Handoffs must carry real action supervision through the official transform.
                if "actions" not in batch:
                    raise ValueError(f"handoff rollout batch for {suite} has no actions")
                report["rollout_branch_batches"]["handoff"][suite] = sample_summary(batch)
        if loss_status["call"]["active"]:
            selected, batch = selected_rollout_batch(
                rollout_store, call_records, call_transform,
                batch_size=args.batch_size, seed=args.seed, step=0,
            )
            if "actions" in batch:
                raise ValueError("call rollout observations unexpectedly contain actions")
            targets, observed = rollout_store.call_targets(selected, policy_order=POLICY_IDS)
            targets, observed = validate_call_supervision(targets, observed, len(selected))
            report["rollout_branch_batches"]["call"] = {
                "observation": sample_summary(batch),
                "budgets": sample_summary(np.asarray([r.budget_steps for r in selected], np.float32)),
                "targets": sample_summary(np.asarray(targets, np.float32)),
                "observed_mask": sample_summary(np.asarray(observed, np.bool_)),
                "observed_count": int(np.asarray(observed).sum()),
            }
    print(json.dumps(report, indent=2, sort_keys=True))


class TransformedDataset:
    def __init__(self, dataset, transform): self.dataset, self.transform = dataset, transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return self.transform(dict(self.dataset[index]))


def make_transform(norm_dir: Path, *, require_actions: bool = True):
    # Importing these is safe; importantly, this is not training.config/data_loader.
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.models import tokenizer as tokenizer_api
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.shared import normalize

    stats = normalize.load(norm_dir.parent)
    return transforms.compose((
        LiberoInputs(model_type=model_api.ModelType.PI05),
        # Call records are observation-only.  Non-strict mode still normalizes
        # every present official key, while permitting the intentionally absent
        # action leaf; demonstrations and handoffs keep the original strict gate.
        transforms.Normalize(stats, use_quantiles=True, strict=require_actions),
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(
            tokenizer_api.PaligemmaTokenizer(
                200, tokenizer_path=Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
            ),
            discrete_state_input=False,
        ),
        transforms.PadStatesAndActions(32),
    ))


def make_loaders(args: argparse.Namespace, datasets: dict[str, Any], transform, *, steps: dict[str, int],
                 train: bool, persistent_workers: bool = True):
    from torch.utils.data import DataLoader
    result = {}
    for i, suite in enumerate(SUITES):
        ds = TransformedDataset(datasets[suite], transform)
        if train:
            sampler = DeterministicBatchSampler(len(ds), args.batch_size, args.seed + i, steps[suite])
            workers = args.num_workers
        else:
            # Metadata-only selection; first ten positions cover all suite tasks, then frames spread within task.
            sampler = FixedIndexBatchSampler(datasets[suite].diagnostic_indices(args.batch_size))
            workers = 0
        result[suite] = DataLoader(ds, batch_sampler=sampler, num_workers=workers,
            collate_fn=numpy_collate, multiprocessing_context="spawn" if workers else None,
            persistent_workers=bool(workers and persistent_workers), worker_init_fn=worker_init_cpu_only if workers else None)
    return result


def next_cycling(loaders, iterators, suite):
    try: return next(iterators[suite])
    except (KeyError, StopIteration):
        iterators[suite] = iter(loaders[suite])
        return next(iterators[suite])


def observation_and_actions(batch):
    from openpi.models.model import Observation
    return Observation.from_dict(batch), batch["actions"]


def observation_only(batch):
    from openpi.models.model import Observation
    return Observation.from_dict(batch)


def validate_call_supervision(targets: Any, observed: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.bool_)
    expected = (batch_size, len(POLICY_IDS))
    if targets.shape != expected or observed.shape != expected:
        raise ValueError(
            f"call labels must have target/mask shape {expected}, got {targets.shape}/{observed.shape}"
        )
    if not np.all(observed.sum(axis=1) == 1):
        raise ValueError("each call label must observe exactly its one attempted policy")
    if not np.all(np.isin(targets[observed], (0.0, 1.0))):
        raise ValueError("observed call targets must be binary outcomes")
    return targets, observed


def scalar_metrics(metrics):
    import jax
    return {k: float(np.asarray(jax.device_get(v))) for k, v in metrics.items()}


def tree_is_finite(tree) -> bool:
    import jax
    import jax.numpy as jnp
    checks = jax.tree.leaves(jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), tree))
    if not checks:
        return True
    return bool(np.asarray(jax.device_get(jnp.stack(checks).all())))


def put_batch_on_mesh(batch, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec
    from openpi.training import sharding
    placement = NamedSharding(mesh, PartitionSpec(sharding.DATA_AXIS))
    return jax.tree.map(lambda x: jax.device_put(x, placement), batch)


def current_git_sha() -> str:
    repo = os.environ.get("REPO_DIR")
    if not repo:
        raise RuntimeError("REPO_DIR is required even when invoking the Python entrypoint directly")
    actual = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True,
                            text=True, capture_output=True).stdout.strip()
    expected = os.environ.get("EXPECTED_COMMIT")
    if expected and actual != expected:
        raise RuntimeError(f"actual git HEAD {actual} does not match EXPECTED_COMMIT")
    return actual


def execution_commit_provenance() -> dict[str, Any]:
    value = os.environ.get("PI05_ALLOW_UNPUBLISHED_COMMIT", "0")
    if value == "0":
        return {
            "execution_commit_policy": "canonical_upstream_required",
            "canonical_publication_verified": True,
        }
    if value == "1":
        return {
            "execution_commit_policy": "user_approved_local_commit",
            "canonical_publication_verified": False,
        }
    raise ValueError("PI05_ALLOW_UNPUBLISHED_COMMIT must be exactly 0 or 1")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_loss_args(args)
    commit_provenance = execution_commit_provenance()
    if args.resume != bool(args.from_checkpoint):
        raise ValueError("--resume and --from-checkpoint must be specified together")
    rollout_store = None
    rollout_summary = None
    if args.rollout_manifest is not None:
        # Parsing, provenance checks, sample checksums, and label-policy validation
        # deliberately happen before model parameters or an output directory are touched.
        from openpi.training.plugin_rollouts import load_rollout_manifest

        rollout_store = load_rollout_manifest(args.rollout_manifest)
        rollout_summary = rollout_store.summary()
    loss_status, handoff_records, call_records = resolve_loss_status(args, rollout_store)
    validation_handoff_records, validation_call_records = validation_loss_records(rollout_store)
    print(json.dumps({"resolved_loss_config": loss_status}, sort_keys=True))
    norm_path, _, norm_hash, base_hash = require_inputs(args)
    if rollout_summary is not None:
        check_rollout_binding(rollout_summary, norm_hash, base_hash)
    git_sha = current_git_sha()
    train_raw, val_raw, dataset_manifest = load_datasets(args)
    if args.preflight_only:
        run_preflight(
            args, train_raw, val_raw, dataset_manifest, norm_hash, base_hash, norm_path, git_sha,
            rollout_store=rollout_store, rollout_summary=rollout_summary, loss_status=loss_status,
            handoff_records=handoff_records, call_records=call_records,
        )
        return

    if os.environ.get("JAX_PROCESS_COUNT", "1") != "1":
        raise RuntimeError("this entrypoint expects one 8-GPU process")
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY must be injected through the environment")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"no-overwrite: output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    import jax
    import jax.numpy as jnp
    import optax
    import wandb
    from openpi.training import plugin_bank, plugin_objectives, sharding

    if len(jax.devices()) != 8:
        raise RuntimeError(f"exactly 8 JAX devices required, got {jax.devices()}")
    if args.batch_size <= 0 or args.batch_size % 8:
        raise ValueError("global batch size must be positive and divisible by 8")
    mesh = sharding.make_mesh(8)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=5e-5, warmup_steps=100,
        decay_steps=UPDATES_PER_SUITE, end_value=5e-6,
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(
        schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10,
    ))
    loss_config = plugin_objectives.LossConfig(
        handoff_weight=args.handoff_weight, call_weight=args.call_weight
    )
    params_path = args.base_checkpoint / "params"
    graphdef, frozen, adapters = plugin_bank.initialize_bank(str(params_path), args.seed, mesh)
    opt_states = plugin_bank.initialize_optimizer_states(tx, adapters)
    call_active = bool(loss_status["call"]["active"])
    head_tx = None
    head_params = None
    head_opt_state = None
    head_updates = 0
    if call_active:
        head_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                1e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4
            ),
        )
        head_seed = jax.random.fold_in(jax.random.key(args.seed), 2081)
        head_params = plugin_objectives.init_call_head(head_seed)
        head_opt_state = head_tx.init(head_params)

    manifest_core = {
        "schema": 1, "run_name": "PI05-libero-test", "project": "physicalrsi",
        "git_sha": git_sha, "seed": args.seed,
        **commit_provenance,
        "suites": list(SUITES), "total_updates": TOTAL_UPDATES, "updates_per_suite": UPDATES_PER_SUITE,
        "global_batch_size": args.batch_size, "fsdp_devices": 8, "lora_rank": 32, "ema": False,
        "optimizer": {"name": "adamw", "b1": .9, "b2": .95, "eps": 1e-8,
                      "weight_decay": 1e-10, "clip_norm": 1.0},
        "call_head_optimizer": {
            "name": "adamw", "learning_rate": 1e-4, "b1": 0.9, "b2": 0.999,
            "eps": 1e-8, "weight_decay": 1e-4, "clip_norm": 1.0,
        },
        "sampler": {"name": "per-suite deterministic epoch permutation", "drop_last": True,
                    "resume_offset": "suite_updates * global_batch_size", "workers_per_suite": args.num_workers},
        "schedule": {"name": "warmup_cosine", "warmup": 100, "peak": 5e-5,
                     "end": 5e-6, "steps": 1000},
        "holdout_semantics": "adapter-training diagnostic only; not unseen-policy validation or official success",
        "stage_semantics": {
            "full_experimental_joint_losses": (
                "stage B experimental full-coverage joint-loss training; routing head remains uncalibrated"
            ),
            "partial_experimental_joint_losses": (
                "experimental partial joint-loss training; see per-target disabled reasons; "
                "no full-handoff claim"
            ),
            "stage_a_fm_only": "stage A: four independent FM LoRA adapters; joint losses disabled",
        }[loss_status["mode"]],
        "stage_selection": loss_status["stage_selection"],
        "fm_only_requested": bool(args.fm_only),
        "resolved_loss_config": loss_status,
        "loss_config": dataclasses.asdict(loss_config),
        "rollout_manifest_summary": rollout_summary,
        "heldout_real_label_counts": {
            "handoff_by_target": {
                suite: len(validation_handoff_records[suite]) for suite in SUITES
            },
            "call": len(validation_call_records),
            "call_by_policy": {
                policy_id: sum(
                    record.attempted_policy_id == policy_id
                    for record in validation_call_records
                )
                for policy_id in POLICY_IDS
            },
        },
        "label_policy_bundle_ids": (
            [] if rollout_summary is None else rollout_summary["policy_bundle_ids"]
        ),
        "routing_head_status": "experimental_uncalibrated",
        "routing_use_warning": (
            "Fresh outcome validation of the final updated bank is required; "
            "this checkpoint does not claim calibrated routing."
        ),
        "norm_stats_sha256": norm_hash, "base_inventory_sha256": base_hash,
        "dataset_manifest": dataset_manifest,
    }
    manifest_core["runner_manifest_sha256"] = canonical_hash(manifest_core)
    manifest_path = args.output_dir / "run_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume requires existing {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != manifest_core:
            raise RuntimeError("resume run_manifest.json does not match current run")
    else:
        # Output was proven absent above, so this cannot silently overwrite a run.
        manifest_path.write_text(json.dumps(manifest_core, indent=2, sort_keys=True) + "\n")
    steps = {s: 0 for s in SUITES}
    if args.resume:
        adapters, opt_states, steps, saved_manifest = plugin_bank.load_bank(
            args.from_checkpoint, adapters, opt_states,
            expected_base_checkpoint_path=str(params_path),
            expected_norm_stats_hash=norm_hash, expected_base_manifest_hash=base_hash,
        )
        saved_extra = saved_manifest.get("metadata_extra", saved_manifest)
        if saved_extra.get("runner_manifest_sha256") != manifest_core["runner_manifest_sha256"]:
            raise RuntimeError("resume manifest does not match current run")
        if call_active:
            auxiliary = plugin_bank.load_auxiliary_state(
                args.from_checkpoint,
                {
                    "head_params": head_params,
                    "head_opt_state": head_opt_state,
                    "updates": np.asarray(0, dtype=np.int64),
                },
            )
            head_params = auxiliary["head_params"]
            head_opt_state = auxiliary["head_opt_state"]
            head_updates = int(np.asarray(auxiliary["updates"]))
            if head_updates != sum(steps.values()):
                raise ValueError(
                    "call-head update count does not match restored global actor update count"
                )

    transform = make_transform(norm_path)
    call_transform = make_transform(norm_path, require_actions=False)
    train_loaders = make_loaders(args, train_raw, transform, steps=steps, train=True)
    # Fixed val loaders and fixed RNG make every 400-step evaluation comparable.
    val_loaders = make_loaders(args, val_raw, transform, steps=steps, train=False)
    train_iters: dict[str, Any] = {}; val_batches = {
        s: observation_and_actions(next(iter(val_loaders[s]))) for s in SUITES
    }
    step_fn = plugin_bank.make_step(graphdef, tx, mesh)
    handoff_step_fn = None
    if any(item["active"] for item in loss_status["handoff"].values()):
        handoff_step_fn = plugin_objectives.make_handoff_step(
            graphdef, tx, mesh, loss_config.handoff_weight
        )
    feature_fn = call_step_fn = None
    if call_active:
        feature_fn = plugin_objectives.make_feature_extractor(graphdef, mesh)
        call_step_fn = plugin_objectives.make_call_step(head_tx, loss_config.call_weight)
    eval_fn = plugin_bank.make_eval_step(graphdef, mesh)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        LOGGER.warning("signal %s received; saving after this optimizer step", signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    wandb_id_path = args.output_dir / "wandb_run_id.txt"
    if args.resume:
        if not wandb_id_path.is_file(): raise FileNotFoundError(f"resume requires {wandb_id_path}")
        run = wandb.init(project="physicalrsi", name="PI05-libero-test", config=manifest_core,
                         id=wandb_id_path.read_text().strip(), resume="must")
    else:
        run = wandb.init(project="physicalrsi", name="PI05-libero-test", config=manifest_core)
        wandb_id_path.write_text(run.id + "\n")
    global_step = sum(steps.values())
    last_metrics: dict[str, dict[str, float]] = {}

    def save_checkpoint(step: int):
        dest = args.output_dir / "checkpoints" / f"step_{step:08d}"
        if dest.exists(): raise FileExistsError(f"checkpoint already exists: {dest}")
        auxiliary_state = None
        if call_active:
            auxiliary_state = {
                "head_params": head_params,
                "head_opt_state": head_opt_state,
                "updates": np.asarray(head_updates, dtype=np.int64),
            }
        plugin_bank.save_bank(dest, adapters, opt_states, steps,
            base_checkpoint_path=str(params_path), norm_stats_hash=norm_hash,
            base_manifest_hash=base_hash, metadata_extra=manifest_core,
            auxiliary_state=auxiliary_state)

    while global_step < TOTAL_UPDATES:
        expected_suite = SUITES[global_step % len(SUITES)]
        raw_batch = next_cycling(train_loaders, train_iters, expected_suite)
        obs, actions = put_batch_on_mesh(observation_and_actions(raw_batch), mesh)
        step_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), global_step)
        handoff_active = bool(loss_status["handoff"][expected_suite]["active"])
        handoff_count = 0
        if handoff_active:
            selected_handoff, handoff_batch = selected_rollout_batch(
                rollout_store, handoff_records[expected_suite], transform,
                batch_size=args.batch_size,
                seed=args.seed + SUITES.index(expected_suite), step=steps[expected_suite],
            )
            handoff_obs, handoff_actions = put_batch_on_mesh(
                observation_and_actions(handoff_batch), mesh
            )
            new_adapter, new_opt, metrics = handoff_step_fn(
                frozen, adapters[expected_suite], opt_states[expected_suite],
                obs, actions, handoff_obs, handoff_actions, step_rng,
            )
            candidate_adapters = dict(adapters)
            candidate_opts = dict(opt_states)
            candidate_adapters[expected_suite] = new_adapter
            candidate_opts[expected_suite] = new_opt
            selected = expected_suite
            handoff_count = len(selected_handoff)
        else:
            # This is the original stage-A actor path, including when neither
            # auxiliary loss has labels.  No extra actor optimizer update occurs.
            candidate_adapters, candidate_opts, selected, metrics = plugin_bank.update_selected_bank(
                global_step, step_fn, frozen, adapters, opt_states, obs, actions, step_rng
            )
        if selected != expected_suite: raise RuntimeError(f"round-robin mismatch: {selected} != {expected_suite}")
        values = scalar_metrics(metrics)
        values["actor_loss"] = values["loss"]
        values.setdefault("fm_loss", values["loss"])
        values.setdefault("handoff_loss", 0.0)
        values.update({
            "handoff_count": float(handoff_count),
            "handoff_active": float(handoff_active),
            "call_count": 0.0,
            "call_active": float(call_active),
            "call_loss": 0.0,
            "call_weighted_loss": 0.0,
        })

        candidate_head_params = head_params
        candidate_head_opt_state = head_opt_state
        if call_active:
            selected_calls, call_batch = selected_rollout_batch(
                rollout_store, call_records, call_transform,
                batch_size=args.batch_size, seed=args.seed, step=global_step,
            )
            call_obs = observation_only(call_batch)
            budgets = np.asarray([record.budget_steps for record in selected_calls], dtype=np.float32)
            targets, observed = rollout_store.call_targets(
                selected_calls, policy_order=POLICY_IDS
            )
            targets, observed = validate_call_supervision(targets, observed, len(selected_calls))
            call_obs, budgets, targets, observed = put_batch_on_mesh(
                (call_obs, budgets, targets, observed), mesh
            )
            call_rng = jax.random.fold_in(step_rng, 1)
            features = feature_fn(
                frozen, candidate_adapters[selected], call_obs, budgets, call_rng
            )
            candidate_head_params, candidate_head_opt_state, call_metrics = call_step_fn(
                head_params, head_opt_state, features, targets, observed
            )
            values.update(scalar_metrics(call_metrics))
            values["call_count"] = float(len(selected_calls))

        values["joint_loss"] = values["actor_loss"] + values["call_weighted_loss"]

        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"non-finite metrics before state commit at step {global_step}: {values}")
        if not (tree_is_finite(candidate_adapters[selected]) and tree_is_finite(candidate_opts[selected])):
            raise FloatingPointError(f"non-finite actor candidate before state commit at step {global_step}")
        if call_active and not (
            tree_is_finite(candidate_head_params) and tree_is_finite(candidate_head_opt_state)
        ):
            raise FloatingPointError(f"non-finite call-head candidate before state commit at step {global_step}")
        adapters, opt_states = candidate_adapters, candidate_opts
        if call_active:
            head_params, head_opt_state = candidate_head_params, candidate_head_opt_state
            head_updates += 1
        steps[selected] += 1; global_step += 1
        last_metrics[selected] = values
        if global_step % 20 == 0:
            logs = {f"train/{s}/{k}": v for s, m in last_metrics.items() for k, v in m.items()}
            logs.update({f"train/{s}/updates": steps[s] for s in SUITES})
            LOGGER.info("step=%d per_suite=%s", global_step,
                        {s: {"updates": steps[s], **last_metrics.get(s, {})} for s in SUITES})
            run.log(logs, step=global_step)
        if global_step % EVAL_EVERY == 0:
            eval_logs = {}
            for i, suite in enumerate(SUITES):
                erng = jax.random.fold_in(jax.random.PRNGKey(args.seed), i)
                vobs, vactions = put_batch_on_mesh(val_batches[suite], mesh)
                em = eval_fn(frozen, adapters[suite], vobs, vactions, erng)
                eval_logs.update({f"adapter_holdout_diagnostic/{suite}/{k}": v
                                  for k, v in scalar_metrics(em).items()})
            for i, suite in enumerate(SUITES):
                records = validation_handoff_records[suite]
                prefix = f"handoff_holdout_diagnostic/{suite}"
                eval_logs[f"{prefix}/available"] = float(bool(records))
                eval_logs[f"{prefix}/record_count"] = float(len(records))
                if records:
                    selected_handoff, handoff_batch = selected_rollout_batch(
                        rollout_store, records, transform,
                        batch_size=args.batch_size, seed=args.seed + 10_000 + i, step=0,
                    )
                    hobs, hactions = put_batch_on_mesh(
                        observation_and_actions(handoff_batch), mesh
                    )
                    hrng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 10_000 + i)
                    hm = eval_fn(frozen, adapters[suite], hobs, hactions, hrng)
                    eval_logs[f"{prefix}/loss"] = scalar_metrics(hm)["loss"]
                    eval_logs[f"{prefix}/sample_count"] = float(len(selected_handoff))

            call_prefix = "call_holdout_diagnostic_uncalibrated"
            call_eval_available = bool(call_active and validation_call_records)
            eval_logs[f"{call_prefix}/available"] = float(call_eval_available)
            eval_logs[f"{call_prefix}/record_count"] = float(len(validation_call_records))
            if call_eval_available:
                selected_calls, call_batch = selected_rollout_batch(
                    rollout_store, validation_call_records, call_transform,
                    batch_size=args.batch_size, seed=args.seed + 20_000, step=0,
                )
                call_obs = observation_only(call_batch)
                budgets = np.asarray(
                    [record.budget_steps for record in selected_calls], dtype=np.float32
                )
                targets, observed = rollout_store.call_targets(
                    selected_calls, policy_order=POLICY_IDS
                )
                targets, observed = validate_call_supervision(
                    targets, observed, len(selected_calls)
                )
                call_obs, budgets, targets, observed = put_batch_on_mesh(
                    (call_obs, budgets, targets, observed), mesh
                )
                crng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 20_000)
                # Action-expert adapters do not affect the prefix feature path;
                # use a fixed complete graph state for deterministic diagnostics.
                features = feature_fn(frozen, adapters[SUITES[0]], call_obs, budgets, crng)
                logits = plugin_objectives.logits_call_head(head_params, features)
                call_loss, observed_count = plugin_objectives.masked_call_loss(
                    logits, targets, observed
                )
                safe_logits = jnp.where(observed, logits, 0.0)
                safe_targets = jnp.where(observed, targets, 0.0)
                squared_error = jnp.square(jax.nn.sigmoid(safe_logits) - safe_targets)
                brier = jnp.sum(jnp.where(observed, squared_error, 0.0)) / jnp.maximum(
                    observed_count, 1.0
                )
                eval_logs.update({
                    f"{call_prefix}/loss": float(np.asarray(jax.device_get(call_loss))),
                    f"{call_prefix}/brier": float(np.asarray(jax.device_get(brier))),
                    f"{call_prefix}/observed_count": float(
                        np.asarray(jax.device_get(observed_count))
                    ),
                    f"{call_prefix}/sample_count": float(len(selected_calls)),
                })
            run.log(eval_logs, step=global_step)
        if global_step == 4 or global_step % SAVE_EVERY == 0 or stop_requested:
            save_checkpoint(global_step)
        if stop_requested: break
    run.finish()


if __name__ == "__main__":
    main()
