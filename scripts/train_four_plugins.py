#!/usr/bin/env python3
"""Train four independent pi0.5 LoRA adapters in one round-robin job.

This entrypoint deliberately does not import openpi.training.config or data_loader:
the pinned OpenPI checkout expects an older LeRobot API.  Dataset access is through
openpi.training.plugin_data, which reads the local v3 dataset directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import numpy as np


SUITES = ("spatial", "object", "goal", "long")
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


def parse_args() -> argparse.Namespace:
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
    return p.parse_args()


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


def sample_summary(x: dict[str, Any]) -> dict[str, Any]:
    def summarize(v):
        if isinstance(v, dict): return {k: summarize(x) for k, x in v.items()}
        if isinstance(v, np.ndarray): return {"shape": list(v.shape), "dtype": str(v.dtype)}
        if isinstance(v, list): return {"length": len(v), "first": str(v[0])[:160] if v else None}
        return str(v)[:160]
    return {k: summarize(v) for k, v in x.items()}


def run_preflight(args: argparse.Namespace, train, val, manifest, norm_hash: str, base_hash: str,
                  norm_path: Path, git_sha: str) -> None:
    report: dict[str, Any] = {"dataset_manifest": manifest, "norm_stats_sha256": norm_hash,
                              "base_inventory_sha256": base_hash, "git_sha": git_sha, "suites": {}}
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
    loaders = make_loaders(args, train, transform, steps={s: 0 for s in SUITES},
                           train=True, persistent_workers=False)
    for suite in SUITES:
        batch = next(iter(loaders[suite]))
        report["suites"][suite]["transformed_batch"] = sample_summary(batch)
    print(json.dumps(report, indent=2, sort_keys=True))


class TransformedDataset:
    def __init__(self, dataset, transform): self.dataset, self.transform = dataset, transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return self.transform(dict(self.dataset[index]))


def make_transform(norm_dir: Path):
    # Importing these is safe; importantly, this is not training.config/data_loader.
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.models import tokenizer as tokenizer_api
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.shared import normalize

    stats = normalize.load(norm_dir.parent)
    return transforms.compose((
        LiberoInputs(model_type=model_api.ModelType.PI05),
        transforms.Normalize(stats, use_quantiles=True, strict=True),
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


def scalar_metrics(metrics):
    import jax
    return {k: float(np.asarray(jax.device_get(v))) for k, v in metrics.items()}


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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.resume != bool(args.from_checkpoint):
        raise ValueError("--resume and --from-checkpoint must be specified together")
    norm_path, _, norm_hash, base_hash = require_inputs(args)
    git_sha = current_git_sha()
    train_raw, val_raw, dataset_manifest = load_datasets(args)
    if args.preflight_only:
        run_preflight(args, train_raw, val_raw, dataset_manifest, norm_hash, base_hash, norm_path, git_sha)
        return

    if os.environ.get("JAX_PROCESS_COUNT", "1") != "1":
        raise RuntimeError("this entrypoint expects one 8-GPU process")
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY must be injected through the environment")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"no-overwrite: output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    import jax
    import optax
    import wandb
    from openpi.training import plugin_bank
    from openpi.training import sharding

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
    params_path = args.base_checkpoint / "params"
    graphdef, frozen, adapters = plugin_bank.initialize_bank(str(params_path), args.seed, mesh)
    opt_states = plugin_bank.initialize_optimizer_states(tx, adapters)

    manifest_core = {
        "schema": 1, "run_name": "PI05-libero-test", "project": "physicalrsi",
        "git_sha": git_sha, "seed": args.seed,
        "suites": list(SUITES), "total_updates": TOTAL_UPDATES, "updates_per_suite": UPDATES_PER_SUITE,
        "global_batch_size": args.batch_size, "fsdp_devices": 8, "lora_rank": 32, "ema": False,
        "optimizer": {"name": "adamw", "b1": .9, "b2": .95, "eps": 1e-8,
                      "weight_decay": 1e-10, "clip_norm": 1.0},
        "sampler": {"name": "per-suite deterministic epoch permutation", "drop_last": True,
                    "resume_offset": "suite_updates * global_batch_size", "workers_per_suite": args.num_workers},
        "schedule": {"name": "warmup_cosine", "warmup": 100, "peak": 5e-5,
                     "end": 5e-6, "steps": 1000},
        "holdout_semantics": "adapter-training diagnostic only; not unseen-policy validation or official success",
        "stage_semantics": "stage A: four independent FM LoRA adapters; no joint-loss claim",
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

    transform = make_transform(norm_path)
    train_loaders = make_loaders(args, train_raw, transform, steps=steps, train=True)
    # Fixed val loaders and fixed RNG make every 400-step evaluation comparable.
    val_loaders = make_loaders(args, val_raw, transform, steps=steps, train=False)
    train_iters: dict[str, Any] = {}; val_batches = {
        s: observation_and_actions(next(iter(val_loaders[s]))) for s in SUITES
    }
    step_fn = plugin_bank.make_step(graphdef, tx, mesh)
    eval_fn = plugin_bank.make_eval_step(graphdef, mesh)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logging.warning("signal %s received; saving after this optimizer step", signum)
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
        plugin_bank.save_bank(dest, adapters, opt_states, steps,
            base_checkpoint_path=str(params_path), norm_stats_hash=norm_hash,
            base_manifest_hash=base_hash, metadata_extra=manifest_core)

    while global_step < TOTAL_UPDATES:
        expected_suite = SUITES[global_step % len(SUITES)]
        raw_batch = next_cycling(train_loaders, train_iters, expected_suite)
        obs, actions = put_batch_on_mesh(observation_and_actions(raw_batch), mesh)
        step_rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), global_step)
        candidate_adapters, candidate_opts, selected, metrics = plugin_bank.update_selected_bank(
            global_step, step_fn, frozen, adapters, opt_states, obs, actions, step_rng
        )
        if selected != expected_suite: raise RuntimeError(f"round-robin mismatch: {selected} != {expected_suite}")
        values = scalar_metrics(metrics)
        if not all(np.isfinite(values[k]) for k in ("loss", "grad_norm", "adapter_norm")):
            raise FloatingPointError(f"non-finite metrics before state commit at step {global_step}: {values}")
        adapters, opt_states = candidate_adapters, candidate_opts
        steps[selected] += 1; global_step += 1
        last_metrics[selected] = values
        if global_step % 20 == 0:
            logs = {f"train/{s}/{k}": v for s, m in last_metrics.items() for k, v in m.items()}
            logs.update({f"train/{s}/updates": steps[s] for s in SUITES})
            logging.info("step=%d per_suite=%s", global_step,
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
            run.log(eval_logs, step=global_step)
        if global_step == 4 or global_step % SAVE_EVERY == 0 or stop_requested:
            save_checkpoint(global_step)
        if stop_requested: break
    run.finish()


if __name__ == "__main__":
    main()
