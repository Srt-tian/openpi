#!/usr/bin/env python3
"""Thin external-plan driver around the reviewed PI0.5 LIBERO evaluator."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from pi05_shard_spec import PLUGIN_IDS, SUITES, load_manifest, validate_case


def load_base(path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("pi05_eval_base_for_shard", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, choices=range(5), required=True)
    parser.add_argument("--rollout-workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--arm", choices=("base", "plugin"), required=True)
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--service-uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physicalrsi-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-base-graph", default="original_pi05_libero")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--api-key-env", default="OPENPI_API_KEY")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def fixed_batch_plan(manifest, worker_id: int, arm: str, suite: str | None) -> list[dict]:
    if arm == "plugin" and suite is None:
        raise ValueError("plugin batch requires exactly one --suite and one fixed service")
    if arm == "base" and suite is not None:
        raise ValueError("base batch covers the worker plan; do not filter it by suite")
    cases = [validate_case(case, manifest["seed"]) for case in manifest["workers"][str(worker_id)]["cases"]]
    if suite is not None:
        cases = [case for case in cases if case["suite"] == suite]
    if not cases:
        raise ValueError("fixed service batch has no cases")
    policy_id = "base" if arm == "base" else PLUGIN_IDS[suite]
    return [{"arm": arm, "policy_id": policy_id, **case} for case in cases]


def episode_path(episodes: Path, index: int, item: dict) -> Path:
    return episodes / f"{index:04d}_{item['arm']}_{item['suite']}_{item['task_id']}_{item['init_id']}.json"


def run_rows(base, plan, args, videos, episodes):
    """Sequential fallback; parallel_rollouts provides the reviewed spawn implementation."""
    if args.rollout_workers > 1:
        try:
            from parallel_rollouts import run_episodes_ordered
        except ImportError as exc:
            raise RuntimeError("parallel_rollouts.py is required for --rollout-workers > 1") from exc
        return run_episodes_ordered(args.base_eval, plan, vars(args), episodes, videos, args.rollout_workers)
    return [base.run_episode(item, args, videos, episode_path(episodes, index, item))
            for index, item in enumerate(plan)]


def main(argv=None):
    args = parse_args(argv)
    manifest = load_manifest(args.plan_json)
    if args.seed != manifest["seed"]:
        raise ValueError("CLI seed differs from frozen plan")
    plan = fixed_batch_plan(manifest, args.worker_id, args.arm, args.suite)
    if not args.execute:
        print(json.dumps({"execute": False, "worker_id": args.worker_id, "plan": plan}, indent=2))
        return 0
    if args.output is None:
        raise ValueError("--output is required with --execute")
    base = load_base(args.base_eval)
    policy_id, checkpoint_sha = base.validate_execute_args(args, plan)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("evaluation output must be a fresh path")
    init_assets = base.inspect_init_assets(plan)
    service_identity = base.validate_service_metadata(args, policy_id, checkpoint_sha)
    args.verified_service_identity = service_identity
    output.mkdir(parents=True)
    episodes, videos = output / "episodes", output / "videos"
    episodes.mkdir()
    videos.mkdir()
    receipt = {
        "schema": "pi05_libero_shard_batch.v1",
        "global_plan_sha256": manifest["manifest_sha256"],
        "worker_id": args.worker_id,
        "rollout_workers": args.rollout_workers,
        "arm": args.arm,
        "suite": args.suite,
        "policy_id": policy_id,
        "service_identity": service_identity,
        "official_init_assets": init_assets,
        "plan": plan,
    }
    receipt_hash = base.canonical_hash(receipt)
    receipt["manifest_sha256"] = receipt_hash
    base.atomic_json(output / "manifest.json", receipt)
    rows = run_rows(base, plan, args, videos, episodes)
    base.atomic_json(output / "summary.json", base.summarize(rows, receipt_hash, len(plan)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
