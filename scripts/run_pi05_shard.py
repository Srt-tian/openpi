#!/usr/bin/env python3
"""Build, run, and aggregate the fixed five-GPU PI0.5 LIBERO extension."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from pi05_shard_spec import (
    PLUGIN_IDS,
    SUITES,
    build_manifest,
    case_rows_from_outputs,
    load_episode_rows,
    load_manifest,
    source_checksum,
    validate_final_rows,
    validate_reused_rows,
    case_for,
)


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stop(child) -> None:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)


def add_build(subparsers):
    parser = subparsers.add_parser("build-plan")
    parser.add_argument("--reused-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def add_run(subparsers):
    parser = subparsers.add_parser("run-worker")
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, choices=range(5), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--plugins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physicalrsi-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rollout-workers", type=int, choices=(1, 2, 3), default=2)


def add_aggregate(subparsers):
    parser = subparsers.add_parser("aggregate")
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    add_build(subs)
    add_run(subs)
    add_aggregate(subs)
    return parser.parse_args(argv)


def build_plan(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = build_manifest(args.reused_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, manifest)


def gpu_locks(runtime_root: Path, gpu_id: int):
    handles = []
    lock_dir = runtime_root / "tmp"
    if not lock_dir.is_dir():
        raise FileNotFoundError(lock_dir)
    for name in (f"physicalrsi-gpu{gpu_id}-queue.lock", f"physicalrsi-gpu{gpu_id}-slot1-queue.lock"):
        handle = (lock_dir / name).open("a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    return handles


def ensure_gpu_idle(gpu_id: int) -> None:
    processes = subprocess.check_output([
        "nvidia-smi", "--id", str(gpu_id), "--query-compute-apps=pid", "--format=csv,noheader"
    ], text=True).strip()
    if processes:
        raise RuntimeError(f"GPU {gpu_id} has existing compute processes; not taking over")


def worker_expected_pairs(plan, worker_id: int) -> set[str]:
    return {case["id"] for case in plan["workers"][str(worker_id)]["cases"]}


def validate_worker_rows(rows, expected_ids: set[str]) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    for row in rows:
        case_id, arm = row.get("case_id"), row.get("arm")
        case = case_for(row.get("suite"), row.get("task_id"), row.get("init_id"))
        expected_policy = "base" if arm == "base" else PLUGIN_IDS[case["suite"]] if arm == "plugin" else None
        if case_id not in expected_ids or arm not in ("base", "plugin") or arm in pairs.setdefault(case_id, {}):
            raise RuntimeError("worker output contains unexpected or duplicate case arm")
        if (case_id != case["id"] or row.get("joint_task_number") != case["joint_task_number"]
                or row.get("policy_seed") != case["policy_seed"] or row.get("policy_id") != expected_policy
                or not isinstance(row.get("success"), bool)
                or row.get("status") not in ("success", "failure", "error")):
            raise RuntimeError("worker output identity or outcome fields are invalid")
        pairs[case_id][arm] = row
    if set(pairs) != expected_ids or any(set(pair) != {"base", "plugin"} for pair in pairs.values()):
        raise RuntimeError("worker output does not contain its exact complete pair set")
    return pairs


def run_worker(args) -> None:
    plan = load_manifest(args.plan_json)
    root = args.code.resolve()
    git = lambda *parts: subprocess.check_output(["git", "-C", str(root), *parts], text=True).strip()
    if git("rev-parse", "HEAD") != args.expected_commit or git("status", "--porcelain"):
        raise RuntimeError("execution checkout must be clean at the expected commit")
    if args.output.exists():
        raise FileExistsError("fresh worker output is required; no implicit resume")
    locks = gpu_locks(args.runtime_root, args.gpu_id)
    ensure_gpu_idle(args.gpu_id)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True)
    manifest_sha = hashlib.sha256((args.plugins / "manifest.json").read_bytes()).hexdigest()
    cases = plan["workers"][str(args.worker_id)]["cases"]
    suites = [suite for suite in SUITES if any(case["suite"] == suite for case in cases)]
    batches = [("base", "base", None), *[(PLUGIN_IDS[suite], "plugin", suite) for suite in suites]]
    expected_episodes = len(cases) * 2
    record = {
        "schema": "pi05_libero_shard_worker.v1",
        "status": "starting",
        "started_unix": time.time(),
        "global_plan_sha256": plan["manifest_sha256"],
        "worker_id": args.worker_id,
        "gpu_id": args.gpu_id,
        "rollout_workers": args.rollout_workers,
        "code": str(root),
        "commit": args.expected_commit,
        "checkpoint_sha256": manifest_sha,
        "planned_pairs": len(cases),
        "planned_episodes": expected_episodes,
        "retry_policy": "none",
        "batches": [],
    }
    save(args.output / "controller.json", record)
    policy_env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
        policy_env.pop(name, None)
    policy_env.update({
        "PYTHONPATH": f"{root}/src:{root}/packages/openpi-client/src",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OPENPI_DATA_HOME": "/home/magiclab/.cache/openpi",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    eval_env = dict(os.environ)
    eval_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    server = evaluator = None

    def interrupted(signum, frame):
        raise InterruptedError(f"worker controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for policy_id, arm, suite in batches:
            batch_cases = cases if arm == "base" else [case for case in cases if case["suite"] == suite]
            batch = {"policy_id": policy_id, "arm": arm, "suite": suite, "planned": len(batch_cases), "status": "loading"}
            record["batches"].append(batch)
            record.update(status="running", current_policy=policy_id)
            save(args.output / "controller.json", record)
            command = [str(args.policy_python), str(root / "scripts/serve_plugin_policy.py"),
                "--base-checkpoint", str(args.base), "--plugin-checkpoint", str(args.plugins),
                "--policy-id", policy_id, "--port", str(args.port), "--allow-verified-base-relocation"]
            with (args.output / f"server_{policy_id}.log").open("w") as log:
                server = subprocess.Popen(command, cwd=root, env=policy_env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                batch["server_pid"] = server.pid
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"{policy_id} service exited {server.returncode} before readiness")
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"{policy_id} service not ready within 900 seconds")
                        time.sleep(2)
                batch["status"] = "evaluating"
                save(args.output / "controller.json", record)
                eval_args = [str(root / "scripts/eval_pi05_shard.py"), "--execute",
                    "--base-eval", str(root / "scripts/eval_pi05_plugins.py"),
                    "--plan-json", str(args.plan_json.resolve()), "--worker-id", str(args.worker_id),
                    "--rollout-workers", str(args.rollout_workers), "--arm", arm,
                    "--service-uri", f"ws://127.0.0.1:{args.port}", "--timeout-seconds", "300",
                    "--expected-checkpoint-sha256", manifest_sha,
                    "--physicalrsi-root", str(args.physicalrsi_root),
                    "--output", str(args.output / policy_id)]
                if suite:
                    eval_args += ["--suite", suite]
                venv_site = args.policy_python.parent.parent / "lib/python3.11/site-packages"
                shell = 'source "$1/bin/env.sh"\nexport PYTHONPATH="$2/scripts:$2/packages/openpi-client/src:$PYTHONPATH:$3"\nshift 3\nexec "$LIBERO_EVAL_ROOT/runtime/python311/bin/python3.11" "$@"'
                with (args.output / f"eval_{policy_id}.log").open("w") as evlog:
                    evaluator = subprocess.Popen(["bash", "-c", shell, "shard-eval", str(args.runtime_root),
                        str(root), str(venv_site), *eval_args], env=eval_env, stdout=evlog,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    exit_code = evaluator.wait(timeout=21600)
                batch["eval_exit_code"] = exit_code
                if exit_code:
                    raise RuntimeError(f"{policy_id} evaluator exited {exit_code}")
                summary = json.loads((args.output / policy_id / "summary.json").read_text())
                if summary.get("complete") is not True or summary.get("episodes") != len(batch_cases):
                    raise RuntimeError(f"{policy_id} batch coverage incomplete; stop without retry")
                batch.update(status="complete", episodes=summary["episodes"], successes=summary["successes"], errors=summary["errors"])
                save(args.output / "controller.json", record)
                if summary["errors"]:
                    raise RuntimeError(f"{policy_id} infrastructure errors; preserve and stop without retry")
                stop(server)
                server = None
        rows = case_rows_from_outputs(args.output)
        paired = validate_worker_rows(rows, worker_expected_pairs(plan, args.worker_id))
        result = {
            "episodes": len(rows), "pairs": len(paired), "errors": 0,
            "base_successes": sum(pair["base"]["success"] for pair in paired.values()),
            "plugin_successes": sum(pair["plugin"]["success"] for pair in paired.values()),
            "claim_scope": "sharded_development_extension_not_official_score",
        }
        save(args.output / "paired_summary.json", result)
        record.update(status="complete", result=result)
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500])
        raise
    finally:
        stop(evaluator)
        stop(server)
        record["ended_unix"] = time.time()
        save(args.output / "controller.json", record)
        del locks


def aggregate(args) -> None:
    plan = load_manifest(args.plan_json)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.worker_output) != 5:
        raise ValueError("aggregate requires exactly five --worker-output directories")
    reused_root = Path(plan["reused_source"])
    reused_rows, checksum = load_episode_rows(reused_root)
    if checksum != plan["reused_source_checksum"]:
        raise ValueError("reused pilot02 source checksum changed")
    validate_reused_rows(reused_rows)
    new_rows = []
    seen_worker_ids = set()
    for root in args.worker_output:
        controller = json.loads((root / "controller.json").read_text())
        worker_id = controller.get("worker_id")
        if controller.get("status") != "complete" or controller.get("global_plan_sha256") != plan["manifest_sha256"]:
            raise ValueError("worker output is incomplete or belongs to another plan")
        if worker_id in seen_worker_ids or worker_id not in range(5):
            raise ValueError("worker outputs must cover IDs 0..4 exactly once")
        seen_worker_ids.add(worker_id)
        rows = case_rows_from_outputs(root)
        validate_worker_rows(rows, worker_expected_pairs(plan, worker_id))
        new_rows.extend(rows)
    if seen_worker_ids != set(range(5)):
        raise ValueError("worker outputs must cover IDs 0..4")
    paired = validate_final_rows([*reused_rows, *new_rows])
    errors = sum(row.get("status") == "error" for pair in paired.values() for row in pair.values())
    if errors:
        raise RuntimeError("aggregate contains infrastructure errors; no score emitted")
    result = {
        "schema": "pi05_libero_400pair_extension.v1",
        "global_plan_sha256": plan["manifest_sha256"],
        "reused_source_checksum": checksum,
        "episodes": 800,
        "pairs": 400,
        "reused_pairs": 80,
        "new_pairs": 320,
        "errors": 0,
        "base_successes": sum(pair["base"]["success"] for pair in paired.values()),
        "plugin_successes": sum(pair["plugin"]["success"] for pair in paired.values()),
        "recovered": sorted(key for key, pair in paired.items() if pair["plugin"]["success"] and not pair["base"]["success"]),
        "regressed": sorted(key for key, pair in paired.items() if pair["base"]["success"] and not pair["plugin"]["success"]),
        "claim_scope": "full_40_task_x_10_init_paired_development_evaluation_not_official_50_init_score",
    }
    args.output.mkdir(parents=True)
    save(args.output / "paired_summary.json", result)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "build-plan":
        build_plan(args)
    elif args.command == "run-worker":
        run_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
