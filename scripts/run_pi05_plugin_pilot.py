#!/usr/bin/env python3
"""One GPU, five fixed-policy services, paired development episodes; no retries."""
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
import sys
import time


def init_count_arg(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("init count must be an integer in 1..50") from exc
    if not 1 <= parsed <= 50:
        raise argparse.ArgumentTypeError("init count must be an integer in 1..50")
    return parsed


def coverage_expectations(init_count):
    if isinstance(init_count, bool) or not 1 <= init_count <= 50:
        raise ValueError("init_count must be in 1..50")
    return {
        "base_batch": 8 * init_count,
        "plugin_batch": 2 * init_count,
        "base_total": 8 * init_count,
        "plugin_total": 8 * init_count,
        "pairs": 8 * init_count,
        "episodes": 16 * init_count,
    }


def validate_pair_coverage(rows, init_count):
    expected = coverage_expectations(init_count)
    paired = {}
    for row in rows:
        pair = paired.setdefault(row["case_id"], {})
        if row["arm"] in pair:
            raise RuntimeError("pair coverage contains a duplicate arm")
        pair[row["arm"]] = row
    base_count = sum(row["arm"] == "base" for row in rows)
    plugin_count = sum(row["arm"] == "plugin" for row in rows)
    if (len(rows) != expected["episodes"] or len(paired) != expected["pairs"]
            or base_count != expected["base_total"] or plugin_count != expected["plugin_total"]
            or any(set(pair) != {"base", "plugin"} for pair in paired.values())):
        raise RuntimeError("pair coverage is incomplete or duplicated")
    return paired


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--plugins", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physicalrsi-root", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--policy-python", type=Path, required=True)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--init-count", type=init_count_arg, default=2)
    args = p.parse_args()
    expected = coverage_expectations(args.init_count)
    root = args.code.resolve()
    git = lambda *a: subprocess.check_output(["git", "-C", str(root), *a], text=True).strip()
    if git("rev-parse", "HEAD") != args.expected_commit or git("status", "--porcelain"):
        raise RuntimeError("execution checkout must be clean at the expected commit")
    if args.output.exists():
        raise FileExistsError("fresh output is required; no implicit resume")
    lock_handles = []
    for name in ("physicalrsi-gpu0-queue.lock", "physicalrsi-gpu0-slot1-queue.lock"):
        handle = (args.runtime_root / "tmp" / name).open("a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_handles.append(handle)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if processes:
        raise RuntimeError("GPU has existing compute processes; not taking over")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True)
    manifest_sha = hashlib.sha256((args.plugins / "manifest.json").read_bytes()).hexdigest()
    batches = [("base", "base", None), ("spatial", "plugin", "libero_spatial"),
               ("object", "plugin", "libero_object"), ("goal", "plugin", "libero_goal"),
               ("long", "plugin", "libero_10")]
    record = {"schema": 1, "status": "starting", "started_unix": time.time(),
              "code": str(root), "commit": args.expected_commit,
              "base": str(args.base), "plugins": str(args.plugins),
              "checkpoint_sha256": manifest_sha, "init_count": args.init_count,
              "planned_episodes": expected["episodes"],
              "scope": "development_pilot_not_official_score", "batches": []}
    save(args.output / "controller.json", record)
    policy_env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
        policy_env.pop(name, None)
    policy_env.update({"PYTHONPATH": f"{root}/src:{root}/packages/openpi-client/src",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OPENPI_DATA_HOME": "/home/magiclab/.cache/openpi",
        "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    server = evaluator = None
    def interrupted(signum, frame):
        raise InterruptedError(f"controller received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for policy_id, arm, suite in batches:
            batch = {"policy_id": policy_id, "arm": arm, "suite": suite, "status": "loading"}
            record["batches"].append(batch)
            record.update(status="running", current_policy=policy_id)
            save(args.output / "controller.json", record)
            command = [str(args.policy_python), str(root / "scripts/serve_plugin_policy.py"),
                "--base-checkpoint", str(args.base), "--plugin-checkpoint", str(args.plugins),
                "--policy-id", policy_id, "--port", str(args.port), "--allow-verified-base-relocation"]
            with (args.output / f"server_{policy_id}.log").open("w") as log:
                server = subprocess.Popen(command, cwd=root, env=policy_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                batch["server_pid"] = server.pid
                save(args.output / "controller.json", record)
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
                eval_args = [str(root / "scripts/eval_pi05_plugins.py"), "--execute", "--arm", arm,
                    "--service-uri", f"ws://127.0.0.1:{args.port}", "--timeout-seconds", "300",
                    "--expected-checkpoint-sha256", manifest_sha, "--physicalrsi-root", str(args.physicalrsi_root),
                    "--output", str(args.output / policy_id), "--init-count", str(args.init_count)]
                if suite:
                    eval_args += ["--suite", suite]
                venv_site = args.policy_python.parent.parent / "lib/python3.11/site-packages"
                # Existing LIBERO stack owns NumPy/torch; append only to resolve websocket dependencies.
                shell = 'source "$1/bin/env.sh"\nexport PYTHONPATH="$2/packages/openpi-client/src:$PYTHONPATH:$3"\nshift 3\nexec "$LIBERO_EVAL_ROOT/runtime/python311/bin/python3.11" "$@"'
                with (args.output / f"eval_{policy_id}.log").open("w") as evlog:
                    evaluator = subprocess.Popen(["bash", "-c", shell, "pilot-eval", str(args.runtime_root),
                        str(root), str(venv_site), *eval_args], stdout=evlog, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    exit_code = evaluator.wait(timeout=7200)
                batch["eval_exit_code"] = exit_code
                if exit_code:
                    raise RuntimeError(f"{policy_id} evaluator exited {exit_code}")
                summary = json.loads((args.output / policy_id / "summary.json").read_text())
                expected_batch = expected["base_batch"] if policy_id == "base" else expected["plugin_batch"]
                if summary.get("complete") is not True or summary.get("episodes") != expected_batch:
                    raise RuntimeError(f"{policy_id} batch coverage is incomplete; stop without retry")
                batch.update(status="complete", episodes=summary["episodes"],
                    successes=summary["successes"], errors=summary["errors"])
                save(args.output / "controller.json", record)
                if summary["errors"]:
                    raise RuntimeError(f"{policy_id} infrastructure errors; preserve cases and stop without retry")
                stop(server)
                server = None
        rows = []
        for policy_id, _, _ in batches:
            for path in sorted((args.output / policy_id / "episodes").glob("*.json")):
                rows.append(json.loads(path.read_text()))
        paired = validate_pair_coverage(rows, args.init_count)
        result = {"episodes": expected["episodes"], "pairs": expected["pairs"],
            "init_count": args.init_count, "errors": sum(row["status"] == "error" for row in rows),
            "base_successes": sum(pair["base"]["success"] for pair in paired.values()),
            "plugin_successes": sum(pair["plugin"]["success"] for pair in paired.values()),
            "recovered": [key for key, pair in paired.items() if pair["plugin"]["success"] and not pair["base"]["success"]],
            "regressed": [key for key, pair in paired.items() if pair["base"]["success"] and not pair["plugin"]["success"]],
            "claim_scope": "paired_development_pilot_not_official_score"}
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


if __name__ == "__main__":
    main()
