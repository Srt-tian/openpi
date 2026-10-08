#!/usr/bin/env python3
"""Run frozen PI05 harness batches with one service and one simulator per GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from run_pi05_shard import ensure_gpu_idle, gpu_locks, save, stop


def validate_summary(summary, cases, policy, mode):
    expected = {(c["suite"], c["task_id"], c["init_id"], c.get("replicate_id", 0)) for c in cases}
    rows = summary.get("cases", [])
    actual = {(c["suite"], c["task_id"], c["init_id"], c.get("replicate_id", 0)) for c in rows}
    if (len(expected) != len(cases) or len(rows) != len(cases) or actual != expected
            or not summary.get("complete") or summary.get("errors") != 0
            or summary.get("planned") != len(cases) or summary.get("completed") != len(cases)
            or any(c.get("policy_id") != policy or c.get("status") == "error" for c in rows)):
        raise RuntimeError("batch coverage, policy identity, or error accounting failed")
    if mode == "parity" and summary.get("parity_equal") is not True:
        raise RuntimeError("exact loop parity failed")


def checked_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("batch input must be a checked-in file under the execution checkout")
    return path


def check_checkout(root, expected):
    git = lambda *args: subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != expected or git("status", "--porcelain"):
        raise ValueError("execution checkout is not clean at its frozen commit")


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("code", "job", "base", "plugins", "output", "physicalrsi-root", "runtime-root", "policy-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-physicalrsi-commit", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = args.code.resolve()
    check_checkout(root, args.expected_commit)
    check_checkout(args.physicalrsi_root, args.expected_physicalrsi_commit)
    job_path = checked_path(root, args.job)
    job = json.loads(job_path.read_text())
    if job.get("schema") != "pi05_harness_worker.v1" or not job.get("batches"):
        raise ValueError("invalid worker job")
    if args.output.exists():
        raise FileExistsError("fresh output required; no implicit retry or resume")
    locks = gpu_locks(args.runtime_root, args.gpu_id)
    ensure_gpu_idle(args.gpu_id)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    manifest_bytes = (args.plugins / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    batches = []
    names = set()
    for batch in job["batches"]:
        name = batch["name"]
        if not name.replace("_", "").replace("-", "").isalnum() or name in names:
            raise ValueError("invalid or duplicate batch name")
        names.add(name)
        if batch["mode"] not in ("parity", "harness", "legacy"):
            raise ValueError("unknown evaluation mode")
        paths = {key: checked_path(root, batch[key]) for key in ("registry", "routes", "cases")}
        routes = json.loads(paths["routes"].read_text())
        cases = json.loads(paths["cases"].read_text())["cases"]
        policies = {routes["tasks"][f"{case['suite']}/{case['task_id']}"] for case in cases}
        if len(policies) != 1:
            raise ValueError("one service batch must select one policy")
        policy = next(iter(policies))
        if routes["identities"][policy]["checkpoint_sha256"] != manifest_sha:
            raise ValueError("checkpoint does not match frozen route identity")
        expected_adapter = None if policy == "base" else manifest["banks"][policy]["adapter_sha256"]
        if routes["identities"][policy]["adapter_sha256"] != expected_adapter:
            raise ValueError("adapter does not match frozen route identity")
        batches.append((batch, paths, policy, cases))
    args.output.mkdir(parents=True)
    record = {"schema": "pi05_harness_worker_status.v1", "status": "starting",
              "started_unix": time.time(), "controller_pid": os.getpid(),
              "commit": args.expected_commit, "physicalrsi_commit": args.expected_physicalrsi_commit,
              "job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
              "checkpoint_sha256": manifest_sha, "retry_policy": "none", "batches": [],
              "services_per_gpu": 1, "simulators_per_gpu": 1}
    save(args.output / "controller.json", record)
    policy_env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
        policy_env.pop(key, None)
    policy_env.update(PYTHONPATH=f"{root}/src:{root}/packages/openpi-client/src",
                      PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                      CUDA_VISIBLE_DEVICES=str(args.gpu_id), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      OPENPI_DATA_HOME=str(Path.home() / ".cache/openpi"), OMP_NUM_THREADS="4",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    eval_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu_id))
    server = evaluator = None

    def interrupted(signum, _frame):
        raise InterruptedError(f"controller signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for batch_index, (batch, paths, policy, cases) in enumerate(batches):
            check_checkout(root, args.expected_commit)
            check_checkout(args.physicalrsi_root, args.expected_physicalrsi_commit)
            if hashlib.sha256((args.plugins / "manifest.json").read_bytes()).hexdigest() != manifest_sha:
                raise ValueError("plugin manifest changed after preflight")
            state = {"name": batch["name"], "policy_id": policy, "mode": batch["mode"],
                     "planned_cases": len(cases), "status": "loading"}
            record["batches"].append(state)
            record.update(status="running", current_batch=batch["name"])
            command = [str(args.policy_python), str(root / "scripts/serve_plugin_policy.py"),
                       "--base-checkpoint", str(args.base), "--plugin-checkpoint", str(args.plugins),
                       "--policy-id", policy, "--port", str(args.port), "--allow-verified-base-relocation"]
            state["service_reused_from_previous_batch"] = server is not None
            if server is None:
                with (args.output / f"server_{batch['name']}.log").open("w") as log:
                    server = subprocess.Popen(command, cwd=root, env=policy_env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            state["server_pid"] = server.pid
            save(args.output / "controller.json", record)
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"service exited before readiness ({server.returncode})")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("service not ready within 900 seconds")
                    time.sleep(2)
            eval_args = [str(root / "scripts/run_pi05_harness_eval.py"),
                         "--roborsi-root", str(args.physicalrsi_root),
                         "--eval-helpers", str(root / "scripts/eval_pi05_plugins.py"),
                         "--host", "127.0.0.1", "--port", str(args.port),
                         "--output", str(args.output / batch["name"]), "--mode", batch["mode"],
                         "--timeout-seconds", "300", "--max-episode-seconds", "1200",
                         "--record-payload-hashes"]
            for key, path in paths.items():
                eval_args.extend(["--" + key, str(path)])
            # Do not resolve policy_python: venv symlinks must keep their venv context.
            site = args.policy_python.parent.parent / "lib/python3.11/site-packages"
            shell = ('source "$1/bin/env.sh"\n'
                     'export PYTHONPATH="$2/scripts:$2/packages/openpi-client/src:$PYTHONPATH:$3"\n'
                     'shift 3\nexec "$LIBERO_EVAL_ROOT/runtime/python311/bin/python3.11" "$@"')
            state["status"] = "evaluating"
            with (args.output / f"eval_{batch['name']}.log").open("w") as log:
                evaluator = subprocess.Popen(["bash", "-c", shell, "harness-eval",
                                              str(args.runtime_root), str(root), str(site), *eval_args],
                                             env=eval_env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                state["evaluator_pid"] = evaluator.pid
                save(args.output / "controller.json", record)
                code = evaluator.wait(timeout=21600)
            state["eval_exit_code"] = code
            if code:
                raise RuntimeError(f"evaluation exited {code}; preserving evidence without retry")
            summary = json.loads((args.output / batch["name"] / "summary.json").read_text())
            validate_summary(summary, cases, policy, batch["mode"])
            state.update(status="complete", summary=summary)
            # Same-policy control/probe arms share one loaded model graph so a
            # cold-load numeric change cannot masquerade as a harness effect.
            next_policy = batches[batch_index + 1][2] if batch_index + 1 < len(batches) else None
            if next_policy != policy:
                stop(server)
                server = None
            save(args.output / "controller.json", record)
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        if record["batches"] and record["batches"][-1]["status"] != "complete":
            record["batches"][-1].update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        stop(evaluator)
        stop(server)
        record["ended_unix"] = time.time()
        save(args.output / "controller.json", record)
        del locks


if __name__ == "__main__":
    main()
