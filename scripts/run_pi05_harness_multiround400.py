#!/usr/bin/env python3
"""Create-only orchestration for approved, matched PI0.5 400-pair rounds.

This is deliberately a thin control-plane wrapper around the checked-in worker,
artifact exporter, and aggregate entrypoints.  It never builds candidate configs,
chooses seeds, retries a rollout, trains, or promotes a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from urllib.parse import urlparse


SCHEMA = "pi05_harness_multiround400.v1"
ROUND_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
TOP = {"schema", "round_id", "round_dir", "canonical", "candidate_registry",
       "baseline_registry", "changed_tasks", "case_plan", "task_catalog", "routes",
       "aggregate_script", "export_script", "weights", "runtime", "workers", "collection"}
WORKER = {"worker_id", "host", "user", "control_path", "checkout", "job", "output",
          "policy_python", "gpu_id", "port"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"create-only artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def checked_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("source input must be a nonempty checkout-relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"source input is missing or escapes checkout: {relative}")
    return path


def exact_keys(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} schema mismatch")
    return value


def case_key(row: dict) -> tuple[str, int, int, int]:
    required = {"suite", "task_id", "init_id", "replicate_id",
                "joint_task_number", "official_cap"}
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("case schema mismatch; wildcards and extra seed fields are forbidden")
    suite, task, init, rep = (row["suite"], row["task_id"], row["init_id"],
                              row["replicate_id"])
    if (suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
            or type(task) is not int or not 0 <= task < 10
            or type(init) is not int or not 0 <= init < 10
            or type(rep) is not int or rep != 0
            or type(row["joint_task_number"]) is not int
            or type(row["official_cap"]) is not int or row["official_cap"] <= 0):
        raise ValueError("case is outside the fixed official 40x10 replicate-0 protocol")
    return suite, task, init, rep


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_and_validate(spec_path: Path, source: Path) -> dict:
    source = source.resolve()
    spec = exact_keys(json.loads(spec_path.read_text()), TOP, "round spec")
    if spec["schema"] != SCHEMA or not isinstance(spec["round_id"], str) \
            or not ROUND_ID.fullmatch(spec["round_id"]):
        raise ValueError("invalid round schema or round_id")
    round_dir = Path(spec["round_dir"])
    if not round_dir.is_absolute():
        raise ValueError("round_dir must be absolute")

    canonical = exact_keys(spec["canonical"], {"repository", "branch", "commit"}, "canonical")
    parsed = urlparse(canonical["repository"])
    if (parsed.scheme not in {"https", "ssh"} or parsed.username or parsed.password
            or not canonical["branch"].startswith("feature/")
            or not SHA1.fullmatch(canonical["commit"])):
        raise ValueError("canonical provenance must be a credential-free URL, feature branch, full SHA")
    upstream = git_value(source, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote_name = upstream.split("/", 1)[0] if "/" in upstream else ""
    if (git_value(source, "rev-parse", "HEAD") != canonical["commit"]
            or git_value(source, "rev-parse", "@{upstream}") != canonical["commit"]
            or git_value(source, "branch", "--show-current") != canonical["branch"]
            or not upstream.endswith("/" + canonical["branch"])
            or not remote_name
            or git_value(source, "remote", "get-url", remote_name).rstrip("/")
                != canonical["repository"].rstrip("/")
            or git_value(source, "status", "--porcelain")):
        raise ValueError("source checkout is not clean at the explicit canonical branch/upstream")

    paths = {name: checked_file(source, spec[name]) for name in
             ("candidate_registry", "baseline_registry", "case_plan", "task_catalog",
              "routes", "aggregate_script", "export_script")}
    baseline = json.loads(paths["baseline_registry"].read_text())
    candidate = json.loads(paths["candidate_registry"].read_text())
    if (baseline.get("schema") != 1 or candidate.get("schema") != 1
            or set(baseline.get("tasks", {})) != set(candidate.get("tasks", {}))
            or len(baseline["tasks"]) != 40):
        raise ValueError("baseline/candidate registries must explicitly cover the same LIBERO-40")
    changed = spec["changed_tasks"]
    if (not isinstance(changed, list) or not changed or len(changed) != len(set(changed))
            or any(key not in baseline["tasks"] for key in changed)):
        raise ValueError("changed_tasks must be an explicit unique nonempty registry subset")
    for key in baseline["tasks"]:
        old = checked_file(source, str(Path(spec["baseline_registry"]).parent / baseline["tasks"][key]))
        new = checked_file(source, str(Path(spec["candidate_registry"]).parent / candidate["tasks"][key]))
        if key not in changed and old.read_bytes() != new.read_bytes():
            raise ValueError(f"unchanged task config differs: {key}")

    plan = json.loads(paths["case_plan"].read_text())
    if set(plan) != {"schema", "cases"} or plan["schema"] != "pi05_harness_candidate400_cases.v1" \
            or not isinstance(plan["cases"], list) or len(plan["cases"]) != 400:
        raise ValueError("case plan must be the explicit canonical 400-pair schema")
    expected = {case_key(row) for row in plan["cases"]}
    canonical_cases = {(suite, task, init, 0)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        for task in range(10) for init in range(10)}
    if len(expected) != 400 or expected != canonical_cases:
        raise ValueError("case plan is not exact unique LIBERO-40 x official-init-10")

    catalog = json.loads(paths["task_catalog"].read_text())
    catalog_keys = {row.get("key") for row in catalog.get("tasks", [])}
    if catalog.get("schema") != 1 or catalog_keys != set(baseline["tasks"]):
        raise ValueError("task catalog is not the matching explicit LIBERO-40")
    routes = json.loads(paths["routes"].read_text())
    if (set(routes.get("tasks", {})) != set(baseline["tasks"])
            or set(routes["tasks"].values()) != {"base"}
            or routes.get("identities", {}).get("base", {}).get("adapter_sha256") is not None):
        raise ValueError("round must use the frozen native-base route with adapter null")

    weights = exact_keys(spec["weights"], {"base_checkpoint", "plugin_checkpoint",
        "plugin_manifest_sha256", "policy_id", "adapter_sha256"}, "weights")
    if (weights["policy_id"] != "base" or weights["adapter_sha256"] is not None
            or not SHA256.fullmatch(weights["plugin_manifest_sha256"])):
        raise ValueError("only explicit native-base identity is accepted")
    runtime = exact_keys(spec["runtime"], {"physicalrsi_root", "physicalrsi_commit",
        "runtime_root"}, "runtime")
    if not SHA1.fullmatch(runtime["physicalrsi_commit"]):
        raise ValueError("physicalrsi_commit must be a full SHA")

    workers = spec["workers"]
    if not isinstance(workers, list) or len(workers) != 4:
        raise ValueError("exactly four workers are required")
    ids, outputs, ports, covered = set(), set(), set(), set()
    for worker in workers:
        exact_keys(worker, WORKER, "worker")
        wid = worker["worker_id"]
        if type(wid) is not int or not 0 <= wid < 4 or wid in ids:
            raise ValueError("worker_id must uniquely cover 0..3")
        ids.add(wid)
        for field in ("control_path", "checkout", "output", "policy_python"):
            if not Path(worker[field]).is_absolute():
                raise ValueError(f"worker {field} must be absolute")
        if (not isinstance(worker["host"], str) or not worker["host"]
                or not isinstance(worker["user"], str) or not worker["user"]
                or type(worker["gpu_id"]) is not int or worker["gpu_id"] < 0
                or type(worker["port"]) is not int or not 1024 <= worker["port"] <= 65535
                or worker["output"] in outputs or (worker["host"], worker["port"]) in ports):
            raise ValueError("invalid/duplicate worker host, output, gpu, or port")
        outputs.add(worker["output"]); ports.add((worker["host"], worker["port"]))
        job_path = checked_file(source, worker["job"])
        job = json.loads(job_path.read_text())
        if job.get("schema") != "pi05_harness_worker.v1" or len(job.get("batches", [])) != 2:
            raise ValueError("worker job must contain exactly matched control/candidate batches")
        control, trial = job["batches"]
        if (control.get("name") != f"worker{wid}_control"
                or trial.get("name") != f"worker{wid}_candidate"
                or control.get("mode") != "harness" or trial.get("mode") != "harness"
                or control.get("cases") != trial.get("cases")
                or control.get("routes") != spec["routes"] or trial.get("routes") != spec["routes"]
                or control.get("registry") != spec["baseline_registry"]
                or trial.get("registry") != spec["candidate_registry"]):
            raise ValueError("worker job is not an exact matched native-control/candidate mapping")
        cases_doc = json.loads(checked_file(source, control["cases"]).read_text())
        if set(cases_doc) != {"schema", "cases"} or len(cases_doc["cases"]) != 100:
            raise ValueError("each worker must have exactly 100 cases")
        keys = [case_key(row) for row in cases_doc["cases"]]
        if len(set(keys)) != 100 or covered.intersection(keys):
            raise ValueError("worker cases contain duplicates or overlap")
        covered.update(keys)
        worker["job_sha256"] = digest(job_path)
    if ids != set(range(4)) or covered != expected:
        raise ValueError("four jobs do not partition exact 400-pair coverage")

    collection = exact_keys(spec["collection"], {"collector_worker_id", "relay_dir",
        "collection_root", "max_compressed_bytes"}, "collection")
    if (collection["collector_worker_id"] not in ids
            or not Path(collection["relay_dir"]).is_absolute()
            or not Path(collection["collection_root"]).is_absolute()
            or type(collection["max_compressed_bytes"]) is not int
            or not 0 < collection["max_compressed_bytes"] <= 500_000_000):
        raise ValueError("invalid bounded collection configuration")
    spec["_source_hashes"] = {name: digest(path) for name, path in paths.items()}
    return spec


class Commands:
    def run(self, argv: list[str], *, input_text: str | None = None,
            capture: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(argv, input=input_text, text=True, check=True,
                              capture_output=capture)


def ssh_argv(worker: dict) -> list[str]:
    return ["ssh", "-S", worker["control_path"], "-o", "ControlMaster=no",
            f"{worker['user']}@{worker['host']}", "bash", "-s"]


def ssh(worker: dict, script: str, commands: Commands) -> str:
    return commands.run(ssh_argv(worker), input_text=script).stdout.strip()


def remote_preflight(spec: dict, worker: dict, commands: Commands) -> dict:
    q = shlex.quote
    checkout, output = worker["checkout"], worker["output"]
    script = f"""set -e
cd {q(checkout)}
test "$(git rev-parse HEAD)" = {q(spec['canonical']['commit'])}
test "$(git rev-parse @{{upstream}})" = {q(spec['canonical']['commit'])}
test "$(git branch --show-current)" = {q(spec['canonical']['branch'])}
test "$(git rev-parse --is-shallow-repository)" = false
test -z "$(git status --porcelain)"
test "$(sha256sum {q(checkout + '/' + worker['job'])} | awk '{{print $1}}')" = {q(worker['job_sha256'])}
test ! -e {q(output)}
test -d {q(spec['weights']['base_checkpoint'])}
test -f {q(spec['weights']['plugin_checkpoint'] + '/manifest.json')}
test -x {q(worker['policy_python'])}
test "$(sha256sum {q(spec['weights']['plugin_checkpoint'] + '/manifest.json')} | awk '{{print $1}}')" = {q(spec['weights']['plugin_manifest_sha256'])}
test -d {q(spec['runtime']['physicalrsi_root'] + '/.git')}
test "$(git -C {q(spec['runtime']['physicalrsi_root'])} rev-parse HEAD)" = {q(spec['runtime']['physicalrsi_commit'])}
test -z "$(git -C {q(spec['runtime']['physicalrsi_root'])} status --porcelain)"
test -x {q(spec['runtime']['runtime_root'] + '/runtime/python311/bin/python3.11')}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
flock -n {q(spec['runtime']['runtime_root'] + '/tmp/physicalrsi-gpu' + str(worker['gpu_id']) + '-queue.lock')} -c true
flock -n {q(spec['runtime']['runtime_root'] + '/tmp/physicalrsi-gpu' + str(worker['gpu_id']) + '-slot1-queue.lock')} -c true
python3 - <<'PY'
import hashlib,json,socket
manifest={q(spec['weights']['plugin_checkpoint'] + '/manifest.json')}
routes={q(checkout + '/' + spec['routes'])}
m=hashlib.sha256(open(manifest,'rb').read()).hexdigest(); r=json.load(open(routes))
assert r['identities']['base']['checkpoint_sha256']==m
assert r['identities']['base']['adapter_sha256'] is None
s=socket.socket(); s.bind(('127.0.0.1',{worker['port']})); s.close()
PY
echo '{{"worker_id":{worker['worker_id']},"ok":true}}'
"""
    return json.loads(ssh(worker, script, commands).splitlines()[-1])


def prepare(spec: dict, spec_path: Path, commands: Commands) -> Path:
    round_dir = Path(spec["round_dir"])
    if round_dir.exists():
        raise FileExistsError("round_dir must be fresh; rounds are immutable")
    round_dir.mkdir(parents=True)
    save_create_only(round_dir / "round_spec.json", json.loads(spec_path.read_text()))
    receipts = []
    for worker in spec["workers"]:
        receipt = remote_preflight(spec, worker, commands)
        receipts.append(receipt)
        save_create_only(round_dir / f"preflight_worker{worker['worker_id']}.json", receipt)
    save_create_only(round_dir / "preflight.json", {
        "schema": "pi05_harness_multiround400.preflight.v1", "ok": True,
        "round_id": spec["round_id"], "canonical": spec["canonical"],
        "source_hashes": spec["_source_hashes"], "workers": receipts,
        "fixed_pairs": 400, "fixed_episodes": 800, "retry_policy": "none",
        "seed_selection": "forbidden", "training": "forbidden"})
    return round_dir


def worker_command(spec: dict, worker: dict) -> list[str]:
    c = worker["checkout"]
    return ["python3", c + "/scripts/run_pi05_harness_worker.py",
        "--code", c, "--job", c + "/" + worker["job"],
        "--base", spec["weights"]["base_checkpoint"],
        "--plugins", spec["weights"]["plugin_checkpoint"],
        "--output", worker["output"],
        "--physicalrsi-root", spec["runtime"]["physicalrsi_root"],
        "--runtime-root", spec["runtime"]["runtime_root"],
        "--policy-python", worker["policy_python"],
        "--expected-commit", spec["canonical"]["commit"],
        "--expected-physicalrsi-commit", spec["runtime"]["physicalrsi_commit"],
        "--gpu-id", str(worker["gpu_id"]), "--port", str(worker["port"])]


def launch(spec: dict, commands: Commands) -> list[dict]:
    round_dir = Path(spec["round_dir"])
    if not (round_dir / "preflight.json").is_file():
        raise ValueError("preflight receipt is required")
    rows = []
    for worker in spec["workers"]:
        log, pidfile = worker["output"] + ".controller.log", worker["output"] + ".controller.pid"
        command = shlex.join(worker_command(spec, worker))
        script = (f"set -e\ntest ! -e {shlex.quote(worker['output'])}\n"
                  f"test ! -e {shlex.quote(pidfile)}\n"
                  f"nohup {command} >{shlex.quote(log)} 2>&1 </dev/null &\n"
                  f"pid=$!\necho \"$pid\" >{shlex.quote(pidfile)}\necho \"$pid\"\n")
        pid = int(ssh(worker, script, commands).splitlines()[-1])
        row = {"worker_id": worker["worker_id"], "controller_pid": pid,
               "output": worker["output"], "log": log}
        rows.append(row)
        save_create_only(round_dir / f"launch_worker{worker['worker_id']}.json", row)
    save_create_only(round_dir / "launch.json", {"schema": "pi05_harness_multiround400.launch.v1",
        "round_id": spec["round_id"], "launched_unix": time.time(), "workers": rows})
    return rows


def status(spec: dict, commands: Commands) -> list[dict]:
    launch_doc = json.loads((Path(spec["round_dir"]) / "launch.json").read_text())
    pids = {row["worker_id"]: row["controller_pid"] for row in launch_doc["workers"]}
    rows = []
    for worker in spec["workers"]:
        script = f"""python3 - <<'PY'
import glob,json,os
root={worker['output']!r}; pid={pids[worker['worker_id']]}
p=root+'/controller.json'
doc=json.load(open(p)) if os.path.isfile(p) else {{}}
print(json.dumps({{'worker_id':{worker['worker_id']},'alive':os.path.isdir('/proc/'+str(pid)),
 'controller_status':doc.get('status','not_created'),
 'episodes':len(glob.glob(root+'/*/episodes/*.json'))}}))
PY
"""
        rows.append(json.loads(ssh(worker, script, commands).splitlines()[-1]))
    return rows


def wait_terminal(spec: dict, commands: Commands, poll_seconds: int) -> list[dict]:
    if poll_seconds < 10 or poll_seconds > 300:
        raise ValueError("poll_seconds must be 10..300")
    while True:
        rows = status(spec, commands)
        if all(row["controller_status"] == "complete" and not row["alive"]
               and row["episodes"] == 200 for row in rows):
            save_create_only(Path(spec["round_dir"]) / "terminal.json", {
                "schema": "pi05_harness_multiround400.terminal.v1", "complete": True,
                "episodes": 800, "workers": rows, "ended_unix": time.time()})
            return rows
        failed = [row for row in rows if row["controller_status"] == "failed"
                  or (not row["alive"] and row["controller_status"] != "complete")]
        if failed:
            save_create_only(Path(spec["round_dir"]) / "terminal.json", {
                "schema": "pi05_harness_multiround400.terminal.v1", "complete": False,
                "workers": rows, "failed": failed, "ended_unix": time.time()})
            raise RuntimeError("worker failed; evidence preserved and no retry attempted")
        time.sleep(poll_seconds)


def failure_decision(aggregate: dict, spec: dict) -> tuple[dict, dict]:
    pairs = aggregate.get("pairs", [])
    recovered = [{k: row.get(k) for k in ("suite", "task_id", "init_id")}
                 for row in pairs if row.get("outcome") == "recovered"]
    regressed = [{k: row.get(k) for k in ("suite", "task_id", "init_id")}
                 for row in pairs if row.get("outcome") == "regressed"]
    candidate_failures = [{k: row.get(k) for k in ("suite", "task_id", "init_id")}
                          for row in pairs if row.get("candidate_success") is False]
    causal_failures = [{"case": {k: row.get(k) for k in ("suite", "task_id", "init_id")},
                        "confounds": row.get("causal_confounds", [])}
                       for row in pairs if row.get("causal_gate_pass") is False]
    failures = {"schema": "pi05_harness_multiround400.failures.v1",
        "round_id": spec["round_id"], "candidate_failures": candidate_failures,
        "regressions": regressed, "recoveries": recovered, "causal_failures": causal_failures}
    decision = {"schema": "pi05_harness_multiround400.decision.v1",
        "round_id": spec["round_id"], "score_complete": aggregate.get("score_complete") is True,
        "exact_prefix_attribution_complete": aggregate.get("exact_prefix_attribution_complete") is True,
        "score": aggregate.get("score"), "changed_tasks": spec["changed_tasks"],
        "regression_count": len(regressed), "recovery_count": len(recovered),
        "promotion_performed": False, "eligible_for_promotion": False,
        "decision": "manual_review_required",
        "note": "No automatic promotion, training, retry, best-of-N selection, or next round."}
    return failures, decision


def finalize(spec: dict, commands: Commands) -> dict:
    round_dir = Path(spec["round_dir"])
    terminal = json.loads((round_dir / "terminal.json").read_text())
    if terminal.get("complete") is not True:
        raise ValueError("only a complete 800-episode terminal receipt can be finalized")
    archive_rows = []
    for worker in spec["workers"]:
        receipt = str(Path(worker["output"]).with_name(Path(worker["output"]).name + "_artifact_receipt.json"))
        archive = f"/tmp/{spec['round_id']}_worker{worker['worker_id']}.json_inventory.tgz"
        exporter = worker["checkout"] + "/" + spec["export_script"]
        script = f"""set -e
test ! -e {shlex.quote(receipt)}
python3 {shlex.quote(exporter)} --worker-output {shlex.quote(worker['output'])} --output {shlex.quote(receipt)}
test ! -e {shlex.quote(archive)}
cd {shlex.quote(str(Path(worker['output']).parent))}
{{ find {shlex.quote(Path(worker['output']).name)} -type f -name '*.json' -print0; printf '%s\\0' {shlex.quote(Path(receipt).name)}; }} | tar --null -czf {shlex.quote(archive)} --files-from=-
if tar -tzf {shlex.quote(archive)} | grep -Eq '\\.(mp4|avi|mov)$'; then exit 9; fi
python3 - <<'PY'
import hashlib,json,os
p={archive!r}; h=hashlib.sha256()
with open(p,'rb') as stream:
    for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
digest=h.hexdigest()
print(json.dumps({{'bytes':os.path.getsize(p),'sha256':digest}}))
PY
"""
        row = json.loads(ssh(worker, script, commands).splitlines()[-1])
        row.update(worker_id=worker["worker_id"], archive=archive, receipt=receipt)
        archive_rows.append(row)
        save_create_only(round_dir / f"export_worker{worker['worker_id']}.json", row)
    total = sum(row["bytes"] for row in archive_rows)
    plan = {"schema": "pi05_harness_multiround400.collection.v1", "archives": archive_rows,
            "total_compressed_bytes": total,
            "max_compressed_bytes": spec["collection"]["max_compressed_bytes"],
            "video_transfer": "forbidden"}
    save_create_only(round_dir / "collection_plan.json", plan)
    if total > spec["collection"]["max_compressed_bytes"]:
        raise RuntimeError("bounded collection limit exceeded; no archive transfer attempted")

    relay = Path(spec["collection"]["relay_dir"])
    if relay.exists():
        raise FileExistsError("relay_dir must be fresh")
    relay.mkdir(parents=True)
    ordered = sorted(archive_rows, key=lambda row: row["bytes"])
    for index, row in enumerate(ordered):
        worker = next(w for w in spec["workers"] if w["worker_id"] == row["worker_id"])
        source = f"{worker['user']}@{worker['host']}:{row['archive']}"
        commands.run(["scp", "-o", f"ControlPath={worker['control_path']}",
                      "-o", "ControlMaster=no", source, str(relay) ])
        local = relay / Path(row["archive"]).name
        if local.stat().st_size != row["bytes"] or digest(local) != row["sha256"]:
            raise RuntimeError("archive transfer receipt mismatch")
        if index == 0:
            save_create_only(round_dir / "collection_pilot.json", {
                "schema": "pi05_harness_multiround400.collection_pilot.v1",
                "worker_id": row["worker_id"], "bytes": row["bytes"],
                "sha256": row["sha256"], "verified": True})

    collector = next(w for w in spec["workers"]
                     if w["worker_id"] == spec["collection"]["collector_worker_id"])
    collection_root = spec["collection"]["collection_root"]
    ssh(collector, f"set -e\ntest ! -e {shlex.quote(collection_root)}\nmkdir -p {shlex.quote(collection_root)}\n", commands)
    for row in archive_rows:
        local = relay / Path(row["archive"]).name
        commands.run(["scp", "-o", f"ControlPath={collector['control_path']}",
            "-o", "ControlMaster=no", str(local),
            f"{collector['user']}@{collector['host']}:/tmp/{local.name}"])
        ssh(collector, f"set -e\ncd {shlex.quote(collection_root)}\ntar -xzf {shlex.quote('/tmp/' + local.name)}\n", commands)

    aggregate_remote = collection_root + "/aggregate.json"
    args = ["python3", collector["checkout"] + "/" + spec["aggregate_script"]]
    for worker in spec["workers"]:
        args += ["--worker-output", collection_root + "/" + Path(worker["output"]).name]
    for worker in spec["workers"]:
        args += ["--artifact-receipt", collection_root + "/" +
                 Path(worker["output"]).name + "_artifact_receipt.json"]
    args += ["--case-plan", collector["checkout"] + "/" + spec["case_plan"],
             "--task-catalog", collector["checkout"] + "/" + spec["task_catalog"],
             "--routes", collector["checkout"] + "/" + spec["routes"],
             "--output", aggregate_remote]
    aggregate_exit = int(ssh(collector,
        "set -e\ntest ! -e " + shlex.quote(aggregate_remote) + "\nset +e\n" +
        shlex.join(args) + "\nrc=$?\nset -e\ntest -f " + shlex.quote(aggregate_remote) +
        "\necho \"$rc\"\n", commands).splitlines()[-1])
    local_aggregate = round_dir / "aggregate.json"
    commands.run(["scp", "-o", f"ControlPath={collector['control_path']}",
        "-o", "ControlMaster=no", f"{collector['user']}@{collector['host']}:{aggregate_remote}",
        str(local_aggregate)])
    aggregate = json.loads(local_aggregate.read_text())
    failures, decision = failure_decision(aggregate, spec)
    decision["aggregate_exit_code"] = aggregate_exit
    save_create_only(round_dir / "failure_regressions.json", failures)
    save_create_only(round_dir / "decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("preflight", "launch", "status", "wait", "finalize", "run"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    spec = load_and_validate(args.spec, args.source_checkout)
    commands = Commands()
    if args.command in ("preflight", "run"):
        prepare(spec, args.spec, commands)
    if args.command in ("launch", "run"):
        launch(spec, commands)
    if args.command == "status":
        print(json.dumps(status(spec, commands), indent=2, sort_keys=True)); return 0
    if args.command in ("wait", "run"):
        wait_terminal(spec, commands, args.poll_seconds)
    if args.command in ("finalize", "run"):
        print(json.dumps(finalize(spec, commands), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
