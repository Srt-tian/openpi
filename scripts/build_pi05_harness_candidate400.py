#!/usr/bin/env python3
"""Build the frozen native-vs-two-control PI0.5 LIBERO-400 plan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import pi05_harness_backend as backend


PATCH = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("PI05_TEST_PHYSICALRSI_ROOT",
    "/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007"))
BASELINE = PATCH / "configs/pi05_harness"
GOAL3 = PATCH / "configs/pi05_goal3_early_probe_validation"
LONG8 = PATCH / "configs/pi05_closed_lift_intent_veto"
OUTPUT = PATCH / "configs/pi05_harness_candidate400"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
GOAL_CONTROL = {"kind":"response_probe_v1","enabled":True,"lift_z_command":.2,
    "max_lift_steps":20,"lift_target_m":.025,"native_reserve_steps":80,
    "minimum_actual":100}
LONG_CONTROL = {"kind":"closed_dwell_lift_v1","enabled":True,
                "veto_native_upward_intent":True}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def worker_cases(worker: int) -> list[dict]:
    if type(worker) is not int or not 0 <= worker < 4:
        raise ValueError("worker must be 0..3")
    return [{"suite": suite, "task_id": task_id, "init_id": init_id,
             "replicate_id": 0}
            for ordinal, suite in enumerate(SUITES) for task_id in range(10)
            if (ordinal * 10 + task_id) % 4 == worker for init_id in range(10)]


def build(output: Path = OUTPUT, baseline: Path = BASELINE, goal3: Path = GOAL3,
          long8: Path = LONG8, source: Path = SOURCE) -> Path:
    output, baseline, goal3, long8, source = map(
        lambda p: Path(p).resolve(), (output, baseline, goal3, long8, source))
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    base_registry = json.loads((baseline / "registry.json").read_text())
    catalog = json.loads((baseline / "task_catalog.json").read_text())
    rows = {row["key"]: row for row in catalog.get("tasks", [])}
    if len(rows) != 40 or set(base_registry.get("tasks", {})) != set(rows):
        raise ValueError("baseline is not the explicit LIBERO-40 registry")
    sources = {"libero_goal/3": goal3, "libero_10/8": long8}
    expected_controls = {"libero_goal/3": GOAL_CONTROL, "libero_10/8": LONG_CONTROL}
    chosen = {}
    for key, root in sources.items():
        registry = json.loads((root / "registry.json").read_text())
        config = json.loads((root / registry["tasks"][key]).read_text())
        if config.get("pi05_control") != expected_controls[key]:
            raise ValueError(f"reviewed candidate changed: {key}")
        chosen[key] = root / registry["tasks"][key]
    api = backend.import_roborsi(source)
    from roborsi.self_harness.core import digest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_candidate400.", dir=output.parent))
    try:
        digests = {}
        for key, relative in sorted(base_registry["tasks"].items()):
            source_file = chosen.get(key, baseline / relative)
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
            config = json.loads(target.read_text())
            if key not in sources and "pi05_control" in config:
                raise ValueError(f"baseline control unexpectedly enabled: {key}")
            digests[key] = digest(config)
        registry = dict(base_registry)
        registry["name"] = "pi05-harness-candidate400"
        registry["metadata"] = {"task_config_sha256": digests,
            "status": "provisional_pending_closed_lift_intent_veto_110",
            "changed_tasks": ["libero_goal/3", "libero_10/8"],
            "routing_inputs": ["suite", "task_index"],
            "oracle_inputs": [],
            "score_scope": "paired_native_vs_candidate_full_libero400",
            "attribution_warning": "score coverage is distinct from exact-prefix attribution"}
        dump(temporary / "registry.json", registry)
        dump(temporary / "task_catalog.json", catalog)
        case_plan = []
        for joint, suite in enumerate(SUITES):
            for task_id in range(10):
                row = rows[f"{suite}/{task_id}"]
                for init_id in range(10):
                    case_plan.append({"suite":suite,"task_id":task_id,"init_id":init_id,
                        "replicate_id":0,"joint_task_number":joint*10+task_id,
                        "official_cap":row["max_steps"]})
        dump(temporary / "case_plan.json",
             {"schema":"pi05_harness_candidate400_cases.v1","cases":case_plan})
        observed = set()
        for worker in range(4):
            cases = worker_cases(worker); observed.update(
                (x["suite"],x["task_id"],x["init_id"]) for x in cases)
            dump(temporary / f"cases_worker{worker}.json",
                 {"schema":"pi05_harness_cases.v1","cases":cases})
            common = {"routes":"configs/pi05_harness/routes_base.json",
                "cases":f"configs/pi05_harness_candidate400/cases_worker{worker}.json",
                "mode":"harness"}
            dump(temporary / f"job_worker{worker}.json",
                 {"schema":"pi05_harness_worker.v1","batches":[
                    {**common,"name":f"worker{worker}_control",
                     "registry":"configs/pi05_harness/registry.json"},
                    {**common,"name":f"worker{worker}_candidate",
                     "registry":"configs/pi05_harness_candidate400/registry.json"}]})
        if len(observed) != 400 or any(len(worker_cases(i)) != 100 for i in range(4)):
            raise AssertionError("worker coverage is not exact LIBERO-400")
        inputs = {key:{"instruction":row["instruction"],"max_steps":row["max_steps"]}
                  for key,row in rows.items()}
        proposal = api.TaskHarnessRegistry(temporary/"registry.json",{"pi05"}).materialize(inputs)
        if proposal["task_config_sha256"] != digests or len(proposal["harnesses"]) != 40:
            raise AssertionError("candidate registry failed materialization")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return output


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-root",type=Path,default=OUTPUT)
    p.add_argument("--baseline-root",type=Path,default=BASELINE);p.add_argument("--goal3-root",type=Path,default=GOAL3)
    p.add_argument("--long8-root",type=Path,default=LONG8);p.add_argument("--source-root",type=Path,default=SOURCE)
    a=p.parse_args();print(build(a.output_root,a.baseline_root,a.goal3_root,a.long8_root,a.source_root))
