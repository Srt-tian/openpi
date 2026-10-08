#!/usr/bin/env python3
"""Create the task-wide goal/3 strong-response-probe 19-pair research plan."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

import pi05_harness_backend as backend


SOURCE = Path(os.environ.get("PI05_TEST_PHYSICALRSI_ROOT",
    "/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007"))
PATCH = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PATCH / "configs/pi05_harness"
DEFAULT_OUTPUT = PATCH / "configs/pi05_strong_response_probe"
CONTROL = {"kind": "response_probe_v1", "enabled": True,
           "lift_z_command": .2, "max_lift_steps": 20,
           "lift_target_m": .025, "native_reserve_steps": 80}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def case(init_id: int, replicate_id: int) -> dict:
    return {"suite": "libero_goal", "task_id": 3,
            "init_id": init_id, "replicate_id": replicate_id}


def worker_cases(index: int) -> list[dict]:
    if index == 0:
        return [case(4, replicate) for replicate in range(5)]
    if index == 1:
        return [case(4, replicate) for replicate in range(5, 10)]
    if index == 2:
        return [case(init_id, 0) for init_id in (0, 1, 2, 3, 5)]
    if index == 3:
        return [case(init_id, 0) for init_id in (6, 7, 8, 9)]
    raise ValueError("worker index must be 0..3")


def build(output: Path = DEFAULT_OUTPUT, baseline: Path = DEFAULT_BASELINE,
          source: Path = SOURCE) -> Path:
    output, baseline, source = output.resolve(), baseline.resolve(), source.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    registry = json.loads((baseline / "registry.json").read_text())
    catalog = json.loads((baseline / "task_catalog.json").read_text())
    routes = json.loads((baseline / "routes_base.json").read_text())
    rows = {row["key"]: row for row in catalog["tasks"]}
    if len(rows) != 40 or set(registry.get("tasks", {})) != set(rows):
        raise ValueError("baseline must be a complete explicit LIBERO-40 registry")
    if (set(routes.get("tasks", {})) != set(rows)
            or set(routes["tasks"].values()) != {"base"}
            or routes.get("identities", {}).get("base", {}).get("adapter_sha256") is not None):
        raise ValueError("strong probe requires the complete native-base route identity")
    api = backend.import_roborsi(source)
    from roborsi.self_harness.core import digest
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_strong_response_probe.", dir=output.parent))
    try:
        config_digests = {}
        for key, relative in sorted(registry["tasks"].items()):
            config = json.loads((baseline / relative).read_text())
            if "pi05_control" in config:
                raise ValueError("baseline task config already contains pi05_control")
            if key == "libero_goal/3":
                config["pi05_control"] = copy.deepcopy(CONTROL)
                config["reasoning"] = (
                    "Task-wide stronger response probe for goal/3 after the bounded probe's "
                    "mechanical cue covered 9/9 hard failures but only lifted about 0.0038 m."
                )
                config["evidence"] = {
                    "scope": "posthoc_repeat_diagnostic_not_full400_score",
                    "full_repeat_verified": {"base": "103/140", "plugins": "106/140"},
                    "shared_hard_case": {"base": "1/10", "plugin": "0/10"},
                    "observations": {"old_probe_hard_failure_cue_coverage": "9/9",
                                     "old_probe_hard_success_cue_coverage": "0/1",
                                     "old_probe_commanded_lift_m": 0.05,
                                     "old_probe_actual_lift_approx_m": 0.0038},
                    "forbidden_oracles": ["object_state", "grasp_state", "init_id_routing"],
                }
                config["proposal"] = {"candidate": "strong_response_probe_v1",
                    "rationale": "Stop on 0.025 m actual EEF rise or 20 lift actions; this is a "
                                 "mechanical response test, not a grasp or drawer-success certificate."}
                config["status"] = "unvalidated_research_probe"
            dump(temporary / relative, config)
            config_digests[key] = digest(config)
        new_registry = copy.deepcopy(registry)
        new_registry["name"] = "pi05-goal3-strong-response-probe"
        new_registry["metadata"] = {"task_config_sha256": config_digests,
                                    "control_scope": "task_wide_libero_goal_3"}
        dump(temporary / "registry.json", new_registry)
        dump(temporary / "task_catalog.json", catalog)
        for index in range(4):
            cases = worker_cases(index)
            cases_name = f"cases_worker{index}.json"
            dump(temporary / cases_name, {"schema": "pi05_harness_cases.v1", "cases": cases})
            common = {"routes": "configs/pi05_harness/routes_base.json",
                      "cases": f"configs/pi05_strong_response_probe/{cases_name}", "mode": "harness"}
            dump(temporary / f"job_worker{index}.json", {
                "schema": "pi05_harness_worker.v1", "batches": [
                    {**common, "name": f"worker{index}_control",
                     "registry": "configs/pi05_harness/registry.json"},
                    {**common, "name": f"worker{index}_probe",
                     "registry": "configs/pi05_strong_response_probe/registry.json"},
                ]})
        tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                 for key, row in rows.items()}
        proposal = api.TaskHarnessRegistry(temporary / "registry.json", {"pi05"}).materialize(tasks)
        if proposal["task_config_sha256"] != config_digests or len(proposal["harnesses"]) != 40:
            raise AssertionError("response-probe registry failed full materialization")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(build(args.output_root, args.baseline_root, args.source_root))
