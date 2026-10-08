#!/usr/bin/env python3
"""Create the task-wide goal/3 response-probe registry and 19-pair research plan."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

import pi05_harness_backend as backend


SOURCE = Path("/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007")
PATCH = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PATCH / "configs/pi05_harness"
DEFAULT_OUTPUT = PATCH / "configs/pi05_response_probe"
CONTROL = {"kind": "response_probe_v1", "enabled": True}


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
        return [case(init_id, 0) for init_id in (0, 1, 2)]
    if index == 3:
        return [case(init_id, 0) for init_id in (3, 5, 6)]
    if index == 4:
        return [case(init_id, 0) for init_id in (7, 8, 9)]
    raise ValueError("worker index must be 0..4")


def build(output: Path = DEFAULT_OUTPUT, baseline: Path = DEFAULT_BASELINE,
          source: Path = SOURCE) -> Path:
    output, baseline, source = output.resolve(), baseline.resolve(), source.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    registry = json.loads((baseline / "registry.json").read_text())
    catalog = json.loads((baseline / "task_catalog.json").read_text())
    rows = {row["key"]: row for row in catalog["tasks"]}
    if len(rows) != 40 or set(registry.get("tasks", {})) != set(rows):
        raise ValueError("baseline must be a complete explicit LIBERO-40 registry")
    api = backend.import_roborsi(source)
    from roborsi.self_harness.core import digest
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_response_probe.", dir=output.parent))
    try:
        config_digests = {}
        for key, relative in sorted(registry["tasks"].items()):
            config = json.loads((baseline / relative).read_text())
            if "pi05_control" in config:
                raise ValueError("baseline task config already contains pi05_control")
            if key == "libero_goal/3":
                config["pi05_control"] = copy.deepcopy(CONTROL)
                config["reasoning"] = (
                    "Task-wide response probe for goal/3, motivated by a shared hard case with "
                    "base 1/10 and plugin 0/10; robot z span was about 0.001 despite z command "
                    "about -0.85, open persisted over 30 steps, and aperture was about 0.08."
                )
                config["evidence"] = {
                    "scope": "posthoc_repeat_diagnostic_not_full400_score",
                    "full_repeat_verified": {"base": "103/140", "plugins": "106/140"},
                    "shared_hard_case": {"base": "1/10", "plugin": "0/10"},
                    "observations": {"robot_z_span_approx": 0.001, "z_command_approx": -0.85,
                                     "open_steps_gt": 30, "aperture_approx": 0.08},
                    "forbidden_oracles": ["object_state", "grasp_state", "init_id_routing"],
                }
                config["proposal"] = {"candidate": "response_probe_v1",
                    "rationale": "Fixed goal/3 task-level control; research screening only."}
                config["status"] = "unvalidated_research_probe"
            dump(temporary / relative, config)
            config_digests[key] = digest(config)
        new_registry = copy.deepcopy(registry)
        new_registry["name"] = "pi05-goal3-response-probe"
        new_registry["metadata"] = {"task_config_sha256": config_digests,
                                    "control_scope": "task_wide_libero_goal_3"}
        dump(temporary / "registry.json", new_registry)
        dump(temporary / "task_catalog.json", catalog)
        for index in range(5):
            cases = worker_cases(index)
            cases_name = f"cases_worker{index}.json"
            dump(temporary / cases_name, {"schema": "pi05_harness_cases.v1", "cases": cases})
            common = {"routes": "configs/pi05_harness/routes_base.json",
                      "cases": f"configs/pi05_response_probe/{cases_name}", "mode": "harness"}
            dump(temporary / f"job_worker{index}.json", {
                "schema": "pi05_harness_worker.v1", "batches": [
                    {**common, "name": f"worker{index}_control",
                     "registry": "configs/pi05_harness/registry.json"},
                    {**common, "name": f"worker{index}_probe",
                     "registry": "configs/pi05_response_probe/registry.json"},
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
