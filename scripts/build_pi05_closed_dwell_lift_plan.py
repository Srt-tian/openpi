#!/usr/bin/env python3
"""Create the fixed long8 closed-dwell-lift research registry and paired plan."""
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
DEFAULT_NAMESPACE = "pi05_closed_dwell_lift_base"
DEFAULT_OUTPUT = PATCH / "configs" / DEFAULT_NAMESPACE
CONTROL = {"kind": "closed_dwell_lift_v1", "enabled": True}
PRIMARY = (0, 2, 3, 5, 8)
OTHER = (1, 4, 6, 7, 9)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def worker_cases(worker: int) -> list[dict]:
    if type(worker) is not int or not 0 <= worker < 5:
        raise ValueError("worker must be 0..4")
    def case(init_id: int, replicate_id: int) -> dict:
        return {"suite": "libero_10", "task_id": 8,
                "init_id": init_id, "replicate_id": replicate_id}
    return [case(PRIMARY[worker], replicate) for replicate in range(10)] + [
        case(OTHER[worker], 0)]


def build(output: Path = DEFAULT_OUTPUT, baseline: Path = DEFAULT_BASELINE,
          source: Path = SOURCE, *, namespace: str = DEFAULT_NAMESPACE,
          policy_id: str = "base") -> Path:
    if not namespace.replace("_", "").isalnum() or policy_id not in {"base", "long"}:
        raise ValueError("simple namespace and policy_id base or long required")
    output, baseline, source = output.resolve(), baseline.resolve(), source.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    base_registry = json.loads((baseline / "registry.json").read_text())
    catalog = json.loads((baseline / "task_catalog.json").read_text())
    base_routes = json.loads((baseline / "routes_base.json").read_text())
    plugin_routes = json.loads((baseline / "routes_plugins.json").read_text())
    rows = {row["key"]: row for row in catalog["tasks"]}
    if len(rows) != 40 or set(base_registry.get("tasks", {})) != set(rows):
        raise ValueError("baseline must be a complete explicit LIBERO-40 registry")
    if base_routes.get("identities") != plugin_routes.get("identities"):
        raise ValueError("base and plugin routes disagree on frozen identities")
    api = backend.import_roborsi(source)
    from roborsi.self_harness.core import digest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_closed_dwell_lift.", dir=output.parent))
    try:
        digests = {}
        for key, relative in sorted(base_registry["tasks"].items()):
            config = json.loads((baseline / relative).read_text())
            if "pi05_control" in config:
                raise ValueError("baseline task config contains pi05_control")
            if key == "libero_10/8":
                config["pi05_control"] = copy.deepcopy(CONTROL)
            dump(temporary / relative, config)
            digests[key] = digest(config)
        registry = copy.deepcopy(base_registry)
        registry["name"] = namespace.replace("_", "-")
        registry["metadata"] = {
            "task_config_sha256": digests,
            "candidate_status": "posthoc_unvalidated_research_candidate",
            "control_scope": "task_wide_libero_10_8_no_init_routing",
            "evidence": {
                "hypothesis_source": "plugin50_posthoc_cue_only_requires_native_base_validation",
                "current_native_reference": {"base": "26/50", "hard_init_2_base": "1/10"},
                "posthoc_60_dwell_cue": {"failed_coverage": "9/15",
                                          "successful_false_trigger": "1/35"},
                "constraint": "pose_still_moves_so_modify_z_only",
                "not_a_certificate": "no_object_or_grasp_success_oracle",
            },
        }
        dump(temporary / "registry.json", registry)
        dump(temporary / "task_catalog.json", catalog)
        if policy_id == "base":
            route_path = "configs/pi05_harness/routes_base.json"
        else:
            route_tasks = {key: ("long" if key == "libero_10/8" else "base") for key in sorted(rows)}
            route = {"schema": "pi05_harness_routes.v1", "name": "pi05-long8-plugin-only",
                     "tasks": route_tasks, "identities": copy.deepcopy(base_routes["identities"]),
                     "selection_basis": "fixed_long8_research_screen",
                     "validation_status": "unvalidated", "routing_inputs": ["suite", "task_index"]}
            if plugin_routes["tasks"].get("libero_10/8") != "long":
                raise ValueError("plugin route does not identify the frozen long policy")
            dump(temporary / "routes_long8_plugin.json", route)
            route_path = f"configs/{namespace}/routes_long8_plugin.json"
        tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                 for key, row in rows.items()}
        proposal = api.TaskHarnessRegistry(temporary / "registry.json", {"pi05"}).materialize(tasks)
        if proposal["task_config_sha256"] != digests or len(proposal["harnesses"]) != 40:
            raise AssertionError("closed-dwell-lift registry failed complete materialization")
        for worker in range(5):
            cases_name = f"cases_worker{worker}.json"
            dump(temporary / cases_name,
                 {"schema": "pi05_harness_cases.v1", "cases": worker_cases(worker)})
            common = {"routes": route_path,
                      "cases": f"configs/{namespace}/{cases_name}", "mode": "harness"}
            dump(temporary / f"job_worker{worker}.json", {
                "schema": "pi05_harness_worker.v1", "batches": [
                    {**common, "name": f"worker{worker}_control",
                     "registry": "configs/pi05_harness/registry.json"},
                    {**common, "name": f"worker{worker}_assist",
                     "registry": f"configs/{namespace}/registry.json"},
                ]})
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--policy-id", choices=("base", "long"), default="base")
    args = parser.parse_args()
    destination = args.output_root or PATCH / "configs" / args.namespace
    print(build(destination, args.baseline_root, args.source_root,
                namespace=args.namespace, policy_id=args.policy_id))
