#!/usr/bin/env python3
"""Create task-wide goal/3 instruction-lease research registries and jobs."""
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
DEFAULT_OUTPUT = PATCH / "configs/pi05_instruction_lease"
LEASES = {"lease40": 40, "lease80": 80}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def control(steps: int) -> dict:
    return {"kind": "instruction_lease_v1", "enabled": True,
            "instruction": "open the top drawer", "lease_steps": steps}


def build(output: Path = DEFAULT_OUTPUT, baseline: Path = DEFAULT_BASELINE,
          source: Path = SOURCE) -> Path:
    output, baseline, source = output.resolve(), baseline.resolve(), source.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    base_registry = json.loads((baseline / "registry.json").read_text())
    catalog = json.loads((baseline / "task_catalog.json").read_text())
    rows = {row["key"]: row for row in catalog["tasks"]}
    if len(rows) != 40 or set(base_registry.get("tasks", {})) != set(rows):
        raise ValueError("baseline must be a complete explicit LIBERO-40 registry")
    api = backend.import_roborsi(source)
    from roborsi.self_harness.core import digest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_instruction_lease.", dir=output.parent))
    try:
        tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                 for key, row in rows.items()}
        for variant, steps in LEASES.items():
            digests = {}
            for key, relative in sorted(base_registry["tasks"].items()):
                config = json.loads((baseline / relative).read_text())
                if "pi05_control" in config:
                    raise ValueError("baseline task config contains pi05_control")
                if key == "libero_goal/3":
                    config["pi05_control"] = control(steps)
                dump(temporary / variant / relative, config)
                digests[key] = digest(config)
            registry = copy.deepcopy(base_registry)
            registry["name"] = f"pi05-goal3-instruction-{variant}"
            registry["metadata"] = {
                "task_config_sha256": digests,
                "candidate_status": "posthoc_unvalidated_research_candidate",
                "evidence": {"source": "goal3_phase_screen", "hard_case": "8/10",
                             "unconditional_regressions": 3,
                             "claim_scope": "research_screen_not_full400_score"},
                "control_scope": "task_wide_libero_goal_3_no_init_routing",
            }
            dump(temporary / variant / "registry.json", registry)
            dump(temporary / variant / "task_catalog.json", catalog)
            proposal = api.TaskHarnessRegistry(
                temporary / variant / "registry.json", {"pi05"}).materialize(tasks)
            if proposal["task_config_sha256"] != digests or len(proposal["harnesses"]) != 40:
                raise AssertionError(f"{variant} failed complete registry materialization")
        for worker in range(5):
            common = {"routes": "configs/pi05_harness/routes_base.json",
                      "cases": f"configs/pi05_response_probe/cases_worker{worker}.json",
                      "mode": "harness"}
            registries = [
                ("control", "configs/pi05_harness/registry.json"),
                *[(name, f"configs/pi05_instruction_lease/{name}/registry.json")
                  for name in LEASES],
            ]
            dump(temporary / f"job_worker{worker}.json", {
                "schema": "pi05_harness_worker.v1", "batches": [
                    {**common, "name": f"worker{worker}_{name}", "registry": registry}
                    for name, registry in registries
                ]})
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
