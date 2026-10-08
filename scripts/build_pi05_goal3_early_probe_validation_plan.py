#!/usr/bin/env python3
"""Freeze the 100-pair goal/3 validation plan without changing the candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pi05_harness_backend as backend


SOURCE = Path(os.environ.get(
    "PI05_TEST_PHYSICALRSI_ROOT",
    "/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007",
))
PATCH = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATED = PATCH / "configs/pi05_early_strong_response_probe"
DEFAULT_OUTPUT = PATCH / "configs/pi05_goal3_early_probe_validation"
CONTROL = {
    "kind": "response_probe_v1", "enabled": True,
    "lift_z_command": .2, "max_lift_steps": 20,
    "lift_target_m": .025, "native_reserve_steps": 80,
    "minimum_actual": 100,
}
KNOWN_19 = {(init_id, replicate_id) for init_id in range(10)
            for replicate_id in (range(10) if init_id == 4 else (0,))}
ALL_100 = tuple((init_id, replicate_id) for init_id in range(10)
                for replicate_id in range(10))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case(init_id: int, replicate_id: int) -> dict:
    return {"suite": "libero_goal", "task_id": 3,
            "init_id": init_id, "replicate_id": replicate_id}


def cohort(init_id: int, replicate_id: int) -> str:
    return ("known_19_reproduction" if (init_id, replicate_id) in KNOWN_19
            else "noise_expansion_81")


def worker_cases(index: int) -> list[dict]:
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError("worker index must be 0..3")
    return [case(*key) for key in ALL_100[index::4]]


def build(output: Path = DEFAULT_OUTPUT, validated: Path = DEFAULT_VALIDATED,
          source: Path = SOURCE) -> Path:
    output, validated, source = output.resolve(), validated.resolve(), source.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    registry_path = validated / "registry.json"
    registry = json.loads(registry_path.read_text())
    catalog_path = validated / "task_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    rows = {row["key"]: row for row in catalog.get("tasks", [])}
    tasks = registry.get("tasks", {})
    if len(rows) != 40 or set(tasks) != set(rows):
        raise ValueError("validated source must be a complete explicit LIBERO-40 registry")
    goal3 = json.loads((validated / tasks["libero_goal/3"]).read_text())
    if goal3.get("pi05_control") != CONTROL:
        raise ValueError("validated goal/3 candidate parameters changed")
    if any("pi05_control" in json.loads((validated / relative).read_text())
           for key, relative in tasks.items() if key != "libero_goal/3"):
        raise ValueError("validated source enables control outside goal/3")

    temporary = Path(tempfile.mkdtemp(
        prefix=".pi05_goal3_early_probe_validation.", dir=output.parent))
    try:
        # Registry, catalog, and every task config are immutable experiment inputs.
        for relative in ("registry.json", "task_catalog.json"):
            shutil.copyfile(validated / relative, temporary / relative)
        source_hashes = {}
        for key, relative in sorted(tasks.items()):
            source_file, target_file = validated / relative, temporary / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target_file)
            if sha256(source_file) != sha256(target_file):
                raise AssertionError(f"byte copy changed task config {key}")
            source_hashes[key] = sha256(source_file)

        plan_cases = [{**case(*key), "cohort": cohort(*key)} for key in ALL_100]
        dump(temporary / "case_plan.json", {
            "schema": "pi05_response_probe_case_plan.v1",
            "task": {"suite": "libero_goal", "task_id": 3},
            "cases": plan_cases,
        })
        observed = set()
        for index in range(4):
            cases = worker_cases(index)
            observed.update((row["init_id"], row["replicate_id"]) for row in cases)
            cases_name = f"cases_worker{index}.json"
            dump(temporary / cases_name, {
                "schema": "pi05_harness_cases.v1", "cases": cases})
            common = {
                "routes": "configs/pi05_harness/routes_base.json",
                "cases": f"configs/pi05_goal3_early_probe_validation/{cases_name}",
                "mode": "harness",
            }
            dump(temporary / f"job_worker{index}.json", {
                "schema": "pi05_harness_worker.v1",
                "batches": [
                    {**common, "name": f"worker{index}_control",
                     "registry": "configs/pi05_harness/registry.json"},
                    {**common, "name": f"worker{index}_probe",
                     "registry": "configs/pi05_goal3_early_probe_validation/registry.json"},
                ],
            })
        if observed != set(ALL_100):
            raise AssertionError("worker partition is not the exact 100-pair grid")

        api = backend.import_roborsi(source)
        task_inputs = {key: {"instruction": row["instruction"],
                             "max_steps": row["max_steps"]}
                       for key, row in rows.items()}
        proposal = api.TaskHarnessRegistry(
            temporary / "registry.json", {"pi05"}).materialize(task_inputs)
        if (proposal["task_config_sha256"]
                != registry.get("metadata", {}).get("task_config_sha256")
                or len(proposal["harnesses"]) != 40):
            raise AssertionError("copied registry failed full materialization")
        if any(sha256(validated / relative) != source_hashes[key]
               for key, relative in tasks.items()):
            raise AssertionError("validated input changed during build")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validated-root", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(build(args.output_root, args.validated_root, args.source_root))
