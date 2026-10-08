#!/usr/bin/env python3
"""Create-only frozen 14-case, two-arm PI0.5 repeat-probe plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile


SCHEMA_CASES = "pi05_harness_cases.v1"
SCHEMA_JOB = "pi05_harness_worker.v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "configs/pi05_repeat_probe"
REGISTRY = "configs/pi05_harness/registry.json"
ROUTES_BASE = "configs/pi05_harness/routes_base.json"
ROUTES_PLUGINS = "configs/pi05_harness/routes_plugins.json"
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ASSIGNMENTS = {
    0: (("libero_10", 8, 0), ("libero_10", 8, 2)),
    1: (("libero_10", 8, 3), ("libero_10", 8, 5)),
    2: (("libero_10", 8, 8), ("libero_10", 9, 3)),
    3: (("libero_spatial", 3, 6), ("libero_spatial", 7, 2),
        ("libero_spatial", 5, 2), ("libero_goal", 3, 4)),
    4: (("libero_object", 4, 1), ("libero_object", 2, 7),
        ("libero_object", 5, 9), ("libero_goal", 9, 5)),
}
REPLICATES = range(10)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def joint_task_number(suite: str, task_id: int) -> int:
    return SUITE_ORDER.index(suite) * 10 + task_id


def seed(suite: str, task_id: int, init_id: int, replicate_id: int) -> int:
    return 7 + joint_task_number(suite, task_id) * 50 + init_id + replicate_id * 1000000007


def cases(rows: tuple[tuple[str, int, int], ...]) -> list[dict]:
    return [{"suite": suite, "task_id": task, "init_id": init,
             "replicate_id": rep}
            for suite, task, init in rows for rep in REPLICATES]


def batch(name: str, routes: str, case_path: str) -> dict:
    return {"name": name, "mode": "legacy", "registry": REGISTRY,
            "routes": routes, "cases": case_path}


def validate_plan(plan: dict[int, list[dict]]) -> dict:
    union = {row for rows in ASSIGNMENTS.values() for row in rows}
    if len(union) != 14 or sum(map(len, ASSIGNMENTS.values())) != 14:
        raise AssertionError("assignments must contain exactly 14 unique cases")
    identities = set()
    counts = {"base": 0, "plugin": 0}
    for worker, batches in plan.items():
        assigned = set(ASSIGNMENTS[worker])
        for item in batches:
            arm = "base" if item["routes"] == ROUTES_BASE else "plugin"
            payload = item.pop("_cases")
            for case in payload:
                key = (case["suite"], case["task_id"], case["init_id"])
                if key not in assigned or case["replicate_id"] not in REPLICATES:
                    raise AssertionError("batch escaped its frozen worker assignment")
                identity = (arm, *key, case["replicate_id"])
                if identity in identities:
                    raise AssertionError("duplicate arm/case/replicate")
                identities.add(identity)
                counts[arm] += 1
    expected = {(arm, *row, rep) for arm in ("base", "plugin")
                for row in union for rep in REPLICATES}
    if identities != expected:
        raise AssertionError("plan is not the exact 2 x 14 x 10 grid")
    return {"union_cases": 14, "replicates_per_case": 10,
            "arms": ["base", "plugin"], "base_episodes": counts["base"],
            "plugin_episodes": counts["plugin"], "episodes": len(identities),
            "unique_arm_case_replicates": len(identities),
            "exact_2x14x10_coverage": True,
            "policy_seed_formula": "7 + joint40_task_number * 50 + init_id + replicate_id * 1000000007",
            "ambient_seed_formula": "7 + joint40_task_number * 50 + init_id; fixed across replicates",
            "execution_backend": "legacy execute_policy_steps; independent of harness migration parity gate",
            "inference_call_seed_stride": 1000003,
            "environment_seed": 7, "routing_scope": "task only; never init or replicate"}


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_repeat_probe.", dir=output.parent))
    try:
        plan: dict[int, list[dict]] = {}
        for worker, assigned in ASSIGNMENTS.items():
            all_rows = cases(assigned)
            all_path = f"configs/pi05_repeat_probe/cases_worker{worker}_all.json"
            dump(temporary / f"cases_worker{worker}_all.json",
                 {"schema": SCHEMA_CASES, "cases": all_rows})
            batches = [batch(f"worker{worker}_base", ROUTES_BASE, all_path)]
            batches[0]["_cases"] = all_rows
            for suite in SUITE_ORDER:
                suite_rows = tuple(row for row in assigned if row[0] == suite)
                if not suite_rows:
                    continue
                relative = f"configs/pi05_repeat_probe/cases_worker{worker}_{suite}.json"
                payload = cases(suite_rows)
                dump(temporary / f"cases_worker{worker}_{suite}.json",
                     {"schema": SCHEMA_CASES, "cases": payload})
                item = batch(f"worker{worker}_plugin_{suite}", ROUTES_PLUGINS, relative)
                item["_cases"] = payload
                batches.append(item)
            plan[worker] = batches
        metadata = validate_plan(plan)
        for worker, batches in plan.items():
            clean = [{key: value for key, value in item.items() if key != "_cases"}
                     for item in batches]
            dump(temporary / f"job_worker{worker}.json",
                 {"schema": SCHEMA_JOB, "batches": clean})
        dump(temporary / "metadata.json", {"schema": "pi05_repeat_probe.metadata.v1",
                                            **metadata})
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    print(build(parser.parse_args().output_root))
