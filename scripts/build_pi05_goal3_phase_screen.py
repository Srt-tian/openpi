#!/usr/bin/env python3
"""Create three fixed, task-wide goal/3 phase-screen registries and jobs."""
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
DEFAULT_OUTPUT = PATCH / "configs/pi05_goal3_phase_screen"
ORIGINAL = "open the top drawer and put the bowl inside"
PREFIX = "open the top drawer"
VARIANTS = {
    "prefix100": [(PREFIX, 100, "next"), (ORIGINAL, 200, "abort")],
    "prefix150": [(PREFIX, 150, "next"), (ORIGINAL, 150, "abort")],
    "sequential": [("open the top drawer first, then put the bowl inside it", 300, "abort")],
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prefix_variants(budgets: list[int]) -> dict:
    if not budgets or any(type(value) is not int or not 1 <= value < 300 for value in budgets):
        raise ValueError("prefix budgets must be unique integers in 1..299")
    if len(set(budgets)) != len(budgets):
        raise ValueError("prefix budgets must be unique integers in 1..299")
    return {f"prefix{value}": [(PREFIX, value, "next"),
                                (ORIGINAL, 300 - value, "abort")] for value in budgets}


def harness(name: str, variants: dict) -> dict:
    return {"schema": 1, "name": f"pi05-goal3-{name}", "remember": [], "stages": [
        {"skill": "pi05", "instruction": instruction, "max_steps": steps,
         "until": None, "on_timeout": timeout}
        for instruction, steps, timeout in variants[name]
    ]}


def build(output: Path = DEFAULT_OUTPUT, baseline: Path = DEFAULT_BASELINE,
          source: Path = SOURCE, *, variants: dict | None = None,
          namespace: str = "pi05_goal3_phase_screen", include_probe: bool = True) -> Path:
    variants = VARIANTS if variants is None else variants
    if not variants or not namespace.replace("_", "").isalnum():
        raise ValueError("variants and a simple namespace are required")
    output, baseline, source = output.resolve(), baseline.resolve(), source.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    base_registry = json.loads((baseline / "registry.json").read_text())
    catalog = json.loads((baseline / "task_catalog.json").read_text())
    rows = {row["key"]: row for row in catalog["tasks"]}
    if len(rows) != 40 or set(base_registry.get("tasks", {})) != set(rows):
        raise ValueError("baseline must be a complete explicit LIBERO-40 registry")
    if rows["libero_goal/3"]["instruction"] != ORIGINAL or rows["libero_goal/3"]["max_steps"] != 300:
        raise ValueError("unexpected goal/3 catalog contract")
    api = backend.import_roborsi(source)
    from roborsi.self_harness.core import digest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_goal3_phase_screen.", dir=output.parent))
    try:
        tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                 for key, row in rows.items()}
        for variant in variants:
            digests = {}
            for key, relative in sorted(base_registry["tasks"].items()):
                config = json.loads((baseline / relative).read_text())
                if "pi05_control" in config:
                    raise ValueError("baseline task config contains pi05_control")
                if key == "libero_goal/3":
                    config["harness"] = harness(variant, variants)
                    config["reasoning"] = (
                        "Unvalidated task-wide research candidate from audited probe01 phase-error "
                        "evidence; a fixed-duration prefix is not a certificate that its subgoal succeeded."
                    )
                    evidence = {
                        "source": "audited_probe01_phase_error",
                        "scope": "research_screen_not_full400_score",
                        "phase_semantics": "fixed_duration_not_subgoal_success_certificate",
                        "routing": "task_wide_no_init_image_state_or_object_oracle",
                    }
                    if namespace == "pi05_goal3_early_prefix":
                        evidence.update(
                            prior_phase_screen={"hard_cases": "8/10",
                                                "unconditional_regressions": 3},
                            later_instruction_lease="lease40_and_lease80_zero_net_gain",
                            current_hypothesis="earlier_fixed_handoff")
                    config["evidence"] = evidence
                    config["proposal"] = {"candidate": variant,
                        "rationale": "Screen a fixed goal/3 prompt/phase hypothesis without oracle transitions."}
                    config["status"] = "unvalidated_research_candidate"
                dump(temporary / variant / relative, config)
                digests[key] = digest(config)
            registry = copy.deepcopy(base_registry)
            registry["name"] = f"pi05-goal3-phase-screen-{variant}"
            registry["metadata"] = {"task_config_sha256": digests,
                                    "candidate_status": "unvalidated_research_candidate"}
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
                *([("probe", "configs/pi05_response_probe/registry.json")] if include_probe else []),
                *[(name, f"configs/{namespace}/{name}/registry.json")
                  for name in variants],
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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--prefix-budgets", nargs="+", type=int)
    parser.add_argument("--namespace", default="pi05_goal3_phase_screen")
    parser.add_argument("--without-probe", action="store_true")
    args = parser.parse_args()
    selected = prefix_variants(args.prefix_budgets) if args.prefix_budgets else None
    destination = args.output_root or PATCH / "configs" / args.namespace
    print(build(destination, args.baseline_root, args.source_root, variants=selected,
                namespace=args.namespace, include_probe=not args.without_probe))
