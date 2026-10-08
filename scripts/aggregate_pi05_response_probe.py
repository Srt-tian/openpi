#!/usr/bin/env python3
"""Strict paired aggregation and causal-prefix audit for PI0.5 response probe."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

TASK = ("libero_goal", 3)
CAP = 300
EXPECTED = {(init, rep) for init in range(10)
            for rep in (range(10) if init == 4 else (0,))}
FULL_VALIDATION = {(init, rep) for init in range(10) for rep in range(10)}
KNOWN_COHORT = "known_19_reproduction"
EXPANSION_COHORT = "noise_expansion_81"


def exact(left, right) -> bool:
    return type(left) is type(right) and left == right


def arm(name: str) -> str:
    hits = [value for value in ("control", "probe") if value in name]
    if len(hits) != 1:
        raise ValueError(f"batch name must identify one arm: {name}")
    return hits[0]


def episode_key(case: dict, expected: set[tuple[int, int]] = EXPECTED) -> tuple[int, int]:
    if (case.get("suite"), case.get("task_id")) != TASK:
        raise ValueError("episode is not libero_goal/3")
    key = (case.get("init_id"), case.get("replicate_id"))
    if (any(type(value) is not int for value in key) or key not in expected):
        raise ValueError(f"episode outside frozen paired plan: {key}")
    return key


def load_case_plan(path: Path) -> tuple[set[tuple[int, int]], dict[tuple[int, int], str]]:
    value = json.loads(path.read_text())
    if (not isinstance(value, dict)
            or set(value) != {"schema", "task", "cases"}
            or value.get("schema") != "pi05_response_probe_case_plan.v1"):
        raise ValueError("case plan has invalid top-level schema")
    task = value.get("task")
    if (not isinstance(task, dict) or set(task) != {"suite", "task_id"}
            or (task.get("suite"), task.get("task_id")) != TASK):
        raise ValueError("case plan task must be exactly libero_goal/3")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 100:
        raise ValueError("explicit validation case plan must contain exactly 100 cases")
    cohorts = {}
    required = {"suite", "task_id", "init_id", "replicate_id", "cohort"}
    for index, row in enumerate(cases):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"case plan row {index} has invalid schema")
        if (row.get("suite"), row.get("task_id")) != TASK:
            raise ValueError(f"case plan row {index} has wrong task")
        init_id, replicate_id = row.get("init_id"), row.get("replicate_id")
        if (type(init_id) is not int or type(replicate_id) is not int
                or not 0 <= init_id < 10 or not 0 <= replicate_id < 10):
            raise ValueError(f"case plan row {index} has invalid init/replicate")
        key = (init_id, replicate_id)
        if key in cohorts:
            raise ValueError(f"duplicate case-plan key: {key}")
        expected_cohort = KNOWN_COHORT if key in EXPECTED else EXPANSION_COHORT
        if row.get("cohort") != expected_cohort:
            raise ValueError(f"case plan row {index} has invalid cohort")
        cohorts[key] = expected_cohort
    if set(cohorts) != FULL_VALIDATION:
        raise ValueError("case plan is not the exact init0..9 x replicate0..9 grid")
    return set(cohorts), cohorts


def manual_info(episode: dict) -> tuple[int | None, int, list[str], dict, list[str]]:
    report = episode.get("runner", {}).get("report", {})
    skill = report.get("skills", {}).get("pi05", {})
    provenance = skill.get("provenance", skill)
    errors = []
    parameters = provenance.get("parameters")
    if parameters is None:  # Backward-compatible audit for the frozen original probe.
        parameters = {"lift_z_command": .05, "max_lift_steps": 8,
                      "lift_target_m": .02, "native_reserve_steps": 20,
                      "minimum_actual": 120, "max_manual_actions": 14,
                      "trigger_remaining_steps": 39}
    elif isinstance(parameters, dict) and "minimum_actual" not in parameters:
        parameters = {**parameters, "minimum_actual": 120}
    expected_keys = {"lift_z_command", "max_lift_steps", "lift_target_m",
                     "native_reserve_steps", "max_manual_actions",
                     "trigger_remaining_steps", "minimum_actual"}
    valid_parameters = (isinstance(parameters, dict) and set(parameters) == expected_keys
        and type(parameters["lift_z_command"]) in (int, float)
        and math.isfinite(parameters["lift_z_command"])
        and 0 < parameters["lift_z_command"] <= .2
        and type(parameters["max_lift_steps"]) is int
        and 1 <= parameters["max_lift_steps"] <= 20
        and type(parameters["lift_target_m"]) in (int, float)
        and math.isfinite(parameters["lift_target_m"])
        and 0 < parameters["lift_target_m"] <= .025
        and type(parameters["native_reserve_steps"]) is int
        and 20 <= parameters["native_reserve_steps"] <= 80
        and type(parameters["minimum_actual"]) is int
        and 60 <= parameters["minimum_actual"] <= 120
        and parameters["max_manual_actions"] == 4 + parameters["max_lift_steps"] + 2
        and parameters["trigger_remaining_steps"] == 5 + parameters["max_manual_actions"]
            + parameters["native_reserve_steps"])
    if not valid_parameters:
        errors.append("invalid response-probe provenance parameters")
        parameters = {"max_manual_actions": 0, "native_reserve_steps": CAP}
    if provenance.get("kind") != "pi05_response_probe_v1":
        errors.append("probe provenance kind mismatch")
    if provenance.get("execution_reconciled") is not True:
        errors.append("probe execution not reconciled")
    rows = provenance.get("executed_rows", [])
    emitted = provenance.get("emitted_rows", [])
    manual = [row for row in rows if row.get("executed") is True and row.get("kind") != "native"]
    emitted_manual = [row for row in emitted if row.get("kind") != "native"]
    kinds = [row.get("kind") for row in manual]
    if any(kind not in {"close", "lift", "settle", "reopen"} for kind in kinds):
        errors.append("unknown manual action kind")
    maximum = parameters["max_manual_actions"]
    reserve = parameters["native_reserve_steps"]
    if (provenance.get("manual_actions_emitted") not in range(maximum + 1)
            or provenance.get("manual_actions_emitted") != len(emitted_manual)
            or len(manual) > maximum):
        errors.append("manual action limit exceeded")
    steps = [row.get("expected_actual_step") for row in manual]
    if any(type(step) is not int for step in steps) or steps != sorted(set(steps)):
        errors.append("manual actual steps invalid")
    for row in manual:
        if (not isinstance(row.get("action"), list) or len(row["action"]) != 7
                or not isinstance(row.get("post_state8"), list) or len(row["post_state8"]) != 8):
            errors.append("manual row lacks action7/post_state8")
            break
    first = steps[0] if steps else None
    if first is not None and first % 5:
        errors.append("first manual step is not an empty five-action-cache boundary")
    if first is not None and first + provenance.get("manual_actions_emitted", 99) + reserve > CAP:
        errors.append(f"{reserve}-action natural reserve violated")
    if manual and not provenance.get("events"):
        errors.append("manual actions lack trigger event")
    executed_indices = {row.get("emission_index") for row in manual}
    for row in emitted_manual:
        index = row.get("emission_index")
        if index in executed_indices:
            if row.get("executed") is not True:
                errors.append("executed manual emission is not marked executed")
        elif row.get("truncated_before_execution") is not True:
            errors.append("unexecuted manual emission lacks explicit truncation")
    trace = report.get("trace", [])
    trace_by_step = {row.get("step"): row for row in trace}
    for row in manual:
        step, traced = row.get("expected_actual_step"), trace_by_step.get(row.get("expected_actual_step"))
        if (not isinstance(traced, dict) or traced.get("skill") != "pi05"
                or not exact(traced.get("action"), row.get("action"))):
            errors.append(f"manual row does not match main trace at step {step}")
        following = trace_by_step.get(step + 1) if type(step) is int else None
        if following is not None and not exact(following.get("state"), row.get("post_state8")):
            errors.append(f"manual post_state8 does not match next trace state at step {step}")
    return first, len(manual), kinds, parameters, errors


def trace_prefix(episode: dict, stop: int | None):
    trace = episode.get("runner", {}).get("report", {}).get("trace", [])
    selected = trace if stop is None else [row for row in trace if row.get("step", -1) < stop]
    return [{"step": row.get("step"), "state": row.get("state"),
             "action": row.get("action"), "skill": row.get("skill")} for row in selected]


def call_prefix(episode: dict, stop: int | None, field: str):
    rows = episode.get("runner", {}).get(field, [])
    if stop is None:
        return rows
    count = (stop + 4) // 5
    return rows[:count]


def causal_pair(control: dict, probe: dict) -> dict:
    first, manual_count, kinds, parameters, errors = manual_info(probe)
    ctrace, ptrace = trace_prefix(control, first), trace_prefix(probe, first)
    if not exact(ctrace, ptrace):
        errors.append("pre-intervention action/state trace differs")
    for field in ("policy_calls", "payload_hashes"):
        expected_calls = first // 5 if first is not None and first % 5 == 0 else None
        if expected_calls is not None and (len(control.get("runner", {}).get(field, [])) < expected_calls
                                           or len(probe.get("runner", {}).get(field, [])) < expected_calls):
            errors.append(f"pre-intervention {field} shorter than cache-derived prefix")
        if not exact(call_prefix(control, first, field), call_prefix(probe, first, field)):
            errors.append(f"pre-intervention {field} differs")
    c_report, p_report = control["runner"]["report"], probe["runner"]["report"]
    if first is None:
        if not exact(trace_prefix(control, None), trace_prefix(probe, None)):
            errors.append("untriggered full traces differ")
        if (c_report.get("success"), c_report.get("status")) != (p_report.get("success"), p_report.get("status")):
            errors.append("untriggered outcome differs")
    return {"triggered": first is not None, "first_manual_actual_step": first,
            "manual_actions_executed": manual_count, "manual_kinds": kinds,
            "response_probe_parameters": parameters,
            "causal_gate_pass": not errors, "confounds": errors}


def load(worker_roots: list[Path], expected: set[tuple[int, int]] = EXPECTED,
         allowed_worker_counts: tuple[int, ...] = (4, 5)
         ) -> tuple[dict, list[str]]:
    episodes, errors, summary_count = {}, [], 0
    if len(worker_roots) not in allowed_worker_counts:
        errors.append(f"expected worker-output count in {allowed_worker_counts}")
    for root in worker_roots:
        controller = json.loads((root / "controller.json").read_text())
        batches = controller.get("batches", [])
        if controller.get("status") != "complete":
            errors.append(f"{root}: controller not complete")
        if len(batches) != 2 or [arm(row.get("name", "")) for row in batches] != ["control", "probe"]:
            errors.append(f"{root}: expected adjacent control/probe batches")
        elif batches[0].get("server_pid") != batches[1].get("server_pid"):
            errors.append(f"{root}: control/probe did not reuse one server_pid")
        elif batches[1].get("service_reused_from_previous_batch") is not True:
            errors.append(f"{root}: probe batch lacks explicit service reuse receipt")
        for summary_path in sorted(root.glob("*/summary.json")):
            summary_count += 1
            batch_arm = arm(summary_path.parent.name)
            summary = json.loads(summary_path.read_text())
            manifest = json.loads((summary_path.parent / "manifest.json").read_text())
            manifest_identity = manifest.get("verified_service_identity")
            if (manifest.get("fixed_policy_id") != "base"
                    or not isinstance(manifest_identity, dict)
                    or manifest_identity.get("policy_id") != "base"):
                errors.append(f"{summary_path}: manifest is not verified base policy")
            if (summary.get("mode") != "harness" or summary.get("complete") is not True
                    or summary.get("errors") != 0
                    or summary.get("planned") != summary.get("completed")):
                errors.append(f"{summary_path}: incomplete/error/non-harness summary")
            for row in summary.get("cases", []):
                key = episode_key(row.get("case", row), expected)
                if row.get("policy_id") != "base":
                    errors.append(f"{summary_path}: non-base policy")
                path = (summary_path.parent / row["episode"]).resolve()
                if not path.is_relative_to(summary_path.parent.resolve()):
                    errors.append(f"{summary_path}: episode path escape")
                    continue
                value = json.loads(path.read_text())
                if key != episode_key(value.get("case", {}), expected):
                    errors.append(f"{path}: summary/episode case mismatch")
                report = value.get("runner", {}).get("report", {})
                calls = value.get("runner", {}).get("policy_calls", [])
                if (not isinstance(calls, list) or not calls
                        or any(call.get("metadata") != manifest_identity for call in calls)):
                    errors.append(f"{path}: policy call actual identity differs from manifest")
                trace = report.get("trace", [])
                if (type(report.get("steps")) is not int or report["steps"] != len(trace)
                        or not 0 < report["steps"] <= CAP):
                    errors.append(f"{path}: report steps/trace/cap mismatch")
                if report.get("success") is not row.get("success"):
                    errors.append(f"{path}: summary/report success mismatch")
                slot = (batch_arm, key)
                if slot in episodes:
                    raise ValueError(f"duplicate episode {slot}")
                episodes[slot] = {"value": value, "row": row, "identity": manifest_identity}
    if summary_count != 2 * len(worker_roots):
        errors.append("expected exactly two batch summaries per worker")
    return episodes, errors


def summarize(rows: list[dict]) -> dict:
    outcomes = Counter(row["outcome"] for row in rows)
    return {"pairs": len(rows),
            "control_successes": sum(row["control_success"] for row in rows),
            "probe_successes": sum(row["probe_success"] for row in rows),
            "paired_outcomes": dict(outcomes)}


def aggregate(worker_roots: list[Path],
              expected_cases: set[tuple[int, int]] = EXPECTED,
              cohorts: dict[tuple[int, int], str] | None = None) -> dict:
    allowed_worker_counts = (4,) if cohorts is not None else (4, 5)
    episodes, errors = load(worker_roots, expected_cases, allowed_worker_counts)
    expected_slots = {(value, key) for value in ("control", "probe")
                      for key in expected_cases}
    if set(episodes) != expected_slots:
        errors.append(f"coverage is not exact {len(expected_cases)} pairs x 2 arms")
    identities = [entry["identity"] for entry in episodes.values()]
    if identities and (not isinstance(identities[0], dict)
                       or any(item != identities[0] for item in identities)
                       or identities[0].get("policy_id") != "base"):
        errors.append("actual base checkpoint identities differ")
    pairs, outcomes, regressions = [], Counter(), []
    for key in sorted(expected_cases):
        if ("control", key) not in episodes or ("probe", key) not in episodes:
            continue
        c, p = episodes[("control", key)], episodes[("probe", key)]
        ccase, pcase = c["value"]["case"], p["value"]["case"]
        if (ccase.get("policy_seed"), ccase.get("ambient_seed")) != (pcase.get("policy_seed"), pcase.get("ambient_seed")):
            errors.append(f"{key}: paired seeds differ")
        cs, ps = bool(c["row"].get("success")), bool(p["row"].get("success"))
        label = "both" if cs and ps else "recovered" if ps else "regressed" if cs else "neither"
        outcomes[label] += 1
        gate = causal_pair(c["value"], p["value"])
        if not gate["causal_gate_pass"]:
            errors.append(f"{key}: confounded: {', '.join(gate['confounds'])}")
        if key[0] != 4 and label == "regressed":
            regressions.append({"init_id": key[0], "replicate_id": key[1]})
        pairs.append({"init_id": key[0], "replicate_id": key[1],
                      "cohort": (cohorts or {}).get(key),
                      "control_success": cs, "probe_success": ps,
                      "outcome": label, **gate})
    hard = [row for row in pairs if row["init_id"] == 4]
    parameter_sets = {json.dumps(row["response_probe_parameters"], sort_keys=True)
                      for row in pairs}
    if len(parameter_sets) > 1:
        errors.append("response-probe parameters differ across paired records")
    candidate = json.loads(next(iter(parameter_sets))) if len(parameter_sets) == 1 else None
    known = [row for row in pairs if (row["init_id"], row["replicate_id"]) in EXPECTED]
    expansion = [row for row in pairs
                 if (row["init_id"], row["replicate_id"]) not in EXPECTED]
    per_init = {str(init_id): summarize(
        [row for row in pairs if row["init_id"] == init_id]) for init_id in range(10)}
    return {"schema": "pi05_response_probe.aggregate.v1", "complete": not errors,
            "candidate": candidate,
            "coverage": {"expected_pairs": len(expected_cases),
                         "observed_pairs": len(pairs),
                         "episodes": len(episodes), "errors": errors},
            "hard_case_init4": {"pairs": len(hard),
                "control_successes": sum(row["control_success"] for row in hard),
                "probe_successes": sum(row["probe_success"] for row in hard)},
            "paired_outcomes": dict(outcomes), "other_init_regressions": regressions,
            "validation_breakdown": {
                "all_expected": summarize(pairs),
                KNOWN_COHORT: summarize(known),
                EXPANSION_COHORT: summarize(expansion),
                "per_init": per_init,
            },
            "claim_gate": "No harness-benefit claim unless every pair passes the exact causal-prefix gate.",
            "pairs": pairs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-output", action="append", type=Path, required=True)
    parser.add_argument("--case-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output is create-only")
    expected, cohorts = ((EXPECTED, None) if args.case_plan is None
                         else load_case_plan(args.case_plan))
    result = aggregate(args.worker_output, expected, cohorts)
    result["coverage"]["case_plan_sha256"] = (None if args.case_plan is None else
        hashlib.sha256(args.case_plan.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
