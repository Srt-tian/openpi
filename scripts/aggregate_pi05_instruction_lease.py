#!/usr/bin/env python3
"""Strict three-arm aggregation for PI0.5 goal3 instruction leases."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from aggregate_pi05_goal3_phase_screen import initial_receipt, valid_initial

ARMS = ("control", "lease40", "lease80")
ORIGINAL = "open the top drawer and put the bowl inside"
LEASED = "open the top drawer"
EXPECTED = {(init, rep) for init in range(10)
            for rep in (range(10) if init == 4 else (0,))}


def arm(name: str) -> str:
    hits = [value for value in ARMS if value in name]
    if len(hits) != 1:
        raise ValueError(f"batch name does not identify one arm: {name}")
    return hits[0]


def case_key(case: dict) -> tuple[int, int]:
    if (case.get("suite"), case.get("task_id")) != ("libero_goal", 3):
        raise ValueError("non-goal3 case")
    result = (case.get("init_id"), case.get("replicate_id"))
    if result not in EXPECTED:
        raise ValueError("case outside frozen inventory")
    ambient = 7 + 23 * 50 + result[0]
    if (case.get("ambient_seed"), case.get("policy_seed")) != (
            ambient, ambient + result[1] * 1000000007):
        raise ValueError("case seed protocol mismatch")
    return result


def lease_provenance(value: dict) -> dict:
    raw = value.get("runner", {}).get("report", {}).get("skills", {}).get("pi05", {})
    return raw.get("provenance", raw)


def audit_episode(value: dict, which: str, identity: dict) -> list[str]:
    errors, report = [], value.get("runner", {}).get("report", {})
    trace = report.get("trace", [])
    if (type(report.get("steps")) is not int or report["steps"] != len(trace)
            or not 0 < len(trace) <= 300
            or [row.get("step") for row in trace] != list(range(len(trace)))):
        errors.append("trace incomplete/nonconsecutive/over cap")
    harness = value.get("harness", {})
    stages = harness.get("stages", [])
    if (harness.get("remember") != [] or len(stages) != 1
            or stages[0].get("skill") != "pi05" or stages[0].get("instruction") != ORIGINAL
            or stages[0].get("max_steps") != 300 or stages[0].get("until") is not None
            or stages[0].get("on_timeout") != "abort"
            or any(row.get("stage") != 0 or row.get("skill") != "pi05" for row in trace)):
        errors.append("original single-stage harness contract changed")
    calls = value.get("runner", {}).get("policy_calls", [])
    receipts = value.get("runner", {}).get("payload_hashes", [])
    expected_calls = (len(trace) + 4) // 5
    if len(calls) != len(receipts) or len(calls) != expected_calls:
        errors.append("call/hash count is not ceil(steps/5)")
    seed0 = value.get("case", {}).get("policy_seed")
    for index, (call, receipt) in enumerate(zip(calls, receipts)):
        seed = seed0 + index * 1000003
        if (call.get("inference_call") != index or receipt.get("inference_call") != index
                or call.get("policy_seed") != seed or receipt.get("policy_seed") != seed
                or call.get("metadata") != identity or call.get("response_valid") is not True
                or receipt.get("status") != "ok"):
            errors.append("call seed/identity/receipt invalid")
            break
    if which == "control":
        if any(row.get("prompt") != ORIGINAL for row in receipts):
            errors.append("control prompt changed")
        return errors
    planned = 40 if which == "lease40" else 80
    provenance = lease_provenance(value)
    if (provenance.get("kind") != "instruction_lease_v1"
            or provenance.get("configured_instruction") != LEASED
            or provenance.get("planned_lease_steps") != planned
            or provenance.get("original_native_reserve") != 80
            or provenance.get("execution_reconciled") is not True
            or provenance.get("actions_transformed") is not False
            or provenance.get("oracle_inputs") != []):
        errors.append("lease provenance contract invalid")
    prompt_records = provenance.get("prompt_call_records", [])
    emitted, executed = provenance.get("emitted_rows", []), provenance.get("executed_rows", [])
    actual, first = provenance.get("actual_leased_steps"), provenance.get("first_changed_prompt_step")
    if type(actual) is not int or not 0 <= actual <= planned:
        errors.append("actual leased-step bound violated")
    if len(prompt_records) != len(calls) or len(emitted) != len(calls) * 5 or len(executed) != len(trace):
        errors.append("prompt/emitted/executed cardinality mismatch")
    trace_by_step = {row.get("step"): row for row in trace}
    executed_by_emission = {row.get("emission_index"): row for row in executed}
    for index, record in enumerate(prompt_records):
        expected_prompt = LEASED if record.get("leased") else ORIGINAL
        if (record.get("prompt_call_index") != index or record.get("delegate_call_index") != index
                or record.get("actual_step") != index * 5 or record.get("emitted_steps") != 5
                or record.get("planned_prompt") != expected_prompt
                or record.get("used_prompt") != expected_prompt
                or record.get("original_prompt") != ORIGINAL
                or index >= len(receipts) or receipts[index].get("prompt") != expected_prompt):
            errors.append("prompt call does not match actual payload")
            break
        executed_for_call = sum(row.get("prompt_call_index") == index for row in executed)
        if record.get("executed_steps") != executed_for_call:
            errors.append("prompt-call executed_steps differs from executed rows")
            break
    for row in emitted:
        step, index = row.get("expected_actual_step"), row.get("emission_index")
        if index in executed_by_emission:
            executed_row = executed_by_emission[index]
            traced = trace_by_step.get(step)
            if (row.get("executed") is not True or executed_row != row
                    or not isinstance(traced, dict) or traced.get("skill") != "pi05"
                    or traced.get("action") != executed_row.get("action")):
                errors.append("executed emission differs from trace")
                break
        elif row.get("truncated_before_execution") is not True:
            errors.append("unexecuted emission lacks explicit terminal truncation")
            break
    changed = [row for row in prompt_records if row.get("leased") is True]
    if changed:
        events = provenance.get("events", [])
        if (type(first) is not int or first % 5 or first != changed[0].get("actual_step")
                or first < 5 or prompt_records[first // 5 - 1].get("leased") is not False
                or first + planned + 80 > 300):
            errors.append("first changed prompt/grace/reserve boundary invalid")
        expected_actual = min(planned, len(trace) - first) if type(first) is int else -1
        expected_changed_steps = list(range(first, min(first + planned, len(trace)), 5)) if type(first) is int else []
        if (actual != expected_actual
                or [row.get("actual_step") for row in changed] != expected_changed_steps):
            errors.append("changed prompts are not one continuous full-or-terminal-truncated lease")
        cue = events[-1].get("cue", {}) if events else {}
        recheck = events[-1].get("recheck", {}) if events else {}
        cue_context, recheck_context = cue.get("context", {}), recheck.get("context", {})
        if (first < 125 or cue_context.get("actual_executed") != first - 5
                or recheck_context.get("actual_executed") != first
                or any(context.get("actual_executed", -1) + context.get("remaining_episode", -1) != 300
                       or context.get("stage_executed") != context.get("actual_executed")
                       or context.get("stage_executed", -1) + context.get("remaining_stage", -1) != 300
                       or context.get("stage_index") != 0
                       for context in (cue_context, recheck_context))):
            errors.append("event cue/recheck does not prove five-step grace and fixed 300 budget")
        if (not events or (actual == planned and events[-1].get("status") not in
                           {"lease_complete", "lease_complete_at_terminal"})
                or (actual < planned and events[-1].get("status") != "terminal_truncated_during_lease")):
            errors.append("lease completion/truncation status inconsistent")
    elif first is not None or actual != 0:
        errors.append("untriggered lease has changed-prompt metadata")
    leased_executed = sum(row.get("kind") == "leased" for row in executed)
    if leased_executed != actual:
        errors.append("actual lease count differs from executed rows")
    return errors


def prefix(value: dict, stop: int | None):
    runner, trace = value.get("runner", {}), value.get("runner", {}).get("report", {}).get("trace", [])
    count = None if stop is None else stop // 5
    return {"trace": trace if stop is None else trace[:stop],
            "calls": runner.get("policy_calls", []) if count is None else runner.get("policy_calls", [])[:count],
            "hashes": runner.get("payload_hashes", []) if count is None else runner.get("payload_hashes", [])[:count]}


def causal(control: dict, candidate: dict) -> list[str]:
    provenance = lease_provenance(candidate)
    first = provenance.get("first_changed_prompt_step")
    errors = []
    if prefix(control, first) != prefix(candidate, first):
        errors.append("pre-lease trace/calls/payload hashes differ")
    if first is None:
        cr, pr = control["runner"]["report"], candidate["runner"]["report"]
        if prefix(control, None) != prefix(candidate, None):
            errors.append("untriggered whole-episode evidence differs")
        if (cr.get("success"), cr.get("status")) != (pr.get("success"), pr.get("status")):
            errors.append("untriggered outcome differs")
    return errors


def load(roots: list[Path]):
    episodes, errors, summaries = {}, [], 0
    if len(roots) != 5:
        errors.append("expected five workers")
    for root in roots:
        controller = json.loads((root / "controller.json").read_text())
        batches = controller.get("batches", [])
        if (controller.get("status") != "complete" or len(batches) != 3
                or [arm(row.get("name", "")) for row in batches] != list(ARMS)
                or len({row.get("server_pid") for row in batches}) != 1
                or any(row.get("service_reused_from_previous_batch") is not True for row in batches[1:])):
            errors.append(f"{root}: controller/service reuse invalid")
        for path in sorted(root.glob("*/summary.json")):
            summaries += 1; which = arm(path.parent.name)
            summary = json.loads(path.read_text()); manifest = json.loads((path.parent / "manifest.json").read_text())
            identity = manifest.get("verified_service_identity")
            if (summary.get("mode") != "harness" or summary.get("complete") is not True
                    or summary.get("errors") != 0 or summary.get("planned") != summary.get("completed")
                    or manifest.get("fixed_policy_id") != "base" or not isinstance(identity, dict)):
                errors.append(f"{path}: summary/manifest invalid")
            for row in summary.get("cases", []):
                pair = case_key(row); episode_path = (path.parent / row["episode"]).resolve()
                if not episode_path.is_relative_to(path.parent.resolve()): errors.append("episode path escape"); continue
                value = json.loads(episode_path.read_text())
                report = value["runner"]["report"]
                if (pair != case_key(value.get("case", {})) or row.get("policy_id") != "base"
                        or identity.get("policy_id") != "base"
                        or row.get("success") is not report.get("success")
                        or row.get("steps") != report.get("steps")):
                    errors.append(f"{episode_path}: case/summary mismatch")
                slot = (which, pair)
                if slot in episodes: raise ValueError(f"duplicate {slot}")
                errors.extend(f"{episode_path}: {message}" for message in audit_episode(value, which, identity))
                episodes[slot] = {"value": value, "row": row, "identity": identity}
    if summaries != 15: errors.append("expected 15 summaries")
    return episodes, errors


def aggregate(roots: list[Path]) -> dict:
    episodes, errors = load(roots); expected = {(a, p) for a in ARMS for p in EXPECTED}
    if set(episodes) != expected: errors.append("coverage is not 19 x 3")
    identities = [row["identity"] for row in episodes.values()]
    if identities and any(value != identities[0] for value in identities): errors.append("service identities differ")
    stats, pairs = {a: Counter() for a in ARMS}, []
    for pair in sorted(EXPECTED):
        if any((a, pair) not in episodes for a in ARMS): continue
        control = episodes[("control", pair)]; initial = initial_receipt(control["value"])
        try: initial_valid = valid_initial(initial)
        except (AttributeError, TypeError): initial_valid = False
        if not initial_valid: errors.append(f"control/{pair}: invalid initial receipt")
        base = bool(control["row"].get("success")); outcome = {}
        for a in ARMS:
            item = episodes[(a, pair)]; success = bool(item["row"].get("success")); outcome[a] = success
            candidate_initial = initial_receipt(item["value"])
            try: candidate_valid = valid_initial(candidate_initial)
            except (AttributeError, TypeError): candidate_valid = False
            if not candidate_valid: errors.append(f"{a}/{pair}: invalid initial receipt")
            if candidate_initial != initial: errors.append(f"{a}/{pair}: initial observation differs")
            stats[a]["hard" if pair[0] == 4 else "other"] += int(success)
            stats[a]["original"] += int(success and pair[1] == 0)
            if a != "control":
                label = "both" if base and success else "recovered" if success else "regressed" if base else "neither"
                stats[a][label] += 1
                errors.extend(f"{a}/{pair}: {message}" for message in causal(control["value"], item["value"]))
        pairs.append({"init_id": pair[0], "replicate_id": pair[1], "success": outcome})
    return {"schema":"pi05_instruction_lease.aggregate.v1","complete":not errors,
        "coverage":{"expected_episodes":57,"observed_episodes":len(episodes),"errors":errors},
        "arm_scores":{a:{"hard_init4":[stats[a]["hard"],10],"other_nine_inits":[stats[a]["other"],9],
          "original_ten_init_rep0":[stats[a]["original"],10],"paired_vs_control":({k:stats[a][k] for k in ("both","recovered","regressed","neither")} if a!="control" else None)} for a in ARMS},
        "evidence_limit":"Response actions are hash-recorded only; raw service responses were not independently recomputed.",
        "selection_warning":"Arms remain separate; no best-of-N pooled score.","pairs":pairs}


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--worker-output",action="append",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError("output is create-only")
    result=aggregate(a.worker_output);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return 0 if result["complete"] else 1


if __name__ == "__main__": raise SystemExit(main())
