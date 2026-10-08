#!/usr/bin/env python3
"""Strict paired aggregation for the PI0.5 closed-dwell lift screen."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from aggregate_pi05_goal3_phase_screen import initial_receipt, valid_initial

ARMS = ("control", "assist")
REPEATED = {0, 2, 3, 5, 8}
EXPECTED = {(init, rep) for init in range(10)
            for rep in (range(10) if init in REPEATED else (0,))}
CAP = 520


def arm(name):
    hits = [a for a in ARMS if a in name]
    if len(hits) != 1:
        raise ValueError(f"batch name does not identify one arm: {name}")
    return hits[0]


def case_key(case):
    key = (case.get("init_id"), case.get("replicate_id"))
    ambient = 7 + 38 * 50 + key[0] if type(key[0]) is int else None
    if ((case.get("suite"), case.get("task_id"), case.get("joint_task_number"))
            != ("libero_10", 8, 38) or key not in EXPECTED
            or case.get("ambient_seed") != ambient
            or case.get("policy_seed") != ambient + key[1] * 1000000007):
        raise ValueError("case/task/init/seed differs from frozen inventory")
    return key


def provenance(value):
    raw = value.get("runner", {}).get("report", {}).get("skills", {}).get("pi05", {})
    return raw.get("provenance", raw)


def audit_episode(value, which, identity):
    errors = []
    report = value.get("runner", {}).get("report", {})
    trace = report.get("trace", [])
    if (type(report.get("steps")) is not int or report["steps"] != len(trace)
            or not 0 < len(trace) <= CAP
            or [r.get("step") for r in trace] != list(range(len(trace)))):
        errors.append("trace incomplete/nonconsecutive/over cap")
    harness = value.get("harness", {})
    stages = harness.get("stages", [])
    if (harness.get("remember") != [] or len(stages) != 1
            or stages[0].get("skill") != "pi05" or stages[0].get("max_steps") != CAP
            or stages[0].get("until") is not None or stages[0].get("on_timeout") != "abort"
            or any(r.get("stage") != 0 or r.get("skill") != "pi05" for r in trace)):
        errors.append("single-stage task-520 harness contract invalid")
    calls = value.get("runner", {}).get("policy_calls", [])
    receipts = value.get("runner", {}).get("payload_hashes", [])
    if len(calls) != len(receipts) or len(calls) != (len(trace) + 4) // 5:
        errors.append("call/hash count is not ceil(steps/5)")
    seed0 = value.get("case", {}).get("policy_seed")
    for i, (call, receipt) in enumerate(zip(calls, receipts)):
        seed = seed0 + i * 1000003
        if (call.get("inference_call") != i or receipt.get("inference_call") != i
                or call.get("policy_seed") != seed or receipt.get("policy_seed") != seed
                or call.get("metadata") != identity or call.get("response_valid") is not True
                or receipt.get("status") != "ok"):
            errors.append("call seed/identity/receipt invalid"); break
    if which == "control":
        return errors
    p = provenance(value)
    emitted, executed, chunks = p.get("emitted_rows", []), p.get("executed_rows", []), p.get("chunks", [])
    if (p.get("kind") != "closed_dwell_lift_v1" or p.get("execution_reconciled") is not True
            or p.get("native_reserve") != 80 or p.get("mechanical_proxy_only") is not True
            or p.get("grasp_or_task_success_certificate") is not False or p.get("oracle_inputs") != []):
        errors.append("assist provenance contract invalid")
    veto_enabled = p.get("veto_native_upward_intent", False)
    if type(veto_enabled) is not bool:
        errors.append("intent-veto parameter is not bool")
    if len(executed) != len(trace):
        errors.append("executed rows do not cover trace")
    executed_by_index = {r.get("emission_index"): r for r in executed}
    for row in emitted:
        match = executed_by_index.get(row.get("emission_index"))
        if match is None:
            if row.get("truncated_before_execution") is not True:
                errors.append("unexecuted emission lacks terminal truncation"); break
            continue
        step = row.get("expected_actual_step")
        action = row.get("executed_action")
        if (match != row or row.get("executed") is not True or type(step) is not int
                or step >= len(trace) or trace[step].get("action") != action
                or trace[step].get("state") != row.get("pre_state8")):
            errors.append("executed row differs from trace/state"); break
        raw = row.get("raw_action")
        if not isinstance(raw, list) or len(raw) != 7 or not isinstance(action, list) or len(action) != 7:
            errors.append("raw/executed action shape invalid"); break
        expected = list(raw)
        if row.get("kind") == "assist": expected[2] = max(expected[2], .2)
        if action != expected or row.get("modified") is not (action != raw) or action[6] != raw[6]:
            errors.append("assist is not exact z-only max(raw_z,.2)"); break
    modified = [r for r in executed if r.get("modified") is True]
    first = p.get("first_changed_action_step")
    slots = p.get("assist_slots_executed")
    if (len(modified) != p.get("modification_count") or len(modified) > 10
            or type(slots) is not int or not 0 <= slots <= 10
            or slots != sum(r.get("kind") == "assist" for r in executed)
            or any(r.get("kind") != "assist" or r.get("raw_action", [0] * 7)[6] < .5
                   for r in modified)
            or sum(r.get("modified") is True for r in emitted) != p.get("emitted_modification_count")):
        errors.append("assist modification/slot accounting invalid")
    attempted = p.get("attempted") is True
    cue = p.get("cue", {}); context = cue.get("context", {}) if isinstance(cue, dict) else {}
    if attempted:
        cue_actual = context.get("actual_executed", -1)
        if (cue.get("window") != 60 or cue.get("closed_command_rows", -1) < 57
                or not all(value <= .012 for value in cue.get("xyz_ptp_m", [1, 1, 1]))
                or not .012 <= cue.get("aperture_m", -1) <= .070
                or cue_actual < 120 or cue_actual + context.get("remaining_episode", -1) != CAP
                or context.get("stage_executed") + context.get("remaining_stage", -1) != CAP
                or context.get("stage_index") != 0
                or min(context.get("remaining_episode", -1), context.get("remaining_stage", -1)) < 90):
            errors.append("assist cue/reserve invalid")
    if modified:
        cue_actual = context.get("actual_executed", -1)
        if (first != modified[0].get("expected_actual_step") or first < cue_actual
                or cue_actual + 10 + 80 > CAP or not attempted):
            errors.append("assist first-change/reserve invalid")
    elif first is not None:
        errors.append("first change recorded without modified action")
    reason, incoming = p.get("veto_reason"), p.get("native_incoming_z")
    cue_row = next((r for r in executed
                    if r.get("expected_actual_step") == context.get("actual_executed")), None)
    if veto_enabled and attempted and (not isinstance(cue_row, dict)
            or cue_row.get("raw_action", [None] * 7)[2] != incoming):
        errors.append("cue-time native z does not match executed-row provenance")
    if reason is not None:
        if (not veto_enabled or reason != "incoming_native_upward_intent"
                or type(incoming) not in (int, float) or incoming <= 0
                or not attempted or modified or slots != 0 or first is not None
                or any(r.get("kind") != "native" for r in executed)):
            errors.append("intent veto provenance/action contract invalid")
    elif veto_enabled and attempted:
        if type(incoming) not in (int, float) or incoming > 0:
            errors.append("enabled intent veto failed to record/apply cue-time native z")
    elif not attempted and (reason is not None or incoming is not None):
        errors.append("intent veto evidence exists without cue")
    if (len(chunks) != len(calls)
            or any(type(r.get("cache_chunk_inference_index")) is not int
                   or not 0 <= r["cache_chunk_inference_index"] < len(chunks)
                   for r in emitted + executed)
            or sum(c.get("emitted_rows", -1) for c in chunks) != len(emitted)
            or sum(c.get("executed_rows", -1) for c in chunks) != len(executed)):
        errors.append("chunk count/index/totals invalid")
    for i, chunk in enumerate(chunks):
        if (chunk.get("chunk_index") != i or chunk.get("delegate_call_index") != i
                or chunk.get("inference_actual_step") != i * 5
                or chunk.get("emitted_rows") != sum(r.get("cache_chunk_inference_index") == i for r in emitted)
                or chunk.get("executed_rows") != sum(r.get("cache_chunk_inference_index") == i for r in executed)):
            errors.append("chunk/call/cache accounting invalid"); break
    return errors


def causal(control, candidate):
    first = provenance(candidate).get("first_changed_action_step")
    ct, at = control["runner"]["report"]["trace"], candidate["runner"]["report"]["trace"]
    stop = len(ct) if first is None else first
    errors = []
    if ct[:stop] != at[:stop]: errors.append("pre-change trace differs")
    # The call which produced an in-chunk first change is still pre-intervention input evidence.
    ncall = first // 5 + 1 if first is not None else len(control["runner"].get("policy_calls", []))
    for field in ("policy_calls", "payload_hashes"):
        if control["runner"].get(field, [])[:ncall] != candidate["runner"].get(field, [])[:ncall]:
            errors.append(f"pre-change {field} differ")
    if first is None:
        if ct != at or control["runner"].get("policy_calls") != candidate["runner"].get("policy_calls") \
                or control["runner"].get("payload_hashes") != candidate["runner"].get("payload_hashes"):
            errors.append("unchanged whole-episode evidence differs")
        cr, ar = control["runner"]["report"], candidate["runner"]["report"]
        if (cr.get("success"), cr.get("status")) != (ar.get("success"), ar.get("status")):
            errors.append("unchanged outcome differs")
    return errors


def aggregate(roots):
    episodes, errors, summaries = {}, [], 0
    if len(roots) not in (4, 5): errors.append("expected four candidate or five legacy workers")
    for root in roots:
        controller = json.loads((root / "controller.json").read_text()); batches = controller.get("batches", [])
        if (controller.get("status") != "complete" or len(batches) != 2
                or [arm(r.get("name", "")) for r in batches] != list(ARMS)
                or len({r.get("server_pid") for r in batches}) != 1
                or batches[1].get("service_reused_from_previous_batch") is not True):
            errors.append(f"{root}: controller/service reuse invalid")
        for sp in sorted(root.glob("*/summary.json")):
            summaries += 1; which = arm(sp.parent.name); summary = json.loads(sp.read_text())
            manifest = json.loads((sp.parent / "manifest.json").read_text()); identity = manifest.get("verified_service_identity")
            if (summary.get("mode") != "harness" or summary.get("complete") is not True or summary.get("errors") != 0
                    or summary.get("planned") != summary.get("completed") or manifest.get("fixed_policy_id") != "base"
                    or not isinstance(identity, dict) or identity.get("policy_id") != "base"
                    or identity.get("base_graph") != "original_pi05_libero" or identity.get("adapter_sha256") is not None):
                errors.append(f"{sp}: summary/base identity invalid")
            for row in summary.get("cases", []):
                key = case_key(row); ep = (sp.parent / row["episode"]).resolve()
                if not ep.is_relative_to(sp.parent.resolve()): errors.append("episode path escape"); continue
                value = json.loads(ep.read_text()); report = value.get("runner", {}).get("report", {})
                if (key != case_key(value.get("case", {})) or row.get("policy_id") != "base"
                        or row.get("success") is not report.get("success") or row.get("steps") != report.get("steps")):
                    errors.append(f"{ep}: case/summary mismatch")
                slot = (which, key)
                if slot in episodes: raise ValueError(f"duplicate {slot}")
                errors += [f"{ep}: {e}" for e in audit_episode(value, which, identity)]
                episodes[slot] = {"value": value, "row": row, "identity": identity}
    if summaries != 2 * len(roots): errors.append("expected two summaries per worker")
    expected = {(a, k) for a in ARMS for k in EXPECTED}
    if set(episodes) != expected: errors.append("coverage is not exact 55 x 2")
    identities = [v["identity"] for v in episodes.values()]
    if identities and any(v != identities[0] for v in identities): errors.append("service identities differ")
    veto_parameters = {provenance(v["value"]).get("veto_native_upward_intent", False)
                       for (which, _), v in episodes.items() if which == "assist"}
    if len(veto_parameters) != 1 or any(type(v) is not bool for v in veto_parameters):
        errors.append("intent-veto parameter differs across assist records")
    stats, pairs = {a: Counter() for a in ARMS}, []
    for key in sorted(EXPECTED):
        if any((a, key) not in episodes for a in ARMS): continue
        c, a = episodes[("control", key)], episodes[("assist", key)]
        ci, ai = initial_receipt(c["value"]), initial_receipt(a["value"])
        try: valid = valid_initial(ci) and valid_initial(ai)
        except (AttributeError, TypeError): valid = False
        if not valid or ci != ai: errors.append(f"{key}: invalid/different initial observation")
        errors += [f"assist/{key}: {e}" for e in causal(c["value"], a["value"])]
        outcomes = {name: bool(episodes[(name, key)]["row"].get("success")) for name in ARMS}
        for name, success in outcomes.items():
            stats[name]["hard2" if key[0] == 2 else "repeated40" if key[0] in REPEATED else "other5"] += success
            stats[name]["original10"] += success and key[1] == 0
        label = "both" if outcomes["control"] and outcomes["assist"] else "recovered" if outcomes["assist"] else "regressed" if outcomes["control"] else "neither"
        stats["assist"][label] += 1; pairs.append({"init_id": key[0], "replicate_id": key[1], "success": outcomes})
    return {"schema":"pi05_closed_dwell_lift.aggregate.v1","complete":not errors,
            "veto_native_upward_intent": (next(iter(veto_parameters)) if len(veto_parameters) == 1 else None),
            "coverage":{"expected_episodes":110,"observed_episodes":len(episodes),"errors":errors},
            "arm_scores":{n:{"hard_init2":[stats[n]["hard2"],10],"other_four_repeated":[stats[n]["repeated40"],40],
              "other_five_inits":[stats[n]["other5"],5],"original_ten_init_rep0":[stats[n]["original10"],10],
              "paired_vs_control":({k:stats[n][k] for k in ("both","recovered","regressed","neither")} if n=="assist" else None)} for n in ARMS},
            "evidence_limit":"Service responses are hash-recorded; raw responses were not independently recomputed.","pairs":pairs}


def main():
    p=argparse.ArgumentParser();p.add_argument("--worker-output",action="append",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError("output is create-only")
    result=aggregate(a.worker_output);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return 0 if result["complete"] else 1


if __name__ == "__main__": raise SystemExit(main())
