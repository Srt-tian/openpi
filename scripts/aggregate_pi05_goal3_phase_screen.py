#!/usr/bin/env python3
"""Strict configurable-arm aggregation for the PI0.5 goal3 language-phase screen."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from aggregate_pi05_response_probe import causal_pair

ARMS = ("control", "probe", "prefix100", "prefix150", "sequential")
VALID_ARMS = ("control", "probe", "prefix25", "prefix50", "prefix75",
              "prefix100", "prefix150", "sequential")
ORIGINAL = "open the top drawer and put the bowl inside"
OPEN = "open the top drawer"
SEQUENTIAL = "open the top drawer first, then put the bowl inside it"
EXPECTED = {(init, rep) for init in range(10)
            for rep in (range(10) if init == 4 else (0,))}


def validate_arms(arms) -> tuple[str, ...]:
    values = tuple(arms)
    if (not values or "control" not in values or len(set(values)) != len(values)
            or any(value not in VALID_ARMS for value in values)):
        raise ValueError(f"arms must be unique members of {VALID_ARMS}: {values}")
    return values


def arm(name: str, arms=ARMS) -> str:
    arms = validate_arms(arms)
    hits = [value for value in arms if value in name]
    if len(hits) != 1:
        raise ValueError(f"batch name does not select exactly one arm: {name}")
    return hits[0]


def key(case: dict) -> tuple[int, int]:
    if (case.get("suite"), case.get("task_id")) != ("libero_goal", 3):
        raise ValueError("non-goal3 case")
    value = (case.get("init_id"), case.get("replicate_id"))
    if value not in EXPECTED:
        raise ValueError("case outside frozen 19-pair inventory")
    ambient = 7 + 23 * 50 + value[0]
    if (case.get("ambient_seed"), case.get("policy_seed")) != (
            ambient, ambient + value[1] * 1000000007):
        raise ValueError("case seeds differ from frozen protocol")
    return value


def expected_stages(which: str):
    if which in ("control", "probe"):
        return [(ORIGINAL, 300)]
    if which.startswith("prefix") and which in VALID_ARMS:
        budget = int(which.removeprefix("prefix"))
        return [(OPEN, budget), (ORIGINAL, 300 - budget)]
    if which == "sequential":
        return [(SEQUENTIAL, 300)]
    raise ValueError(f"unknown arm: {which}")


def audit_episode(value: dict, which: str, manifest_identity: dict) -> list[str]:
    errors, report = [], value.get("runner", {}).get("report", {})
    trace = report.get("trace", [])
    if (type(report.get("steps")) is not int or report["steps"] != len(trace)
            or not 0 < report["steps"] <= 300
            or [row.get("step") for row in trace] != list(range(len(trace)))):
        errors.append("incomplete/nonconsecutive/over-cap trace")
    stages = value.get("harness", {}).get("stages", [])
    actual = [(row.get("instruction"), row.get("max_steps")) for row in stages]
    if actual != expected_stages(which):
        errors.append("harness stage instruction/budget mismatch")
    if (value.get("harness", {}).get("remember") != []
            or any(row.get("skill") != "pi05" or row.get("until") is not None
                   or row.get("on_timeout") != ("abort" if index == len(stages) - 1 else "next")
                   for index, row in enumerate(stages))):
        errors.append("harness stage control contract mismatch")
    stage_ids = [row.get("stage") for row in trace]
    if stage_ids and stage_ids[0] != 0:
        errors.append("trace does not begin in stage0")
    expected = expected_stages(which)
    boundary = expected[0][1] if len(expected) == 2 else None
    if boundary is not None:
        expected_ids = [0 if step < boundary else 1 for step in range(len(trace))]
        if stage_ids != expected_ids:
            errors.append("stage switch is not at the fixed boundary")
    elif any(stage != 0 for stage in stage_ids):
        errors.append("single-stage arm changed stage")
    calls = value.get("runner", {}).get("policy_calls", [])
    hashes = value.get("runner", {}).get("payload_hashes", [])
    if len(calls) != len(hashes) or not calls:
        errors.append("policy call/hash recording incomplete")
    if which != "probe" and len(calls) != (len(trace) + 4) // 5:
        errors.append("non-probe call count is not ceil(steps/5)")
    episode_seed = value.get("case", {}).get("policy_seed")
    for index, (call, receipt) in enumerate(zip(calls, hashes)):
        seed = episode_seed + index * 1000003
        if (call.get("inference_call") != index or receipt.get("inference_call") != index
                or call.get("policy_seed") != seed or receipt.get("policy_seed") != seed
                or call.get("metadata") != manifest_identity
                or call.get("response_valid") is not True or receipt.get("status") != "ok"):
            errors.append("noncontinuous inference seed/call identity")
            break
        step = index * 5
        stage = 1 if boundary is not None and step >= boundary else 0
        prompt = expected[stage][0]
        if receipt.get("prompt") != prompt:
            errors.append("payload prompt does not match stage instruction")
            break
    return errors


def initial_receipt(value: dict) -> dict:
    rows = value.get("runner", {}).get("payload_hashes", [])
    if not rows:
        return {}
    row = rows[0]
    return {name: row.get(name) for name in
            ("observation_image", "observation_wrist_image", "observation_state")}


def valid_initial(receipt: dict) -> bool:
    specs = {"observation_image": ([224, 224, 3], "uint8"),
             "observation_wrist_image": ([224, 224, 3], "uint8"),
             "observation_state": ([8], "float64")}
    if set(receipt) != set(specs):
        return False
    for name, (shape, dtype) in specs.items():
        item = receipt.get(name)
        digest = item.get("sha256") if isinstance(item, dict) else None
        if (item.get("shape") != shape or item.get("dtype") != dtype
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            return False
    return True


def load(roots: list[Path], arms=ARMS):
    arms = validate_arms(arms)
    episodes, errors, summaries = {}, [], 0
    if len(roots) != 5:
        errors.append("expected five worker outputs")
    for root in roots:
        controller = json.loads((root / "controller.json").read_text())
        batches = controller.get("batches", [])
        if (controller.get("status") != "complete" or len(batches) != len(arms)
                or [arm(row.get("name", ""), arms) for row in batches] != list(arms)):
            errors.append(f"{root}: controller/batch order incomplete")
        elif (len({row.get("server_pid") for row in batches}) != 1
              or any(row.get("service_reused_from_previous_batch") is not True for row in batches[1:])):
            errors.append(f"{root}: selected arms did not reuse one service")
        for path in sorted(root.glob("*/summary.json")):
            summaries += 1
            which = arm(path.parent.name, arms)
            summary = json.loads(path.read_text())
            manifest = json.loads((path.parent / "manifest.json").read_text())
            identity = manifest.get("verified_service_identity")
            if (summary.get("mode") != "harness" or summary.get("complete") is not True
                    or summary.get("errors") != 0
                    or summary.get("planned") != summary.get("completed")
                    or manifest.get("fixed_policy_id") != "base"
                    or not isinstance(identity, dict) or identity.get("policy_id") != "base"):
                errors.append(f"{path}: summary/manifest invalid")
            for row in summary.get("cases", []):
                pair = key(row)
                episode_path = (path.parent / row["episode"]).resolve()
                if not episode_path.is_relative_to(path.parent.resolve()):
                    errors.append(f"{path}: episode path escape")
                    continue
                value = json.loads(episode_path.read_text())
                if pair != key(value.get("case", {})) or row.get("policy_id") != "base":
                    errors.append(f"{episode_path}: case/policy mismatch")
                if (row.get("success") is not value.get("runner", {}).get("report", {}).get("success")
                        or row.get("steps") != value.get("runner", {}).get("report", {}).get("steps")):
                    errors.append(f"{episode_path}: summary/report mismatch")
                slot = (which, pair)
                if slot in episodes:
                    raise ValueError(f"duplicate episode {slot}")
                local_errors = audit_episode(value, which, identity)
                errors.extend(f"{episode_path}: {message}" for message in local_errors)
                episodes[slot] = {"value": value, "row": row, "identity": identity}
    if summaries != 5 * len(arms):
        errors.append(f"expected {5 * len(arms)} summaries")
    return episodes, errors


def aggregate(roots: list[Path], arms=ARMS) -> dict:
    arms = validate_arms(arms)
    episodes, errors = load(roots, arms)
    expected = {(which, pair) for which in arms for pair in EXPECTED}
    if set(episodes) != expected:
        errors.append(f"coverage is not exact {len(arms)} arms x 19 cases")
    identities = [row["identity"] for row in episodes.values()]
    if identities and (not isinstance(identities[0], dict)
                       or any(value != identities[0] for value in identities)):
        errors.append("actual service identities differ across arms/workers")
    pairs, stats = [], {which: Counter() for which in arms}
    for pair in sorted(EXPECTED):
        if any((which, pair) not in episodes for which in arms):
            continue
        control = episodes[("control", pair)]
        initial = initial_receipt(control["value"])
        if not valid_initial(initial):
            errors.append(f"control/{pair}: initial RGB/state receipt invalid")
        for which in arms:
            if which == "control":
                continue
            candidate = episodes[(which, pair)]
            candidate_initial = initial_receipt(candidate["value"])
            if not valid_initial(candidate_initial):
                errors.append(f"{which}/{pair}: initial RGB/state receipt invalid")
            if candidate_initial != initial:
                errors.append(f"{which}/{pair}: initial RGB/state hashes differ from control")
            ccase, pcase = control["value"]["case"], candidate["value"]["case"]
            if (ccase.get("ambient_seed"), ccase.get("policy_seed")) != (
                    pcase.get("ambient_seed"), pcase.get("policy_seed")):
                errors.append(f"{which}/{pair}: paired seeds differ")
        if ("probe", pair) in episodes:
            gate = causal_pair(control["value"], episodes[("probe", pair)]["value"])
            if not gate["causal_gate_pass"]:
                errors.append(f"probe/{pair}: causal gate failed: {gate['confounds']}")
        outcomes = {}
        base = bool(control["row"].get("success"))
        for which in arms:
            success = bool(episodes[(which, pair)]["row"].get("success"))
            outcomes[which] = success
            bucket = "hard10" if pair[0] == 4 else "other9"
            stats[which][bucket] += int(success)
            stats[which]["original10"] += int(success and pair[1] == 0)
            if which != "control":
                label = "both" if base and success else "recovered" if success else "regressed" if base else "neither"
                stats[which][label] += 1
        pairs.append({"init_id": pair[0], "replicate_id": pair[1], "success": outcomes})
    return {"schema": "pi05_goal3_phase_screen.aggregate.v1", "complete": not errors,
            "coverage": {"expected_episodes": len(arms) * 19, "observed_episodes": len(episodes), "errors": errors},
            "arm_scores": {which: {"hard_init4": [stats[which]["hard10"], 10],
                "other_nine_inits": [stats[which]["other9"], 9],
                "original_ten_init_rep0": [stats[which]["original10"], 10],
                "paired_vs_control": {name: stats[which][name] for name in
                    ("both", "recovered", "regressed", "neither")} if which != "control" else None}
                for which in arms},
            "selection_warning": f"{len(arms)} predeclared arms are reported separately; no best-of-N pooled score.",
            "pairs": pairs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-output", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output is create-only")
    result = aggregate(args.worker_output, args.arms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
