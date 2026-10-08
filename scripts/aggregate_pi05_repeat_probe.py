#!/usr/bin/env python3
"""Strict aggregation for the frozen PI0.5 14-case paired repeat probe."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable

import build_pi05_repeat_probe as plan


SUMMARY_SCHEMA = "pi05_harness_eval.summary.v1"
OUTPUT_SCHEMA = "pi05_repeat_probe.aggregate.v1"


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> dict:
    if trials == 0:
        return {"successes": successes, "trials": 0, "estimate": None,
                "lower": None, "upper": None, "confidence": 0.95}
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return {"successes": successes, "trials": trials, "estimate": p,
            "lower": max(0.0, center - half), "upper": min(1.0, center + half),
            "confidence": 0.95}


def exact_paired_pvalue(recovered: int, regressed: int) -> float:
    discordant = recovered + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(recovered, regressed) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def discover(inputs: Iterable[Path]) -> list[Path]:
    found = set()
    for value in inputs:
        if value.is_file():
            found.add(value.resolve())
        elif value.is_dir():
            found.update(path.resolve() for path in value.rglob("summary.json"))
        else:
            raise FileNotFoundError(value)
    return sorted(found)


def load_rows(inputs: Iterable[Path]) -> list[dict]:
    rows = []
    for path in discover(inputs):
        value = json.loads(path.read_text())
        if value.get("schema") != SUMMARY_SCHEMA or not isinstance(value.get("cases"), list):
            continue
        for raw in value["cases"]:
            required = {"policy_id", "suite", "task_id", "init_id", "replicate_id", "status", "success"}
            if not isinstance(raw, dict) or not required <= set(raw):
                raise ValueError(f"malformed repeat episode row in {path}")
            row = {key: raw[key] for key in required}
            row["source_summary"] = str(path)
            rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    expected_cases = {row for values in plan.ASSIGNMENTS.values() for row in values}
    normalized = {}
    for row in rows:
        policy = row["policy_id"]
        case = (row["suite"], row["task_id"], row["init_id"])
        expected_plugin = {"libero_spatial": "spatial", "libero_object": "object",
                           "libero_goal": "goal", "libero_10": "long"}.get(row["suite"])
        if policy == "base":
            arm = "base"
        elif policy == expected_plugin:
            arm = "plugin"
        else:
            raise ValueError("episode policy_id is neither base nor its suite plugin")
        rep = row["replicate_id"]
        if case not in expected_cases or type(rep) is not int or rep not in plan.REPLICATES:
            raise ValueError("episode outside frozen 14 x 10 inventory")
        key = (arm, *case, rep)
        if key in normalized:
            raise ValueError(f"duplicate arm/case/replicate: {key}")
        if type(row["success"]) is not bool or not isinstance(row["status"], str):
            raise ValueError("invalid status/success types")
        normalized[key] = row
    expected = {(arm, *case, rep) for arm in ("base", "plugin")
                for case in expected_cases for rep in plan.REPLICATES}
    missing = sorted(expected - set(normalized))
    extra = sorted(set(normalized) - expected)
    results = []
    totals = defaultdict(int)
    for suite, task, init in sorted(expected_cases):
        marginal = {}
        paired = {"both_succeeded": 0, "recovered_plugin_only": 0,
                  "regressed_base_only": 0, "neither_succeeded": 0,
                  "error_pairs": 0, "complete_pairs": 0}
        for arm in ("base", "plugin"):
            arm_rows = [normalized.get((arm, suite, task, init, rep)) for rep in plan.REPLICATES]
            errors = sum(row is not None and row["status"] == "error" for row in arm_rows)
            valid = [row for row in arm_rows if row is not None and row["status"] != "error"]
            successes = sum(row["success"] is True for row in valid)
            marginal[arm] = {**wilson(successes, len(valid)), "errors": errors,
                             "missing": sum(row is None for row in arm_rows)}
            totals[f"{arm}_successes"] += successes
            totals[f"{arm}_valid"] += len(valid)
            totals["errors"] += errors
        for rep in plan.REPLICATES:
            base = normalized.get(("base", suite, task, init, rep))
            plugin = normalized.get(("plugin", suite, task, init, rep))
            if base is None or plugin is None or base["status"] == "error" or plugin["status"] == "error":
                paired["error_pairs"] += 1
                continue
            paired["complete_pairs"] += 1
            label = ("both_succeeded" if base["success"] and plugin["success"] else
                     "recovered_plugin_only" if plugin["success"] else
                     "regressed_base_only" if base["success"] else "neither_succeeded")
            paired[label] += 1
        pvalue = exact_paired_pvalue(paired["recovered_plugin_only"], paired["regressed_base_only"])
        paired["exact_mcnemar_two_sided_p"] = pvalue
        paired["exact_paired_binomial_two_sided_p"] = pvalue
        paired["inference_status"] = "exploratory; n=10 has low power; multiplicity unadjusted; no efficacy claim"
        results.append({"case_id": f"{suite}/{task}/{init}", "suite": suite,
                        "task_id": task, "init_id": init,
                        "marginal_wilson_95": marginal, "paired": paired})
    errors = totals["errors"]
    complete = len(normalized) == 280 and not missing and not extra and errors == 0
    return {"schema": OUTPUT_SCHEMA, "complete": complete,
            "completion_rule": "exactly 280 unique planned episodes and zero infrastructure errors",
            "coverage": {"expected": 280, "observed_unique": len(normalized),
                         "missing": len(missing), "extra": len(extra), "errors": errors,
                         "errors_retained_in_planned_denominator": True,
                         "errors_counted_as_policy_failures": False},
            "totals": dict(totals),
            "statistics_scope": "Wilson intervals are marginal only; exact McNemar/paired-binomial tests are exploratory, low-power at 10 pairs, and multiplicity-unadjusted.",
            "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("aggregate output is create-only")
    result = aggregate(load_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
