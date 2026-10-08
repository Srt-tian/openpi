#!/usr/bin/env python3
"""Pure-data contract for the five-GPU PI0.5 LIBERO extension."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PLUGIN_IDS = dict(zip(SUITES, ("spatial", "object", "goal", "long"), strict=True))
REUSED_TASKS = {
    "libero_spatial": (0, 9),
    "libero_object": (0, 4),
    "libero_goal": (0, 3),
    "libero_10": (0, 8),
}
WORKER_TASK_COUNTS = (7, 7, 6, 6, 6)
CASE_KEYS = {"suite", "task_id", "init_id", "joint_task_number", "policy_seed", "id"}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def case_for(suite: str, task_id: int, init_id: int, seed: int = 7) -> dict[str, Any]:
    if suite not in SUITES or isinstance(task_id, bool) or not 0 <= task_id < 10:
        raise ValueError("case has invalid LIBERO suite/task")
    if isinstance(init_id, bool) or not 0 <= init_id < 10:
        raise ValueError("extension cases require official init_id in 0..9")
    joint = SUITES.index(suite) * 10 + task_id
    return {
        "suite": suite,
        "task_id": task_id,
        "init_id": init_id,
        "joint_task_number": joint,
        "policy_seed": seed + joint * 50 + init_id,
        "id": f"{suite}/{task_id}/{init_id}",
    }


def validate_case(value: Any, seed: int = 7) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CASE_KEYS:
        raise ValueError(f"case must contain exactly {sorted(CASE_KEYS)}")
    expected = case_for(value.get("suite"), value.get("task_id"), value.get("init_id"), seed)
    if value != expected:
        raise ValueError(f"case fields do not match fixed derivation for {expected['id']}")
    return expected


def reused_case_ids() -> list[str]:
    return [
        case_for(suite, task_id, init_id)["id"]
        for suite in SUITES
        for task_id in REUSED_TASKS[suite]
        for init_id in range(10)
    ]


def remaining_tasks() -> list[tuple[str, int]]:
    return [
        (suite, task_id)
        for suite in SUITES
        for task_id in range(10)
        if task_id not in REUSED_TASKS[suite]
    ]


def worker_cases(worker_id: int, seed: int = 7) -> list[dict[str, Any]]:
    if isinstance(worker_id, bool) or not 0 <= worker_id < 5:
        raise ValueError("worker_id must be in 0..4")
    tasks = [task for index, task in enumerate(remaining_tasks()) if index % 5 == worker_id]
    return [case_for(suite, task_id, init_id, seed) for suite, task_id in tasks for init_id in range(10)]


def episode_files(root: Path) -> list[Path]:
    paths = sorted(root.resolve().rglob("episodes/*.json"))
    if not paths:
        raise ValueError("reused source has no episode JSON files")
    return paths


def source_checksum(root: Path, paths: list[Path] | None = None) -> str:
    root = root.resolve()
    digest = hashlib.sha256()
    for path in paths or episode_files(root):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def load_episode_rows(root: Path) -> tuple[list[dict[str, Any]], str]:
    paths = episode_files(root)
    rows = [json.loads(path.read_text()) for path in paths]
    return rows, source_checksum(root, paths)


def validate_reused_rows(rows: list[dict[str, Any]]) -> None:
    expected_ids = set(reused_case_ids())
    seen: set[tuple[str, str]] = set()
    for row in rows:
        case = case_for(row.get("suite"), row.get("task_id"), row.get("init_id"))
        if row.get("case_id") != case["id"] or row.get("joint_task_number") != case["joint_task_number"]:
            raise ValueError("reused episode identity does not match the frozen case")
        if row.get("policy_seed") != case["policy_seed"]:
            raise ValueError("reused episode policy seed does not match the frozen case")
        arm = row.get("arm")
        expected_policy = "base" if arm == "base" else PLUGIN_IDS[case["suite"]] if arm == "plugin" else None
        if row.get("policy_id") != expected_policy:
            raise ValueError("reused episode arm/policy is invalid")
        if not isinstance(row.get("success"), bool) or row.get("status") not in ("success", "failure"):
            raise ValueError("reused pilot02 must contain completed, non-error outcomes")
        if (row["status"] == "success") != row["success"]:
            raise ValueError("reused episode status/success disagree")
        key = (case["id"], arm)
        if key in seen:
            raise ValueError("reused source contains a duplicate case arm")
        seen.add(key)
    expected = {(case_id, arm) for case_id in expected_ids for arm in ("base", "plugin")}
    if seen != expected:
        raise ValueError("reused source must contain both arms for all 80 prior cases")
    successes = {
        arm: sum(row["success"] is True for row in rows if row["arm"] == arm)
        for arm in ("base", "plugin")
    }
    if successes != {"base": 74, "plugin": 77}:
        raise ValueError("reused source does not match pilot02's frozen 74/80 and 77/80 outcomes")


def build_manifest(reused_source: Path, seed: int = 7) -> dict[str, Any]:
    if seed != 7:
        raise ValueError("extension is pinned to seed 7 to match pilot02")
    rows, checksum = load_episode_rows(reused_source)
    validate_reused_rows(rows)
    workers = {str(i): {"task_count": WORKER_TASK_COUNTS[i], "cases": worker_cases(i, seed)} for i in range(5)}
    manifest = {
        "schema": "pi05_libero_fivegpu_plan.v1",
        "seed": seed,
        "init_ids": list(range(10)),
        "assignment": "remaining_joint_task_order_round_robin_mod_5",
        "worker_task_counts": list(WORKER_TASK_COUNTS),
        "reused_tasks": {suite: list(REUSED_TASKS[suite]) for suite in SUITES},
        "reused_case_ids": reused_case_ids(),
        "reused_source": str(reused_source.resolve()),
        "reused_source_checksum": checksum,
        "reused_episode_count": len(rows),
        "workers": workers,
        "total_pairs": 400,
        "new_pairs": 320,
        "reused_pairs": 80,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("plan manifest must be an object")
    unsigned = dict(value)
    claimed_hash = unsigned.pop("manifest_sha256", None)
    if claimed_hash != canonical_hash(unsigned):
        raise ValueError("plan manifest checksum mismatch")
    expected_keys = {
        "schema", "seed", "init_ids", "assignment", "worker_task_counts", "reused_tasks",
        "reused_case_ids", "reused_source", "reused_source_checksum", "reused_episode_count",
        "workers", "total_pairs", "new_pairs", "reused_pairs", "manifest_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("plan manifest contains missing or unknown fields")
    if value.get("schema") != "pi05_libero_fivegpu_plan.v1" or value.get("seed") != 7:
        raise ValueError("unsupported plan schema or seed")
    if (value.get("init_ids") != list(range(10))
            or value.get("assignment") != "remaining_joint_task_order_round_robin_mod_5"
            or value.get("reused_tasks") != {suite: list(REUSED_TASKS[suite]) for suite in SUITES}
            or value.get("reused_episode_count") != 160
            or value.get("total_pairs") != 400
            or value.get("new_pairs") != 320
            or value.get("reused_pairs") != 80):
        raise ValueError("plan constants differ from the frozen extension protocol")
    if value.get("reused_case_ids") != reused_case_ids():
        raise ValueError("plan does not reuse the exact complete pilot02 case set")
    if value.get("worker_task_counts") != list(WORKER_TASK_COUNTS):
        raise ValueError("worker task counts must be 7/7/6/6/6")
    all_ids: set[str] = set()
    for worker_id in range(5):
        entry = value.get("workers", {}).get(str(worker_id), {})
        cases = [validate_case(case, 7) for case in entry.get("cases", [])]
        if cases != worker_cases(worker_id, 7) or entry.get("task_count") != WORKER_TASK_COUNTS[worker_id]:
            raise ValueError(f"worker {worker_id} does not match fixed round-robin assignment")
        ids = {case["id"] for case in cases}
        if len(ids) != len(cases) or all_ids.intersection(ids):
            raise ValueError("new case assignments contain duplicates")
        all_ids.update(ids)
    if len(all_ids) != 320 or all_ids.intersection(value["reused_case_ids"]):
        raise ValueError("new/reused case coverage is not an exact 400-case partition")
    checksum = value.get("reused_source_checksum")
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)):
        raise ValueError("reused source checksum is invalid")
    if not isinstance(value.get("reused_source"), str) or not value["reused_source"]:
        raise ValueError("reused source path is invalid")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    return validate_manifest(json.loads(path.read_text()))


def case_rows_from_outputs(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(root.resolve().rglob("episodes/*.json"))]


def validate_final_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        case = case_for(row.get("suite"), row.get("task_id"), row.get("init_id"))
        if row.get("case_id") != case["id"]:
            raise ValueError("aggregate row case identity mismatch")
        arm = row.get("arm")
        pair = paired.setdefault(case["id"], {})
        if arm not in ("base", "plugin") or arm in pair:
            raise ValueError("aggregate contains invalid or duplicate arm")
        pair[arm] = row
    expected_ids = {case_for(suite, task, init_id)["id"] for suite in SUITES for task in range(10) for init_id in range(10)}
    if set(paired) != expected_ids or any(set(pair) != {"base", "plugin"} for pair in paired.values()):
        raise ValueError("aggregate must contain exactly 400 complete pairs")
    return paired
