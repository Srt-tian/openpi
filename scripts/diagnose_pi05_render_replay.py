#!/usr/bin/env python3
"""Diagnose LIBERO replay determinism with frozen actions and no policy service."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")
EXPECTED_CASES = {
    ("libero_10", 8, 0, 2): 420,
    ("libero_10", 8, 1, 0): 160,
}
ENVIRONMENT_SEED = 7
SETTLING_STEPS = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--eval-helpers", type=Path, required=True)
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 4), required=True)
    return parser.parse_args(argv)


def configure_threads(value: int, *, require_fresh_numpy: bool = True) -> dict[str, str]:
    if type(value) is not int or value not in (1, 4):
        raise ValueError("thread condition must be exactly 1 or 4")
    if require_fresh_numpy and "numpy" in sys.modules:
        raise RuntimeError("thread variables must be set before importing numpy")
    result = {name: str(value) for name in THREAD_VARS}
    os.environ.update(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_helpers(path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("_pi05_render_replay_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    for name in ("make_environment", "load_official_init_states", "observation_payload"):
        if not hasattr(module, name):
            raise ImportError(f"eval helpers missing {name}")
    return module


def _hash_array(value: Any, np) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {"sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "shape": list(array.shape), "dtype": str(array.dtype)}


def _state8(raw: dict[str, Any], helpers, np):
    return np.asarray(helpers.observation_payload(
        raw, "diagnostic-only", "base", 0)["observation/state"], dtype=np.float64)


def _sim(env):
    direct = getattr(env, "sim", None)
    if direct is not None:
        return direct
    inner = getattr(env, "env", None)
    value = getattr(inner, "sim", None)
    if value is None:
        raise AttributeError("LIBERO environment does not expose sim")
    return value


def _sim_state(sim, np):
    value = sim.get_state()
    if hasattr(value, "flatten"):
        value = value.flatten()
    return np.asarray(value, dtype=np.float64).reshape(-1).copy()


def _warmstart(sim, np):
    value = getattr(getattr(sim, "data", None), "qacc_warmstart", None)
    return None if value is None else np.asarray(value, dtype=np.float64).reshape(-1).copy()


def load_action_case(path: Path, np) -> dict[str, Any]:
    value = json.loads(path.read_text())
    case = value.get("case", {})
    key = (case.get("suite"), case.get("task_id"), case.get("init_id"),
           case.get("replicate_id", 0))
    if key not in EXPECTED_CASES or case.get("policy_id") != "base":
        raise ValueError("episode is not one of the two frozen base-control diagnostics")
    control = value.get("pi05_control")
    if control != {"kind": "response_probe_v1", "enabled": False}:
        raise ValueError("source episode must be the unmodified control arm")
    actions = np.asarray(value.get("runner", {}).get("environment_actions"), dtype=np.float64)
    horizon = EXPECTED_CASES[key]
    if (actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < horizon
            or not np.isfinite(actions[:horizon]).all()):
        raise ValueError("source episode lacks the frozen finite 7D action prefix")
    ambient = 7 + 38 * 50 + key[2]
    if case.get("ambient_seed") != ambient:
        raise ValueError("source ambient seed differs from frozen protocol")
    return {"path": path.resolve(), "sha256": sha256_file(path), "key": key,
            "case": case, "horizon": horizon, "actions": actions[:horizon].copy()}


def capture_step(raw, env, helpers, np, step: int) -> dict[str, Any]:
    state = _state8(raw, helpers, np)
    sim = _sim(env)
    sim_state = _sim_state(sim, np)
    warmstart = _warmstart(sim, np)
    row = {"step": step, "state8": _hash_array(state, np),
           "state8_value": state.tolist(), "sim_state": _hash_array(sim_state, np),
           "sim_state_value": sim_state.tolist(),
           "qacc_warmstart": None if warmstart is None else _hash_array(warmstart, np),
           "qacc_warmstart_value": None if warmstart is None else warmstart.tolist()}
    if step % 5 == 0:
        payload = helpers.observation_payload(raw, "diagnostic-only", "base", 0)
        row["images"] = {
            "policy_agent224": _hash_array(payload["observation/image"], np),
            "policy_wrist224": _hash_array(payload["observation/wrist_image"], np),
            "raw_wrist256": _hash_array(raw["robot0_eye_in_hand_image"], np),
        }
        row["image_values"] = {
            "policy_agent224": np.asarray(payload["observation/image"]).copy(),
            "policy_wrist224": np.asarray(payload["observation/wrist_image"]).copy(),
            "raw_wrist256": np.asarray(raw["robot0_eye_in_hand_image"]).copy(),
        }
    return row


def replay_once(source: dict[str, Any], helpers, np) -> dict[str, Any]:
    case, env = source["case"], None
    random.seed(case["ambient_seed"])
    np.random.seed(case["ambient_seed"])
    records = []
    try:
        env, _, task = helpers.make_environment(case["suite"], case["task_id"], ENVIRONMENT_SEED)
        initial_states, asset = helpers.load_official_init_states(task)
        settling = getattr(helpers, "SETTLING_STEPS", SETTLING_STEPS)
        dummy = np.asarray(getattr(helpers, "DUMMY_ACTION", DUMMY_ACTION), dtype=np.float64)
        if settling != SETTLING_STEPS or not np.array_equal(dummy, DUMMY_ACTION):
            raise ValueError("eval helper official settling contract changed")
        env.reset()
        raw = env.set_init_state(initial_states[case["init_id"]])
        for _ in range(settling):
            raw, _, _, _ = env.step(dummy.tolist())
        for step, action in enumerate(source["actions"]):
            records.append(capture_step(raw, env, helpers, np, step))
            raw, _, _, _ = env.step(action.tolist())
        return {"records": records, "init_asset": asset,
                "settling_steps": SETTLING_STEPS, "environment_seed": ENVIRONMENT_SEED}
    finally:
        if env is not None:
            env.close()


def _numeric_diff(left: Any, right: Any, np) -> dict[str, Any]:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    delta = a - b
    return {"max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
            "equal": bool(np.array_equal(a, b))}


def _image_diff(left: Any, right: Any, np) -> dict[str, Any]:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return {"shape_equal": False, "left_shape": list(a.shape), "right_shape": list(b.shape)}
    delta = a.astype(np.float64) - b.astype(np.float64)
    pixels = np.any(a != b, axis=-1) if a.ndim >= 3 else a != b
    return {"shape_equal": True, "different_pixels": int(np.count_nonzero(pixels)),
            "different_values": int(np.count_nonzero(a != b)),
            "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0}


def compare_replays(first: dict[str, Any], second: dict[str, Any], actions, np):
    a, b = first["records"], second["records"]
    if len(a) != len(b):
        raise ValueError("replay lengths differ")
    first_field = {name: None for name in
                   ("state8", "sim_state", "qacc_warmstart",
                    "policy_agent224", "policy_wrist224", "raw_wrist256")}
    image_detail = None
    for x, y in zip(a, b):
        step = x["step"]
        for name in ("state8", "sim_state", "qacc_warmstart"):
            if first_field[name] is None and x[name] != y[name]:
                left = x.get(name + "_value"); right = y.get(name + "_value")
                first_field[name] = {"step": step,
                    "numeric": None if left is None or right is None else _numeric_diff(left, right, np)}
        if "images" in x:
            for name in ("policy_agent224", "policy_wrist224", "raw_wrist256"):
                if first_field[name] is None and x["images"][name] != y["images"][name]:
                    detail = _image_diff(x["image_values"][name], y["image_values"][name], np)
                    first_field[name] = {"step": step, "image": detail}
                    if image_detail is None:
                        image_detail = {"step": step, "field": name, "metrics": detail,
                            "left": x["image_values"][name], "right": y["image_values"][name],
                            "same_action": True, "action7": actions[step].tolist(),
                            "action_sha256": _hash_array(actions[step], np)["sha256"],
                            "state8": _numeric_diff(x["state8_value"], y["state8_value"], np),
                            "sim_state": _numeric_diff(x["sim_state_value"], y["sim_state_value"], np)}
    steps = [value["step"] for value in first_field.values() if value is not None]
    return {"first_difference_by_field": first_field,
            "first_difference_step": min(steps) if steps else None,
            "first_image_difference": image_detail}


def _json_safe_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    result = dict(comparison)
    detail = result.get("first_image_difference")
    if detail is not None:
        detail = dict(detail); detail.pop("left", None); detail.pop("right", None)
        result["first_image_difference"] = detail
    return result


def compact_hash_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain per-step hashes while excluding simulator/image arrays."""
    result = []
    for row in records:
        compact = {name: row[name] for name in
                   ("step", "state8", "sim_state", "qacc_warmstart")}
        if "images" in row:
            compact["images"] = row["images"]
        result.append(compact)
    return result


def save_first_difference_pngs(comparison, directory: Path, stem: str):
    detail = comparison.get("first_image_difference")
    if detail is None:
        return []
    import imageio.v2 as imageio
    paths = []
    for label, value in (("A", detail["left"]), ("B", detail["right"])):
        path = directory / f"{stem}_{detail['field']}_step{detail['step']:03d}_{label}.png"
        imageio.imwrite(path, value)
        paths.append(path.name)
    return paths


def execute(args) -> dict[str, Any]:
    thread_env = configure_threads(args.threads)
    import numpy as np
    helpers = import_helpers(args.eval_helpers)
    if args.output.exists():
        raise FileExistsError("diagnostic output is create-only")
    if len(args.episode) != 2:
        raise ValueError("exactly two frozen control episodes are required")
    sources = [load_action_case(path, np) for path in args.episode]
    if {source["key"] for source in sources} != set(EXPECTED_CASES):
        raise ValueError("diagnostic episodes must cover the exact two frozen cases")
    args.output.mkdir(parents=True)
    rows = []
    for source in sorted(sources, key=lambda item: item["key"]):
        first = replay_once(source, helpers, np)
        second = replay_once(source, helpers, np)
        if first["init_asset"] != second["init_asset"]:
            raise ValueError("official init asset changed between replays")
        comparison = compare_replays(first, second, source["actions"], np)
        stem = f"init{source['key'][2]}_rep{source['key'][3]}_threads{args.threads}"
        pngs = save_first_difference_pngs(comparison, args.output, stem)
        rows.append({"case": source["case"], "horizon": source["horizon"],
                     "source_episode": str(source["path"]),
                     "source_episode_sha256": source["sha256"],
                     "init_asset": first["init_asset"],
                     "replay_hash_records": {
                         "A": compact_hash_records(first["records"]),
                         "B": compact_hash_records(second["records"]),
                     },
                     "comparison": _json_safe_comparison(comparison), "pngs": pngs})
    result = {"schema": "pi05_render_replay_diagnostic.v1",
              "label": "diagnostic_action_replay_not_policy_eval",
              "excluded_from_scores": True, "policy_service_loaded": False,
              "best_of_n": False, "condition_threads": args.threads,
              "thread_environment": thread_env, "replays_per_case": 2,
              "eval_helpers": str(args.eval_helpers.resolve()),
              "eval_helpers_sha256": sha256_file(args.eval_helpers), "cases": rows}
    (args.output / "diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
