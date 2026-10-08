"""Spawn-isolated, ordered episode execution; uncaught failures keep their denominator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


_BASE_CACHE: dict[str, Any] = {}


def rollout_workers_arg(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("rollout workers must be an integer in 1..3") from exc
    if not 1 <= parsed <= 3:
        raise argparse.ArgumentTypeError("rollout workers must be an integer in 1..3")
    return parsed


def _episode_path(episodes: Path, index: int, item: dict[str, Any]) -> Path:
    return episodes / f"{index:04d}_{item['arm']}_{item['suite']}_{item['task_id']}_{item['init_id']}.json"


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_base(path: str):
    resolved = Path(path).resolve()
    cache_key = str(resolved)
    if cache_key in _BASE_CACHE:
        return _BASE_CACHE[cache_key]
    name = "_pi05_eval_worker_" + hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import evaluator module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_episode", None)):
        raise AttributeError("evaluator module does not expose run_episode")
    _BASE_CACHE[cache_key] = module
    return module


def _run_planned_episode(base, index, item, args_values, episodes, videos):
    module = _load_base(base)
    planned = dict(item)
    planned["plan_index"] = index
    row = module.run_episode(planned, SimpleNamespace(**args_values), Path(videos),
                             _episode_path(Path(episodes), index, planned))
    if not isinstance(row, dict):
        raise TypeError("run_episode must return a result dict")
    return index, row


def _uncaught_error_row(index, item, error):
    return {
        **item,
        "plan_index": index,
        "case_id": item.get("id", f"{item.get('suite')}/{item.get('task_id')}/{item.get('init_id')}"),
        "status": "error",
        "success": False,
        "error_type": type(error).__name__,
        "error": str(error)[:1000],
        "failure_scope": "uncaught_rollout_worker_exception_no_retry",
    }


def run_episodes_ordered(base, plan, args, episodes, videos, workers):
    """Run spawn-isolated episodes and return rows in exact frozen-plan order."""
    worker_count = rollout_workers_arg(workers)
    base_path = str(Path(base).resolve())
    episode_root, video_root = Path(episodes).resolve(), Path(videos).resolve()
    args_values = dict(args) if isinstance(args, dict) else dict(vars(args))
    planned = [dict(item) for item in plan]
    rows: list[dict[str, Any] | None] = [None] * len(planned)
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(_run_planned_episode, base_path, index, item, args_values,
                        str(episode_root), str(video_root)): (index, item)
            for index, item in enumerate(planned)
        }
        for future in as_completed(futures):
            index, item = futures[future]
            try:
                returned_index, row = future.result()
                if returned_index != index:
                    raise ValueError("rollout worker returned the wrong plan index")
            except BaseException as error:
                row = _uncaught_error_row(index, item, error)
                _atomic_json(_episode_path(episode_root, index, item), row)
            rows[index] = row
    if any(row is None for row in rows):
        raise AssertionError("every planned rollout must produce a row")
    return rows
