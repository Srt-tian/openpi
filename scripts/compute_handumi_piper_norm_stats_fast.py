"""Compute Piper joint-space normalization stats without decoding videos."""

from __future__ import annotations

import argparse
import collections
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize
from openpi.training import config as training_config


JOINT_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False])


def _load_episode(path: pathlib.Path, *, action_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["frame_index", "observation.state", "action"]).sort_by("frame_index")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    absolute_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.shape != absolute_action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(f"Unexpected state/action shapes in {path}: {state.shape}, {absolute_action.shape}")

    frame_count = len(state)
    future_indices = np.minimum(
        np.arange(frame_count)[:, None] + np.arange(action_horizon)[None, :], frame_count - 1
    )
    actions = absolute_action[future_indices]
    actions[..., JOINT_MASK] -= state[:, None, JOINT_MASK]
    return state, actions.astype(np.float32, copy=False)


def _full_batches(paths: list[pathlib.Path], *, action_horizon: int, batch_size: int):
    state_buffer: collections.deque[np.ndarray] = collections.deque()
    action_buffer: collections.deque[np.ndarray] = collections.deque()
    buffered = 0
    for path in paths:
        states, actions = _load_episode(path, action_horizon=action_horizon)
        state_buffer.append(states)
        action_buffer.append(actions)
        buffered += len(states)
        while buffered >= batch_size:
            remaining = batch_size
            state_parts = []
            action_parts = []
            while remaining:
                take = min(remaining, len(state_buffer[0]))
                state_parts.append(state_buffer[0][:take])
                action_parts.append(action_buffer[0][:take])
                if take == len(state_buffer[0]):
                    state_buffer.popleft()
                    action_buffer.popleft()
                else:
                    state_buffer[0] = state_buffer[0][take:]
                    action_buffer[0] = action_buffer[0][take:]
                buffered -= take
                remaining -= take
            yield np.concatenate(state_parts), np.concatenate(action_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_handumi_tblock_piper_joint_lora")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = training_config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    root = pathlib.Path(data_config.root)
    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if len(paths) != 194:
        raise ValueError(f"Expected 194 episode parquet files, found {len(paths)}")

    output_dir = config.assets_dirs / data_config.repo_id
    output_file = output_dir / "norm_stats.json"
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_file}")

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    frames = batches = 0
    start = time.monotonic()
    for states, actions in _full_batches(
        paths,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
    ):
        stats["state"].update(states)
        stats["actions"].update(actions)
        frames += len(states)
        batches += 1

    normalize.save(output_dir, {key: value.get_statistics() for key, value in stats.items()})
    print(
        f"HANDUMI_PIPER_NORM_OK episodes={len(paths)} frames={frames} batches={batches} "
        f"dropped={51757 - frames} elapsed_s={time.monotonic() - start:.2f} output={output_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
