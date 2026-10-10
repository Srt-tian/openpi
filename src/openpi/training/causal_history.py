"""Causal physical-action histories layered on the verified LIBERO v3 reader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def normalize_physical_actions(actions: np.ndarray, action_stats: Any) -> np.ndarray:
    """Apply the base policy's quantile transform to raw environment 7D actions."""
    value = np.asarray(actions, dtype=np.float32)
    if value.shape[-1:] != (7,) or not np.isfinite(value).all():
        raise ValueError("physical actions must be finite with trailing dimension 7")
    q01 = np.asarray(action_stats.q01, dtype=np.float32)[..., :7]
    q99 = np.asarray(action_stats.q99, dtype=np.float32)[..., :7]
    if q01.shape != (7,) or q99.shape != (7,) or not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError("action quantiles must be finite 7D arrays")
    # Deliberately identical to transforms.Normalize._normalize_quantile: no clip
    # and no special treatment of the gripper component.
    return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


class CausalHistoryDataset:
    """Adds bounded expert history without changing the parent's current target."""

    def __init__(self, dataset: Any, action_stats: Any, max_history: int):
        if type(max_history) is not int or not 0 <= max_history <= 520:
            raise ValueError("max_history must be an integer in 0..520")
        self.dataset, self.action_stats, self.max_history = dataset, action_stats, max_history

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pos = int(np.searchsorted(self.dataset._ends, index, side="right"))
        episode = self.dataset.episodes[pos]
        start = 0 if pos == 0 else int(self.dataset._ends[pos - 1])
        frame = index - start
        table = self.dataset._load_episode(episode)
        if frame < 0 or frame >= episode.length:
            raise ValueError("frame is outside its episode")
        first = max(0, frame - self.max_history)
        command = table.action[first:frame]
        pre = table.state[first:frame]
        post = table.state[first + 1 : frame + 1]
        if not (len(command) == len(pre) == len(post)):
            raise AssertionError("causal history alignment failed")
        if pre.shape != (len(command), 8) or post.shape != (len(command), 8):
            raise ValueError("historical states must be aligned 8D rows")
        if not np.isfinite(pre).all() or not np.isfinite(post).all():
            raise ValueError("historical states must be finite")
        item = dict(self.dataset[index])
        item["causal_history"] = {
            "pre_state": pre.copy(),
            "executed_action_raw": command.copy(),
            "executed_action_normalized": normalize_physical_actions(command, self.action_stats),
            "post_state": post.copy(),
            "mask": np.ones(len(command), dtype=np.bool_),
        }
        item["causal_history_provenance"] = {
            "source": "expert_demonstration",
            "episode_index": episode.index,
            "current_frame_index": frame,
            "first_history_frame_index": first,
            "available_past": frame,
            "returned": len(command),
            "max_history": self.max_history,
            "truncated": first > 0,
            "omitted_past": first,
            "burn_in_executed": 0,
            "history_state_warmed": False,
            "runtime_source_differs": "runtime records policy actions actually executed in the environment",
        }
        return item


@dataclass(frozen=True)
class ExecutedTransition:
    step: int
    pre_state: np.ndarray
    action: np.ndarray
    post_state: np.ndarray


class RuntimeExecutedHistory:
    """Episode-local callback sink; predicted but unexecuted actions never enter it."""

    def __init__(self, max_history: int):
        if type(max_history) is not int or not 0 <= max_history <= 520:
            raise ValueError("max_history must be an integer in 0..520")
        self.max_history, self._episode, self._rows = max_history, None, []
        self._next_step = 0

    def begin_episode(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("nonempty episode_id required")
        self._episode, self._rows, self._next_step = episode_id, [], 0

    def record_executed(self, episode_id: str, step: int, pre_state, action, post_state) -> None:
        if episode_id != self._episode:
            raise ValueError("cross-episode history event")
        if type(step) is not int or step != self._next_step:
            raise ValueError("executed steps must be unique and contiguous")
        pre, act, post = map(lambda x: np.asarray(x, dtype=np.float32), (pre_state, action, post_state))
        if pre.shape != (8,) or post.shape != (8,) or act.shape != (7,):
            raise ValueError("transition must be state8/action7/state8")
        if not all(np.isfinite(x).all() for x in (pre, act, post)):
            raise ValueError("transition must be finite")
        self._next_step += 1
        if self.max_history:
            self._rows.append(ExecutedTransition(step, pre.copy(), act.copy(), post.copy()))
            if len(self._rows) > self.max_history:
                self._rows.pop(0)

    def last(self) -> tuple[ExecutedTransition, ...]:
        return tuple(ExecutedTransition(x.step, x.pre_state.copy(), x.action.copy(), x.post_state.copy()) for x in self._rows)
