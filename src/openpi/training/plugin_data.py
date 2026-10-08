"""Small, dependency-light LeRobot v3 reader for the four LIBERO suites.

This intentionally does not use ``lerobot.common``: the OpenPI revision used by
plugin training pins a reader which predates the v3 on-disk format.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np


VIDEO_KEYS = ("observation.images.image", "observation.images.image2")
SUITE_TASKS = {
    "libero_10": tuple(range(0, 10)),
    "libero_goal": tuple(range(10, 20)),
    "libero_object": tuple(range(20, 30)),
    "libero_spatial": tuple(range(30, 40)),
}
EXPECTED_TASKS = (
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put the yellow and white mug in the microwave and close it",
    "turn on the stove and put the moka pot on it",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both moka pots on the stove",
    "put both the cream cheese box and the butter in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "pick up the book and place it in the back compartment of the caddy",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
    "open the top drawer and put the bowl inside",
    "put the cream cheese in the bowl",
    "put the wine bottle on top of the cabinet",
    "push the plate to the front of the stove",
    "turn on the stove",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "open the middle drawer of the cabinet",
    "pick up the orange juice and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the alphabet soup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
)


@dataclass(frozen=True)
class Episode:
    index: int
    task: int
    length: int
    dataset_from: int
    dataset_to: int
    data_chunk: int
    data_file: int
    # key -> (chunk, file, from_timestamp, to_timestamp)
    videos: dict[str, tuple[int, int, float, float]]


@dataclass(frozen=True)
class _EpisodeTable:
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    frame_index: np.ndarray
    index: np.ndarray
    task_index: np.ndarray


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise ImportError("LIBERO v3 reading requires pyarrow") from exc
    return pq


def _match_timestamp(target: float, timestamps: Iterable[float], tolerance: float) -> int:
    values = list(timestamps)
    if not values:
        raise ValueError("video decode returned no timestamped frames")
    index = min(range(len(values)), key=lambda i: abs(values[i] - target))
    error = abs(values[index] - target)
    if error > tolerance:
        raise ValueError(
            f"nearest decoded frame is {error:.6f}s from {target:.6f}s "
            f"(tolerance {tolerance:.6f}s)"
        )
    return index


class _VideoDecoder:
    """Per-process PyAV handles plus a bounded, forward-prefetched frame LRU."""

    def __init__(self, max_frames: int = 64, max_open: int = 4, prefetch: int = 8):
        self.max_frames = max_frames
        self.max_open = max_open
        self.prefetch = prefetch
        self._pid = os.getpid()
        self._frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._open: OrderedDict[str, tuple[Any, Any]] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_frames"] = OrderedDict()
        state["_open"] = OrderedDict()
        state["_pid"] = None
        return state

    def _reset_after_fork(self) -> None:
        if self._pid == os.getpid():
            return
        for container, _ in self._open.values():
            container.close()
        self._frames.clear()
        self._open.clear()
        self._pid = os.getpid()

    def _container(self, path: Path):
        self._reset_after_fork()
        key = str(path)
        if key in self._open:
            value = self._open.pop(key)
            self._open[key] = value
            return value
        try:
            import av
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise ImportError("LIBERO video decoding requires PyAV") from exc
        container = av.open(key)
        stream = container.streams.video[0]
        self._open[key] = (container, stream)
        while len(self._open) > self.max_open:
            old, _ = self._open.popitem(last=False)[1]
            old.close()
        return container, stream

    @staticmethod
    def _key(path: Path, timestamp: float) -> tuple[str, int]:
        return str(path), round(timestamp * 1_000_000)

    def get(self, path: Path, timestamp: float, fps: float, upper: float) -> np.ndarray:
        key = self._key(path, timestamp)
        if key in self._frames:
            value = self._frames.pop(key)
            self._frames[key] = value
            return value
        container, stream = self._container(path)
        tolerance = 0.51 / fps
        seek_time = max(0.0, timestamp - 2.0)
        container.seek(int(seek_time / float(stream.time_base)), stream=stream, backward=True)
        # Episode end is exclusive. Relative parquet timestamps may be float32:
        # an exact end such as 1023.0 can become 1022.9999996 after adding the
        # video offset. A fixed epsilon then admits a nonexistent end frame.
        # Quantize only optional prefetch admission to the frame grid. Always
        # decode the requested frame under the unchanged strict PTS tolerance.
        wanted = [timestamp]
        wanted.extend(
            value for i in range(1, self.prefetch)
            if (value := timestamp + i / fps) < upper - 0.5 / fps
        )
        found_times: list[float] = []
        found_frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            if frame_time < timestamp - tolerance:
                continue
            if frame_time > wanted[-1] + tolerance:
                break
            found_times.append(frame_time)
            found_frames.append(frame.to_ndarray(format="rgb24"))
        for wanted_time in wanted:
            nearest = _match_timestamp(wanted_time, found_times, tolerance)
            cache_key = self._key(path, wanted_time)
            self._frames[cache_key] = found_frames[nearest]
            self._frames.move_to_end(cache_key)
        while len(self._frames) > self.max_frames:
            self._frames.popitem(last=False)
        return self._frames[key]


def _validate_tasks(rows: list[dict[str, Any]]) -> dict[int, str]:
    mapping = {int(row["task_index"]): str(row["__index_level_0__"]) for row in rows}
    expected = dict(enumerate(EXPECTED_TASKS))
    if mapping != expected:
        missing = sorted(set(expected) - set(mapping))
        extra = sorted(set(mapping) - set(expected))
        changed = [i for i in sorted(set(mapping) & set(expected)) if mapping[i] != expected[i]]
        raise ValueError(f"unexpected LIBERO task mapping: missing={missing}, extra={extra}, changed={changed}")
    return mapping


def _validate_info(info: dict[str, Any]) -> float:
    if info.get("codebase_version") != "v3.0" or info.get("total_tasks") != 40:
        raise ValueError("expected the verified 40-task LeRobot v3.0 LIBERO dataset")
    expected = {
        "observation.images.image": ("video", [256, 256, 3]),
        "observation.images.image2": ("video", [256, 256, 3]),
        "observation.state": ("float32", [8]),
        "action": ("float32", [7]),
    }
    features = info.get("features", {})
    for key, (dtype, shape) in expected.items():
        feature = features.get(key, {})
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"unexpected feature metadata for {key}: {feature}")
    fps = float(info.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"invalid dataset fps: {fps}")
    return fps


def _read_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import json

    pq = _pyarrow_parquet()
    info = json.loads((root / "meta/info.json").read_text())
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    episode_files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise FileNotFoundError("no episode metadata parquet files")
    episodes = [row for path in episode_files for row in pq.read_table(path).to_pylist()]
    return info, tasks, episodes


def _episodes_from_rows(rows: list[dict[str, Any]]) -> list[Episode]:
    episodes = []
    for row in rows:
        length = int(row["length"])
        dataset_from = int(row["dataset_from_index"])
        dataset_to = int(row["dataset_to_index"])
        if length <= 0 or dataset_to - dataset_from != length:
            raise ValueError(f"episode {row['episode_index']}: inconsistent frame bounds")
        videos = {
            key: (
                int(row[f"videos/{key}/chunk_index"]),
                int(row[f"videos/{key}/file_index"]),
                float(row[f"videos/{key}/from_timestamp"]),
                float(row[f"videos/{key}/to_timestamp"]),
            )
            for key in VIDEO_KEYS
        }
        episodes.append(Episode(
            index=int(row["episode_index"]), task=-1, length=length,
            dataset_from=dataset_from, dataset_to=dataset_to,
            data_chunk=int(row["data/chunk_index"]), data_file=int(row["data/file_index"]), videos=videos,
        ))
    if len({episode.index for episode in episodes}) != len(episodes):
        raise ValueError("duplicate episode_index in metadata")
    return sorted(episodes, key=lambda episode: episode.index)


class LiberoV3Dataset:
    """Map-style dataset whose indices sample frames uniformly."""

    def __init__(self, root: Path, episodes: list[Episode], prompts: dict[int, str], fps: float, horizon: int):
        self.root = Path(root)
        self.episodes = tuple(episodes)
        self.prompts = prompts
        self.fps = fps
        self.horizon = horizon
        self._ends = np.cumsum([episode.length for episode in episodes], dtype=np.int64)
        self._tables: OrderedDict[int, _EpisodeTable] = OrderedDict()
        self._videos = _VideoDecoder()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tables"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return int(self._ends[-1]) if len(self._ends) else 0

    def diagnostic_indices(self, batch_size: int = 32) -> list[int]:
        """Choose deterministic, frame-spread indices, round-robin across tasks."""
        tasks = sorted({episode.task for episode in self.episodes})
        if batch_size < len(tasks):
            raise ValueError(f"batch_size {batch_size} cannot cover {len(tasks)} tasks")
        intervals: dict[int, list[tuple[int, int]]] = {task: [] for task in tasks}
        previous = 0
        for end, episode in zip(self._ends.tolist(), self.episodes, strict=True):
            intervals[episode.task].append((previous, int(end) - previous))
            previous = int(end)
        quotas = {
            task: batch_size // len(tasks) + (position < batch_size % len(tasks))
            for position, task in enumerate(tasks)
        }
        per_task: dict[int, list[int]] = {}
        for task in tasks:
            total = sum(length for _, length in intervals[task])
            offsets = [
                min(total - 1, int((sample + 0.5) * total / quotas[task]))
                for sample in range(quotas[task])
            ]
            selected = []
            for offset in offsets:
                for start, length in intervals[task]:
                    if offset < length:
                        selected.append(start + offset)
                        break
                    offset -= length
            per_task[task] = selected
        return [
            per_task[task][round_index]
            for round_index in range(max(quotas.values()))
            for task in tasks
            if round_index < len(per_task[task])
        ]

    def _load_episode(self, episode: Episode) -> _EpisodeTable:
        if episode.index in self._tables:
            table = self._tables.pop(episode.index)
            self._tables[episode.index] = table
            return table
        pq = _pyarrow_parquet()
        path = self.root / f"data/chunk-{episode.data_chunk:03d}/file-{episode.data_file:03d}.parquet"
        columns = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
        raw = pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode.index)])
        values = {name: raw[name].to_pylist() for name in columns}
        if len(values["episode_index"]) != episode.length or set(values["episode_index"]) != {episode.index}:
            raise ValueError(f"episode {episode.index}: parquet row count/identity mismatch")
        order = np.argsort(np.asarray(values["frame_index"], dtype=np.int64))
        table = _EpisodeTable(
            state=np.asarray(values["observation.state"], dtype=np.float32)[order],
            action=np.asarray(values["action"], dtype=np.float32)[order],
            timestamp=np.asarray(values["timestamp"], dtype=np.float64)[order],
            frame_index=np.asarray(values["frame_index"], dtype=np.int64)[order],
            index=np.asarray(values["index"], dtype=np.int64)[order],
            task_index=np.asarray(values["task_index"], dtype=np.int64)[order],
        )
        if table.state.shape != (episode.length, 8) or table.action.shape != (episode.length, 7):
            raise ValueError(f"episode {episode.index}: invalid state/action shapes")
        finite = (
            np.isfinite(table.state).all()
            and np.isfinite(table.action).all()
            and np.isfinite(table.timestamp).all()
        )
        if not finite:
            raise ValueError(f"episode {episode.index}: non-finite numeric value")
        if not np.array_equal(table.frame_index, np.arange(episode.length)):
            raise ValueError(f"episode {episode.index}: frame_index is not contiguous")
        if not np.array_equal(table.index, np.arange(episode.dataset_from, episode.dataset_to)):
            raise ValueError(f"episode {episode.index}: global index range differs from metadata")
        tasks = set(table.task_index.tolist())
        if len(tasks) != 1:
            raise ValueError(f"episode {episode.index}: expected one task, got {tasks}")
        if episode.task >= 0 and tasks != {episode.task}:
            raise ValueError(f"episode {episode.index}: task differs from split metadata")
        self._tables[episode.index] = table
        while len(self._tables) > 8:
            self._tables.popitem(last=False)
        return table

    def _image(self, episode: Episode, key: str, relative_timestamp: float) -> np.ndarray:
        chunk, file, start, stop = episode.videos[key]
        absolute = start + relative_timestamp
        if absolute < start - 1e-7 or absolute >= stop + 1e-7:
            raise ValueError(f"episode {episode.index}: video timestamp outside episode interval")
        path = self.root / f"videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4"
        try:
            image = self._videos.get(path, absolute, self.fps, stop)
        except ValueError as exc:
            raise ValueError(
                f"episode={episode.index} camera={key} video={path} "
                f"requested={absolute:.9f}s interval=[{start:.9f}, {stop:.9f}): {exc}"
            ) from exc
        if image.dtype != np.uint8 or image.shape != (256, 256, 3):
            raise ValueError(f"unexpected decoded image: dtype={image.dtype}, shape={image.shape}")
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_pos = bisect_right(self._ends, index)
        episode = self.episodes[episode_pos]
        start = 0 if episode_pos == 0 else int(self._ends[episode_pos - 1])
        frame = index - start
        table = self._load_episode(episode)
        action_indices = np.minimum(frame + np.arange(self.horizon), episode.length - 1)
        timestamp = float(table.timestamp[frame])
        return {
            "observation/image": self._image(episode, VIDEO_KEYS[0], timestamp),
            "observation/wrist_image": self._image(episode, VIDEO_KEYS[1], timestamp),
            "observation/state": table.state[frame],
            "actions": table.action[action_indices],
            "prompt": self.prompts[episode.task],
        }


def _partition_episodes(episodes: list[Episode], seed: int, holdout_per_task: int):
    by_task: dict[int, list[Episode]] = {task: [] for task in range(40)}
    for episode in episodes:
        if episode.task not in by_task:
            raise ValueError(f"episode {episode.index}: invalid task {episode.task}")
        by_task[episode.task].append(episode)
    train, val = {}, {}
    for task, task_episodes in by_task.items():
        if len(task_episodes) <= holdout_per_task:
            raise ValueError(f"task {task} has only {len(task_episodes)} episodes")
        shuffled = sorted(task_episodes, key=lambda episode: episode.index)
        random.Random((seed << 8) + task).shuffle(shuffled)
        val[task] = sorted(shuffled[:holdout_per_task], key=lambda episode: episode.index)
        train[task] = sorted(shuffled[holdout_per_task:], key=lambda episode: episode.index)
    return train, val


def build_suite_datasets(root: str | Path, horizon: int = 10, seed: int = 42, holdout_per_task: int = 2):
    """Build disjoint frame-uniform train/validation datasets for all suites."""
    root = Path(root)
    if horizon <= 0 or holdout_per_task <= 0:
        raise ValueError("horizon and holdout_per_task must be positive")
    info, task_rows, episode_rows = _read_metadata(root)
    fps = _validate_info(info)
    prompts = _validate_tasks(task_rows)
    episodes = _episodes_from_rows(episode_rows)
    if len(episodes) != int(info.get("total_episodes", -1)):
        raise ValueError("episode count differs from info.json")
    if [episode.index for episode in episodes] != list(range(len(episodes))):
        raise ValueError("episode indices are not contiguous from zero")
    if sum(episode.length for episode in episodes) != int(info.get("total_frames", -1)):
        raise ValueError("frame count differs from info.json")

    # Task is stored in frame parquet rather than episode metadata. Read only one
    # scalar column per shared data file, then attach the verified task to episodes.
    pq = _pyarrow_parquet()
    attached = []
    for episode in episodes:
        path = root / f"data/chunk-{episode.data_chunk:03d}/file-{episode.data_file:03d}.parquet"
        column = pq.read_table(path, columns=["task_index"], filters=[("episode_index", "=", episode.index)])
        tasks = set(column["task_index"].to_pylist())
        if len(tasks) != 1 or next(iter(tasks)) not in prompts:
            raise ValueError(f"episode {episode.index}: invalid task values {tasks}")
        attached.append(Episode(**{**episode.__dict__, "task": int(next(iter(tasks))) }))

    train_by_task, val_by_task = _partition_episodes(attached, seed, holdout_per_task)
    train, val = {}, {}
    suites = {}
    for suite, task_ids in SUITE_TASKS.items():
        suite_train = [episode for task in task_ids for episode in train_by_task[task]]
        suite_val = [episode for task in task_ids for episode in val_by_task[task]]
        train[suite] = LiberoV3Dataset(root, suite_train, prompts, fps, horizon)
        val[suite] = LiberoV3Dataset(root, suite_val, prompts, fps, horizon)
        suites[suite] = {
            "task_indices": list(task_ids),
            "train_episode_indices": [episode.index for episode in suite_train],
            "val_episode_indices": [episode.index for episode in suite_val],
            "train_frames": sum(episode.length for episode in suite_train),
            "val_frames": sum(episode.length for episode in suite_val),
        }
    covered = {task for task_ids in SUITE_TASKS.values() for task in task_ids}
    if covered != set(range(40)) or any(len(task_ids) != 10 for task_ids in SUITE_TASKS.values()):
        raise AssertionError("suite definitions do not cover exactly 40 tasks")
    manifest = {
        "format": "lerobot-v3", "root": str(root), "fps": fps, "horizon": horizon,
        "seed": seed, "holdout_per_task": holdout_per_task,
        "task_mapping": {str(index): prompt for index, prompt in prompts.items()}, "suites": suites,
    }
    return train, val, manifest
