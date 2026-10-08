import json
import pickle
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from openpi.training import plugin_data


def _episode(index, task, length=3, start=0.0):
    videos = {
        key: (2, 7, start, start + length / 10.0)
        for key in plugin_data.VIDEO_KEYS
    }
    return plugin_data.Episode(
        index=index,
        task=task,
        length=length,
        dataset_from=index * 100,
        dataset_to=index * 100 + length,
        data_chunk=0,
        data_file=0,
        videos=videos,
    )


def _table(episode):
    return plugin_data._EpisodeTable(
        state=np.zeros((episode.length, 8), dtype=np.float32),
        action=np.arange(episode.length * 7, dtype=np.float32).reshape(episode.length, 7),
        timestamp=np.arange(episode.length, dtype=np.float64) / 10,
        frame_index=np.arange(episode.length),
        index=np.arange(episode.dataset_from, episode.dataset_to),
        task_index=np.full(episode.length, episode.task),
    )


class PluginDataTest(unittest.TestCase):
    @staticmethod
    def _mock_video(times):
        frames = [
            SimpleNamespace(
                pts=round(time * 10_000),
                to_ndarray=lambda format, value=index: np.full((2, 2, 3), value, dtype=np.uint8),
            )
            for index, time in enumerate(times)
        ]
        container = mock.Mock()
        container.decode.side_effect = lambda stream: iter(frames)
        stream = SimpleNamespace(time_base=Fraction(1, 10_000))
        return container, stream

    def test_prefetch_excludes_float32_rounded_exclusive_video_end(self):
        path = Path("/video.mp4")
        for requested in (997.8 + float(np.float32(24.9)), 1022.7, 1022.7 + 4e-7):
            with self.subTest(requested=requested):
                decoder = plugin_data._VideoDecoder()
                container, stream = self._mock_video([1022.7, 1022.8, 1022.9])
                with mock.patch.object(decoder, "_container", return_value=(container, stream)):
                    image = decoder.get(path, requested, 10.0, 1023.0)
                    repeated = decoder.get(path, requested, 10.0, 1023.0)
                np.testing.assert_array_equal(image, np.zeros((2, 2, 3), dtype=np.uint8))
                np.testing.assert_array_equal(repeated, image)
                self.assertEqual(len(decoder._frames), 3)
                self.assertNotIn(decoder._key(path, 1023.0), decoder._frames)
                self.assertEqual(container.seek.call_count, 1)

    def test_prefetch_does_not_cache_the_next_episode_boundary(self):
        decoder = plugin_data._VideoDecoder()
        path = Path("/video.mp4")
        container, stream = self._mock_video([1022.7, 1022.8, 1022.9, 1023.0])
        with mock.patch.object(decoder, "_container", return_value=(container, stream)):
            decoder.get(path, 1022.6999996, 10.0, 1023.0)
        self.assertEqual(len(decoder._frames), 3)
        self.assertNotIn(decoder._key(path, 1023.0), decoder._frames)

    def test_requested_frame_is_never_replaced_by_out_of_tolerance_neighbor(self):
        decoder = plugin_data._VideoDecoder()
        container, stream = self._mock_video([12.8])
        with mock.patch.object(decoder, "_container", return_value=(container, stream)):
            with self.assertRaisesRegex(ValueError, r"0.100000s.*tolerance 0.051000s"):
                decoder.get(Path("/video.mp4"), 12.7, 10.0, 14.0)

    def test_interior_prefetch_still_uses_strict_timestamp_matching(self):
        decoder = plugin_data._VideoDecoder(prefetch=3)
        container, stream = self._mock_video([12.7, 12.9])
        with mock.patch.object(decoder, "_container", return_value=(container, stream)):
            with self.assertRaisesRegex(ValueError, r"0.100000s.*tolerance 0.051000s"):
                decoder.get(Path("/video.mp4"), 12.7, 10.0, 14.0)

    def test_action_chunk_clamps_inside_episode(self):
        first, second = _episode(0, 0), _episode(1, 1)
        tables = {0: _table(first), 1: _table(second)}
        dataset = plugin_data.LiberoV3Dataset(
            "/unused", [first, second], dict(enumerate(plugin_data.EXPECTED_TASKS)), 10.0, 10
        )
        image = lambda *args: np.zeros((256, 256, 3), dtype=np.uint8)
        with (
            mock.patch.object(dataset, "_load_episode", side_effect=lambda episode: tables[episode.index]),
            mock.patch.object(dataset, "_image", side_effect=image),
        ):
            item = dataset[first.length - 1]
        expected_last = tables[0].action[-1]
        np.testing.assert_array_equal(item["actions"], np.repeat(expected_last[None], 10, axis=0))
        self.assertEqual(item["prompt"], plugin_data.EXPECTED_TASKS[0])
        self.assertEqual(item["observation/state"].shape, (8,))

    def test_deterministic_holdout_is_disjoint_and_covers_every_task(self):
        episodes = [_episode(task * 10 + offset, task) for task in range(40) for offset in range(4)]
        train, val = plugin_data._partition_episodes(episodes, seed=42, holdout_per_task=2)
        train_again, val_again = plugin_data._partition_episodes(episodes, seed=42, holdout_per_task=2)
        for task in range(40):
            train_ids = {episode.index for episode in train[task]}
            val_ids = {episode.index for episode in val[task]}
            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertEqual((len(train_ids), len(val_ids)), (2, 2))
            self.assertEqual(val_ids, {episode.index for episode in val_again[task]})
            self.assertEqual(train_ids, {episode.index for episode in train_again[task]})

    def test_diagnostic_indices_spread_over_every_task_without_loading(self):
        episodes = [_episode(task, task, length=task + 2) for task in range(10)]
        dataset = plugin_data.LiberoV3Dataset(
            "/unused", episodes, dict(enumerate(plugin_data.EXPECTED_TASKS)), 10.0, 10
        )
        indices = dataset.diagnostic_indices(32)
        self.assertEqual(len(indices), 32)
        tasks = []
        for index in indices:
            episode_position = np.searchsorted(dataset._ends, index, side="right")
            tasks.append(dataset.episodes[episode_position].task)
        self.assertEqual(tasks[:10], list(range(10)))
        self.assertEqual(set(tasks), set(range(10)))
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            dataset.diagnostic_indices(9)

    def test_task_mapping_fails_closed_on_changed_name_or_index(self):
        rows = [
            {"task_index": index, "__index_level_0__": prompt}
            for index, prompt in enumerate(plugin_data.EXPECTED_TASKS)
        ]
        self.assertEqual(plugin_data._validate_tasks(rows), dict(enumerate(plugin_data.EXPECTED_TASKS)))
        rows[3] = {"task_index": 3, "__index_level_0__": "almost the same"}
        with self.assertRaisesRegex(ValueError, r"changed=\[3\]"):
            plugin_data._validate_tasks(rows)

    def test_video_offset_and_timestamp_tolerance(self):
        episode = _episode(5, 0, start=12.5)
        dataset = plugin_data.LiberoV3Dataset(
            "/dataset", [episode], dict(enumerate(plugin_data.EXPECTED_TASKS)), 10.0, 10
        )
        calls = []

        def fake_get(path, timestamp, fps, upper):
            calls.append((str(path), timestamp, fps, upper))
            return np.zeros((256, 256, 3), dtype=np.uint8)

        with mock.patch.object(dataset._videos, "get", side_effect=fake_get):
            dataset._image(episode, plugin_data.VIDEO_KEYS[0], 0.2)
        self.assertEqual(calls, [(
            "/dataset/videos/observation.images.image/chunk-002/file-007.mp4",
            12.7,
            10.0,
            12.8,
        )])
        self.assertEqual(plugin_data._match_timestamp(12.7, [12.649, 12.701], 0.051), 1)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            plugin_data._match_timestamp(12.7, [12.64, 12.76], 0.051)

    def test_decoder_pickle_drops_process_local_handles_and_manifest_contract(self):
        decoder = plugin_data._VideoDecoder()
        decoder._frames[("video", 1)] = np.zeros((1,), dtype=np.uint8)
        restored = pickle.loads(pickle.dumps(decoder))
        self.assertFalse(restored._frames)
        self.assertFalse(restored._open)
        manifest = {
            "task_mapping": {str(index): prompt for index, prompt in enumerate(plugin_data.EXPECTED_TASKS)},
            "suites": {name: {"task_indices": list(tasks)} for name, tasks in plugin_data.SUITE_TASKS.items()},
        }
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertEqual(set().union(*plugin_data.SUITE_TASKS.values()), set(range(40)))

    def test_video_error_identifies_episode_camera_path_and_requested_time(self):
        episode = _episode(5, 0, start=12.5)
        dataset = plugin_data.LiberoV3Dataset(
            "/dataset", [episode], dict(enumerate(plugin_data.EXPECTED_TASKS)), 10.0, 10
        )
        with mock.patch.object(dataset._videos, "get", side_effect=ValueError("PTS mismatch")):
            with self.assertRaisesRegex(ValueError, "episode=5 camera=observation.images.image") as caught:
                dataset._image(episode, plugin_data.VIDEO_KEYS[0], 0.2)
        self.assertIn("file-007.mp4", str(caught.exception))
        self.assertIn("requested=12.700000000s", str(caught.exception))
        self.assertIn("PTS mismatch", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
