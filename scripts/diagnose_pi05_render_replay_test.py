#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import diagnose_pi05_render_replay as target


class FakeSim:
    def __init__(self, env):
        self.env = env
        self.data = SimpleNamespace(qacc_warmstart=np.zeros(3, dtype=np.float64))

    def get_state(self):
        return np.asarray([self.env.step_number, self.env.offset], dtype=np.float64)


class FakeEnv:
    def __init__(self, differing=False):
        self.differing = differing
        self.step_number = 0
        self.offset = 0.
        self.sim = FakeSim(self)
        self.closed = False

    def raw(self):
        wrist = np.zeros((256, 256, 3), dtype=np.uint8)
        if self.differing and self.step_number == 25:
            wrist[4, 4, 1] = 2
        return {"state8": np.asarray([self.step_number] + [0.] * 7),
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": wrist}

    def reset(self):
        self.step_number = 0

    def set_init_state(self, value):
        del value
        return self.raw()

    def step(self, action):
        self.offset += float(action[0])
        self.step_number += 1
        return self.raw(), 0., False, {}

    def close(self):
        self.closed = True


class Helpers:
    def __init__(self):
        self.calls = 0
        self.envs = []

    def make_environment(self, suite, task_id, seed):
        self.calls += 1
        env = FakeEnv(differing=self.calls == 2)
        self.envs.append(env)
        return env, None, SimpleNamespace(language="fake")

    def load_official_init_states(self, task):
        del task
        return [np.zeros(2) for _ in range(50)], {
            "file": "fake.init", "sha256": "f" * 64, "count": 50, "shape": [50, 2]}

    def observation_payload(self, raw, instruction, policy_id, policy_seed):
        del instruction, policy_id, policy_seed
        wrist = raw["robot0_eye_in_hand_image"][::2, ::2][:224, :224]
        agent = raw["agentview_image"][::2, ::2][:224, :224]
        return {"observation/image": agent, "observation/wrist_image": wrist,
                "observation/state": raw["state8"]}


def source(horizon=20):
    actions = np.zeros((horizon, 7), dtype=np.float64)
    actions[:, 0] = .1
    return {"case": {"suite": "libero_10", "task_id": 8, "init_id": 0,
                     "replicate_id": 2, "ambient_seed": 1907, "policy_id": "base"},
            "actions": actions, "horizon": horizon}


class RenderReplayTest(unittest.TestCase):
    def test_replays_fresh_env_and_finds_raw_wrist_difference(self):
        helpers = Helpers()
        first = target.replay_once(source(), helpers, np)
        second = target.replay_once(source(), helpers, np)
        comparison = target.compare_replays(first, second, source()["actions"], np)
        self.assertEqual(comparison["first_difference_step"], 15)
        image = comparison["first_image_difference"]
        self.assertEqual((image["step"], image["field"]), (15, "policy_wrist224"))
        self.assertEqual(image["metrics"]["different_pixels"], 1)
        self.assertEqual(image["metrics"]["max_abs"], 2.)
        self.assertTrue(image["same_action"])
        self.assertTrue(image["state8"]["equal"])
        self.assertTrue(all(env.closed for env in helpers.envs))
        compact = target.compact_hash_records(first["records"])
        self.assertEqual(len(compact), 20)
        self.assertNotIn("sim_state_value", compact[0])
        self.assertIn("images", compact[0])

    def test_identical_replays_have_no_difference(self):
        helpers = Helpers()
        first = target.replay_once(source(), helpers, np)
        helpers.calls = 0
        second = target.replay_once(source(), helpers, np)
        comparison = target.compare_replays(first, second, source()["actions"], np)
        self.assertIsNone(comparison["first_difference_step"])
        self.assertIsNone(comparison["first_image_difference"])

    def test_source_is_exact_control_case_and_prefix(self):
        actions = [[0.] * 7 for _ in range(420)]
        value = {"case": {"suite": "libero_10", "task_id": 8, "init_id": 0,
                          "replicate_id": 2, "ambient_seed": 1907, "policy_id": "base"},
                 "pi05_control": {"kind": "response_probe_v1", "enabled": False},
                 "runner": {"environment_actions": actions}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"; path.write_text(json.dumps(value))
            loaded = target.load_action_case(path, np)
            self.assertEqual((loaded["key"], len(loaded["actions"])),
                             (("libero_10", 8, 0, 2), 420))
            value["pi05_control"]["enabled"] = True; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "control arm"):
                target.load_action_case(path, np)

    def test_thread_condition_is_exact(self):
        self.assertEqual(set(target.configure_threads(1, require_fresh_numpy=False)),
                         set(target.THREAD_VARS))
        for value in (0, 2, True, 4.0):
            with self.assertRaises(ValueError):
                target.configure_threads(value, require_fresh_numpy=False)


if __name__ == "__main__":
    unittest.main()
