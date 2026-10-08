#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import pi05_harness_backend as backend


PHYSICALRSI_ROOT = Path(os.environ.get(
    "PI05_TEST_PHYSICALRSI_ROOT",
    "/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007",
))
EVAL_SCRIPT = Path(os.environ.get(
    "PI05_TEST_EVAL_SCRIPT",
    "/home/user/tian_ws/pi05_plugin_eval_20261008/patch/scripts/eval_pi05_plugins.py",
))


def raw_observation(value: float = 0.0):
    return {
        "agentview_image": np.full((4, 4, 3), int(value) % 255, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((4, 4, 3), int(value + 1) % 255, dtype=np.uint8),
        "state": np.arange(8, dtype=np.float64) + value,
        "raw": "must_not_reach_policy",
        "world_from_model": np.eye(4),
        "live_state": np.ones(8),
    }


class FakeLibero:
    def __init__(self, done_policy_step=None):
        self.env = SimpleNamespace(
            action_spec=(np.full(7, -1.0), np.full(7, 1.0))
        )
        self.done_policy_step = done_policy_step
        self.all_actions = []
        self.policy_actions = []
        self.settling = True

    def reset(self):
        return raw_observation()

    def set_init_state(self, state):
        self.state = state
        self.settling = True
        return raw_observation()

    def step(self, action):
        value = np.asarray(action, dtype=np.float64)
        self.all_actions.append(value.copy())
        if len(self.all_actions) > backend.SETTLING_STEPS:
            self.settling = False
            self.policy_actions.append(value.copy())
        step = len(self.policy_actions)
        done = self.done_policy_step is not None and step == self.done_policy_step
        # Return done during the final settling step too; the adapter must ignore it.
        if len(self.all_actions) == backend.SETTLING_STEPS:
            done = True
        return raw_observation(float(step)), 0.0, done, {}

    def close(self):
        self.closed = True


class FakeHelpers:
    STEP_CAPS = {"libero_object": 280}

    def __init__(self, done_policy_step=None):
        self.done_policy_step = done_policy_step
        self.environments = []

    def make_environment(self, suite, task_id, environment_seed):
        assert (suite, task_id, environment_seed) == ("libero_object", 4, 7)
        env = FakeLibero(self.done_policy_step)
        self.environments.append(env)
        task = SimpleNamespace(language="pick up the ketchup and place it in the basket")
        return env, object(), task

    @staticmethod
    def load_official_init_states(task):
        del task
        return list(range(50)), {"file": "fake.init", "count": 50, "sha256": "0" * 64}

    @staticmethod
    def observation_payload(raw, instruction, policy_id, policy_seed):
        # Deliberately include forbidden keys to prove the environment adapter
        # whitelists rather than forwarding the builder result wholesale.
        return {
            "observation/image": raw["agentview_image"],
            "observation/wrist_image": raw["robot0_eye_in_hand_image"],
            "observation/state": raw["state"],
            "prompt": instruction,
            "policy_id": policy_id,
            "policy_seed": policy_seed,
            "raw": raw,
            "world_from_model": raw["world_from_model"],
            "live_state": raw["live_state"],
        }

    class EpisodeIdentityTransport:
        def __init__(self, inner, expected_metadata):
            self.inner = inner
            self.expected_metadata = expected_metadata
            self.verified_metadata = None

        def infer(self, payload):
            response = self.inner.infer(payload)
            self.verified_metadata = self.expected_metadata.copy()
            return response

        def close(self):
            self.inner.close()


class FakeTransport:
    def __init__(self, chunks):
        self.chunks = deque(np.asarray(chunk, dtype=np.float64) for chunk in chunks)
        self.payloads = []
        self.verified_metadata = {"policy_id": "object", "adapter_sha256": "a" * 64}
        self.closed = False

    def infer(self, payload):
        self.payloads.append(payload)
        return {"actions": self.chunks.popleft().copy()}

    def close(self):
        self.closed = True


def chunk(value=0.0, gripper=0.0):
    result = np.full((5, 7), value, dtype=np.float64)
    result[:, -1] = gripper
    return result


def harness(stages):
    return {"schema": 1, "name": "test", "remember": [], "stages": stages}


def stage(instruction, steps, on_timeout="abort"):
    return {
        "skill": "pi05",
        "instruction": instruction,
        "max_steps": steps,
        "until": None,
        "on_timeout": on_timeout,
    }


class Pi05HarnessBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = backend.import_roborsi(PHYSICALRSI_ROOT)

    def make_environment(self, helpers, policy_seed=207):
        return backend.Pi05HarnessEnvironment(
            self.api.Observation,
            helpers,
            suite="libero_object",
            task_id=4,
            policy_id="object",
            policy_seed=policy_seed,
        )

    def test_replicate_seed_derivation_is_closed_and_replica_zero_is_legacy(self):
        self.assertEqual(backend.policy_seed_for_case(14, 0), 707)
        self.assertEqual(
            backend.policy_seed_for_case(14, 0, 1),
            707 + backend.REPLICATE_SEED_STRIDE,
        )
        for invalid in (True, -1, 10, 1.5):
            with self.assertRaisesRegex(ValueError, "replicate_id"):
                backend.policy_seed_for_case(14, 0, invalid)
        ambient = backend.ambient_seed_for_case(14, 0)
        self.assertEqual(ambient, 707)
        self.assertEqual(ambient, backend.ambient_seed_for_case(14, 0))
        # Replicate 9's policy seed exceeds numpy's legacy uint32 seed domain,
        # but the fixed ambient seed used by simulation remains executable.
        self.assertGreater(backend.policy_seed_for_case(14, 0, 9), 2**32)
        np.random.seed(ambient)

    def test_multistage_reset_keeps_episode_global_seed_sequence(self):
        helpers = FakeHelpers(done_policy_step=4)
        transport = FakeTransport([chunk(0.1), chunk(0.2)])
        skill = backend.Pi05HarnessSkill(
            lambda: transport, policy_id="object", policy_seed=207
        )
        program = harness([
            stage("pick up the ketchup", 2, "next"),
            stage("original instruction", 3),
        ])
        episode = backend.run_with_shared_runner(
            self.api.Runner,
            lambda: self.make_environment(helpers),
            skill,
            program,
            init_id=0,
            official_cap=5,
        )
        self.assertTrue(episode.report["success"])
        self.assertEqual([record["policy_seed"] for record in episode.policy_calls], [207, 1_000_210])
        self.assertEqual([payload["prompt"] for payload in transport.payloads], [
            "pick up the ketchup", "original instruction"
        ])
        np.testing.assert_allclose(np.asarray(episode.environment_actions)[:, 0], [0.1, 0.1, 0.2, 0.2])
        self.assertTrue(transport.closed)

    def test_done_from_final_allowed_action_scores_success(self):
        helpers = FakeHelpers(done_policy_step=3)
        transport = FakeTransport([chunk(0.3)])
        skill = backend.Pi05HarnessSkill(lambda: transport, policy_id="object", policy_seed=207)
        episode = backend.run_with_shared_runner(
            self.api.Runner,
            lambda: self.make_environment(helpers),
            skill,
            harness([stage("original", 3)]),
            init_id=0,
            official_cap=3,
        )
        self.assertTrue(episode.report["success"])
        self.assertEqual(episode.report["status"], "task_success")
        self.assertEqual(episode.report["steps"], 3)

    def test_total_stage_budget_cannot_exceed_official_cap(self):
        skill = backend.Pi05HarnessSkill(
            lambda: FakeTransport([chunk()]), policy_id="object", policy_seed=207
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            backend.run_with_shared_runner(
                self.api.Runner,
                lambda: self.make_environment(FakeHelpers()),
                skill,
                harness([stage("a", 3, "next"), stage("b", 3)]),
                init_id=0,
                official_cap=5,
            )

    def test_nan_action_is_rejected_before_environment_step(self):
        helpers = FakeHelpers()
        bad = chunk()
        bad[0, 0] = np.nan
        skill = backend.Pi05HarnessSkill(
            lambda: FakeTransport([bad]), policy_id="object", policy_seed=207
        )
        episode = backend.run_with_shared_runner(
            self.api.Runner,
            lambda: self.make_environment(helpers),
            skill,
            harness([stage("original", 1)]),
            init_id=0,
            official_cap=1,
        )
        self.assertEqual(episode.report["status"], "error")
        self.assertEqual(episode.report["error_type"], "ValueError")
        self.assertEqual(episode.environment_actions, [])
        self.assertEqual(episode.policy_calls[0]["policy_seed"], 207)
        self.assertFalse(episode.policy_calls[0]["response_valid"])

    def test_identity_transport_factory_wraps_each_episode_connection(self):
        helpers = FakeHelpers()
        inners = []

        def inner_factory():
            value = FakeTransport([chunk()])
            inners.append(value)
            return value

        factory = backend.episode_identity_transport_factory(
            helpers, inner_factory, {"policy_id": "object", "adapter_sha256": "a" * 64}
        )
        first, second = factory(), factory()
        self.assertIsNot(first, second)
        self.assertIsNot(first.inner, second.inner)
        self.assertEqual(len(inners), 2)

    def test_eval_helpers_bind_to_the_exact_requested_file(self):
        module = backend.import_eval_helpers(EVAL_SCRIPT)
        self.assertEqual(Path(module.__file__).resolve(), EVAL_SCRIPT.resolve())
        self.assertIs(backend.import_eval_helpers(EVAL_SCRIPT), module)

    def test_gripper_1_02_is_forwarded_without_client_clip(self):
        helpers = FakeHelpers(done_policy_step=1)
        skill = backend.Pi05HarnessSkill(
            lambda: FakeTransport([chunk(gripper=1.02)]),
            policy_id="object",
            policy_seed=207,
        )
        episode = backend.run_with_shared_runner(
            self.api.Runner,
            lambda: self.make_environment(helpers),
            skill,
            harness([stage("original", 1)]),
            init_id=0,
            official_cap=1,
        )
        self.assertEqual(episode.environment_actions[0][-1], 1.02)
        self.assertEqual(helpers.environments[0].policy_actions[0][-1], 1.02)

    def test_payload_and_observation_exclude_privileged_fields(self):
        helpers = FakeHelpers(done_policy_step=1)
        transport = FakeTransport([chunk()])
        skill = backend.Pi05HarnessSkill(lambda: transport, policy_id="object", policy_seed=207)
        episode = backend.run_with_shared_runner(
            self.api.Runner,
            lambda: self.make_environment(helpers),
            skill,
            harness([stage("original", 1)]),
            init_id=0,
            official_cap=1,
        )
        self.assertEqual(set(transport.payloads[0]), {
            "observation/image", "observation/wrist_image", "observation/state",
            "prompt", "policy_id", "policy_seed",
        })
        self.assertFalse({"raw", "world_from_model", "live_state"} & set(transport.payloads[0]))
        self.assertEqual(episode.report["trace"][0]["signals"], {})

    def test_single_stage_matches_the_old_five_action_loop(self):
        cap = 7
        chunks = [chunk(0.1, -1.0), chunk(0.2, 1.02)]
        helpers = FakeHelpers()
        transport = FakeTransport(chunks)
        skill = backend.Pi05HarnessSkill(lambda: transport, policy_id="object", policy_seed=207)
        episode = backend.run_with_shared_runner(
            self.api.Runner,
            lambda: self.make_environment(helpers),
            skill,
            harness([stage("original", cap)]),
            init_id=0,
            official_cap=cap,
        )

        manual_helpers = FakeHelpers()
        manual_env = self.make_environment(manual_helpers)
        obs = manual_env.reset(0)
        manual_transport = FakeTransport(chunks)
        queue = deque()
        calls = 0
        while len(manual_env.actions) < cap:
            if not queue:
                payload = {
                    "observation/image": obs.policy["observation/image"].copy(),
                    "observation/wrist_image": obs.policy["observation/wrist_image"].copy(),
                    "observation/state": obs.policy["observation/state"].copy(),
                    "prompt": "original",
                    "policy_id": "object",
                    "policy_seed": 207 + calls * backend.CALL_SEED_STRIDE,
                }
                queue.extend(manual_transport.infer(payload)["actions"][:5])
                calls += 1
            obs = manual_env.step(queue.popleft())
            if obs.success:
                break
        np.testing.assert_allclose(episode.environment_actions, manual_env.actions)
        self.assertEqual(
            [record["policy_seed"] for record in episode.policy_calls],
            [payload["policy_seed"] for payload in manual_transport.payloads],
        )
        self.assertEqual(episode.report["success"], obs.success)

    def test_exact_40_task_registry_uses_shared_registry_validation(self):
        tasks = {}
        caps = {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
        }
        for suite, cap in caps.items():
            for task_id in range(10):
                tasks[f"{suite}/{task_id}"] = {
                    "instruction": f"{suite} instruction {task_id}",
                    "max_steps": cap,
                }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps({
                "schema": 1,
                "name": "pi05-test",
                "default_skill": "pi05",
                "tasks": {},
            }))
            proposal = backend.materialize_pi05_registry(PHYSICALRSI_ROOT, path, tasks)
        self.assertEqual(proposal["harnesses"], {})


if __name__ == "__main__":
    unittest.main()
