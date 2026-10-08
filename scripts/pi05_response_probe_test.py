#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest

import numpy as np
import pi05_harness_backend as backend


MODULE = Path(__file__).with_name("pi05_response_probe.py")
SPEC = importlib.util.spec_from_file_location("pi05_response_probe_under_test", MODULE)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def state(z=0.4, aperture=.08):
    return np.asarray([0., 0., z, 0., 0., 0., aperture / 2, -aperture / 2])


def context(actual, remaining=200, stage=0):
    return MappingProxyType({"actual_executed": actual,
        "remaining_episode": remaining, "stage_executed": actual,
        "remaining_stage": remaining, "stage_index": stage})


class Delegate:
    def __init__(self, action=None):
        self.calls = 0
        self.action = np.asarray(action if action is not None else
                                 [0., 0., -.8, 0., 0., 0., -.8])
        self.call_seeds, self.begins, self.resets, self.finals = [], 0, 0, 0
        self.call_records = []
        self.provenance = {"kind": "fake_pi05"}

    def begin_episode(self):
        self.begins += 1

    def reset(self):
        self.resets += 1

    def act(self, observation, instruction, memory):
        del observation, instruction, memory
        self.call_seeds.append(7 + self.calls * 1_000_003)
        self.call_records.append({"policy_seed": self.call_seeds[-1]})
        self.calls += 1
        return np.repeat(self.action[None], 5, axis=0)

    def finalize_episode(self, executed):
        self.finals += 1
        self.final_trace = tuple(executed)


class Driver:
    def __init__(self, wrapped):
        self.skill = wrapped
        wrapped.begin_episode()
        wrapped.reset()
        wrapped.on_reset(state())
        self.actual = 0

    def act(self, remaining=200, stage=0):
        self.skill.set_execution_context(context(self.actual, remaining, stage))
        return self.skill.act(SimpleNamespace(), "goal", {})

    def execute(self, actions, states=None):
        states = states or [state()] * len(actions)
        for action, post in zip(actions, states):
            self.skill.on_execution(action, post)
            self.actual += 1

    def prepare_guard(self):
        # Only the final 30 executed native actions are guard evidence.  The
        # Runner context independently reports the preceding 90 real steps.
        self.actual = 90
        for _ in range(6):
            actions = self.act()
            self.execute(actions)
        assert self.actual == 120


class ResponseProbeTest(unittest.TestCase):
    def test_missing_parameters_preserve_defaults_and_invalid_values_fail(self):
        skill = probe.Pi05ResponseProbeSkill(Delegate())
        self.assertEqual((skill.lift_z_command, skill.max_lift_steps,
                          skill.lift_target_m, skill.native_reserve_steps,
                          skill.minimum_actual), (.05, 8, .02, 20, 120))
        invalid = ({"lift_z_command": 0}, {"lift_z_command": .2001},
                   {"max_lift_steps": True}, {"max_lift_steps": 21},
                   {"lift_target_m": .026}, {"native_reserve_steps": 19},
                   {"native_reserve_steps": 81}, {"minimum_actual": True},
                   {"minimum_actual": 59}, {"minimum_actual": 121})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                probe.Pi05ResponseProbeSkill(Delegate(), **kwargs)

    @staticmethod
    def strong_driver():
        return Driver(probe.Pi05ResponseProbeSkill(
            Delegate(), lift_z_command=.2, max_lift_steps=20,
            lift_target_m=.025, native_reserve_steps=80))

    def enter_strong_lift(self):
        driver = self.strong_driver()
        driver.prepare_guard()
        driver.execute(driver.act(111))
        for aperture in (.078, .074, .070, .065):
            driver.execute(driver.act(106), [state(aperture=aperture)])
        return driver

    def test_strong_probe_stops_on_actual_twenty_five_mm_feedback(self):
        driver = self.enter_strong_lift()
        first = driver.act(102)
        np.testing.assert_array_equal(first[0], [0, 0, .2, 0, 0, 0, 1])
        driver.execute(first, [state(z=.426, aperture=.065)])
        settle = driver.act(101)
        np.testing.assert_array_equal(settle[0], [0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(driver.skill._manual_emitted, 6)  # 4 close + 1 lift + 1 settle

    def test_strong_probe_caps_lift_at_twenty_actions(self):
        driver = self.enter_strong_lift()
        for _ in range(20):
            action = driver.act(200)
            self.assertEqual(float(action[0, 2]), .2)
            driver.execute(action, [state(z=.4, aperture=.065)])
        settle = driver.act(200)
        self.assertEqual(float(settle[0, 2]), 0.)
        self.assertEqual(driver.skill._manual_emitted, 25)

    def test_strong_probe_requires_eighty_step_native_reserve(self):
        short = self.strong_driver()
        short.prepare_guard()
        self.assertEqual(short.act(110).shape, (5, 7))
        self.assertFalse(short.skill._attempted)
        exact = self.strong_driver()
        exact.prepare_guard()
        self.assertEqual(exact.act(111).shape, (5, 7))
        self.assertTrue(exact.skill._attempted)

    def test_minimum_actual_100_waits_then_grace_and_rechecks(self):
        driver = Driver(probe.Pi05ResponseProbeSkill(
            Delegate(), lift_z_command=.2, max_lift_steps=20,
            lift_target_m=.025, native_reserve_steps=80, minimum_actual=100))
        driver.actual = 70
        for _ in range(5):
            driver.execute(driver.act(200))
        self.assertEqual(driver.actual, 95)
        self.assertFalse(driver.skill._attempted)
        driver.execute(driver.act(200))
        self.assertEqual(driver.actual, 100)
        grace = driver.act(200)
        self.assertTrue(driver.skill._attempted)
        self.assertEqual(grace.shape, (5, 7))
        driver.execute(grace)
        manual = driver.act(195)
        np.testing.assert_array_equal(manual[0], [0, 0, 0, 0, 0, 0, 1])

    def test_strong_manual_emission_truncation_and_parameters_are_provenant(self):
        driver = self.enter_strong_lift()
        lift = driver.act(102)
        self.assertEqual(lift.shape, (1, 7))
        trace = tuple((row["expected_actual_step"], row["stage_index"])
                      for row in driver.skill._executed)
        driver.skill.finalize_episode(trace)
        emitted = driver.skill.provenance["emitted_rows"][-1]
        self.assertEqual(emitted["kind"], "lift")
        self.assertTrue(emitted["truncated_before_execution"])
        self.assertEqual(driver.skill.provenance["parameters"], {
            "lift_z_command": .2, "max_lift_steps": 20,
            "lift_target_m": .025, "native_reserve_steps": 80,
            "minimum_actual": 120,
            "max_manual_actions": 26, "trigger_remaining_steps": 111})

    def test_guard_false_real_runner_matches_native_actions_and_seeds(self):
        root = Path(os.environ.get(
            "PI05_TEST_PHYSICALRSI_ROOT",
            "/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007",
        ))
        api = backend.import_roborsi(root)

        class Env:
            action_low, action_high, provenance = np.full(7, -np.inf), np.full(7, np.inf), {}
            def __init__(self):
                self.steps, self.actions = 0, []
            def observation(self):
                return api.Observation({"observation/state": state()}, {},
                                       success=self.steps == 7)
            def reset(self, seed):
                del seed
                self.steps, self.actions = 0, []
                return self.observation()
            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                self.steps += 1
                return self.observation()
            def close(self):
                pass

        harness = {"schema": 1, "name": "parity", "remember": [], "stages": [{
            "skill": "pi05", "instruction": "goal", "max_steps": 10,
            "until": None, "on_timeout": "abort",
        }]}
        native_delegate, native_env = Delegate(), Env()
        native = api.Runner(lambda: native_env, {"pi05": native_delegate},
                            max_steps=10, max_seconds=10, chunk_steps=5).run(harness, 0)
        wrapped_delegate, wrapped_env = Delegate(), Env()
        wrapped = probe.Pi05ResponseProbeSkill(wrapped_delegate)
        tested = api.Runner(
            lambda: probe.Pi05ExecutionFeedbackEnvironment(wrapped_env, wrapped),
            {"pi05": wrapped}, max_steps=10, max_seconds=10, chunk_steps=5,
        ).run(harness, 0)
        self.assertEqual(native["status"], tested["status"])
        self.assertEqual(native_delegate.call_seeds, wrapped_delegate.call_seeds)
        np.testing.assert_array_equal(native_env.actions, wrapped_env.actions)
        self.assertFalse(wrapped.provenance["attempted"])

    def test_guard_false_is_native_action_and_seed_equivalent(self):
        direct, wrapped_delegate = Delegate(), Delegate()
        skill = probe.Pi05ResponseProbeSkill(wrapped_delegate)
        driver = Driver(skill)
        for _ in range(2):
            got = driver.act()
            expected = direct.act(None, "goal", {})
            np.testing.assert_array_equal(got, expected)
            driver.execute(got)
        self.assertEqual(wrapped_delegate.call_seeds, direct.call_seeds)
        self.assertEqual(skill.calls, direct.calls)
        self.assertEqual(skill.call_records, wrapped_delegate.call_records)
        self.assertFalse(skill._attempted)

    def test_close_response_lifts_then_settles_and_preserves_calls(self):
        delegate = Delegate()
        driver = Driver(probe.Pi05ResponseProbeSkill(delegate))
        driver.prepare_guard()
        grace = driver.act(100)
        self.assertEqual(grace.shape, (5, 7))
        driver.execute(grace)
        self.assertEqual(delegate.calls, 7)  # six history calls plus grace
        apertures = [.078, .074, .070, .065]
        for value in apertures:
            action = driver.act(95 - (driver.actual - 125))
            np.testing.assert_array_equal(action[0], [0, 0, 0, 0, 0, 0, 1])
            driver.execute(action, [state(aperture=value)])
        first_lift = driver.act(90)
        np.testing.assert_array_equal(first_lift[0], [0, 0, .05, 0, 0, 0, 1])
        driver.execute(first_lift, [state(z=.41, aperture=.065)])
        second_lift = driver.act(89)
        driver.execute(second_lift, [state(z=.421, aperture=.065)])
        for _ in range(2):
            settle = driver.act(88)
            np.testing.assert_array_equal(settle[0], [0, 0, 0, 0, 0, 0, 1])
            driver.execute(settle, [state(z=.421, aperture=.065)])
        resumed = driver.act(86)
        self.assertEqual(resumed.shape, (5, 7))
        self.assertEqual(delegate.calls, 8)  # manual actions did not consume calls
        event = driver.skill._events[0]
        self.assertTrue(event["close_response"])
        self.assertEqual(event["status"], "manual_complete")

    def test_no_close_response_reopens(self):
        driver = Driver(probe.Pi05ResponseProbeSkill(Delegate()))
        driver.prepare_guard()
        driver.execute(driver.act(100))
        for _ in range(4):
            close = driver.act(95)
            driver.execute(close, [state(aperture=.08)])
        for _ in range(2):
            opened = driver.act(90)
            self.assertEqual(float(opened[0, 6]), -1.)
            driver.execute(opened, [state(aperture=.08)])
        self.assertEqual(driver.skill._manual_emitted, 6)
        self.assertFalse(driver.skill._events[0]["close_response"])

    def test_budget_and_nonfinite_guards(self):
        driver = Driver(probe.Pi05ResponseProbeSkill(Delegate()))
        driver.prepare_guard()
        native = driver.act(38)
        self.assertEqual(native.shape, (5, 7))
        self.assertFalse(driver.skill._attempted)
        with self.assertRaisesRegex(ValueError, "finite 8D"):
            driver.skill.on_execution(native[0], state(z=np.nan))
        # Failed callback did not consume the emitted action.
        self.assertEqual(len(driver.skill._pending), 5)

    def test_recheck_preserves_twenty_native_action_reserve(self):
        delegate = Delegate()
        driver = Driver(probe.Pi05ResponseProbeSkill(delegate))
        driver.prepare_guard()
        grace = driver.act(39)
        driver.execute(grace)
        fallback = driver.act(33)
        self.assertEqual(fallback.shape, (5, 7))
        self.assertEqual(driver.skill._manual_emitted, 0)
        self.assertEqual(driver.skill._events[0]["status"], "permanent_veto_recheck_failed")
        self.assertEqual(delegate.calls, 8)

    def test_attempt_is_once_across_stage_reset(self):
        driver = Driver(probe.Pi05ResponseProbeSkill(Delegate()))
        driver.prepare_guard()
        grace = driver.act(100)
        driver.execute(grace)
        # Make the recheck fail, then cross a stage boundary.  Attempt state is
        # episode-global and cannot re-arm.
        driver.skill._state = state(aperture=.01)
        fallback = driver.act(95)
        driver.execute(fallback)
        driver.skill.reset()
        driver.skill.on_reset(state())
        later = driver.act(90, stage=1)
        self.assertEqual(later.shape, (5, 7))
        self.assertTrue(driver.skill._attempted)
        self.assertEqual(len(driver.skill._events), 1)

    def test_finalize_reconciles_only_executed_prefix(self):
        delegate = Delegate()
        driver = Driver(probe.Pi05ResponseProbeSkill(delegate))
        actions = driver.act()
        driver.execute(actions[:2])
        trace = tuple((row["expected_actual_step"], row["stage_index"])
                      for row in driver.skill._executed)
        driver.skill.finalize_episode(trace)
        self.assertTrue(driver.skill.provenance["execution_reconciled"])
        self.assertEqual(sum(r["executed"] for r in driver.skill.provenance["emitted_rows"]), 2)
        self.assertEqual(sum(r.get("truncated_before_execution", False)
                             for r in driver.skill.provenance["emitted_rows"]), 3)
        self.assertEqual((delegate.begins, delegate.finals), (1, 1))

    def test_environment_callback_occurs_only_after_successful_step(self):
        class Env:
            action_low, action_high, provenance = -np.ones(7), np.ones(7), {}
            def reset(self, init_id):
                del init_id
                return SimpleNamespace(policy={"observation/state": state()})
            def step(self, action):
                del action
                raise RuntimeError("simulator failed")
            def close(self):
                pass
        skill = probe.Pi05ResponseProbeSkill(Delegate())
        skill.begin_episode()
        env = probe.Pi05ExecutionFeedbackEnvironment(Env(), skill)
        env.reset(0)
        skill.reset()
        skill.on_reset(state())
        skill.set_execution_context(context(0))
        action = skill.act(None, "goal", {})[0]
        with self.assertRaisesRegex(RuntimeError, "simulator failed"):
            env.step(action)
        self.assertEqual(len(skill._executed), 0)
        self.assertEqual(len(skill._pending), 5)


if __name__ == "__main__":
    unittest.main()
