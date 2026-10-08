#!/usr/bin/env python3

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
import unittest

import numpy as np

import pi05_closed_dwell_lift as lift
import pi05_harness_backend as backend
import pi05_response_probe as feedback


ROOT = Path(os.environ.get(
    "PI05_TEST_PHYSICALRSI_ROOT",
    "/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007",
))
INSTRUCTION = "open the top drawer and put the bowl inside"


def state(z=.4, aperture=.04):
    return np.asarray([0., 0., z, 0., 0., 0., aperture / 2, -aperture / 2])


class Delegate:
    def __init__(self, open_call=None, z_action=-.4):
        self.open_call = open_call
        self.z_action = z_action
        self.calls = 0
        self.call_records = []
        self.provenance = {"kind": "fake_pi05"}
        self.episodes = []

    def begin_episode(self):
        self.calls = 0
        self.call_records = []

    def reset(self):
        pass

    def act(self, observation, instruction, memory):
        del observation, instruction, memory
        index = self.calls
        action = np.asarray([.01, -.02, self.z_action, .03, -.04, .05,
                             -.7 if index == self.open_call else .8])
        self.call_records.append({"inference_call": index,
                                  "policy_seed": 7 + index * 1_000_003})
        self.calls += 1
        return np.repeat(action[None], 5, axis=0)

    def finalize_episode(self, executed):
        self.episodes.append({"records": list(self.call_records),
                              "trace": tuple(executed)})


class Env:
    action_low = np.full(7, -np.inf)
    action_high = np.full(7, np.inf)
    provenance = {"kind": "fake"}

    def __init__(self, observation_type, terminal, *, moving=False,
                 height_response=False, aperture_response=False):
        self.Observation = observation_type
        self.terminal = terminal
        self.moving = moving
        self.height_response = height_response
        self.aperture_response = aperture_response
        self.steps = 0
        self.z = .4
        self.aperture = .04
        self.actions = []

    def observation(self):
        z = self.z + (self.steps * .001 if self.moving else 0.)
        done = self.steps >= self.terminal
        return self.Observation({"observation/state": state(z, self.aperture)}, {},
                                success=done, terminated=done)

    def reset(self, seed):
        del seed
        self.steps, self.z, self.aperture, self.actions = 0, .4, .04, []
        return self.observation()

    def step(self, action):
        value = np.asarray(action, dtype=np.float64).copy()
        self.actions.append(value)
        if self.height_response and value[2] >= .2:
            self.z += .01
        if self.aperture_response and value[2] >= .2:
            self.aperture = .006
        self.steps += 1
        return self.observation()

    def close(self):
        pass


def harness(cap):
    return {"schema": 1, "name": "closed-dwell-test", "remember": [], "stages": [{
        "skill": "pi05", "instruction": INSTRUCTION, "max_steps": cap,
        "until": None, "on_timeout": "abort",
    }]}


def run_case(*, terminal=230, cap=300, moving=False, height=False,
             aperture=False, open_call=None, wrapped=True, delegate=None,
             veto=False, z_action=-.4):
    api = backend.import_roborsi(ROOT)
    delegate = delegate or Delegate(open_call=open_call, z_action=z_action)
    skill = (lift.Pi05ClosedDwellLiftSkill(
        delegate, veto_native_upward_intent=veto) if wrapped else delegate)
    holder = {}

    def factory():
        env = Env(api.Observation, terminal, moving=moving,
                  height_response=height, aperture_response=aperture)
        holder["env"] = env
        return feedback.Pi05ExecutionFeedbackEnvironment(env, skill) if wrapped else env

    report = api.Runner(factory, {"pi05": skill}, max_steps=cap, max_seconds=10,
                        chunk_steps=5).run(harness(cap), 0)
    return report, holder["env"], skill, delegate


class ClosedDwellLiftTest(unittest.TestCase):
    def test_veto_parameter_is_strict_bool_and_default_is_legacy_false(self):
        self.assertFalse(lift.Pi05ClosedDwellLiftSkill(Delegate()).veto_native_upward_intent)
        for value in (0, 1, None, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lift.Pi05ClosedDwellLiftSkill(Delegate(), veto_native_upward_intent=value)

    def test_veto_has_false_cue_native_parity(self):
        native = run_case(terminal=180, moving=True, wrapped=False, z_action=.1)
        tested = run_case(terminal=180, moving=True, veto=True, z_action=.1)
        np.testing.assert_array_equal(native[1].actions, tested[1].actions)
        self.assertEqual(native[3].episodes[0]["records"], tested[3].episodes[0]["records"])
        self.assertFalse(tested[2].provenance["attempted"])

    def test_positive_native_z_permanently_vetoes_without_cache_or_ghosts(self):
        native = run_case(terminal=124, wrapped=False, z_action=.1)
        report, env, skill, delegate = run_case(terminal=124, veto=True, z_action=.1)
        p = skill.provenance
        self.assertTrue(p["attempted"])
        self.assertEqual(p["veto_reason"], "incoming_native_upward_intent")
        self.assertEqual(p["native_incoming_z"], .1)
        self.assertIsNone(p["first_changed_action_step"])
        self.assertEqual((p["assist_slots_executed"], p["modification_count"]), (0, 0))
        self.assertEqual((len(p["emitted_rows"]), len(p["executed_rows"])), (125, 124))
        self.assertTrue(p["emitted_rows"][-1]["truncated_before_execution"])
        self.assertEqual((report["steps"], len(env.actions), delegate.calls), (124, 124, 25))
        np.testing.assert_array_equal(env.actions, np.repeat(
            np.asarray([[.01, -.02, .1, .03, -.04, .05, .8]]), 124, axis=0))
        np.testing.assert_array_equal(env.actions, native[1].actions)
        self.assertEqual(delegate.episodes[0]["records"], native[3].episodes[0]["records"])
        self.assertTrue(p["execution_reconciled"])

    def test_negative_and_zero_native_z_are_not_vetoed(self):
        for value in (-.4, 0.):
            with self.subTest(value=value):
                _, _, skill, _ = run_case(veto=True, z_action=value)
                self.assertTrue(skill.provenance["attempted"])
                self.assertIsNone(skill.provenance["veto_reason"])
                self.assertEqual(skill.provenance["assist_slots_executed"], 10)
                self.assertEqual(skill.provenance["first_changed_action_step"], 120)

    def test_no_cue_is_exact_native_action_call_and_seed_equivalent(self):
        native = run_case(terminal=180, moving=True, wrapped=False)
        tested = run_case(terminal=180, moving=True)
        np.testing.assert_array_equal(native[1].actions, tested[1].actions)
        self.assertEqual(native[3].episodes[0]["records"], tested[3].episodes[0]["records"])
        self.assertFalse(tested[2].provenance["attempted"])

    def test_trigger_changes_only_z_for_ten_slots_and_keeps_cadence(self):
        report, env, skill, delegate = run_case()
        provenance = skill.provenance
        self.assertEqual((report["steps"], len(env.actions)), (230, 230))
        self.assertEqual(provenance["first_changed_action_step"], 120)
        self.assertEqual(provenance["assist_slots_executed"], 10)
        self.assertEqual(provenance["modification_count"], 10)
        self.assertEqual(provenance["early_stop_reason"], "assist_slot_limit")
        raw = np.asarray(provenance["executed_rows"][120]["raw_action"])
        for index in range(120, 130):
            actual = env.actions[index]
            np.testing.assert_array_equal(actual[[0, 1, 3, 4, 5, 6]],
                                          raw[[0, 1, 3, 4, 5, 6]])
            self.assertEqual(actual[2], .2)
        self.assertEqual(env.actions[119][2], -.4)
        self.assertEqual(env.actions[130][2], -.4)
        self.assertEqual(len(delegate.episodes[0]["records"]), 46)
        self.assertEqual([row["cache_chunk_inference_index"]
                          for row in provenance["executed_rows"][120:130]],
                         [24] * 5 + [25] * 5)
        self.assertTrue(provenance["execution_reconciled"])

    def test_reserve_prevents_trigger(self):
        _, _, skill, _ = run_case(terminal=209, cap=209)
        self.assertFalse(skill.provenance["attempted"])
        self.assertEqual(skill.provenance["modification_count"], 0)

    def test_height_feedback_stops_before_next_action(self):
        _, env, skill, _ = run_case(height=True)
        self.assertEqual(skill.provenance["modification_count"], 3)
        self.assertEqual(skill.provenance["early_stop_reason"], "height_response")
        self.assertEqual(env.actions[123][2], -.4)

    def test_aperture_and_raw_gripper_each_stop_without_mutating_raw(self):
        _, env, skill, _ = run_case(aperture=True)
        self.assertEqual(skill.provenance["modification_count"], 1)
        self.assertEqual(skill.provenance["early_stop_reason"], "aperture_too_narrow")
        self.assertEqual(env.actions[121][2], -.4)
        _, env, skill, _ = run_case(open_call=25)
        self.assertEqual(skill.provenance["modification_count"], 5)
        self.assertEqual(skill.provenance["early_stop_reason"],
                         "incoming_raw_gripper_open")
        np.testing.assert_array_equal(env.actions[125],
                                      [.01, -.02, -.4, .03, -.04, .05, -.7])

    def test_terminal_has_no_ghost_execution_and_reconciles(self):
        report, env, skill, _ = run_case(terminal=124)
        provenance = skill.provenance
        self.assertEqual((report["steps"], len(env.actions)), (124, 124))
        self.assertEqual(len(provenance["executed_rows"]), 124)
        self.assertEqual(provenance["modification_count"], 4)
        self.assertEqual(provenance["emitted_modification_count"], 4)
        self.assertTrue(provenance["execution_reconciled"])

    def test_episode_state_is_independent(self):
        delegate = Delegate()
        first = run_case(terminal=130, delegate=delegate)
        second = run_case(terminal=130, delegate=delegate)
        self.assertEqual(first[2].provenance["first_changed_action_step"], 120)
        self.assertEqual(second[2].provenance["first_changed_action_step"], 120)
        self.assertEqual(delegate.episodes[0]["records"], delegate.episodes[1]["records"])

    def test_guard_uses_pre_state_and_rejects_nonfinite(self):
        rows = deque((np.asarray([0, 0, 0, 0, 0, 0, .8]), state())
                     for _ in range(60))
        context = {"actual_executed": 120, "remaining_episode": 90,
                   "stage_executed": 120, "remaining_stage": 90, "stage_index": 0}
        self.assertIsNotNone(lift.closed_dwell_guard(rows, state(), context))
        bad = list(rows); bad[0][1][0] = np.nan
        self.assertIsNone(lift.closed_dwell_guard(bad, state(), context))


if __name__ == "__main__":
    unittest.main()
