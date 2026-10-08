#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

import pi05_harness_backend as backend
import pi05_instruction_lease as lease
import pi05_response_probe as response


ORIGINAL = "open the top drawer and put the bowl inside"
LEASED = "open the top drawer"
ROOT = Path(os.environ.get(
    "PI05_TEST_PHYSICALRSI_ROOT",
    "/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007",
))


def state(aperture=.08):
    return np.asarray([0., 0., .4, 0., 0., 0., aperture / 2, -aperture / 2])


class Delegate:
    def __init__(self):
        self.calls = 0
        self.call_records = []
        self.prompts = []
        self.episode_prompts = []
        self.begin_count = self.final_count = 0
        self.provenance = {"kind": "fake_pi05"}

    def begin_episode(self):
        self.calls = 0
        self.prompts = []
        self.call_records = []
        self.begin_count += 1

    def reset(self):
        pass

    def act(self, observation, instruction, memory):
        del observation, memory
        seed = 7 + self.calls * 1_000_003
        self.prompts.append(instruction)
        self.call_records.append({"inference_call": self.calls, "policy_seed": seed})
        self.calls += 1
        action = np.asarray([0., 0., -.8, 0., 0., 0., -.8])
        return np.repeat(action[None], 5, axis=0)

    def finalize_episode(self, executed):
        self.final_trace = tuple(executed)
        self.episode_prompts.append(list(self.prompts))
        self.final_count += 1


class Env:
    action_low = np.full(7, -np.inf)
    action_high = np.full(7, np.inf)
    provenance = {"kind": "fake"}

    def __init__(self, observation_type, terminal_step, veto_after_grace=False, stall=True):
        self.Observation = observation_type
        self.terminal_step = terminal_step
        self.veto_after_grace = veto_after_grace
        self.stall = stall
        self.steps = 0
        self.actions = []

    def observation(self):
        aperture = .01 if self.veto_after_grace and self.steps >= 125 else .08
        value = state(aperture)
        if not self.stall:
            value[2] = .4 + self.steps * .001
        return self.Observation({"observation/state": value}, {},
            success=self.steps >= self.terminal_step,
            terminated=self.steps >= self.terminal_step)

    def reset(self, seed):
        del seed
        self.steps, self.actions = 0, []
        return self.observation()

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.steps += 1
        return self.observation()

    def close(self):
        pass


def harness(cap=300):
    return {"schema": 1, "name": "lease-test", "remember": [], "stages": [{
        "skill": "pi05", "instruction": ORIGINAL, "max_steps": cap,
        "until": None, "on_timeout": "abort",
    }]}


def run_case(lease_steps, terminal=220, cap=300, *, veto=False, stall=True,
             delegate=None, wrapped=True):
    api = backend.import_roborsi(ROOT)
    delegate = delegate or Delegate()
    skill = lease.Pi05InstructionLeaseSkill(
        delegate, instruction=LEASED, lease_steps=lease_steps
    ) if wrapped else delegate
    holder = {}

    def factory():
        env = Env(api.Observation, terminal, veto_after_grace=veto, stall=stall)
        holder["env"] = env
        return response.Pi05ExecutionFeedbackEnvironment(env, skill) if wrapped else env

    report = api.Runner(factory, {"pi05": skill}, max_steps=cap, max_seconds=10,
                        chunk_steps=5).run(harness(cap), 0)
    return report, holder["env"], skill, delegate


class InstructionLeaseTest(unittest.TestCase):
    def test_no_trigger_is_native_action_prompt_and_seed_equivalent(self):
        native = run_case(40, terminal=140, stall=False, wrapped=False)
        tested = run_case(40, terminal=140, stall=False)
        np.testing.assert_array_equal(native[1].actions, tested[1].actions)
        self.assertEqual(native[3].episode_prompts, tested[3].episode_prompts)
        self.assertEqual(native[3].call_records, tested[3].call_records)
        self.assertFalse(tested[2].provenance["attempted"])

    def test_lease40_changes_exact_actual_steps_then_restores(self):
        report, env, skill, delegate = run_case(40, terminal=220)
        self.assertEqual(report["steps"], len(env.actions))
        self.assertEqual(skill.provenance["first_changed_prompt_step"], 125)
        self.assertEqual(skill.provenance["actual_leased_steps"], 40)
        self.assertEqual(delegate.episode_prompts[0][25:33], [LEASED] * 8)
        self.assertEqual(delegate.episode_prompts[0][24], ORIGINAL)
        self.assertEqual(delegate.episode_prompts[0][33], ORIGINAL)
        self.assertTrue(skill.provenance["execution_reconciled"])

    def test_lease80_changes_exact_actual_steps_then_restores(self):
        _, _, skill, delegate = run_case(80, terminal=230)
        self.assertEqual(skill.provenance["actual_leased_steps"], 80)
        self.assertEqual(delegate.episode_prompts[0][25:41], [LEASED] * 16)
        self.assertEqual(delegate.episode_prompts[0][41], ORIGINAL)

    def test_terminal_truncates_lease_without_ghost_actions(self):
        report, env, skill, delegate = run_case(40, terminal=137)
        self.assertEqual((report["steps"], len(env.actions)), (137, 137))
        self.assertEqual(skill.provenance["actual_leased_steps"], 12)
        calls = skill.provenance["prompt_call_records"]
        self.assertEqual(calls[-1]["executed_steps"], 2)
        self.assertEqual(skill.provenance["events"][0]["status"],
                         "terminal_truncated_during_lease")
        self.assertEqual(len(delegate.episode_prompts[0]), 28)

    def test_done_on_last_leased_step_is_complete_not_truncated(self):
        _, env, skill, _ = run_case(40, terminal=165)
        self.assertEqual(len(env.actions), 165)
        self.assertEqual(skill.provenance["actual_leased_steps"], 40)
        self.assertEqual(skill.provenance["events"][0]["status"],
                         "lease_complete_at_terminal")

    def test_guard_reserve_and_grace_veto_are_one_shot(self):
        _, _, reserve, reserve_delegate = run_case(40, terminal=200, cap=200)
        self.assertFalse(reserve.provenance["attempted"])
        self.assertTrue(all(prompt == ORIGINAL for prompt in reserve_delegate.episode_prompts[0]))
        _, _, veto, veto_delegate = run_case(40, terminal=220, veto=True)
        self.assertTrue(veto.provenance["attempted"])
        self.assertEqual(len(veto.provenance["events"]), 1)
        self.assertEqual(veto.provenance["events"][0]["status"],
                         "permanent_veto_recheck_failed")
        self.assertTrue(all(prompt == ORIGINAL for prompt in veto_delegate.episode_prompts[0]))

    def test_episode_state_and_delegate_seed_index_reset_independently(self):
        api = backend.import_roborsi(ROOT)
        delegate = Delegate()
        skill = lease.Pi05InstructionLeaseSkill(delegate, instruction=LEASED, lease_steps=40)
        environments = []
        def factory():
            env = Env(api.Observation, 140, stall=False)
            environments.append(env)
            return response.Pi05ExecutionFeedbackEnvironment(env, skill)
        runner = api.Runner(factory, {"pi05": skill}, max_steps=300, max_seconds=10,
                            chunk_steps=5)
        runner.run(harness(), 0)
        runner.run(harness(), 0)
        self.assertEqual((delegate.begin_count, delegate.final_count), (2, 2))
        self.assertEqual(delegate.episode_prompts[0], delegate.episode_prompts[1])
        self.assertEqual(len(environments[0].actions), len(environments[1].actions))

    def test_constructor_rejects_unbounded_instruction_or_lease(self):
        for instruction in ("", " " * 3, "x" * 513):
            with self.assertRaises(ValueError):
                lease.Pi05InstructionLeaseSkill(Delegate(), instruction=instruction, lease_steps=40)
        for steps in (0, 39, 60, 80.0, True):
            with self.assertRaises(ValueError):
                lease.Pi05InstructionLeaseSkill(Delegate(), instruction=LEASED, lease_steps=steps)


if __name__ == "__main__":
    unittest.main()
