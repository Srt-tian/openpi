#!/usr/bin/env python3
"""Physical transport-release witness followed by task-level semantic resume."""
from __future__ import annotations

import copy
from collections import deque
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

KIND = "transport_release_semantic_resume_v1"
ACTION_DIM, STATE_DIM = 7, 8
HOLD_ROWS, RELEASE_ROWS, REPLAN_STEPS = 10, 5, 5
APERTURE_HOLD = (.008, .070)
APERTURE_RELEASE = .075
TRANSPORT_XY_M = .10
NATIVE_RESERVE = 80
CONTEXT_KEYS = frozenset({"actual_executed", "remaining_episode", "stage_executed",
                          "remaining_stage", "stage_index"})


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be finite {size}D")
    return out.copy()


class Pi05TransportReleaseSemanticResumeSkill:
    """Transparent native skill; changes only prompts after a physical proxy."""

    def __init__(self, delegate: Any, *, resume_instruction: str):
        if not isinstance(resume_instruction, str) or not resume_instruction.strip() or len(resume_instruction) > 512:
            raise ValueError("resume_instruction must be a nonempty <=512 character string")
        self.delegate, self.resume_instruction = delegate, resume_instruction
        self.provenance: dict[str, Any] = {}
        self._active = False

    @property
    def calls(self): return self.delegate.calls
    @property
    def call_records(self): return self.delegate.call_records

    def begin_episode(self):
        if self._active: raise RuntimeError("semantic-resume episode is already active")
        self.delegate.begin_episode(); self._active = True
        self._context = None; self._fresh = False; self._state = None
        self._pending = deque(); self._emitted = []; self._executed = []; self._calls = []
        self._hold = 0; self._anchor = None; self._transported = False; self._release = 0
        self._anchor_reset_used = False; self._detection_vetoed = False
        self._witness_step = None; self._resume_step = None; self._resume_call = None
        self._original_instruction = None

    def reset(self):
        if not self._active: raise RuntimeError("semantic-resume episode has not begun")
        for row in self._pending: row["truncated_before_execution"] = True
        self._pending.clear(); self._context = None; self._fresh = False
        self.delegate.reset()

    def on_reset(self, state8: Any):
        if not self._active: raise RuntimeError("semantic-resume episode has not begun")
        self._state = _vector(state8, STATE_DIM, "reset state")

    def set_execution_context(self, context: Mapping[str, Any]):
        if type(context) is not MappingProxyType or set(context) != CONTEXT_KEYS:
            raise ValueError("semantic resume requires Runner's immutable execution context")
        if any(type(context[k]) is not int or context[k] < 0 for k in CONTEXT_KEYS):
            raise ValueError("execution context values must be nonnegative integers")
        self._context, self._fresh = dict(context), True

    @staticmethod
    def _aperture(state): return abs(float(state[6] - state[7]))

    def act(self, observation: Any, instruction: str, memory: dict[str, Any]):
        if not self._active or not self._fresh or self._context is None or self._state is None:
            raise RuntimeError("semantic resume requires a fresh Runner context and reset state")
        if self._pending: raise RuntimeError("prior native emissions remain unexecuted")
        self._fresh = False
        if self._original_instruction is None: self._original_instruction = instruction
        elif instruction != self._original_instruction: raise ValueError("harness instruction changed unexpectedly")
        use_resume = self._resume_step is not None or (self._witness_step is not None and min(
            self._context["remaining_episode"], self._context["remaining_stage"]) >= NATIVE_RESERVE)
        prompt = self.resume_instruction if use_resume else instruction
        if use_resume and self._resume_step is None:
            self._resume_step = self._context["actual_executed"]; self._resume_call = self.delegate.calls
        raw = np.asarray(self.delegate.act(observation, prompt, memory), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != ACTION_DIM or len(raw) < REPLAN_STEPS or not np.isfinite(raw).all():
            raise ValueError("PI0.5 delegate must return a finite >=5x7 chunk")
        self._calls.append({"inference_call": self.delegate.calls - 1,
            "actual_step": self._context["actual_executed"], "prompt": prompt,
            "prompt_changed": prompt != instruction})
        for offset, action in enumerate(raw[:REPLAN_STEPS]):
            row = {"emission_index": len(self._emitted), "expected_actual_step": self._context["actual_executed"] + offset,
                   "stage_index": self._context["stage_index"], "action": action.tolist(), "executed": False}
            self._emitted.append(row); self._pending.append(row)
        return raw

    def on_execution(self, action: Any, post_state8: Any):
        value = _vector(action, ACTION_DIM, "executed action"); state = _vector(post_state8, STATE_DIM, "post state")
        if not self._pending: raise RuntimeError("execution callback has no matching emitted action")
        row = self._pending.popleft()
        if not np.array_equal(value, np.asarray(row["action"])): raise ValueError("executed action differs from emitted action")
        row.update(executed=True, execution_index=len(self._executed), post_state8=state.tolist())
        self._executed.append(row); self._state = state
        if self._witness_step is not None or self._detection_vetoed: return
        aperture = self._aperture(state)
        if self._anchor is None:
            self._hold = self._hold + 1 if value[6] >= .5 and APERTURE_HOLD[0] <= aperture <= APERTURE_HOLD[1] else 0
            if self._hold >= HOLD_ROWS: self._anchor = state[:2].copy()
            return
        if not self._transported:
            if float(np.linalg.norm(state[:2] - self._anchor)) >= TRANSPORT_XY_M: self._transported = True
            elif aperture >= APERTURE_RELEASE:
                if not self._anchor_reset_used:
                    self._anchor_reset_used = True; self._anchor = None; self._hold = 0
                else: self._detection_vetoed = True
            return
        self._release = self._release + 1 if value[6] <= -.5 and aperture >= APERTURE_RELEASE else 0
        if self._release >= RELEASE_ROWS:
            self._witness_step = row["expected_actual_step"] + 1

    def finalize_episode(self, executed_trace=()):
        if not self._active: raise RuntimeError("semantic-resume finalization is single-use")
        expected = tuple((r["expected_actual_step"], r["stage_index"]) for r in self._executed)
        reconciled = tuple(executed_trace) == expected
        for row in self._pending: row["truncated_before_execution"] = True
        self._pending.clear()
        try: self.delegate.finalize_episode(executed_trace)
        finally:
            self.provenance = {"kind": KIND, "delegate": copy.deepcopy(getattr(self.delegate,"provenance",{})),
                "parameters": {"hold_rows": HOLD_ROWS, "hold_aperture_m": list(APERTURE_HOLD),
                    "transport_xy_m": TRANSPORT_XY_M, "release_rows": RELEASE_ROWS,
                    "release_aperture_m": APERTURE_RELEASE, "native_reserve_steps": NATIVE_RESERVE},
                "mechanical_proxy_only": True, "object_or_task_completion_certificate": False,
                "witness_actual_step": self._witness_step, "first_prompt_change_actual_step": self._resume_step,
                "first_prompt_change_call": self._resume_call, "anchor_reset_used": self._anchor_reset_used,
                "detection_vetoed": self._detection_vetoed, "call_prompt_receipts": copy.deepcopy(self._calls),
                "emitted_rows": copy.deepcopy(self._emitted), "executed_rows": copy.deepcopy(self._executed),
                "execution_reconciled": reconciled}
            self._active = False
        if not reconciled: raise ValueError("Runner trace and semantic-resume callbacks disagree")


class FeedbackEnvironment:
    def __init__(self, delegate, skill):
        self.delegate, self.skill = delegate, skill
        self.action_low, self.action_high = delegate.action_low, delegate.action_high
        self.provenance = copy.deepcopy(getattr(delegate, "provenance", {}))
        self.provenance["pi05_semantic_resume_feedback"] = "executed_action_and_post_state8_only"
    def __getattr__(self,name): return getattr(self.delegate,name)
    def reset(self,init_id):
        obs=self.delegate.reset(init_id); self.skill.on_reset(obs.policy["observation/state"]); return obs
    def step(self,action):
        obs=self.delegate.step(action); self.skill.on_execution(action,obs.policy["observation/state"]); return obs
    def close(self): return self.delegate.close()


def environment_factory(factory, skill):
    if not callable(factory): raise TypeError("environment factory must be callable")
    return lambda: FeedbackEnvironment(factory(), skill)
