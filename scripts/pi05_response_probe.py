#!/usr/bin/env python3
"""Execution-feedback-only 7D response probe for the PI0.5 shared Runner."""

from __future__ import annotations

import copy
import math
from collections import deque
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


ACTION_DIM = 7
STATE_DIM = 8
WINDOW = 30
MIN_ACTUAL = 120
TRIGGER_REMAINING = 39       # 5 native grace + 14 manual + 20 native reserve
NATIVE_RESERVE = 20
GRACE = 5
MAX_LIFT = 8
MAX_MANUAL = 14
CONTEXT_KEYS = frozenset({
    "actual_executed", "remaining_episode", "stage_executed",
    "remaining_stage", "stage_index",
})
PARAMETER_DEFAULTS = {"lift_z_command": .05, "max_lift_steps": 8,
                      "lift_target_m": .02, "native_reserve_steps": 20,
                      "minimum_actual": MIN_ACTUAL}


def validated_parameters(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized constructor parameters, rejecting unknown or unsafe values."""
    values = {} if values is None else dict(values)
    if not set(values) <= set(PARAMETER_DEFAULTS):
        raise ValueError("unknown PI0.5 response-probe parameter")
    result = {**PARAMETER_DEFAULTS, **values}
    for name, upper in (("lift_z_command", .2), ("lift_target_m", .025)):
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"{name} must be finite and in (0, {upper}]")
        result[name] = float(value)
    for name, lower, upper in (("max_lift_steps", 1, 20),
                               ("native_reserve_steps", 20, 80),
                               ("minimum_actual", 60, 120)):
        value = result[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return result


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be finite {size}D")
    return out.copy()


def open_downward_stall_guard(
    native_rows: Any,
    state8: Any,
    context: Mapping[str, Any],
    minimum_remaining: int,
    *,
    minimum_actual: int = MIN_ACTUAL,
) -> dict[str, Any] | None:
    """Pure open/downward-stall predicate over executed action/post-state rows."""
    state = np.asarray(state8, dtype=np.float64)
    if (state.shape != (STATE_DIM,) or not np.isfinite(state).all()
            or context.get("actual_executed", -1) < minimum_actual
            or min(context.get("remaining_episode", -1),
                   context.get("remaining_stage", -1)) < minimum_remaining
            or len(native_rows) != WINDOW):
        return None
    actions = np.asarray([row[0] for row in native_rows], dtype=np.float64)
    states = np.asarray([row[1] for row in native_rows], dtype=np.float64)
    if (actions.shape != (WINDOW, ACTION_DIM) or states.shape != (WINDOW, STATE_DIM)
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        return None
    z_span = float(np.ptp(states[:, 2]))
    negative_z = int(np.count_nonzero(actions[:, 2] < -.2))
    aperture = abs(float(state[6] - state[7]))
    if z_span > .005 or np.any(actions[:, 6] > -.5) or negative_z < 24 or aperture < .075:
        return None
    return {"z_span_m": z_span, "negative_z_commands": negative_z,
            "maximum_gripper_command": float(actions[:, 6].max()),
            "aperture_m": aperture, "context": copy.deepcopy(dict(context))}


class Pi05ExecutionFeedbackEnvironment:
    """Transparent environment adapter reporting only action and post-state8."""

    def __init__(self, delegate: Any, probe: "Pi05ResponseProbeSkill"):
        self.delegate, self.probe = delegate, probe
        self.action_low = delegate.action_low
        self.action_high = delegate.action_high
        self.provenance = copy.deepcopy(getattr(delegate, "provenance", {}))
        self.provenance["pi05_response_feedback"] = "executed_action_and_post_state8_only"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def reset(self, init_id: int):
        observation = self.delegate.reset(init_id)
        self.probe.on_reset(observation.policy["observation/state"])
        return observation

    def step(self, action: np.ndarray):
        # Callback deliberately follows a successful real env.step: failed or
        # truncated emissions never become evidence.
        observation = self.delegate.step(action)
        self.probe.on_execution(action, observation.policy["observation/state"])
        return observation

    def close(self):
        return self.delegate.close()


def response_probe_environment_factory(factory: Any, probe: "Pi05ResponseProbeSkill"):
    """Bind the same episode-local probe skill to a transparent env factory."""
    if not callable(factory):
        raise TypeError("environment factory must be callable")
    return lambda: Pi05ExecutionFeedbackEnvironment(factory(), probe)


class Pi05ResponseProbeSkill:
    """One-shot mechanical response probe preserving native PI0.5 chunks."""

    def __init__(self, delegate: Any, *, lift_z_command: float = .05,
                 max_lift_steps: int = 8, lift_target_m: float = .02,
                 native_reserve_steps: int = 20, minimum_actual: int = MIN_ACTUAL):
        parameters = validated_parameters({"lift_z_command": lift_z_command,
            "max_lift_steps": max_lift_steps, "lift_target_m": lift_target_m,
            "native_reserve_steps": native_reserve_steps,
            "minimum_actual": minimum_actual})
        self.delegate = delegate
        self.lift_z_command = parameters["lift_z_command"]
        self.max_lift_steps = parameters["max_lift_steps"]
        self.lift_target_m = parameters["lift_target_m"]
        self.native_reserve_steps = parameters["native_reserve_steps"]
        self.minimum_actual = parameters["minimum_actual"]
        self.max_manual_actions = 4 + max_lift_steps + 2
        self.trigger_remaining = GRACE + self.max_manual_actions + native_reserve_steps
        self.provenance: dict[str, Any] = {}
        self._active = False

    @property
    def calls(self) -> int:
        return self.delegate.calls

    @property
    def call_records(self):
        return self.delegate.call_records
    def begin_episode(self):
        if self._active:
            raise RuntimeError("response probe episode is already active")
        self.delegate.begin_episode()
        self._active = True
        self._context: dict[str, int] | None = None
        self._fresh = False
        self._state: np.ndarray | None = None
        self._reset_states: list[list[float]] = []
        self._native = deque(maxlen=WINDOW)
        self._pending: deque[dict[str, Any]] = deque()
        self._emitted: list[dict[str, Any]] = []
        self._executed: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._attempted = False
        self._grace_pending = False
        self._phase = "native"
        self._phase_emitted = 0
        self._manual_emitted = 0
        self._close_start_aperture: float | None = None
        self._lift_start_z: float | None = None

    def reset(self):
        if not self._active:
            raise RuntimeError("response probe episode has not begun")
        if self._pending:
            for row in self._pending:
                row["truncated_before_execution"] = True
            self._pending.clear()
            if self._grace_pending:
                self._grace_pending = False
                self._events[-1]["status"] = "truncated_during_grace"
        self._context, self._fresh = None, False
        self.delegate.reset()

    def on_reset(self, state8: Any):
        if not self._active:
            raise RuntimeError("response probe episode has not begun")
        self._state = _vector(state8, STATE_DIM, "reset state")
        self._reset_states.append(self._state.tolist())

    def set_execution_context(self, context: Mapping[str, Any]):
        if type(context) is not MappingProxyType or set(context) != CONTEXT_KEYS:
            raise ValueError("response probe requires Runner's immutable execution context")
        if any(type(context[key]) is not int or context[key] < 0 for key in CONTEXT_KEYS):
            raise ValueError("execution context values must be nonnegative integers")
        self._context, self._fresh = dict(context), True

    @staticmethod
    def _aperture(state: np.ndarray) -> float:
        return abs(float(state[6] - state[7]))

    def _guard(self, minimum_remaining: int) -> dict[str, Any] | None:
        if self._context is None or self._state is None:
            return None
        return open_downward_stall_guard(
            self._native, self._state, self._context, minimum_remaining,
            minimum_actual=self.minimum_actual,
        )

    def _record_emission(self, actions: np.ndarray, kind: str):
        assert self._context is not None
        for offset, action in enumerate(actions):
            row = {"emission_index": len(self._emitted), "kind": kind,
                   "stage_index": self._context["stage_index"],
                   "expected_actual_step": self._context["actual_executed"] + offset,
                   "action": action.tolist(), "executed": False}
            self._emitted.append(row)
            self._pending.append(row)

    def _native_act(self, observation: Any, instruction: str, memory: dict[str, Any]):
        raw = np.asarray(self.delegate.act(observation, instruction, memory), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != ACTION_DIM or len(raw) < GRACE or not np.isfinite(raw).all():
            raise ValueError("PI0.5 delegate must return a finite >=5x7 chunk")
        self._record_emission(raw[:GRACE], "native")
        return raw

    def _manual_action(self, grip: float, z: float = 0.0) -> np.ndarray:
        value = np.asarray([[0., 0., z, 0., 0., 0., grip]], dtype=np.float64)
        self._record_emission(value, self._phase)
        self._phase_emitted += 1
        self._manual_emitted += 1
        if self._manual_emitted > self.max_manual_actions:
            raise AssertionError("PI0.5 response probe exceeded its manual-action bound")
        return value

    def _budget_ok(self, manual_remaining: int) -> bool:
        assert self._context is not None
        required = manual_remaining + self.native_reserve_steps
        return min(self._context["remaining_episode"], self._context["remaining_stage"]) >= required

    def _manual_or_fallback(self, observation: Any, instruction: str, memory: dict[str, Any]):
        while True:
            if self._phase == "close":
                if self._phase_emitted < 4:
                    if not self._budget_ok((4 - self._phase_emitted) + self.max_lift_steps + 2):
                        break
                    return self._manual_action(+1.)
                aperture = self._aperture(self._state)
                narrowed = self._close_start_aperture - aperture
                event = self._events[-1]
                event.update(close_end_aperture_m=aperture, close_narrowing_m=narrowed)
                if narrowed >= .010 and .008 <= aperture <= .070:
                    self._phase, self._phase_emitted = "lift", 0
                    self._lift_start_z = float(self._state[2])
                    event["close_response"] = True
                else:
                    self._phase, self._phase_emitted = "reopen", 0
                    event["close_response"] = False
            elif self._phase == "lift":
                if (self._phase_emitted >= self.max_lift_steps
                        or float(self._state[2]) - self._lift_start_z >= self.lift_target_m):
                    self._phase, self._phase_emitted = "settle", 0
                    continue
                if not self._budget_ok((self.max_lift_steps - self._phase_emitted) + 2):
                    break
                return self._manual_action(+1., self.lift_z_command)
            elif self._phase == "settle":
                if self._phase_emitted < 2:
                    if not self._budget_ok(2 - self._phase_emitted):
                        break
                    return self._manual_action(+1.)
                self._events[-1]["status"] = "manual_complete"
                self._phase = "complete"
            elif self._phase == "reopen":
                if self._phase_emitted < 2:
                    if not self._budget_ok(2 - self._phase_emitted):
                        break
                    return self._manual_action(-1.)
                self._events[-1]["status"] = "manual_complete_no_close_response"
                self._phase = "complete"
            if self._phase == "complete":
                return self._native_act(observation, instruction, memory)
        self._events[-1]["status"] = "manual_budget_fallback"
        self._phase = "complete"
        return self._native_act(observation, instruction, memory)

    def act(self, observation: Any, instruction: str, memory: dict[str, Any]):
        if not self._active or not self._fresh or self._context is None or self._state is None:
            raise RuntimeError("response probe requires a fresh Runner context and reset state")
        if self._pending:
            raise RuntimeError("Runner requested an action while prior emissions remain unexecuted")
        self._fresh = False
        if self._phase != "native":
            return self._manual_or_fallback(observation, instruction, memory)
        if self._grace_pending:
            self._grace_pending = False
            guard = self._guard(self.max_manual_actions + self.native_reserve_steps)
            if guard is not None:
                self._events[-1].update(status="manual_started", recheck=guard)
                self._phase, self._phase_emitted, self._manual_emitted = "close", 0, 0
                self._close_start_aperture = self._aperture(self._state)
                return self._manual_or_fallback(observation, instruction, memory)
            self._events[-1]["status"] = "permanent_veto_recheck_failed"
        elif not self._attempted:
            guard = self._guard(self.trigger_remaining)
            if guard is not None:
                self._attempted, self._grace_pending = True, True
                self._events.append({"event_index": 0, "status": "five_native_grace",
                                     "cue": guard, "mechanical_proxy_only": True,
                                     "grasp_or_task_success_certificate": False})
        return self._native_act(observation, instruction, memory)

    def on_execution(self, action: Any, post_state8: Any):
        value = _vector(action, ACTION_DIM, "executed action")
        state = _vector(post_state8, STATE_DIM, "post state")
        if not self._pending:
            raise RuntimeError("execution callback has no matching emitted action")
        row = self._pending.popleft()
        if not np.array_equal(value, np.asarray(row["action"], dtype=np.float64)):
            raise ValueError("executed action differs from the emitted action")
        row.update(executed=True, post_state8=state.tolist(),
                   execution_index=len(self._executed))
        self._executed.append(row)
        self._state = state
        if row["kind"] == "native":
            self._native.append((value, state.copy()))

    def finalize_episode(self, executed_trace=()):
        if not self._active:
            raise RuntimeError("response probe finalization is single-use")
        expected = tuple((row["expected_actual_step"], row["stage_index"])
                         for row in self._executed)
        supplied = tuple(executed_trace)
        reconciled = supplied == expected
        if self._pending:
            for row in self._pending:
                row["truncated_before_execution"] = True
            self._pending.clear()
        try:
            self.delegate.finalize_episode(executed_trace)
        finally:
            self.provenance = {"kind": "pi05_response_probe_v1",
                "delegate": copy.deepcopy(getattr(self.delegate, "provenance", {})),
                "parameters": {"lift_z_command": self.lift_z_command,
                    "max_lift_steps": self.max_lift_steps,
                    "lift_target_m": self.lift_target_m,
                    "native_reserve_steps": self.native_reserve_steps,
                    "minimum_actual": self.minimum_actual,
                    "max_manual_actions": self.max_manual_actions,
                    "trigger_remaining_steps": self.trigger_remaining},
                "mechanical_proxy_only": True, "grasp_or_task_success_certificate": False,
                "attempted": self._attempted, "events": copy.deepcopy(self._events),
                "reset_state8": copy.deepcopy(self._reset_states),
                "emitted_rows": copy.deepcopy(self._emitted),
                "executed_rows": copy.deepcopy(self._executed),
                "execution_reconciled": reconciled, "manual_actions_emitted": self._manual_emitted}
            self._active = False
        if not reconciled:
            raise ValueError("Runner trace and response-probe execution callbacks disagree")
