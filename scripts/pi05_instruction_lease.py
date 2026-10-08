#!/usr/bin/env python3
"""Context-guarded bounded instruction lease for the PI0.5 shared Runner."""

from __future__ import annotations

import copy
from collections import deque
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pi05_response_probe import (
    ACTION_DIM,
    CONTEXT_KEYS,
    GRACE,
    STATE_DIM,
    WINDOW,
    open_downward_stall_guard,
)


KIND = "instruction_lease_v1"
LEASE_STEPS = frozenset((40, 80))
ORIGINAL_RESERVE = 80


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be finite {size}D")
    return out.copy()


class Pi05InstructionLeaseSkill:
    """One-shot prompt lease; all emitted actions remain delegate-native."""

    def __init__(self, delegate: Any, *, instruction: str, lease_steps: int):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 512:
            raise ValueError("lease instruction must be a nonempty string of at most 512 characters")
        if type(lease_steps) is not int or lease_steps not in LEASE_STEPS:
            raise ValueError("lease_steps must be exactly 40 or 80")
        self.delegate = delegate
        self.instruction = instruction
        self.lease_steps = lease_steps
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
            raise RuntimeError("instruction lease episode is already active")
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
        self._prompt_calls: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._attempted = False
        self._grace_pending = False
        self._lease_active = False
        self._leased_executed = 0
        self._first_changed_prompt_step: int | None = None

    def reset(self):
        if not self._active:
            raise RuntimeError("instruction lease episode has not begun")
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
            raise RuntimeError("instruction lease episode has not begun")
        self._state = _vector(state8, STATE_DIM, "reset state")
        self._reset_states.append(self._state.tolist())

    def set_execution_context(self, context: Mapping[str, Any]):
        if type(context) is not MappingProxyType or set(context) != CONTEXT_KEYS:
            raise ValueError("instruction lease requires Runner's immutable execution context")
        if any(type(context[key]) is not int or context[key] < 0 for key in CONTEXT_KEYS):
            raise ValueError("execution context values must be nonnegative integers")
        self._context, self._fresh = dict(context), True

    def _guard(self, minimum_remaining: int) -> dict[str, Any] | None:
        if self._context is None or self._state is None:
            return None
        return open_downward_stall_guard(
            self._native, self._state, self._context, minimum_remaining
        )

    def _delegate_act(
        self,
        observation: Any,
        original_instruction: str,
        memory: dict[str, Any],
        used_instruction: str,
        leased: bool,
    ) -> np.ndarray:
        before = self.delegate.calls
        raw = np.asarray(
            self.delegate.act(observation, used_instruction, memory), dtype=np.float64
        )
        if raw.ndim != 2 or raw.shape[1] != ACTION_DIM or len(raw) < GRACE or not np.isfinite(raw).all():
            raise ValueError("PI0.5 delegate must return a finite >=5x7 chunk")
        if self.delegate.calls != before + 1:
            raise RuntimeError("PI0.5 delegate must consume exactly one call per native chunk")
        assert self._context is not None
        call = {"prompt_call_index": len(self._prompt_calls),
                "delegate_call_index": before,
                "actual_step": self._context["actual_executed"],
                "planned_prompt": self.instruction if leased else original_instruction,
                "used_prompt": used_instruction, "original_prompt": original_instruction,
                "leased": leased, "emitted_steps": GRACE, "executed_steps": 0}
        self._prompt_calls.append(call)
        for offset, action in enumerate(raw[:GRACE]):
            row = {"emission_index": len(self._emitted), "kind": "leased" if leased else "native",
                   "prompt_call_index": call["prompt_call_index"],
                   "stage_index": self._context["stage_index"],
                   "expected_actual_step": self._context["actual_executed"] + offset,
                   "action": action.tolist(), "executed": False}
            self._emitted.append(row)
            self._pending.append(row)
        return raw

    def act(self, observation: Any, instruction: str, memory: dict[str, Any]):
        if (not self._active or not self._fresh or self._context is None
                or self._state is None):
            raise RuntimeError("instruction lease requires a fresh Runner context and reset state")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("Runner original instruction must be nonempty")
        if self._pending:
            raise RuntimeError("Runner requested an action while prior emissions remain unexecuted")
        self._fresh = False
        if self._lease_active:
            remaining = self.lease_steps - self._leased_executed
            if remaining <= 0:
                self._lease_active = False
                self._events[-1]["status"] = "lease_complete"
            elif min(self._context["remaining_episode"], self._context["remaining_stage"]) < (
                    remaining + ORIGINAL_RESERVE):
                self._lease_active = False
                self._events[-1]["status"] = "lease_budget_fallback"
            else:
                return self._delegate_act(
                    observation, instruction, memory, self.instruction, True
                )
        if self._grace_pending:
            self._grace_pending = False
            guard = self._guard(self.lease_steps + ORIGINAL_RESERVE)
            if guard is not None:
                self._lease_active = True
                self._first_changed_prompt_step = self._context["actual_executed"]
                self._events[-1].update(status="lease_started", recheck=guard,
                                        first_changed_prompt_step=self._first_changed_prompt_step)
                return self._delegate_act(
                    observation, instruction, memory, self.instruction, True
                )
            self._events[-1]["status"] = "permanent_veto_recheck_failed"
        elif not self._attempted:
            guard = self._guard(GRACE + self.lease_steps + ORIGINAL_RESERVE)
            if guard is not None:
                self._attempted, self._grace_pending = True, True
                self._events.append({"event_index": 0, "status": "five_native_grace",
                                     "cue": guard, "actions_transformed": False,
                                     "oracle_inputs": []})
        return self._delegate_act(observation, instruction, memory, instruction, False)

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
        self._prompt_calls[row["prompt_call_index"]]["executed_steps"] += 1
        self._state = state
        if row["kind"] == "leased":
            self._leased_executed += 1
            if self._leased_executed > self.lease_steps:
                raise AssertionError("instruction lease exceeded its actual execution bound")
        else:
            self._native.append((value, state.copy()))

    def finalize_episode(self, executed_trace=()):
        if not self._active:
            raise RuntimeError("instruction lease finalization is single-use")
        expected = tuple((row["expected_actual_step"], row["stage_index"])
                         for row in self._executed)
        reconciled = tuple(executed_trace) == expected
        if self._pending:
            for row in self._pending:
                row["truncated_before_execution"] = True
            self._pending.clear()
        if self._lease_active and self._events:
            self._events[-1]["status"] = ("lease_complete_at_terminal"
                if self._leased_executed == self.lease_steps
                else "terminal_truncated_during_lease")
        elif self._grace_pending and self._events:
            self._events[-1]["status"] = "terminal_after_grace_before_recheck"
        try:
            self.delegate.finalize_episode(executed_trace)
        finally:
            self.provenance = {"kind": KIND,
                "delegate": copy.deepcopy(getattr(self.delegate, "provenance", {})),
                "configured_instruction": self.instruction,
                "planned_lease_steps": self.lease_steps,
                "actual_leased_steps": self._leased_executed,
                "original_native_reserve": ORIGINAL_RESERVE,
                "first_changed_prompt_step": self._first_changed_prompt_step,
                "attempted": self._attempted, "events": copy.deepcopy(self._events),
                "prompt_call_records": copy.deepcopy(self._prompt_calls),
                "reset_state8": copy.deepcopy(self._reset_states),
                "emitted_rows": copy.deepcopy(self._emitted),
                "executed_rows": copy.deepcopy(self._executed),
                "execution_reconciled": reconciled, "actions_transformed": False,
                "oracle_inputs": []}
            self._active = False
        if not reconciled:
            raise ValueError("Runner trace and instruction-lease execution callbacks disagree")
