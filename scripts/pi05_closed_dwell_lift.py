#!/usr/bin/env python3
"""Bounded closed-dwell lift assist for the PI0.5 shared Runner."""

from __future__ import annotations

import copy
from collections import deque
import hashlib
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pi05_response_probe import ACTION_DIM, CONTEXT_KEYS, GRACE, STATE_DIM


KIND = "closed_dwell_lift_v1"
WINDOW = 60
MIN_ACTUAL = 120
MIN_REMAINING = 90
NATIVE_RESERVE = 80
MAX_ASSIST_SLOTS = 10
MIN_CLOSED_ROWS = 57
MAX_XYZ_PTP = .012
MIN_APERTURE = .012
MAX_APERTURE = .070
LIFT_DELTA = .2
STOP_HEIGHT_DELTA = .025


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be finite {size}D")
    return out.copy()


def closed_dwell_guard(
    native_rows: Any,
    state8: Any,
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Pure cue over executed native actions and their pre-action state8."""
    state = np.asarray(state8, dtype=np.float64)
    if (state.shape != (STATE_DIM,) or not np.isfinite(state).all()
            or context.get("actual_executed", -1) < MIN_ACTUAL
            or min(context.get("remaining_episode", -1),
                   context.get("remaining_stage", -1)) < MIN_REMAINING
            or len(native_rows) != WINDOW):
        return None
    actions = np.asarray([row[0] for row in native_rows], dtype=np.float64)
    states = np.asarray([row[1] for row in native_rows], dtype=np.float64)
    if (actions.shape != (WINDOW, ACTION_DIM) or states.shape != (WINDOW, STATE_DIM)
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        return None
    xyz_ptp = np.ptp(states[:, :3], axis=0)
    closed_rows = int(np.count_nonzero(actions[:, 6] >= .5))
    aperture = abs(float(state[6] - state[7]))
    if (np.any(xyz_ptp > MAX_XYZ_PTP) or closed_rows < MIN_CLOSED_ROWS
            or not MIN_APERTURE <= aperture <= MAX_APERTURE):
        return None
    return {"xyz_ptp_m": xyz_ptp.tolist(), "closed_command_rows": closed_rows,
            "aperture_m": aperture, "window": WINDOW,
            "context": copy.deepcopy(dict(context)),
            "inputs": "executed_native_action_and_pre_state8_only"}


class Pi05ClosedDwellLiftSkill:
    """One-shot lift-only transform while preserving native inference cadence."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.provenance: dict[str, Any] = {}
        self._active = False

    @property
    def calls(self) -> int:
        return self.delegate.calls

    @property
    def call_records(self):
        return self.delegate.call_records

    @staticmethod
    def _aperture(state: np.ndarray) -> float:
        return abs(float(state[6] - state[7]))

    def begin_episode(self):
        if self._active:
            raise RuntimeError("closed-dwell lift episode is already active")
        self.delegate.begin_episode()
        self._active = True
        self._context: dict[str, int] | None = None
        self._fresh = False
        self._state: np.ndarray | None = None
        self._native = deque(maxlen=WINDOW)
        self._pending: deque[dict[str, Any]] = deque()
        self._cache: deque[dict[str, Any]] = deque()
        self._emitted: list[dict[str, Any]] = []
        self._executed: list[dict[str, Any]] = []
        self._chunks: list[dict[str, Any]] = []
        self._cue: dict[str, Any] | None = None
        self._trigger_z: float | None = None
        self._attempted = False
        self._assist_active = False
        self._assist_slots = 0
        self._modifications_emitted = 0
        self._modifications_executed = 0
        self._first_changed_action_step: int | None = None
        self._early_stop_reason: str | None = None

    def reset(self):
        if not self._active:
            raise RuntimeError("closed-dwell lift episode has not begun")
        for row in self._pending:
            row["truncated_before_execution"] = True
        self._pending.clear()
        self._cache.clear()
        self._context, self._fresh = None, False
        self.delegate.reset()

    def on_reset(self, state8: Any):
        if not self._active:
            raise RuntimeError("closed-dwell lift episode has not begun")
        self._state = _vector(state8, STATE_DIM, "reset state")

    def set_execution_context(self, context: Mapping[str, Any]):
        if type(context) is not MappingProxyType or set(context) != CONTEXT_KEYS:
            raise ValueError("closed-dwell lift requires Runner's immutable execution context")
        if any(type(context[key]) is not int or context[key] < 0 for key in CONTEXT_KEYS):
            raise ValueError("execution context values must be nonnegative integers")
        self._context, self._fresh = dict(context), True

    def _delegate_chunk(self, observation: Any, instruction: str, memory: dict[str, Any]):
        before = self.delegate.calls
        raw = np.asarray(self.delegate.act(observation, instruction, memory), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != ACTION_DIM or len(raw) < GRACE or not np.isfinite(raw).all():
            raise ValueError("PI0.5 delegate must return a finite >=5x7 chunk")
        if self.delegate.calls != before + 1:
            raise RuntimeError("PI0.5 delegate must consume exactly one call per native chunk")
        chunk = {"chunk_index": len(self._chunks), "delegate_call_index": before,
                 "inference_actual_step": self._context["actual_executed"],
                 "raw_action_sha256": hashlib.sha256(
                     np.ascontiguousarray(raw[:GRACE]).tobytes()).hexdigest(),
                 "emitted_rows": 0, "executed_rows": 0}
        self._chunks.append(chunk)
        return raw, chunk

    def _row(self, raw: np.ndarray, action: np.ndarray, chunk: dict[str, Any], kind: str):
        modified = not np.array_equal(raw, action)
        row = {"emission_index": len(self._emitted), "kind": kind,
               "stage_index": self._context["stage_index"],
               "expected_actual_step": self._context["actual_executed"],
               "cache_chunk_inference_index": chunk["delegate_call_index"],
               "raw_action": raw.tolist(), "executed_action": action.tolist(),
               "raw_action_sha256": hashlib.sha256(
                   np.ascontiguousarray(raw).tobytes()).hexdigest(),
               "modified": modified, "executed": False}
        self._emitted.append(row)
        self._pending.append(row)
        chunk["emitted_rows"] += 1
        if modified:
            self._modifications_emitted += 1
        return action[None]

    def _stop_reason(self, raw: np.ndarray) -> str | None:
        if raw[6] < .5:
            return "incoming_raw_gripper_open"
        if float(self._state[2]) - self._trigger_z >= STOP_HEIGHT_DELTA:
            return "height_response"
        aperture = self._aperture(self._state)
        if aperture < .008:
            return "aperture_too_narrow"
        if aperture > MAX_APERTURE:
            return "aperture_too_wide"
        return None

    def _emit_assist_raw(self, raw: np.ndarray, chunk: dict[str, Any]):
        reason = self._stop_reason(raw)
        if reason is not None:
            self._assist_active, self._early_stop_reason = False, reason
            return self._row(raw, raw.copy(), chunk, "assist_early_stop_raw")
        action = raw.copy()
        action[2] = max(float(raw[2]), LIFT_DELTA)
        return self._row(raw, action, chunk, "assist")

    def act(self, observation: Any, instruction: str, memory: dict[str, Any]):
        if (not self._active or not self._fresh or self._context is None
                or self._state is None):
            raise RuntimeError("closed-dwell lift requires fresh Runner context/reset state")
        if self._pending:
            raise RuntimeError("Runner requested action while prior emissions remain unexecuted")
        self._fresh = False
        if self._cache:
            cached = self._cache.popleft()
            if self._assist_active and self._assist_slots < MAX_ASSIST_SLOTS:
                return self._emit_assist_raw(cached["raw"], cached["chunk"])
            return self._row(cached["raw"], cached["raw"].copy(), cached["chunk"], "cached_raw")

        cue = None if self._attempted else closed_dwell_guard(
            self._native, self._state, self._context)
        raw, chunk = self._delegate_chunk(observation, instruction, memory)
        if cue is not None:
            self._attempted = self._assist_active = True
            self._cue, self._trigger_z = cue, float(self._state[2])
        if self._assist_active and self._assist_slots < MAX_ASSIST_SLOTS:
            self._cache.extend({"raw": row.copy(), "chunk": chunk} for row in raw[1:GRACE])
            return self._emit_assist_raw(raw[0].copy(), chunk)
        for offset, action in enumerate(raw[:GRACE]):
            row = {"emission_index": len(self._emitted), "kind": "native",
                   "stage_index": self._context["stage_index"],
                   "expected_actual_step": self._context["actual_executed"] + offset,
                   "cache_chunk_inference_index": chunk["delegate_call_index"],
                   "raw_action": action.tolist(), "executed_action": action.tolist(),
                   "raw_action_sha256": hashlib.sha256(
                       np.ascontiguousarray(action).tobytes()).hexdigest(),
                   "modified": False, "executed": False}
            self._emitted.append(row); self._pending.append(row); chunk["emitted_rows"] += 1
        return raw

    def on_execution(self, action: Any, post_state8: Any):
        value = _vector(action, ACTION_DIM, "executed action")
        state = _vector(post_state8, STATE_DIM, "post state")
        if not self._pending:
            raise RuntimeError("execution callback has no matching emitted action")
        row = self._pending.popleft()
        if not np.array_equal(value, np.asarray(row["executed_action"])):
            raise ValueError("executed action differs from emitted action")
        pre_state = self._state.copy()
        row.update(executed=True, execution_index=len(self._executed),
                   pre_state8=pre_state.tolist(), post_state8=state.tolist())
        self._executed.append(row)
        self._chunks[row["cache_chunk_inference_index"]]["executed_rows"] += 1
        if row["modified"]:
            self._modifications_executed += 1
            if self._first_changed_action_step is None:
                self._first_changed_action_step = row["expected_actual_step"]
        if row["kind"] == "native":
            self._native.append((np.asarray(row["raw_action"]), pre_state))
        elif row["kind"] == "assist":
            self._assist_slots += 1
            if self._assist_slots >= MAX_ASSIST_SLOTS:
                self._assist_active = False
                self._early_stop_reason = "assist_slot_limit"
        self._state = state

    def finalize_episode(self, executed_trace=()):
        if not self._active:
            raise RuntimeError("closed-dwell lift finalization is single-use")
        expected = tuple((row["expected_actual_step"], row["stage_index"])
                         for row in self._executed)
        reconciled = tuple(executed_trace) == expected
        for row in self._pending:
            row["truncated_before_execution"] = True
        self._pending.clear()
        try:
            self.delegate.finalize_episode(executed_trace)
        finally:
            self.provenance = {"kind": KIND,
                "delegate": copy.deepcopy(getattr(self.delegate, "provenance", {})),
                "cue": copy.deepcopy(self._cue), "attempted": self._attempted,
                "first_changed_action_step": self._first_changed_action_step,
                "assist_slots_executed": self._assist_slots,
                "modification_count": self._modifications_executed,
                "emitted_modification_count": self._modifications_emitted,
                "early_stop_reason": self._early_stop_reason,
                "native_reserve": NATIVE_RESERVE,
                "chunks": copy.deepcopy(self._chunks),
                "emitted_rows": copy.deepcopy(self._emitted),
                "executed_rows": copy.deepcopy(self._executed),
                "execution_reconciled": reconciled,
                "mechanical_proxy_only": True,
                "grasp_or_task_success_certificate": False, "oracle_inputs": []}
            self._active = False
        if not reconciled:
            raise ValueError("Runner trace and lift execution callbacks disagree")
