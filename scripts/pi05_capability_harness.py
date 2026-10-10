#!/usr/bin/env python3
"""Fail-closed episode-local capability contracts for the PI0.5 harness."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

KINDS = frozenset({"history_conditioner", "sampler_guidance", "candidate_verifier",
                   "prompt_controller", "action_transform"})
OBSERVABLES = frozenset({"rgb", "wrist_rgb", "state8", "full_prompt",
                         "executed_action7", "history"})
FORBIDDEN = frozenset({"init_id", "seed", "success", "done", "ground_truth",
                       "future", "privileged_state"})
ACTION_OWNERS = frozenset({"sampler_guidance", "candidate_verifier",
                           "prompt_controller", "action_transform"})
HEX64 = re.compile(r"[0-9a-f]{64}")


def validate_manifest(value: Mapping[str, Any], *, base_digest: str) -> dict[str, Any]:
    required = {"schema", "capability_id", "kind", "status", "base_checkpoint_sha256",
                "checkpoint_sha256", "allowed_observables", "invocation_budget", "conflicts"}
    if type(value) is not dict or set(value) != required or value["schema"] != "pi05-capability.v1":
        raise ValueError("malformed capability identity")
    if not isinstance(value["capability_id"], str) or not value["capability_id"]:
        raise ValueError("capability_id required")
    if value["kind"] not in KINDS or value["status"] not in ("available", "disabled", "unavailable"):
        raise ValueError("invalid kind/status")
    if value["base_checkpoint_sha256"] != base_digest:
        raise ValueError("frozen base digest mismatch")
    if not isinstance(base_digest, str) or HEX64.fullmatch(base_digest) is None:
        raise ValueError("base digest must be lowercase sha256")
    if value["status"] == "available" and (not isinstance(value["checkpoint_sha256"], str)
            or HEX64.fullmatch(value["checkpoint_sha256"]) is None):
        raise ValueError("available capability requires checkpoint digest")
    if value["status"] != "available" and value["checkpoint_sha256"] is not None:
        raise ValueError("unavailable capability cannot attest a checkpoint")
    obs = value["allowed_observables"]
    if not isinstance(obs, list) or not set(obs) <= OBSERVABLES or set(obs) & FORBIDDEN:
        raise ValueError("unapproved observable")
    if type(value["invocation_budget"]) is not int or value["invocation_budget"] < 0:
        raise ValueError("invalid invocation budget")
    if not isinstance(value["conflicts"], list) or any(not isinstance(x, str) for x in value["conflicts"]):
        raise ValueError("invalid conflicts")
    return copy.deepcopy(value)


def validate_selection(manifests: Sequence[Mapping[str, Any]], task_config: Mapping[str, Any]):
    if set(task_config) != {"task", "capabilities", "unavailable_policy"}:
        raise ValueError("selection must be task-wide and exact")
    if task_config["unavailable_policy"] != "fail_closed" or any(k in task_config for k in FORBIDDEN):
        raise ValueError("selection must fail closed without routing inputs")
    by_id = {m["capability_id"]: m for m in manifests}
    if len(by_id) != len(manifests): raise ValueError("duplicate capability identity")
    selected = task_config["capabilities"]
    if not isinstance(selected, list) or len(selected) != len(set(selected)):
        raise ValueError("duplicate/invalid selection")
    chosen = []
    for cid in selected:
        m = by_id.get(cid)
        if m is None or m["status"] != "available":
            raise RuntimeError(f"selected capability unavailable: {cid}")
        chosen.append(m)
    owners = [m for m in chosen if m["kind"] in ACTION_OWNERS]
    if len(owners) > 1:
        raise ValueError("only one action-changing owner may be selected")
    ids = {m["capability_id"] for m in chosen}
    if any(ids.intersection(m["conflicts"]) for m in chosen):
        raise ValueError("declared capability conflict")
    return chosen


class ExecutionMemory:
    def __init__(self, max_steps: int = 520):
        if type(max_steps) is not int or not 1 <= max_steps <= 520:
            raise ValueError("memory capacity must be in 1..520")
        self.max_steps = max_steps
        self.episode_id = None
        self.rows: list[dict[str, Any]] = []
        self._next_step = 0

    def begin_episode(self, episode_id: str):
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id required")
        self.episode_id, self.rows, self._next_step = episode_id, [], 0

    def record(self, *, episode_id: str, step: int, action7, pre_state8, post_state8,
               rgb_feature=None):
        if episode_id != self.episode_id or type(step) is not int or step != self._next_step:
            raise ValueError("cross-episode, stale, duplicate, or out-of-order event")
        def vec(x, n, label):
            a=np.asarray(x,dtype=np.float64)
            if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError(f"invalid {label}")
            return a.copy()
        feature = None if rgb_feature is None else vec(rgb_feature, len(rgb_feature), "rgb feature")
        if len(self.rows) >= self.max_steps: raise RuntimeError("execution memory capacity exhausted")
        self.rows.append({"episode_id":episode_id,"step":step,"action7":vec(action7,7,"action"),
                          "pre_state8":vec(pre_state8,8,"pre state"),
                          "post_state8":vec(post_state8,8,"post state"),"rgb_feature":feature})
        self._next_step += 1

    def last(self, n: int):
        if type(n) is not int or n < 0: raise ValueError("invalid history length")
        return tuple(copy.deepcopy(self.rows[-n:] if n else ()))

    def stage_handoff(self):
        """Intentionally does not clear episode history."""


class NativeCapabilityAdapter:
    """Transparent adapter; memory observes only confirmed executions."""
    def __init__(self, delegate, memory: ExecutionMemory, *, episode_id: str):
        self.delegate, self.memory, self.episode_id = delegate, memory, episode_id
    @property
    def calls(self): return self.delegate.calls
    @property
    def call_records(self): return self.delegate.call_records
    def begin_episode(self):
        self.delegate.begin_episode(); self.memory.begin_episode(self.episode_id)
    def reset(self): self.delegate.reset(); self.memory.stage_handoff()
    def act(self, observation, instruction, memory):
        return self.delegate.act(observation, instruction, memory)
    def on_execution(self, *, step, action7, pre_state8, post_state8, rgb_feature=None):
        self.memory.record(episode_id=self.episode_id,step=step,action7=action7,
                           pre_state8=pre_state8,post_state8=post_state8,rgb_feature=rgb_feature)
    def finalize_episode(self, executed_trace=()): return self.delegate.finalize_episode(executed_trace)


@dataclass(frozen=True)
class CandidateProtocol:
    count: int
    namespace: str = "pi05-candidate-v1"
    def seeds(self, base_seed: int, call_index: int):
        if type(self.count) is not int or self.count < 1 or type(base_seed) is not int or base_seed < 0:
            raise ValueError("invalid predetermined candidate protocol")
        if type(call_index) is not int or call_index < 0: raise ValueError("invalid call index")
        children=[]
        for k in range(1,self.count):
            raw=f"{self.namespace}|{base_seed}|{call_index}|{k}".encode()
            children.append(int.from_bytes(hashlib.sha256(raw).digest()[:8],"big") & ((1<<63)-1))
        out=(base_seed,*children)
        if len(set(out)) != self.count or out[0] != base_seed: raise AssertionError("seed protocol broken")
        return out


class InvocationBudget:
    def __init__(self, limit: int):
        if type(limit) is not int or limit < 0: raise ValueError("invalid invocation budget")
        self.limit, self.used = limit, 0
    def consume(self):
        if self.used >= self.limit: raise RuntimeError("capability invocation budget exhausted")
        self.used += 1


def _validate_history(value):
    if not isinstance(value, (tuple, list)): raise ValueError("history must be typed execution rows")
    expected={"episode_id","step","action7","pre_state8","post_state8","rgb_feature"}; last=-1;episode=None
    for row in value:
        if not isinstance(row, Mapping) or set(row)!=expected or type(row["step"]) is not int or row["step"]<=last: raise ValueError("malformed history")
        if episode is None: episode=row["episode_id"]
        if row["episode_id"]!=episode: raise ValueError("cross-episode history")
        for key,n in (("action7",7),("pre_state8",8),("post_state8",8)):
            a=np.asarray(row[key]);
            if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError("malformed history vector")
        last=row["step"]

def select_candidate(chunks, seeds, *, base_seed: int, call_index: int,
                     protocol: CandidateProtocol, observables: Mapping[str, Any], scorer: Callable):
    if tuple(seeds) != protocol.seeds(base_seed, call_index) or len(chunks) != protocol.count:
        raise ValueError("candidate set/seed protocol mismatch")
    if not isinstance(observables, Mapping) or not set(observables) <= OBSERVABLES or set(observables) & FORBIDDEN:
        raise ValueError("candidate verifier received oracle/unapproved input")
    if "history" in observables: _validate_history(observables["history"])
    arrays=[]
    horizon=None
    for x in chunks:
        a=np.asarray(x,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all(): raise ValueError("invalid candidate")
        if len(a)<1 or (horizon is not None and len(a)!=horizon): raise ValueError("candidate horizons differ/empty")
        horizon=len(a);arrays.append(a)
    scores=np.asarray(scorer(tuple(arrays),copy.deepcopy(dict(observables))),dtype=np.float64)
    if scores.shape!=(len(arrays),) or not np.isfinite(scores).all(): raise ValueError("invalid scores")
    index=int(np.argmax(scores))
    return index, arrays[index]
