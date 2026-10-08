#!/usr/bin/env python3
"""Thin PI0.5 backend for the existing PhysicalRSI shared harness Runner.

This module owns only the PI0.5 service and LIBERO observation adapters.  The
harness language, stage transitions, budgets, traces, and success accounting
remain owned by ``roborsi.self_harness.core.Runner``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np


ACTION_DIM = 7
STATE_DIM = 8
REPLAN_STEPS = 5
CALL_SEED_STRIDE = 1_000_003
REPLICATE_SEED_STRIDE = 1_000_000_007
REPLICATE_COUNT = 10
REPLICATE_PROTOCOL = "fixed_case_replicate_0_9_stride_1000000007_v1"
ENVIRONMENT_SEED = 7
SETTLING_STEPS = 10
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float64)
POLICY_KEYS = (
    "observation/image",
    "observation/wrist_image",
    "observation/state",
)


def ambient_seed_for_case(joint_task_number: int, init_id: int) -> int:
    """Official environment/process seed; independent of policy replicates."""
    if type(joint_task_number) is not int or not 0 <= joint_task_number < 40:
        raise ValueError("joint_task_number must be in 0..39")
    if type(init_id) is not int or not 0 <= init_id < 50:
        raise ValueError("init_id must be an official index in 0..49")
    return ENVIRONMENT_SEED + joint_task_number * 50 + init_id


def policy_seed_for_case(
    joint_task_number: int,
    init_id: int,
    replicate_id: int = 0,
) -> int:
    """Derive the only admitted episode seed for a fixed case replicate."""
    if type(replicate_id) is not int or not 0 <= replicate_id < REPLICATE_COUNT:
        raise ValueError("replicate_id must be an integer in 0..9")
    return ambient_seed_for_case(joint_task_number, init_id) + replicate_id * REPLICATE_SEED_STRIDE


@dataclass(frozen=True)
class RoborsiAPI:
    Runner: type
    Observation: type
    TaskHarnessRegistry: type
    initial_harness: Callable[[str, int, str], dict[str, Any]]


@dataclass
class BackendEpisode:
    report: dict[str, Any]
    trace_frames: list[np.ndarray]
    environment_actions: list[np.ndarray]
    policy_calls: list[dict[str, Any]]


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def import_roborsi(root: Path | str) -> RoborsiAPI:
    """Import the requested checkout, refusing a silently preloaded checkout."""
    source = Path(root).resolve() / "src"
    required = (
        source / "roborsi/self_harness/core.py",
        source / "roborsi/self_harness/registry.py",
        source / "roborsi/self_harness/demo.py",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"PhysicalRSI shared harness is incomplete below {source}")
    for name, module in tuple(sys.modules.items()):
        if name == "roborsi" or name.startswith("roborsi."):
            loaded = getattr(module, "__file__", None)
            if loaded is not None and not _is_below(Path(loaded), source):
                raise ImportError(
                    f"refusing to replace already imported {name} from another checkout: {loaded}"
                )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    core = importlib.import_module("roborsi.self_harness.core")
    registry = importlib.import_module("roborsi.self_harness.registry")
    demo = importlib.import_module("roborsi.self_harness.demo")
    for module in (core, registry, demo):
        if not _is_below(Path(module.__file__), source):
            raise ImportError(f"PhysicalRSI import did not bind to requested source: {module.__file__}")
    return RoborsiAPI(core.Runner, core.Observation, registry.TaskHarnessRegistry, demo.initial_harness)


def import_eval_helpers(path: Path | str) -> ModuleType:
    """Load the audited PI0.5 evaluator helpers from one exact file path."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    name = "_pi05_eval_helpers_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ImportError("PI0.5 evaluator helper module path changed")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PI0.5 evaluator helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    for symbol in (
        "make_environment",
        "load_official_init_states",
        "observation_payload",
        "STEP_CAPS",
        "EpisodeIdentityTransport",
    ):
        if not hasattr(module, symbol):
            raise ImportError(f"PI0.5 evaluator helper is missing {symbol}")
    return module


def episode_identity_transport_factory(
    eval_helpers: ModuleType | SimpleNamespace,
    inner_transport_factory: Callable[[], Any],
    expected_metadata: Mapping[str, Any],
) -> Callable[[], Any]:
    """Build a fresh, identity-checked transport for every Runner episode."""
    if not callable(inner_transport_factory):
        raise TypeError("inner_transport_factory must be callable")
    expected = copy.deepcopy(dict(expected_metadata))

    def create():
        return eval_helpers.EpisodeIdentityTransport(inner_transport_factory(), expected)

    return create


class Pi05HarnessEnvironment:
    """Official LIBERO reset/observation adapter without simulator-only inputs."""

    def __init__(
        self,
        observation_type: type,
        eval_helpers: ModuleType | SimpleNamespace,
        *,
        suite: str,
        task_id: int,
        policy_id: str,
        policy_seed: int,
        environment_seed: int = ENVIRONMENT_SEED,
    ):
        if environment_seed != ENVIRONMENT_SEED:
            raise ValueError("official PI0.5 LIBERO environment seed must remain 7")
        if suite not in eval_helpers.STEP_CAPS:
            raise ValueError(f"unknown LIBERO suite: {suite}")
        if type(task_id) is not int or not 0 <= task_id < 10:
            raise ValueError("LIBERO task_id must be in 0..9")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("fixed service policy_id is required")
        self._Observation = observation_type
        self._helpers = eval_helpers
        self.suite_name, self.task_id = suite, task_id
        self.policy_id, self.policy_seed = policy_id, int(policy_seed)
        self.environment_seed = environment_seed
        self.env, self.suite, self.task = eval_helpers.make_environment(
            suite, task_id, environment_seed
        )
        self.initial_states, self.init_asset = eval_helpers.load_official_init_states(self.task)
        if len(self.initial_states) < 50:
            raise ValueError("official LIBERO init-state asset must contain at least 50 states")
        physical_low, physical_high = (
            np.asarray(value, dtype=np.float64) for value in self.env.env.action_spec
        )
        if physical_low.shape != (ACTION_DIM,) or physical_high.shape != (ACTION_DIM,):
            raise ValueError("LIBERO environment does not expose a 7D action contract")
        # Runner still enforces shape and finite values.  The PI0.5 service API
        # is not restricted to action_spec: the unmodified LIBERO/robosuite
        # controller owns its established internal boundary handling.
        self.action_low = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
        self.action_high = np.full(ACTION_DIM, np.inf, dtype=np.float64)
        self.provenance = {
            "kind": "pi05_official_libero",
            "suite": suite,
            "task_id": task_id,
            "task_language": self.task.language,
            "environment_seed": environment_seed,
            "settling_steps": SETTLING_STEPS,
            "step_cap": int(eval_helpers.STEP_CAPS[suite]),
            "success_decision_source": "official_libero_env.step.done",
            "policy_observation": "two_rgb224_state8_only",
            "client_action_transform": "none_finite_7d_passthrough",
            "physical_action_spec": {
                "low": physical_low.tolist(),
                "high": physical_high.tolist(),
                "enforcement_owner": "existing_libero_robosuite_controller",
            },
            "init_asset": copy.deepcopy(self.init_asset),
        }
        self.trace_frames: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self._raw = None
        self._steps = 0

    def _observation(self, raw: Mapping[str, Any], done: bool = False):
        built = self._helpers.observation_payload(
            raw, self.task.language, self.policy_id, self.policy_seed
        )
        policy = {key: np.asarray(built[key]).copy() for key in POLICY_KEYS}
        if policy["observation/state"].shape != (STATE_DIM,):
            raise ValueError("PI0.5 LIBERO state must be 8D")
        if set(policy) != set(POLICY_KEYS):
            raise AssertionError("privileged key entered PI0.5 policy observation")
        self.trace_frames.append(policy["observation/image"].copy())
        return self._Observation(policy, {}, success=bool(done), terminated=bool(done))

    def reset(self, init_id: int):
        if type(init_id) is not int or not 0 <= init_id < 50:
            raise ValueError("official LIBERO init-state index must be in 0..49")
        if init_id >= len(self.initial_states):
            raise IndexError(f"official init state {init_id} unavailable")
        self.env.reset()
        raw = self.env.set_init_state(self.initial_states[init_id])
        for _ in range(SETTLING_STEPS):
            raw, _, _, _ = self.env.step(DUMMY_ACTION.tolist())
        # Settling success is deliberately discarded.  Only a policy action's
        # returned done value may score the episode.
        self._raw, self._steps = raw, 0
        self.trace_frames.clear()
        self.actions.clear()
        return self._observation(raw, False)

    def step(self, action: np.ndarray):
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError("PI0.5 action must be finite 7D")
        self.actions.append(value.copy())
        raw, _, done, _ = self.env.step(value.tolist())
        self._raw = raw
        self._steps += 1
        return self._observation(raw, bool(done))

    def close(self):
        self.env.close()


class Pi05HarnessSkill:
    """Stateless service skill with episode-global explicit inference seeds."""

    def __init__(
        self,
        transport_factory: Callable[[], Any],
        *,
        policy_id: str,
        policy_seed: int,
        call_seed_stride: int = CALL_SEED_STRIDE,
        min_chunk_steps: int = REPLAN_STEPS,
    ):
        if not callable(transport_factory):
            raise TypeError("transport_factory must create one episode-local transport")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("fixed policy_id is required")
        if type(policy_seed) is not int or policy_seed < 0:
            raise ValueError("policy_seed must be a nonnegative integer")
        if type(call_seed_stride) is not int or call_seed_stride < 1:
            raise ValueError("call_seed_stride must be positive")
        if type(min_chunk_steps) is not int or min_chunk_steps < 1:
            raise ValueError("min_chunk_steps must be positive")
        self.transport_factory = transport_factory
        self.policy_id, self.policy_seed = policy_id, policy_seed
        self.call_seed_stride, self.min_chunk_steps = call_seed_stride, min_chunk_steps
        self.transport = None
        self.calls = 0
        self.call_records: list[dict[str, Any]] = []
        self.provenance = {
            "kind": "pi05_harness_service",
            "policy_id": policy_id,
            "policy_seed": policy_seed,
            "episode_seed_protocol": REPLICATE_PROTOCOL,
            "call_seed_stride": call_seed_stride,
            "execute_steps": min_chunk_steps,
            "identity_verified": False,
            "payload_keys": [*POLICY_KEYS, "prompt", "policy_id", "policy_seed"],
            "client_action_transform": "none",
        }

    def begin_episode(self):
        if self.transport is not None:
            raise RuntimeError("PI0.5 skill began a new episode before finalizing the old one")
        self.transport = self.transport_factory()
        self.calls = 0
        self.call_records = []

    def reset(self):
        # Runner invokes reset at every stage boundary.  Its local action queue
        # is cleared there; the whole-episode inference seed index must continue.
        pass

    def act(self, observation, instruction: str, memory: dict[str, Any]) -> np.ndarray:
        del memory
        if self.transport is None:
            raise RuntimeError("PI0.5 episode has not begun")
        source = observation.policy
        if any(key not in source for key in POLICY_KEYS):
            raise ValueError("PI0.5 observation is missing a required policy field")
        state = np.asarray(source["observation/state"])
        images = [np.asarray(source[key]) for key in POLICY_KEYS[:2]]
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("PI0.5 observation state must be finite 8D")
        if any(image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 for image in images):
            raise ValueError("PI0.5 cameras must be HWC uint8")
        call_seed = self.policy_seed + self.calls * self.call_seed_stride
        payload = {
            "observation/image": images[0].copy(),
            "observation/wrist_image": images[1].copy(),
            "observation/state": state.copy(),
            "prompt": str(instruction),
            "policy_id": self.policy_id,
            "policy_seed": call_seed,
        }
        response = self.transport.infer(payload)
        verified = getattr(self.transport, "verified_metadata", None)
        if verified is None:
            raise RuntimeError("actual episode connection metadata was not verified")
        record = {
            "inference_call": self.calls,
            "policy_seed": call_seed,
            "metadata": copy.deepcopy(verified),
            "response_valid": False,
        }
        self.call_records.append(record)
        self.calls += 1
        if not isinstance(response, Mapping) or "actions" not in response:
            raise ValueError("PI0.5 service response is missing actions")
        actions = np.asarray(response["actions"], dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != ACTION_DIM
            or len(actions) < self.min_chunk_steps
            or not np.isfinite(actions).all()
        ):
            raise ValueError("PI0.5 service returned an invalid action chunk")
        record["response_valid"] = True
        return actions

    def finalize_episode(self, executed_trace: tuple[tuple[int, int], ...] = ()):
        del executed_trace
        verified = None if self.transport is None else getattr(self.transport, "verified_metadata", None)
        try:
            if self.transport is not None:
                self.transport.close()
        finally:
            self.transport = None
            self.provenance = {
                **self.provenance,
                "identity_verified": verified is not None,
                "service_metadata": copy.deepcopy(verified),
                "inference_calls": self.calls,
                "call_records": copy.deepcopy(self.call_records),
                "episode_connection_closed": True,
            }


def validate_harness_budget(harness: Mapping[str, Any], official_cap: int, skill_name: str = "pi05") -> None:
    stages = harness.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("harness must contain stages")
    if any(stage.get("skill") != skill_name for stage in stages):
        raise ValueError("PI0.5 backend does not route task stages to another policy")
    total = sum(stage.get("max_steps", 0) for stage in stages)
    if total > official_cap:
        raise ValueError("harness stage budget exceeds the official LIBERO cap")


def run_with_shared_runner(
    runner_type: type,
    env_factory: Callable[[], Any],
    skill: Pi05HarnessSkill,
    harness: Mapping[str, Any],
    *,
    init_id: int,
    official_cap: int,
    max_seconds: float = 1200.0,
    skill_name: str = "pi05",
) -> BackendEpisode:
    """Execute one case; all control-flow semantics remain in shared Runner."""
    validate_harness_budget(harness, official_cap, skill_name)
    holder: dict[str, Any] = {}

    def captured_factory():
        environment = env_factory()
        holder["environment"] = environment
        return environment

    runner = runner_type(
        captured_factory,
        {skill_name: skill},
        max_steps=official_cap,
        max_seconds=max_seconds,
        chunk_steps=REPLAN_STEPS,
    )
    report = runner.run(dict(harness), init_id)
    if (
        report.get("status") == "budget_exhausted"
        and report.get("steps", official_cap) < official_cap
        and report.get("elapsed_s", 0.0) >= max_seconds
    ):
        report.update(status="error", success=False, error_type="InfrastructureTimeout")
    environment = holder["environment"]
    return BackendEpisode(
        report=report,
        trace_frames=list(environment.trace_frames),
        environment_actions=[value.copy() for value in environment.actions],
        policy_calls=copy.deepcopy(skill.call_records),
    )


def run_pi05_harness_episode(
    *,
    physicalrsi_root: Path | str,
    eval_script: Path | str,
    inner_transport_factory: Callable[[], Any],
    expected_service_metadata: Mapping[str, Any],
    suite: str,
    task_id: int,
    init_id: int,
    joint_task_number: int,
    policy_id: str,
    harness: Mapping[str, Any],
    replicate_id: int = 0,
    max_seconds: float = 1200.0,
) -> BackendEpisode:
    api = import_roborsi(physicalrsi_root)
    helpers = import_eval_helpers(eval_script)
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    if suite not in suites or type(task_id) is not int or not 0 <= task_id < 10:
        raise ValueError("suite/task is outside the joint LIBERO-40 inventory")
    expected_joint = suites.index(suite) * 10 + task_id
    if type(joint_task_number) is not int or joint_task_number != expected_joint:
        raise ValueError("joint_task_number does not match suite/task identity")
    if expected_service_metadata.get("policy_id") != policy_id:
        raise ValueError("fixed service identity does not match requested policy_id")
    policy_seed = policy_seed_for_case(joint_task_number, init_id, replicate_id)
    # Replicates vary only explicit policy noise.  Simulator/process RNG stays
    # fixed to the official case seed and remains valid for numpy's uint32 API.
    ambient_seed = ambient_seed_for_case(joint_task_number, init_id)
    random.seed(ambient_seed)
    np.random.seed(ambient_seed)
    transport_factory = episode_identity_transport_factory(
        helpers, inner_transport_factory, expected_service_metadata
    )
    skill = Pi05HarnessSkill(
        transport_factory, policy_id=policy_id, policy_seed=policy_seed
    )
    episode = run_with_shared_runner(
        api.Runner,
        lambda: Pi05HarnessEnvironment(
            api.Observation,
            helpers,
            suite=suite,
            task_id=task_id,
            policy_id=policy_id,
            policy_seed=policy_seed,
        ),
        skill,
        harness,
        init_id=init_id,
        official_cap=int(helpers.STEP_CAPS[suite]),
        max_seconds=max_seconds,
    )
    episode.report["evaluation_seeds"] = {
        "ambient_seed": ambient_seed,
        "policy_episode_seed": policy_seed,
        "replicate_id": replicate_id,
        "replicate_protocol": REPLICATE_PROTOCOL,
        "ambient_seed_formula": "7 + joint40_task_number * 50 + init_id",
        "policy_seed_formula": "ambient_seed + replicate_id * 1000000007",
        "call_seed_formula": "policy_episode_seed + inference_call * 1000003",
    }
    return episode


def materialize_pi05_registry(
    physicalrsi_root: Path | str,
    registry_path: Path | str,
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize an exact 40-task PI0.5 registry with shared validation."""
    api = import_roborsi(physicalrsi_root)
    expected = {f"{suite}/{task_id}" for suite in (
        "libero_spatial", "libero_object", "libero_goal", "libero_10"
    ) for task_id in range(10)}
    if set(tasks) != expected:
        raise ValueError("PI0.5 registry materialization requires the exact LIBERO 40-task set")
    caps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
    for key, task in tasks.items():
        suite = key.split("/", 1)[0]
        if task.get("max_steps") != caps[suite] or not isinstance(task.get("instruction"), str):
            raise ValueError(f"task catalog does not preserve official instruction/cap: {key}")
    return api.TaskHarnessRegistry(registry_path, {"pi05"}).materialize(dict(tasks))


def harness_for_task(
    api: RoborsiAPI,
    proposal: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    instruction: str,
    official_cap: int,
) -> dict[str, Any]:
    key = f"{suite}/{task_id}"
    return copy.deepcopy(
        proposal.get("harnesses", {}).get(
            key, api.initial_harness(instruction, official_cap, "pi05")
        )
    )
