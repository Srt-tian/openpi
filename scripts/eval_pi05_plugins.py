#!/usr/bin/env python3
"""Small paired LIBERO evaluation for a base pi0.5 policy and four suite plugins.

The command is plan-only unless ``--execute`` is supplied.  The inference
service must accept the standard OpenPI LIBERO observation plus ``policy_id``
and ``policy_seed``.  Policy selection is fixed for a whole arm; it is never
chosen from the initial-state index or from observations.
"""

from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib.parse import urlsplit

import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PLUGIN_IDS = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "long",
}
STEP_CAPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
TASK_IDS_BY_SUITE = {
    "libero_spatial": (0, 9),
    "libero_object": (0, 4),
    "libero_goal": (0, 3),
    "libero_10": (0, 8),
}
DEFAULT_INIT_COUNT = 2
SETTLING_STEPS = 10
RENDER_SIZE = 256
MODEL_IMAGE_SIZE = 224
REPLAN_STEPS = 5
CALL_SEED_STRIDE = 1_000_003
SEED_PROTOCOL = "paired_episode_plus_call_1000003_numpy_pcg64_noise_10x32_f32_v1"
FPS = 10
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class Case:
    suite: str
    task_id: int
    init_id: int
    joint_task_number: int
    policy_seed: int

    @property
    def id(self) -> str:
        return f"{self.suite}/{self.task_id}/{self.init_id}"


def validate_init_count(value: int) -> int:
    if isinstance(value, bool) or not 1 <= value <= 50:
        raise ValueError("init_count must be in 1..50")
    return value


def argparse_init_count(value: str) -> int:
    try:
        return validate_init_count(int(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("init count must be an integer in 1..50") from exc


def build_cases(seed: int, init_count: int = DEFAULT_INIT_COUNT) -> list[Case]:
    init_count = validate_init_count(init_count)
    cases = []
    for suite_number, suite in enumerate(SUITES):
        for task_id in TASK_IDS_BY_SUITE[suite]:
            joint_task_number = suite_number * 10 + task_id
            for init_id in range(init_count):
                cases.append(Case(
                    suite=suite,
                    task_id=task_id,
                    init_id=init_id,
                    joint_task_number=joint_task_number,
                    policy_seed=seed + joint_task_number * 50 + init_id,
                ))
    return cases


def execution_plan(
    seed: int,
    suites: tuple[str, ...] = SUITES,
    arm: str | None = None,
    init_count: int = DEFAULT_INIT_COUNT,
) -> list[dict[str, Any]]:
    """Suite-major paired plan: all base cases, then the same plugin cases."""
    cases = build_cases(seed, init_count)
    plan = []
    for suite in suites:
        suite_cases = [case for case in cases if case.suite == suite]
        arms = (("base", "base"), ("plugin", PLUGIN_IDS[suite]))
        for selected_arm, policy_id in arms:
            if arm is not None and selected_arm != arm:
                continue
            for case in suite_cases:
                plan.append({"arm": selected_arm, "policy_id": policy_id, **dataclasses.asdict(case), "id": case.id})
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run simulation; without this, print the plan only")
    parser.add_argument("--service-uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physicalrsi-root", type=Path, required=True)
    parser.add_argument("--suite", action="append", choices=SUITES, dest="suites")
    parser.add_argument("--arm", choices=("base", "plugin"))
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-count", type=argparse_init_count, default=DEFAULT_INIT_COUNT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--api-key-env", default="OPENPI_API_KEY")
    return parser.parse_args(argv)


def expected_policy_ids(plan: list[dict[str, Any]]) -> set[str]:
    return {row["policy_id"] for row in plan}


def validate_execute_args(args: argparse.Namespace, plan: list[dict[str, Any]]) -> tuple[str, str]:
    if args.arm is None:
        raise ValueError("--arm is required with --execute because each service process has one fixed policy_id")
    ids = expected_policy_ids(plan)
    if len(ids) != 1:
        raise ValueError("one execution batch must target exactly one fixed service policy_id")
    checkpoint = args.expected_checkpoint_sha256
    if (not isinstance(checkpoint, str) or len(checkpoint) != 64
            or any(char not in "0123456789abcdef" for char in checkpoint)):
        raise ValueError("--expected-checkpoint-sha256 must be a 64-character lowercase hex digest")
    parsed = urlsplit(args.service_uri)
    if parsed.scheme not in ("ws", "wss") or parsed.username is not None or parsed.password is not None:
        raise ValueError("service URI must be ws(s) without embedded credentials")
    return next(iter(ids)), checkpoint


def validated_metadata_subset(metadata: Any, policy_id: str, checkpoint_sha256: str) -> dict[str, Any]:
    expected = {
        "policy_id": policy_id,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_seed_protocol": SEED_PROTOCOL,
    }
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("service metadata does not match fixed policy/checkpoint/seed protocol")
    adapter = metadata.get("adapter_sha256")
    if policy_id == "base":
        if "adapter_sha256" not in metadata or adapter is not None:
            raise ValueError("base service metadata must attest adapter_sha256=null")
    elif (not isinstance(adapter, str) or len(adapter) != 64
          or any(char not in "0123456789abcdef" for char in adapter)):
        raise ValueError("plugin service metadata has invalid adapter_sha256")
    return {**expected, "adapter_sha256": adapter}


def validate_service_metadata(args: argparse.Namespace, policy_id: str, checkpoint_sha256: str) -> dict[str, Any]:
    """Read and validate service identity before the first policy action."""
    import os
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    key = os.environ.get(args.api_key_env)
    headers = {"Authorization": "Api-Key " + key} if key else None
    with connect(
        args.service_uri,
        compression=None,
        open_timeout=args.timeout_seconds,
        close_timeout=2,
        max_size=32 * 1024 * 1024,
        additional_headers=headers,
    ) as socket:
        metadata = msgpack_numpy.unpackb(socket.recv(timeout=args.timeout_seconds))
    return validated_metadata_subset(metadata, policy_id, checkpoint_sha256)


class EpisodeIdentityTransport:
    """Validate metadata from the actual lazy episode connection before returning its first actions."""

    def __init__(self, inner, expected_metadata: dict[str, Any]):
        self.inner = inner
        self.expected_metadata = dict(expected_metadata)
        self.verified_metadata: dict[str, Any] | None = None

    def infer(self, payload):
        response = self.inner.infer(payload)
        # OpenPITransport receives metadata before sending the infer payload.  It
        # exposes that metadata only once infer returns, so validate it here,
        # before Pi05Skill returns any action to the environment loop.
        subset = validated_metadata_subset(
            self.inner.metadata,
            self.expected_metadata["policy_id"],
            self.expected_metadata["checkpoint_sha256"],
        )
        if subset != self.expected_metadata:
            raise ValueError("episode service metadata differs from preflight connection")
        if self.verified_metadata is not None and subset != self.verified_metadata:
            raise ValueError("episode service metadata changed within one connection")
        self.verified_metadata = subset
        return response

    def close(self):
        self.inner.close()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def quat_to_axis_angle(quaternion: Any) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("invalid LIBERO quaternion")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator


def observation_payload(raw: dict[str, Any], instruction: str, policy_id: str, policy_seed: int) -> dict[str, Any]:
    from openpi_client import image_tools

    def image(key: str) -> np.ndarray:
        flipped = np.ascontiguousarray(np.asarray(raw[key])[::-1, ::-1])
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(flipped, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        )

    state = np.concatenate((
        np.asarray(raw["robot0_eef_pos"], dtype=np.float64),
        quat_to_axis_angle(raw["robot0_eef_quat"]),
        np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64),
    ))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO observation did not produce a finite 8D state")
    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": state,
        "prompt": str(instruction),
        "policy_id": policy_id,
        "policy_seed": int(policy_seed),
    }


def make_environment(suite_name: str, task_id: int, environment_seed: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=RENDER_SIZE, camera_widths=RENDER_SIZE
    )
    # Match the published OpenPI evaluation: the environment seed is fixed at 7;
    # the episode identity comes from the explicit official init-state index.
    env.seed(environment_seed)
    return env, suite, task


def load_official_init_states(task) -> tuple[Any, dict[str, Any]]:
    """Load trusted installed LIBERO assets without Torch's weights-only default."""
    import torch
    from libero.libero import get_libero_path

    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = torch.load(path, map_location="cpu", weights_only=False)
    if len(states) < 50:
        raise ValueError(f"official init-state asset has only {len(states)} states")
    shape = list(states.shape) if hasattr(states, "shape") else [len(states)]
    return states, {
        "file": f"{task.problem_folder}/{task.init_states_file}",
        "sha256": sha256_file(path),
        "count": len(states),
        "shape": shape,
    }


def inspect_init_assets(plan: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from libero.libero import benchmark

    result = {}
    for item in plan:
        key = f'{item["suite"]}/{item["task_id"]}'
        if key in result:
            continue
        suite = benchmark.get_benchmark_dict()[item["suite"]]()
        _, receipt = load_official_init_states(suite.get_task(item["task_id"]))
        result[key] = receipt
    return result


def import_physicalrsi(root: Path):
    source = root.resolve() / "src"
    if not (source / "roborsi/self_harness/pi05.py").is_file():
        raise FileNotFoundError(f"PhysicalRSI pi05 adapter missing below {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from roborsi.self_harness.pi05 import OpenPITransport, Pi05Skill
    return OpenPITransport, Pi05Skill


def execute_policy_steps(
    env,
    raw: dict[str, Any],
    skill,
    instruction: str,
    policy_id: str,
    policy_seed: int,
    action_cap: int,
    frames: list[np.ndarray],
    trace: list[dict[str, Any]],
    *,
    payload_builder=observation_payload,
) -> tuple[bool, int, int]:
    """Execute at most ``action_cap`` actions and retain done from the final action."""
    action_queue: deque[np.ndarray] = deque()
    executed_actions = 0
    inference_calls = 0
    active_inference_call = -1
    active_policy_seed = -1
    done = False
    while executed_actions < action_cap and not done:
        call_seed = policy_seed + inference_calls * CALL_SEED_STRIDE
        payload = payload_builder(raw, instruction, policy_id, call_seed)
        frames.append(payload["observation/image"])
        if not action_queue:
            active_inference_call = inference_calls
            active_policy_seed = call_seed
            actions = np.asarray(skill.act(type("Obs", (), {"policy": payload})(), instruction, {}))
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < REPLAN_STEPS:
                raise ValueError("policy returned an invalid or too-short action chunk")
            action_queue.extend(actions[:REPLAN_STEPS])
            inference_calls += 1
        action = np.asarray(action_queue.popleft(), dtype=np.float64)
        raw, _, done, _ = env.step(action.tolist())
        trace.append({
            "step": executed_actions,
            "proprio8": np.asarray(payload["observation/state"], dtype=np.float64).tolist(),
            "action7": action.tolist(),
            "inference_call": active_inference_call,
            "policy_seed": active_policy_seed,
            "skill": policy_id,
            "done": bool(done),
        })
        executed_actions += 1
    return bool(done), executed_actions, inference_calls


def run_episode(item: dict[str, Any], args: argparse.Namespace, videos: Path, episode_path: Path) -> dict[str, Any]:
    import imageio.v2 as imageio

    OpenPITransport, Pi05Skill = import_physicalrsi(args.physicalrsi_root)
    case = Case(**{key: item[key] for key in ("suite", "task_id", "init_id", "joint_task_number", "policy_seed")})
    random.seed(case.policy_seed)
    np.random.seed(case.policy_seed)
    env = None
    transport = None
    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    started = time.time()
    result = {
        **item,
        "case_id": case.id,
        "status": "error",
        "success": False,
        "environment_seed": args.seed,
        "settling_steps": SETTLING_STEPS,
        "action_cap": STEP_CAPS[case.suite],
        "replan_steps": REPLAN_STEPS,
        "inference_calls": 0,
        "executed_actions": 0,
    }
    try:
        env, suite, task = make_environment(case.suite, case.task_id, args.seed)
        initial_states, init_asset = load_official_init_states(task)
        if case.init_id >= len(initial_states):
            raise IndexError(f"official init state {case.init_id} unavailable")
        result["init_asset"] = init_asset
        env.reset()
        raw = env.set_init_state(initial_states[case.init_id])
        done = False
        for _ in range(SETTLING_STEPS):
            raw, _, done, _ = env.step(DUMMY_ACTION.tolist())
        # Official evaluation ignores success during object-settling actions.
        # Only done returned by a subsequent policy action may score the case.
        done = False
        # A new connection and a fresh queue make episode boundaries explicit.
        inner_transport = OpenPITransport(
            args.service_uri, timeout_s=args.timeout_seconds, api_key_env=args.api_key_env
        )
        transport = EpisodeIdentityTransport(
            inner_transport, args.verified_service_identity
        )
        skill = Pi05Skill(
            transport,
            checkpoint_id=item["policy_id"],
            state_dim=8,
            action_dim=7,
            image_keys=("observation/image", "observation/wrist_image"),
        )
        skill.reset()
        done, result["executed_actions"], result["inference_calls"] = execute_policy_steps(
            env, raw, skill, task.language, item["policy_id"], case.policy_seed,
            STEP_CAPS[case.suite], frames, trace,
        )
        if transport.verified_metadata is None:
            raise RuntimeError("episode completed without verified service metadata")
        result["service_metadata"] = transport.verified_metadata
        result["success"] = bool(done)
        result["status"] = "success" if done else "failure"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    finally:
        if transport is not None:
            transport.close()
        if env is not None:
            env.close()
        result["elapsed_seconds"] = time.time() - started
        result["trace"] = trace
        video_name = f'{item["arm"]}_{item["policy_id"]}_{case.suite}_{case.task_id}_{case.init_id}_{result["status"]}.mp4'
        result["video"] = str(Path("videos") / video_name)
        if frames:
            try:
                imageio.mimwrite(videos / video_name, frames, fps=FPS)
            except Exception as exc:
                result["video_error_type"] = type(exc).__name__
                result["video_error"] = str(exc)[:1000]
        atomic_json(episode_path, result)
    return result


def summarize(rows: list[dict[str, Any]], manifest_hash: str, planned: int) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, int]]] = {}
    present_suites = tuple(suite for suite in SUITES if any(row["suite"] == suite for row in rows))
    for suite in present_suites:
        groups[suite] = {}
        ids_by_arm = {}
        for arm in ("base", "plugin"):
            selected = [row for row in rows if row["suite"] == suite and row["arm"] == arm]
            ids_by_arm[arm] = {row["case_id"] if "case_id" in row else row["id"] for row in selected}
            groups[suite][arm] = {
                "episodes": len(selected),
                "successes": sum(row["success"] is True for row in selected),
                "errors": sum(row["status"] == "error" for row in selected),
            }
        pairing_complete = bool(ids_by_arm["base"]) and ids_by_arm["base"] == ids_by_arm["plugin"]
        groups[suite]["pairing_complete"] = pairing_complete
        groups[suite]["paired_success_delta"] = (
            groups[suite]["plugin"]["successes"] - groups[suite]["base"]["successes"]
            if pairing_complete else None
        )
    return {
        "schema": "pi05_plugin_paired_eval.summary.v1",
        "complete": len(rows) == planned,
        "manifest_sha256": manifest_hash,
        "episodes": len(rows),
        "successes": sum(row["success"] is True for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "suites": groups,
        "claim_scope": "service_development_smoke_not_physicalrsi_harness_migration_or_official_score",
        "pairing_rule": "delta_is_null_unless_base_and_plugin_case_id_sets_are_nonempty_and_identical",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suites = tuple(dict.fromkeys(args.suites or SUITES))
    plan = execution_plan(args.seed, suites, args.arm, args.init_count)
    if not args.execute:
        print(json.dumps({"execute": False, "planned_episodes": len(plan), "plan": plan}, indent=2))
        return 0
    policy_id, checkpoint_sha256 = validate_execute_args(args, plan)
    if args.output is None:
        raise ValueError("--output is required with --execute")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("evaluation output must be a fresh path")
    init_assets = inspect_init_assets(plan)
    service_identity = validate_service_metadata(args, policy_id, checkpoint_sha256)
    args.verified_service_identity = service_identity
    output.mkdir(parents=True)
    episodes = output / "episodes"
    videos = output / "videos"
    episodes.mkdir()
    videos.mkdir()
    manifest = {
        "schema": "pi05_plugin_paired_eval.manifest.v1",
        "protocol": "four_suites_two_tasks_variable_inits_base_then_suite_plugin",
        "service_uri": args.service_uri,
        "service_identity": service_identity,
        "batch_scope": {
            "suites": list(suites),
            "arm": args.arm,
            "policy_id": policy_id,
            "fixed_service_process": True,
        },
        "physicalrsi_root": str(args.physicalrsi_root.resolve()),
        "environment_seed": args.seed,
        "init_count": args.init_count,
        "policy_seed_formula": "seed + joint40_task_number * 50 + init_id + inference_call * 1000003",
        "policy_seed_protocol": SEED_PROTOCOL,
        "policy_selection": "fixed_by_arm_and_suite_never_by_init_or_observation",
        "render_size": RENDER_SIZE,
        "model_image_size": MODEL_IMAGE_SIZE,
        "settling_steps": SETTLING_STEPS,
        "dummy_action": DUMMY_ACTION.tolist(),
        "step_caps": STEP_CAPS,
        "official_init_assets": init_assets,
        "replan_steps": REPLAN_STEPS,
        "success_source": "done_returned_by_libero_env.step",
        "aggregation_note": "single-arm batches have null paired delta until joined with the counterpart by case id",
        "plan": plan,
    }
    manifest_hash = canonical_hash(manifest)
    manifest["manifest_sha256"] = manifest_hash
    atomic_json(output / "manifest.json", manifest)
    rows = []
    for index, item in enumerate(plan):
        episode_path = episodes / f"{index:03d}_{item['arm']}_{item['suite']}_{item['task_id']}_{item['init_id']}.json"
        rows.append(run_episode(item, args, videos, episode_path))
    atomic_json(output / "summary.json", summarize(rows, manifest_hash, len(plan)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
