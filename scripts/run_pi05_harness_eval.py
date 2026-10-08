#!/usr/bin/env python3
"""Execute PI0.5 through the shared PhysicalRSI Runner or audit loop parity.

One invocation targets one already-running, fixed-policy service.  Task routing
is frozen independently from cases and cannot depend on an initial-state index.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

import pi05_harness_backend as backend
import pi05_instruction_lease as instruction_lease
import pi05_response_probe as response_probe


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ROUTES_SCHEMA = "pi05_harness_routes.v1"
CASES_SCHEMA = "pi05_harness_cases.v1"
MANIFEST_SCHEMA = "pi05_harness_eval.manifest.v1"
SUMMARY_SCHEMA = "pi05_harness_eval.summary.v1"
FPS = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roborsi-root", type=Path, required=True)
    parser.add_argument("--eval-helpers", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("harness", "parity", "legacy"))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-episode-seconds", type=float, default=1200.0)
    parser.add_argument("--api-key-env", default="OPENPI_API_KEY")
    parser.add_argument("--record-payload-hashes", action="store_true")
    return parser.parse_args(argv)


def _lower_hex(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _task_key(suite: str, task_id: int) -> str:
    return f"{suite}/{task_id}"


def joint_task_number(suite: str, task_id: int) -> int:
    if suite not in SUITES or type(task_id) is not int or not 0 <= task_id < 10:
        raise ValueError("case is outside the joint LIBERO-40 inventory")
    return SUITES.index(suite) * 10 + task_id


def load_routes(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    required = {"schema", "tasks", "identities"}
    optional = {"name", "selection_basis", "validation_status", "routing_inputs"}
    if (
        not required <= set(value)
        or set(value) - required - optional
        or value["schema"] != ROUTES_SCHEMA
    ):
        raise ValueError("invalid PI0.5 routes schema")
    for key in ("name", "selection_basis", "validation_status"):
        if key in value and (not isinstance(value[key], str) or not value[key]):
            raise ValueError(f"routes metadata {key} must be a nonempty string")
    if "routing_inputs" in value and value["routing_inputs"] != ["suite", "task_index"]:
        raise ValueError("routes routing_inputs must be exactly suite/task_index")
    tasks, identities = value["tasks"], value["identities"]
    if not isinstance(tasks, dict) or not tasks or not isinstance(identities, dict):
        raise ValueError("routes require nonempty task and identity mappings")
    expected_tasks = {
        _task_key(suite, task_id) for suite in SUITES for task_id in range(10)
    }
    if set(tasks) != expected_tasks:
        raise ValueError("routes must freeze the exact LIBERO-40 task set")
    normalized_identities = {}
    for policy_id, identity in identities.items():
        if not isinstance(policy_id, str) or not policy_id or not isinstance(identity, dict):
            raise ValueError("invalid policy identity")
        if set(identity) != {"checkpoint_sha256", "base_graph", "adapter_sha256"}:
            raise ValueError("policy identity fields changed")
        adapter = identity["adapter_sha256"]
        if not _lower_hex(identity["checkpoint_sha256"]):
            raise ValueError("checkpoint_sha256 must be lowercase 64-hex")
        if not isinstance(identity["base_graph"], str) or not identity["base_graph"]:
            raise ValueError("base_graph is required")
        if policy_id == "base":
            if adapter is not None:
                raise ValueError("base policy must route to adapter_sha256=null")
        elif not _lower_hex(adapter):
            raise ValueError("plugin policy requires a lowercase 64-hex adapter_sha256")
        normalized_identities[policy_id] = {"policy_id": policy_id, **identity}
    for key, policy_id in tasks.items():
        parts = key.split("/") if isinstance(key, str) else []
        if (
            len(parts) != 2
            or parts[0] not in SUITES
            or not parts[1].isdigit()
            or not 0 <= int(parts[1]) < 10
        ):
            raise ValueError("routes keys must be suite/task only; init routing is forbidden")
        if not isinstance(policy_id, str) or policy_id not in normalized_identities:
            raise ValueError(f"task route references unknown policy identity: {key}")
    result = {
        "schema": ROUTES_SCHEMA,
        "tasks": dict(tasks),
        "identities": normalized_identities,
    }
    result["metadata"] = {key: copy.deepcopy(value[key]) for key in optional if key in value}
    return result


def load_cases(path: Path | str) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text())
    if set(value) != {"schema", "cases"} or value["schema"] != CASES_SCHEMA:
        raise ValueError("invalid PI0.5 cases schema")
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise ValueError("cases must be a nonempty list")
    result, identities = [], set()
    for raw in value["cases"]:
        if (
            not isinstance(raw, dict)
            or set(raw) not in (
                {"suite", "task_id", "init_id"},
                {"suite", "task_id", "init_id", "replicate_id"},
            )
        ):
            raise ValueError("each case must contain suite/task_id/init_id and optional replicate_id")
        suite, task_id, init_id = raw["suite"], raw["task_id"], raw["init_id"]
        number = joint_task_number(suite, task_id)
        if type(init_id) is not int or not 0 <= init_id < 50:
            raise ValueError("init_id must be in 0..49")
        replicate_id = raw.get("replicate_id", 0)
        policy_seed = backend.policy_seed_for_case(number, init_id, replicate_id)
        case_id = f"{suite}/{task_id}/{init_id}/{replicate_id}"
        if case_id in identities:
            raise ValueError("duplicate case")
        identities.add(case_id)
        result.append({
            "id": case_id,
            "suite": suite,
            "task_id": task_id,
            "init_id": init_id,
            "replicate_id": replicate_id,
            "joint_task_number": number,
            "ambient_seed": backend.ambient_seed_for_case(number, init_id),
            "policy_seed": policy_seed,
        })
    return result


def bind_routes(cases: list[dict[str, Any]], routes: Mapping[str, Any]) -> str:
    policy_ids = set()
    for case in cases:
        key = _task_key(case["suite"], case["task_id"])
        if key not in routes["tasks"]:
            raise ValueError(f"case task has no frozen policy route: {key}")
        case["policy_id"] = routes["tasks"][key]
        policy_ids.add(case["policy_id"])
    if len(policy_ids) != 1:
        raise ValueError("one CLI invocation must target exactly one fixed-policy service")
    return next(iter(policy_ids))


def service_uri(host: str, port: int) -> str:
    if (
        not isinstance(host, str)
        or not host
        or "://" in host
        or "/" in host
        or "@" in host
        or not 1 <= port <= 65535
    ):
        raise ValueError("host/port must identify a websocket endpoint without credentials or path")
    bracketed = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"ws://{bracketed}:{port}"


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()).hexdigest()


def load_task_control_snapshot(registry_path: Path | str) -> dict[str, Any]:
    """Read all 40 task files and admit only the task-level probe control."""
    registry_path = Path(registry_path).resolve()
    document = json.loads(registry_path.read_text())
    expected = {_task_key(suite, task_id) for suite in SUITES for task_id in range(10)}
    tasks = document.get("tasks")
    declared = document.get("metadata", {}).get("task_config_sha256")
    if not isinstance(tasks, dict) or set(tasks) != expected:
        raise ValueError("control registry must contain the exact LIBERO-40 task set")
    if not isinstance(declared, dict) or set(declared) != expected:
        raise ValueError("control registry must declare all 40 task config digests")
    controls, digests = {}, {}
    for key, relative in tasks.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("registry task path must be a nonempty string")
        path = (registry_path.parent / relative).resolve()
        if not path.is_relative_to(registry_path.parent) or not path.is_file():
            raise ValueError("registry task config escapes or is missing")
        config = json.loads(path.read_text())
        if not isinstance(config, dict) or config.get("task") != key:
            raise ValueError("task config identity mismatch")
        found = []

        def scan(value: Any, depth: int = 0):
            if isinstance(value, dict):
                for name, child in value.items():
                    if name == "pi05_control":
                        found.append((depth, child))
                    scan(child, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    scan(child, depth + 1)

        scan(config)
        if len(found) > 1 or (found and found[0][0] != 0):
            raise ValueError("pi05_control is admitted only once at task-config top level")
        if "pi05_control" not in config:
            control = {"kind": "response_probe_v1", "enabled": False}
        else:
            raw = config["pi05_control"]
            response_valid = (isinstance(raw, dict)
                and set(raw) == {"kind", "enabled"}
                and raw.get("kind") == "response_probe_v1"
                and type(raw.get("enabled")) is bool)
            lease_valid = (isinstance(raw, dict)
                and set(raw) == {"kind", "enabled", "instruction", "lease_steps"}
                and raw.get("kind") == instruction_lease.KIND
                and raw.get("enabled") is True
                and isinstance(raw.get("instruction"), str)
                and bool(raw["instruction"].strip()) and len(raw["instruction"]) <= 512
                and type(raw.get("lease_steps")) is int
                and raw["lease_steps"] in instruction_lease.LEASE_STEPS)
            if not response_valid and not lease_valid:
                raise ValueError("invalid task-level pi05_control schema")
            control = dict(raw)
        digest = _canonical_sha256(config)
        if declared[key] != digest:
            raise ValueError(f"declared task config digest mismatch: {key}")
        controls[key], digests[key] = control, digest
    return {"controls": controls, "task_config_sha256": digests}


def validate_control_materialization(
    before: Mapping[str, Any], after: Mapping[str, Any], proposal: Mapping[str, Any]
) -> None:
    expected = proposal.get("task_config_sha256")
    if (before.get("task_config_sha256") != after.get("task_config_sha256")
            or not isinstance(expected, dict)
            or after.get("task_config_sha256") != expected):
        raise ValueError("task config digest changed across registry materialization")


def validate_enabled_control_harness(
    control: Mapping[str, Any], harness: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    if not control["enabled"]:
        return
    instruction, cap = validate_parity_harness(harness, int(task["max_steps"]))
    if harness.get("remember") != [] or instruction != task["instruction"] or cap != task["max_steps"]:
        raise ValueError("response probe requires the original full-cap single-stage task")


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def catalog_tasks() -> dict[str, dict[str, Any]]:
    from libero.libero import benchmark

    result = {}
    for suite_name in SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in range(10):
            task = suite.get_task(task_id)
            result[_task_key(suite_name, task_id)] = {
                "instruction": task.language,
                "max_steps": {
                    "libero_spatial": 220,
                    "libero_object": 280,
                    "libero_goal": 300,
                    "libero_10": 520,
                }[suite_name],
            }
    return result


def preflight_identity(
    helpers,
    *,
    uri: str,
    policy_id: str,
    identity: Mapping[str, Any],
    timeout_seconds: float,
    api_key_env: str,
) -> dict[str, Any]:
    args = SimpleNamespace(
        service_uri=uri,
        timeout_seconds=timeout_seconds,
        api_key_env=api_key_env,
        expected_base_graph=identity["base_graph"],
    )
    verified = helpers.validate_service_metadata(
        args, policy_id, identity["checkpoint_sha256"]
    )
    expected = {**dict(identity), "policy_seed_protocol": helpers.SEED_PROTOCOL}
    if verified != expected:
        raise ValueError("preflight service identity differs from frozen route identity")
    return verified


def _array_hash_receipt(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


class PayloadHashTransport:
    """Read-only request/response digest recorder around an actual transport."""

    def __init__(self, inner: Any, records: list[dict[str, Any]]):
        self.inner, self.records = inner, records

    @property
    def metadata(self):
        return self.inner.metadata

    def infer(self, payload: Mapping[str, Any]):
        record = {
            "inference_call": len(self.records),
            "observation_image": _array_hash_receipt(payload["observation/image"]),
            "observation_wrist_image": _array_hash_receipt(payload["observation/wrist_image"]),
            "observation_state": _array_hash_receipt(payload["observation/state"]),
            "prompt": str(payload["prompt"]),
            "policy_id": str(payload["policy_id"]),
            "policy_seed": int(payload["policy_seed"]),
            "response_actions": None,
            "status": "error",
        }
        self.records.append(record)
        try:
            response = self.inner.infer(payload)
            if isinstance(response, Mapping) and "actions" in response:
                record["response_actions"] = _array_hash_receipt(response["actions"])
            record["status"] = "ok"
            return response
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise

    def close(self):
        self.inner.close()


def make_inner_transport_factory(
    helpers,
    args: argparse.Namespace,
    uri: str,
    payload_hash_records: list[dict[str, Any]] | None = None,
):
    OpenPITransport, _ = helpers.import_physicalrsi(args.roborsi_root)

    def create():
        inner = OpenPITransport(
            uri, timeout_s=args.timeout_seconds, api_key_env=args.api_key_env
        )
        if payload_hash_records is not None:
            return PayloadHashTransport(inner, payload_hash_records)
        return inner

    return create


def _reset_old_environment(helpers, case: Mapping[str, Any]):
    env, _, task = helpers.make_environment(
        case["suite"], case["task_id"], backend.ENVIRONMENT_SEED
    )
    states, init_asset = helpers.load_official_init_states(task)
    if case["init_id"] >= len(states):
        env.close()
        raise IndexError(f"official init state {case['init_id']} unavailable")
    env.reset()
    raw = env.set_init_state(states[case["init_id"]])
    for _ in range(backend.SETTLING_STEPS):
        raw, _, _, _ = env.step(backend.DUMMY_ACTION.tolist())
    return env, task, raw, init_asset


def validate_case_seeds(case: Mapping[str, Any]) -> tuple[int, int]:
    ambient = backend.ambient_seed_for_case(case["joint_task_number"], case["init_id"])
    policy = backend.policy_seed_for_case(
        case["joint_task_number"], case["init_id"], case["replicate_id"]
    )
    if case.get("ambient_seed") != ambient or case.get("policy_seed") != policy:
        raise ValueError("case seeds differ from the fixed replicate protocol")
    return ambient, policy


def _run_old_loop(
    helpers,
    args: argparse.Namespace,
    uri: str,
    expected_identity: Mapping[str, Any],
    case: Mapping[str, Any],
    instruction: str,
    action_cap: int,
) -> dict[str, Any]:
    ambient_seed, _ = validate_case_seeds(case)
    random.seed(ambient_seed)
    np.random.seed(ambient_seed)
    env = transport = None
    frames, trace = [], []
    payload_hashes: list[dict[str, Any]] = []
    try:
        env, task, raw, init_asset = _reset_old_environment(helpers, case)
        OpenPITransport, Pi05Skill = helpers.import_physicalrsi(args.roborsi_root)
        inner = OpenPITransport(
            uri, timeout_s=args.timeout_seconds, api_key_env=args.api_key_env
        )
        if getattr(args, "record_payload_hashes", False):
            inner = PayloadHashTransport(inner, payload_hashes)
        transport = helpers.EpisodeIdentityTransport(inner, dict(expected_identity))
        skill = Pi05Skill(
            transport,
            checkpoint_id=case["policy_id"],
            state_dim=8,
            action_dim=7,
            image_keys=("observation/image", "observation/wrist_image"),
        )
        skill.reset()
        success, steps, calls = helpers.execute_policy_steps(
            env,
            raw,
            skill,
            instruction,
            case["policy_id"],
            case["policy_seed"],
            action_cap,
            frames,
            trace,
        )
        if transport.verified_metadata is None:
            raise RuntimeError("old loop completed without actual connection identity")
        return {
            "status": "success" if success else "failure",
            "success": bool(success),
            "steps": steps,
            "inference_calls": calls,
            "trace": trace,
            "frames": frames,
            "service_metadata": copy.deepcopy(transport.verified_metadata),
            "init_asset": init_asset,
            "payload_hashes": payload_hashes,
        }
    finally:
        if transport is not None:
            transport.close()
        if env is not None:
            env.close()


def validate_parity_harness(harness: Mapping[str, Any], official_cap: int) -> tuple[str, int]:
    stages = harness.get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError("parity mode requires a single-stage materialized harness")
    stage = stages[0]
    if (
        stage.get("skill") != "pi05"
        or stage.get("max_steps") != official_cap
        or stage.get("until") is not None
        or stage.get("on_timeout") != "abort"
        or not isinstance(stage.get("instruction"), str)
    ):
        raise ValueError("parity harness must preserve the full official single-stage contract")
    return stage["instruction"], official_cap


def _run_new_loop(
    api: backend.RoborsiAPI,
    helpers,
    args: argparse.Namespace,
    uri: str,
    expected_identity: Mapping[str, Any],
    case: Mapping[str, Any],
    harness: Mapping[str, Any],
    control: Mapping[str, Any],
    payload_hash_records: list[dict[str, Any]] | None = None,
) -> backend.BackendEpisode:
    ambient_seed, _ = validate_case_seeds(case)
    inner_factory = make_inner_transport_factory(
        helpers, args, uri, payload_hash_records=payload_hash_records
    )
    transport_factory = backend.episode_identity_transport_factory(
        helpers, inner_factory, expected_identity
    )
    delegate = backend.Pi05HarnessSkill(
        transport_factory,
        policy_id=case["policy_id"],
        policy_seed=case["policy_seed"],
    )
    base_environment_factory = lambda: backend.Pi05HarnessEnvironment(
        api.Observation,
        helpers,
        suite=case["suite"],
        task_id=case["task_id"],
        policy_id=case["policy_id"],
        policy_seed=case["policy_seed"],
    )
    if control.get("enabled") and control.get("kind") == "response_probe_v1":
        skill = response_probe.Pi05ResponseProbeSkill(delegate)
        environment_factory = response_probe.response_probe_environment_factory(
            base_environment_factory, skill
        )
    elif control.get("enabled") and control.get("kind") == instruction_lease.KIND:
        skill = instruction_lease.Pi05InstructionLeaseSkill(
            delegate, instruction=control["instruction"], lease_steps=control["lease_steps"]
        )
        environment_factory = response_probe.response_probe_environment_factory(
            base_environment_factory, skill
        )
    elif control == {"kind": "response_probe_v1", "enabled": False}:
        skill, environment_factory = delegate, base_environment_factory
    else:
        raise ValueError("unvalidated PI0.5 control reached execution")
    random.seed(ambient_seed)
    np.random.seed(ambient_seed)
    return backend.run_with_shared_runner(
        api.Runner,
        environment_factory,
        skill,
        harness,
        init_id=case["init_id"],
        official_cap=int(helpers.STEP_CAPS[case["suite"]]),
        max_seconds=args.max_episode_seconds,
    )


def compare_parity(old: Mapping[str, Any], new: backend.BackendEpisode) -> dict[str, Any]:
    old_actions = np.asarray([row["action7"] for row in old["trace"]], dtype=np.float64)
    new_actions = np.asarray(new.environment_actions, dtype=np.float64)
    old_proprio = np.asarray([row["proprio8"] for row in old["trace"]], dtype=np.float64)
    new_proprio = np.asarray([row["state"] for row in new.report["trace"]], dtype=np.float64)
    old_seeds = []
    for row in old["trace"]:
        if not old_seeds or row["policy_seed"] != old_seeds[-1]:
            old_seeds.append(row["policy_seed"])
    new_seeds = [row["policy_seed"] for row in new.policy_calls]

    def exact(left: np.ndarray, right: np.ndarray) -> bool:
        return left.shape == right.shape and bool(np.array_equal(left, right))

    checks = {
        "success": bool(old["success"] == new.report["success"]),
        "actions": exact(old_actions, new_actions),
        "call_seeds": old_seeds == new_seeds,
        "proprio": exact(old_proprio, new_proprio),
    }

    def first_mismatch(left: np.ndarray, right: np.ndarray):
        if left.shape != right.shape:
            return {"left_shape": list(left.shape), "right_shape": list(right.shape)}
        unequal = np.argwhere(left != right)
        if not len(unequal):
            return None
        index = tuple(int(value) for value in unequal[0])
        return {"index": list(index), "old": float(left[index]), "new": float(right[index])}

    return {
        "equal": all(checks.values()),
        "tolerance": 0,
        "checks": checks,
        "first_action_mismatch": None if checks["actions"] else first_mismatch(old_actions, new_actions),
        "first_proprio_mismatch": None if checks["proprio"] else first_mismatch(old_proprio, new_proprio),
        "old_call_seeds": old_seeds,
        "new_call_seeds": new_seeds,
    }


def save_video(path: Path, frames: list[np.ndarray]) -> dict[str, Any]:
    if not frames:
        return {"path": str(path.name), "written": False, "error_type": "NoFrames"}
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, frames, fps=FPS)
        return {"path": str(path.name), "written": True}
    except Exception as exc:
        return {"path": str(path.name), "written": False, "error_type": type(exc).__name__}


def summary(rows: list[dict[str, Any]], planned: int, mode: str) -> dict[str, Any]:
    result = {
        "schema": SUMMARY_SCHEMA,
        "mode": mode,
        "planned": planned,
        "completed": len(rows),
        "complete": len(rows) == planned,
        "errors": sum(row.get("status") == "error" for row in rows),
        "successes": sum(row.get("success") is True for row in rows),
        "cases": rows,
        "execution_backend": mode,
    }
    if mode == "parity":
        result["parity_equal"] = (
            len(rows) == planned
            and not result["errors"]
            and all(row.get("parity", {}).get("equal") is True for row in rows)
        )
        result["parity_rule"] = "success/actions/call_seeds/proprio exact equality; tolerance=0"
    return result


def _case_filename(index: int, case: Mapping[str, Any]) -> str:
    return (
        f"{index:03d}_{case['suite']}_{case['task_id']}_{case['init_id']}"
        f"_r{case['replicate_id']:02d}.json"
    )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeout_seconds <= 0 or args.max_episode_seconds <= 0:
        raise ValueError("timeouts must be positive")
    routes = load_routes(args.routes)
    cases = load_cases(args.cases)
    policy_id = bind_routes(cases, routes)
    if args.mode == "parity" and any(case["replicate_id"] != 0 for case in cases):
        raise ValueError("parity mode admits only replicate_id=0 migration checks")
    control_before = load_task_control_snapshot(args.registry)
    uri = service_uri(args.host, args.port)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("output must be a fresh path")

    api = backend.import_roborsi(args.roborsi_root)
    helpers = backend.import_eval_helpers(args.eval_helpers)
    tasks = catalog_tasks()
    proposal = backend.materialize_pi05_registry(args.roborsi_root, args.registry, tasks)
    control_after = load_task_control_snapshot(args.registry)
    validate_control_materialization(control_before, control_after, proposal)
    task_controls = control_after["controls"]
    enabled_keys = [key for key, value in task_controls.items() if value["enabled"]]
    if enabled_keys and args.mode != "harness":
        raise ValueError("enabled pi05_control is admitted only in harness mode")
    for key in enabled_keys:
        suite, raw_task_id = key.split("/")
        task = tasks[key]
        harness = backend.harness_for_task(
            api, proposal, suite=suite, task_id=int(raw_task_id),
            instruction=task["instruction"], official_cap=task["max_steps"]
        )
        validate_enabled_control_harness(task_controls[key], harness, task)
    expected_identity = routes["identities"][policy_id]
    verified_identity = preflight_identity(
        helpers,
        uri=uri,
        policy_id=policy_id,
        identity=expected_identity,
        timeout_seconds=args.timeout_seconds,
        api_key_env=args.api_key_env,
    )

    output.mkdir(parents=True)
    episodes = output / "episodes"
    videos = output / "videos"
    episodes.mkdir()
    videos.mkdir()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "mode": args.mode,
        "service_uri": uri,
        "fixed_policy_id": policy_id,
        "verified_service_identity": verified_identity,
        "routes_sha256": _sha256(args.routes),
        "cases_sha256": _sha256(args.cases),
        "registry_sha256": _sha256(args.registry),
        "registry": str(args.registry.resolve()),
        "task_controls": task_controls,
        "task_config_sha256": control_after["task_config_sha256"],
        "roborsi_root": str(args.roborsi_root.resolve()),
        "eval_helpers": str(args.eval_helpers.resolve()),
        "cases": cases,
        "official_step_caps": dict(helpers.STEP_CAPS),
        "environment_seed": backend.ENVIRONMENT_SEED,
        "settling_steps": backend.SETTLING_STEPS,
        "replan_steps": backend.REPLAN_STEPS,
        "ambient_seed_formula": "7 + joint40_task_number * 50 + init_id",
        "policy_episode_seed_formula": "ambient_seed + replicate_id * 1000000007",
        "inference_seed_formula": "policy_episode_seed + inference_call * 1000003",
        "replicate_protocol": backend.REPLICATE_PROTOCOL,
        "replicate_id_range": list(range(backend.REPLICATE_COUNT)),
        "success_source": "done returned by policy env.step only; settling done ignored",
        "source_hashes": {
            "backend": _sha256(Path(__file__).with_name("pi05_harness_backend.py")),
            "cli": _sha256(Path(__file__)),
            "response_probe": _sha256(Path(__file__).with_name("pi05_response_probe.py")),
            "instruction_lease": _sha256(Path(__file__).with_name("pi05_instruction_lease.py")),
            "eval_helpers": _sha256(args.eval_helpers),
            "roborsi_core": _sha256(args.roborsi_root / "src/roborsi/self_harness/core.py"),
            "roborsi_registry": _sha256(args.roborsi_root / "src/roborsi/self_harness/registry.py"),
        },
        "parity_harness": "registry_materialized_full_cap_single_stage" if args.mode == "parity" else None,
        "execution_backend": args.mode,
        "payload_hash_recording": bool(args.record_payload_hashes),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output / "manifest.json", manifest)

    rows = []
    atomic_json(output / "summary.json", summary(rows, len(cases), args.mode))
    for index, case in enumerate(cases):
        episode_path = episodes / _case_filename(index, case)
        row = {
            **case,
            "status": "error",
            "success": False,
            "execution_backend": args.mode,
            "episode": str(episode_path.relative_to(output)),
        }
        task_control = copy.deepcopy(task_controls[_task_key(case["suite"], case["task_id"])])
        row["pi05_control"] = task_control
        try:
            task = tasks[_task_key(case["suite"], case["task_id"])]
            if args.mode == "parity":
                selected_harness = backend.harness_for_task(
                    api,
                    proposal,
                    suite=case["suite"],
                    task_id=case["task_id"],
                    instruction=task["instruction"],
                    official_cap=task["max_steps"],
                )
                instruction, action_cap = validate_parity_harness(
                    selected_harness, task["max_steps"]
                )
                old = _run_old_loop(
                    helpers, args, uri, verified_identity, case, instruction, action_cap
                )
                runner_payload_hashes = [] if args.record_payload_hashes else None
                new = _run_new_loop(
                    api,
                    helpers,
                    args,
                    uri,
                    verified_identity,
                    case,
                    selected_harness,
                    task_control,
                    payload_hash_records=runner_payload_hashes,
                )
                parity = compare_parity(old, new)
                old_video = save_video(
                    videos / f"{index:03d}_old.mp4", old.pop("frames")
                )
                new_video = save_video(
                    videos / f"{index:03d}_runner.mp4", new.trace_frames
                )
                evidence = {
                    "case": case,
                    "pi05_control": task_control,
                    "harness": selected_harness,
                    "old": old,
                    "runner": {
                        "report": new.report,
                        "environment_actions": new.environment_actions,
                        "policy_calls": new.policy_calls,
                        "payload_hashes": runner_payload_hashes or [],
                    },
                    "parity": parity,
                    "videos": {"old": old_video, "runner": new_video},
                }
                runner_error = new.report.get("status") == "error"
                row.update(
                    status=(
                        "error" if runner_error
                        else "matched" if parity["equal"]
                        else "mismatch"
                    ),
                    success=bool(new.report["success"]),
                    old_success=bool(old["success"]),
                    parity=parity,
                )
                if runner_error:
                    row["error_type"] = new.report.get("error_type", "RunnerError")
            elif args.mode == "harness":
                harness = backend.harness_for_task(
                    api,
                    proposal,
                    suite=case["suite"],
                    task_id=case["task_id"],
                    instruction=task["instruction"],
                    official_cap=task["max_steps"],
                )
                payload_hashes = [] if args.record_payload_hashes else None
                new = _run_new_loop(
                    api,
                    helpers,
                    args,
                    uri,
                    verified_identity,
                    case,
                    harness,
                    task_control,
                    payload_hash_records=payload_hashes,
                )
                video = save_video(videos / f"{index:03d}_harness.mp4", new.trace_frames)
                evidence = {
                    "case": case,
                    "pi05_control": task_control,
                    "harness": harness,
                    "runner": {
                        "report": new.report,
                        "environment_actions": new.environment_actions,
                        "policy_calls": new.policy_calls,
                        "payload_hashes": payload_hashes or [],
                    },
                    "video": video,
                }
                row.update(
                    status=new.report["status"],
                    success=bool(new.report["success"]),
                    steps=new.report["steps"],
                    execution_backend="harness",
                )
            else:
                old = _run_old_loop(
                    helpers,
                    args,
                    uri,
                    verified_identity,
                    case,
                    task["instruction"],
                    task["max_steps"],
                )
                video = save_video(videos / f"{index:03d}_legacy.mp4", old.pop("frames"))
                evidence = {
                    "case": case,
                    "pi05_control": task_control,
                    "execution_backend": "legacy",
                    "legacy": old,
                    "video": video,
                }
                row.update(
                    status=old["status"],
                    success=bool(old["success"]),
                    steps=old["steps"],
                    execution_backend="legacy",
                )
        except Exception as exc:
            evidence = {
                "case": case,
                "pi05_control": task_control,
                "status": "error",
                "success": False,
                "error_type": type(exc).__name__,
            }
            row.update(status="error", success=False, error_type=type(exc).__name__)
        atomic_json(episode_path, evidence)
        rows.append(row)
        atomic_json(output / "summary.json", summary(rows, len(cases), args.mode))
    return summary(rows, len(cases), args.mode)


def main(argv: list[str] | None = None) -> int:
    result = execute(parse_args(argv))
    if result.get("errors") != 0:
        return 1
    if result.get("mode") == "parity" and result.get("parity_equal") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
