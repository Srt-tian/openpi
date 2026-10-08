#!/usr/bin/env python3
"""Fixed A(old)-B(Runner)-A(old) diagnosis for LIBERO goal task 9 init 5."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

import pi05_harness_backend as backend
import run_pi05_harness_eval as harness_cli


SUITE = "libero_goal"
TASK_ID = 9
INIT_ID = 5
REPLICATE_ID = 0
JOINT_TASK_NUMBER = 29
CASE_ID = "libero_goal/9/5/0"
REPLAY_COUNT = 5
FALLBACK_CALL = 37


class DiscardFrames:
    def append(self, value):
        del value


class ReplayCaptureTransport(harness_cli.PayloadHashTransport):
    """Hash every call and optionally retain an untouched payload snapshot."""

    def __init__(self, inner, records, *, retain_payloads=False):
        super().__init__(inner, records)
        self.retain_payloads = retain_payloads
        self.payloads: list[dict[str, Any]] = []

    def infer(self, payload):
        if self.retain_payloads:
            self.payloads.append({
                "observation/image": np.asarray(payload["observation/image"]).copy(),
                "observation/wrist_image": np.asarray(payload["observation/wrist_image"]).copy(),
                "observation/state": np.asarray(payload["observation/state"]).copy(),
                "prompt": str(payload["prompt"]),
                "policy_id": str(payload["policy_id"]),
                "policy_seed": int(payload["policy_seed"]),
            })
        return super().infer(payload)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--roborsi-root", type=Path, required=True)
    parser.add_argument("--eval-helpers", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-episode-seconds", type=float, default=1200.0)
    parser.add_argument("--api-key-env", default="OPENPI_API_KEY")
    return parser.parse_args(argv)


def fixed_case(policy_id: str) -> dict[str, Any]:
    ambient = backend.ambient_seed_for_case(JOINT_TASK_NUMBER, INIT_ID)
    policy = backend.policy_seed_for_case(JOINT_TASK_NUMBER, INIT_ID, REPLICATE_ID)
    return {
        "id": CASE_ID,
        "suite": SUITE,
        "task_id": TASK_ID,
        "init_id": INIT_ID,
        "replicate_id": REPLICATE_ID,
        "joint_task_number": JOINT_TASK_NUMBER,
        "ambient_seed": ambient,
        "policy_seed": policy,
        "policy_id": policy_id,
    }


def input_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["observation_image"]["sha256"],
        record["observation_wrist_image"]["sha256"],
        record["observation_state"]["sha256"],
        record["prompt"],
        record["policy_id"],
        record["policy_seed"],
    )


def analyze_hash_records(groups: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    order = ("A1", "B", "A2")
    counts = {name: len(groups[name]) for name in order}
    common = min(counts.values())
    first_input_diff = None
    first_response_diff = None
    first_same_input_different_response = None
    for index in range(common):
        records = [groups[name][index] for name in order]
        inputs = [input_signature(record) for record in records]
        responses = [record.get("response_actions", {}).get("sha256")
                     if isinstance(record.get("response_actions"), dict) else None
                     for record in records]
        inputs_equal = inputs[0] == inputs[1] == inputs[2]
        responses_equal = responses[0] == responses[1] == responses[2]
        if not inputs_equal and first_input_diff is None:
            first_input_diff = index
        if not responses_equal and first_response_diff is None:
            first_response_diff = index
        if inputs_equal and not responses_equal and first_same_input_different_response is None:
            first_same_input_different_response = index
    if first_same_input_different_response is not None:
        selected = first_same_input_different_response
        reason = "same_input_different_response"
    elif first_input_diff is not None:
        selected = first_input_diff
        reason = "first_discordant_input"
    else:
        selected = min(FALLBACK_CALL, max(0, common - 1))
        reason = "no_input_discordance_fallback_call37"
    return {
        "call_counts": counts,
        "common_call_count": common,
        "all_inputs_equal_over_common_calls": first_input_diff is None,
        "all_responses_equal_over_common_calls": first_response_diff is None,
        "first_input_hash_difference_call": first_input_diff,
        "first_response_hash_difference_call": first_response_diff,
        "first_same_input_different_response_call": first_same_input_different_response,
        "selected_replay_call": selected,
        "selected_replay_reason": reason,
    }


def compact_old_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "step": row["step"],
        "proprio8": row["proprio8"],
        "action7": row["action7"],
        "inference_call": row["inference_call"],
        "policy_seed": row["policy_seed"],
        "done": row["done"],
    } for row in trace]


def compact_runner_trace(trace: list[dict[str, Any]], policy_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    call_by_step = {}
    for index, record in enumerate(policy_calls):
        call_by_step[index * backend.REPLAN_STEPS] = record["policy_seed"]
    active_seed = None
    result = []
    for row in trace:
        if row["step"] in call_by_step:
            active_seed = call_by_step[row["step"]]
        result.append({
            "step": row["step"],
            "proprio8": row["state"],
            "action7": row["action"],
            "policy_seed": active_seed,
            "stage": row["stage"],
        })
    return result


def _inner_transport(helpers, args, uri):
    OpenPITransport, _ = helpers.import_physicalrsi(args.roborsi_root)
    return OpenPITransport(uri, timeout_s=args.timeout_seconds, api_key_env=args.api_key_env)


def run_old_episode(helpers, args, uri, identity, case, instruction, cap, *, retain_payloads):
    ambient, _ = harness_cli.validate_case_seeds(case)
    random.seed(ambient)
    np.random.seed(ambient)
    env = transport = None
    records: list[dict[str, Any]] = []
    capture = None
    trace: list[dict[str, Any]] = []
    try:
        env, task, raw, init_asset = harness_cli._reset_old_environment(helpers, case)
        _, Pi05Skill = helpers.import_physicalrsi(args.roborsi_root)
        capture = ReplayCaptureTransport(
            _inner_transport(helpers, args, uri), records, retain_payloads=retain_payloads
        )
        transport = helpers.EpisodeIdentityTransport(capture, dict(identity))
        skill = Pi05Skill(
            transport,
            checkpoint_id=case["policy_id"],
            state_dim=8,
            action_dim=7,
            image_keys=("observation/image", "observation/wrist_image"),
        )
        skill.reset()
        success, steps, calls = helpers.execute_policy_steps(
            env, raw, skill, instruction, case["policy_id"], case["policy_seed"],
            cap, DiscardFrames(), trace,
        )
        if transport.verified_metadata is None:
            raise RuntimeError("old diagnostic episode lacks actual connection identity")
        return {
            "result": {"status": "success" if success else "failure", "success": bool(success),
                       "steps": steps, "inference_calls": calls, "init_asset": init_asset,
                       "service_metadata": copy.deepcopy(transport.verified_metadata)},
            "trace": compact_old_trace(trace),
            "hashes": records,
            "payloads": [] if capture is None else capture.payloads,
        }
    finally:
        if transport is not None:
            transport.close()
        if env is not None:
            env.close()


def run_runner_episode(api, helpers, args, uri, identity, case, instruction, cap):
    ambient, _ = harness_cli.validate_case_seeds(case)
    random.seed(ambient)
    np.random.seed(ambient)
    records: list[dict[str, Any]] = []
    capture_holder = {}

    def transport_factory():
        capture = ReplayCaptureTransport(
            _inner_transport(helpers, args, uri), records, retain_payloads=False
        )
        capture_holder["capture"] = capture
        return helpers.EpisodeIdentityTransport(capture, dict(identity))

    skill = backend.Pi05HarnessSkill(
        transport_factory, policy_id=case["policy_id"], policy_seed=case["policy_seed"]
    )
    episode = backend.run_with_shared_runner(
        api.Runner,
        lambda: backend.Pi05HarnessEnvironment(
            api.Observation, helpers, suite=SUITE, task_id=TASK_ID,
            policy_id=case["policy_id"], policy_seed=case["policy_seed"],
        ),
        skill,
        api.initial_harness(instruction, cap, "pi05"),
        init_id=INIT_ID,
        official_cap=cap,
        max_seconds=args.max_episode_seconds,
    )
    episode.trace_frames.clear()
    return {
        "result": episode.report,
        "trace": compact_runner_trace(episode.report["trace"], episode.policy_calls),
        "hashes": records,
        "payloads": [],
    }


def run_aba(old_runner: Callable[[str], dict], new_runner: Callable[[str], dict]):
    order = []
    results = {}
    for name, runner in (("A1", old_runner), ("B", new_runner), ("A2", old_runner)):
        order.append(name)
        results[name] = runner(name)
    return order, results


def replay_payload(transport_factory: Callable[[], Any], payload: Mapping[str, Any], count=REPLAY_COUNT):
    if type(count) is not int or count < 1:
        raise ValueError("positive replay count required")
    transport = transport_factory()
    hashes = []
    original_seed = int(payload["policy_seed"])
    try:
        for _ in range(count):
            response = transport.infer(payload)
            if int(payload["policy_seed"]) != original_seed:
                raise RuntimeError("replay transport mutated policy_seed")
            hashes.append(harness_cli._array_hash_receipt(response["actions"])["sha256"])
    finally:
        transport.close()
    return {"calls": count, "policy_seed": original_seed, "action_hashes": hashes,
            "stable": len(set(hashes)) == 1}


def atomic_json(path: Path, value: Any):
    harness_cli.atomic_json(path, value)


def execute(args):
    if args.timeout_seconds <= 0 or args.max_episode_seconds <= 0:
        raise ValueError("timeouts must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("diagnostic output must be fresh")
    routes = harness_cli.load_routes(args.routes)
    policy_id = routes["tasks"][f"{SUITE}/{TASK_ID}"]
    identity = routes["identities"][policy_id]
    uri = harness_cli.service_uri(args.host, args.port)
    helpers = backend.import_eval_helpers(args.eval_helpers)
    api = backend.import_roborsi(args.roborsi_root)
    verified = harness_cli.preflight_identity(
        helpers, uri=uri, policy_id=policy_id, identity=identity,
        timeout_seconds=args.timeout_seconds, api_key_env=args.api_key_env,
    )
    case = fixed_case(policy_id)
    tasks = harness_cli.catalog_tasks()
    task = tasks[f"{SUITE}/{TASK_ID}"]
    instruction, cap = task["instruction"], task["max_steps"]
    output.mkdir(parents=True)
    episodes = output / "episodes"
    episodes.mkdir()
    manifest = {
        "schema": "pi05_loop_parity_diagnostic.manifest.v1",
        "case": case,
        "sequence": ["A1_old", "B_runner", "A2_old", "selected_payload_replay_x5"],
        "service_uri": uri,
        "service_identity": verified,
        "routes_sha256": harness_cli._sha256(args.routes),
        "eval_helpers_sha256": harness_cli._sha256(args.eval_helpers),
        "backend_sha256": harness_cli._sha256(Path(__file__).with_name("pi05_harness_backend.py")),
        "cli_sha256": harness_cli._sha256(Path(__file__).with_name("run_pi05_harness_eval.py")),
        "diagnostic_sha256": harness_cli._sha256(Path(__file__)),
        "ambient_seed_formula": "7 + joint40_task_number * 50 + init_id",
        "policy_seed_formula": "ambient_seed + replicate_id * 1000000007",
        "call_seed_formula": "policy_seed + inference_call * 1000003",
        "extra_inference_scope": "only selected payload replay after A1/B/A2; exactly five calls",
    }
    atomic_json(output / "manifest.json", manifest)

    def persist(name, value):
        persisted = {key: item for key, item in value.items() if key != "payloads"}
        atomic_json(episodes / f"{name}.json", persisted)
        return value

    old = lambda name: persist(name, run_old_episode(
        helpers, args, uri, verified, case, instruction, cap, retain_payloads=name == "A1"
    ))
    new = lambda name: persist(name, run_runner_episode(
        api, helpers, args, uri, verified, case, instruction, cap
    ))
    order, results = run_aba(old, new)

    diagnosis = analyze_hash_records({name: results[name]["hashes"] for name in order})
    selected = diagnosis["selected_replay_call"]
    retained = results["A1"]["payloads"]
    if not retained or selected >= len(retained):
        raise RuntimeError("selected replay payload was not retained from A1")

    def replay_transport_factory():
        inner = _inner_transport(helpers, args, uri)
        return helpers.EpisodeIdentityTransport(inner, dict(verified))

    replay = replay_payload(replay_transport_factory, retained[selected], REPLAY_COUNT)
    output_value = {
        "schema": "pi05_loop_parity_diagnostic.result.v1",
        "case": case,
        "execution_order": order,
        "episode_results": {name: results[name]["result"] for name in order},
        "diagnosis": diagnosis,
        "selected_payload": {
            "source": "A1",
            "inference_call": selected,
            "input_hashes": results["A1"]["hashes"][selected],
        },
        "replay": replay,
    }
    atomic_json(output / "diagnosis.json", output_value)
    return output_value


def main(argv=None):
    execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
