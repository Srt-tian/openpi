#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import pi05_harness_backend as backend
import run_pi05_harness_eval as cli


CHECKPOINT = "c" * 64
ADAPTER = "a" * 64
EVAL_SCRIPT = Path(os.environ.get(
    "PI05_TEST_EVAL_SCRIPT",
    "/home/user/tian_ws/pi05_plugin_eval_20261008/patch/scripts/eval_pi05_plugins.py",
))


def route_document(tasks=None):
    if tasks is None:
        tasks = {
            f"{suite}/{task_id}": "object"
            for suite in cli.SUITES
            for task_id in range(10)
        }
    return {
        "schema": cli.ROUTES_SCHEMA,
        "tasks": tasks,
        "identities": {
            "object": {
                "checkpoint_sha256": CHECKPOINT,
                "base_graph": "pi05_lora",
                "adapter_sha256": ADAPTER,
            }
        },
    }


def case_document(cases=None):
    return {
        "schema": cli.CASES_SCHEMA,
        "cases": cases or [{"suite": "libero_object", "task_id": 4, "init_id": 0}],
    }


def fake_report(actions, states, success=False):
    return {
        "success": success,
        "status": "task_success" if success else "budget_exhausted",
        "steps": len(actions),
        "trace": [
            {"state": list(state), "action": list(action), "step": index, "stage": 0,
             "skill": "pi05", "signals": {}}
            for index, (state, action) in enumerate(zip(states, actions))
        ],
    }


class CliValidationTest(unittest.TestCase):
    def write_json(self, directory, name, value):
        path = Path(directory) / name
        path.write_text(json.dumps(value))
        return path

    def test_routes_are_task_only_and_identity_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self.write_json(directory, "valid.json", route_document())
            loaded = cli.load_routes(valid)
            self.assertEqual(loaded["tasks"]["libero_object/4"], "object")
            invalid_tasks = dict(route_document()["tasks"])
            invalid_tasks["libero_object/4/0"] = invalid_tasks.pop("libero_object/4")
            invalid = route_document(invalid_tasks)
            bad = self.write_json(directory, "bad.json", invalid)
            with self.assertRaisesRegex(ValueError, "exact LIBERO-40"):
                cli.load_routes(bad)

    def test_routes_accept_only_the_documented_nonrouting_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            value = route_document()
            value.update({
                "name": "failure-union-replicates",
                "selection_basis": "frozen union of failed cases",
                "validation_status": "proposal-only",
                "routing_inputs": ["suite", "task_index"],
            })
            valid = self.write_json(directory, "metadata.json", value)
            loaded = cli.load_routes(valid)
            self.assertEqual(loaded["metadata"]["validation_status"], "proposal-only")
            value["routing_inputs"] = ["suite", "task_index", "init_id"]
            bad = self.write_json(directory, "bad_inputs.json", value)
            with self.assertRaisesRegex(ValueError, "routing_inputs"):
                cli.load_routes(bad)
            value["routing_inputs"] = ["suite", "task_index"]
            value["unknown"] = True
            unknown = self.write_json(directory, "unknown.json", value)
            with self.assertRaisesRegex(ValueError, "schema"):
                cli.load_routes(unknown)

    def test_cases_are_explicit_unique_official_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, "cases.json", case_document())
            cases = cli.load_cases(path)
            self.assertEqual(cases[0]["joint_task_number"], 14)
            self.assertEqual(cases[0]["policy_seed"], 707)
            self.assertEqual(cases[0]["ambient_seed"], 707)
            self.assertEqual(cases[0]["replicate_id"], 0)
            replicated = self.write_json(directory, "replicated.json", case_document([
                {"suite": "libero_object", "task_id": 4, "init_id": 0, "replicate_id": 9}
            ]))
            replicated_case = cli.load_cases(replicated)[0]
            self.assertEqual(
                replicated_case["policy_seed"],
                707 + 9 * backend.REPLICATE_SEED_STRIDE,
            )
            self.assertEqual(replicated_case["ambient_seed"], 707)
            self.assertEqual(cli.validate_case_seeds(replicated_case), (707, replicated_case["policy_seed"]))
            duplicate = case_document([
                {"suite": "libero_object", "task_id": 4, "init_id": 0},
                {"suite": "libero_object", "task_id": 4, "init_id": 0, "replicate_id": 0},
            ])
            bad = self.write_json(directory, "duplicate.json", duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cli.load_cases(bad)

    def test_replicate_id_rejects_bool_negative_and_out_of_range(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, invalid in enumerate((True, -1, 10)):
                path = self.write_json(directory, f"bad{index}.json", case_document([
                    {"suite": "libero_object", "task_id": 4, "init_id": 0,
                     "replicate_id": invalid}
                ]))
                with self.assertRaisesRegex(ValueError, "replicate_id"):
                    cli.load_cases(path)

    def test_one_invocation_cannot_mix_fixed_service_policies(self):
        cases = [
            {"suite": "libero_object", "task_id": 4},
            {"suite": "libero_goal", "task_id": 3},
        ]
        routes = {
            "tasks": {"libero_object/4": "object", "libero_goal/3": "goal"}
        }
        with self.assertRaisesRegex(ValueError, "one fixed-policy"):
            cli.bind_routes(cases, routes)

    def test_endpoint_rejects_embedded_scheme_credentials_or_path(self):
        self.assertEqual(cli.service_uri("127.0.0.1", 8000), "ws://127.0.0.1:8000")
        for host in ("ws://127.0.0.1", "user@host", "host/path"):
            with self.assertRaises(ValueError):
                cli.service_uri(host, 8000)

    def test_parity_requires_materialized_full_cap_single_stage(self):
        valid = {
            "stages": [{"skill": "pi05", "instruction": "original", "max_steps": 280,
                        "until": None, "on_timeout": "abort"}]
        }
        self.assertEqual(cli.validate_parity_harness(valid, 280), ("original", 280))
        invalid = {"stages": valid["stages"] * 2}
        with self.assertRaisesRegex(ValueError, "single-stage"):
            cli.validate_parity_harness(invalid, 280)

    def test_preflight_identity_uses_real_seed_protocol_validator(self):
        helpers = backend.import_eval_helpers(EVAL_SCRIPT)
        identity = {
            "policy_id": "object",
            "checkpoint_sha256": CHECKPOINT,
            "base_graph": "pi05_lora",
            "adapter_sha256": ADAPTER,
        }
        raw_metadata = {
            **identity,
            "policy_seed_protocol": helpers.SEED_PROTOCOL,
        }
        # Exercise the evaluator's real identity validator, while replacing
        # only its network preflight with that already-validated result.
        validated = helpers.validated_metadata_subset(
            raw_metadata, "object", CHECKPOINT, "pi05_lora"
        )
        with mock.patch.object(helpers, "validate_service_metadata", return_value=validated):
            result = cli.preflight_identity(
                helpers,
                uri="ws://127.0.0.1:8000",
                policy_id="object",
                identity=identity,
                timeout_seconds=30.0,
                api_key_env="OPENPI_API_KEY",
            )
        self.assertEqual(result["policy_seed_protocol"], helpers.SEED_PROTOCOL)
        wrong = {**raw_metadata, "policy_seed_protocol": "wrong"}
        with self.assertRaisesRegex(ValueError, "metadata"):
            helpers.validated_metadata_subset(wrong, "object", CHECKPOINT, "pi05_lora")


class ParityComparisonTest(unittest.TestCase):
    def old_result(self, actions, states, seeds, success=False):
        return {
            "success": success,
            "trace": [
                {"action7": list(action), "proprio8": list(state), "policy_seed": seed}
                for action, state, seed in zip(actions, states, seeds)
            ],
        }

    def new_result(self, actions, states, call_seeds, success=False):
        return backend.BackendEpisode(
            report=fake_report(actions, states, success),
            trace_frames=[],
            environment_actions=[np.asarray(action) for action in actions],
            policy_calls=[{"policy_seed": seed} for seed in call_seeds],
        )

    def test_exact_parity(self):
        actions = [[0, 0, 0, 0, 0, 0, 1.02], [1, 0, 0, 0, 0, 0, -1]]
        states = [list(range(8)), list(range(1, 9))]
        old = self.old_result(actions, states, [707, 707], True)
        new = self.new_result(actions, states, [707], True)
        comparison = cli.compare_parity(old, new)
        self.assertTrue(comparison["equal"])
        self.assertEqual(comparison["tolerance"], 0)

    def test_first_difference_is_reported_not_hidden(self):
        actions = [[0, 0, 0, 0, 0, 0, 1.02]]
        states = [list(range(8))]
        old = self.old_result(actions, states, [707])
        changed = [row[:] for row in actions]
        changed[0][6] = 1.0200000001
        new = self.new_result(changed, states, [707])
        comparison = cli.compare_parity(old, new)
        self.assertFalse(comparison["equal"])
        self.assertEqual(comparison["first_action_mismatch"]["index"], [0, 6])


class ExecutePersistenceTest(unittest.TestCase):
    def make_args(self, directory, mode, cases):
        root = Path(directory)
        routes_path = root / "routes.json"
        cases_path = root / "cases.json"
        registry_path = root / "registry.json"
        helper_path = root / "helpers.py"
        routes_path.write_text(json.dumps(route_document()))
        cases_path.write_text(json.dumps(case_document(cases)))
        registry_path.write_text("{}")
        helper_path.write_text("")
        return SimpleNamespace(
            roborsi_root=root / "roborsi",
            eval_helpers=helper_path,
            registry=registry_path,
            routes=routes_path,
            cases=cases_path,
            host="127.0.0.1",
            port=8000,
            output=root / "output",
            mode=mode,
            timeout_seconds=30.0,
            max_episode_seconds=1200.0,
            api_key_env="OPENPI_API_KEY",
            record_payload_hashes=False,
        )

    @staticmethod
    def patches(tasks, run_new):
        api = SimpleNamespace(
            Runner=object,
            Observation=object,
            initial_harness=lambda instruction, cap, skill: {
                "schema": 1,
                "name": "official",
                "remember": [],
                "stages": [{"skill": skill, "instruction": instruction, "max_steps": cap,
                            "until": None, "on_timeout": "abort"}],
            },
        )
        helpers = SimpleNamespace(STEP_CAPS={
            "libero_spatial": 220, "libero_object": 280,
            "libero_goal": 300, "libero_10": 520,
        })
        return (
            mock.patch.object(cli.backend, "import_roborsi", return_value=api),
            mock.patch.object(cli.backend, "import_eval_helpers", return_value=helpers),
            mock.patch.object(cli, "catalog_tasks", return_value=tasks),
            mock.patch.object(cli.backend, "materialize_pi05_registry", return_value={"harnesses": {}}),
            mock.patch.object(cli, "preflight_identity", return_value={
                "policy_id": "object", "checkpoint_sha256": CHECKPOINT,
                "base_graph": "pi05_lora", "adapter_sha256": ADAPTER,
            }),
            mock.patch.object(cli, "_run_new_loop", side_effect=run_new),
            mock.patch.object(cli, "save_video", return_value={"written": False}),
            mock.patch.object(cli, "_sha256", return_value="f" * 64),
        )

    def test_harness_mode_keeps_error_case_and_continues(self):
        cases = [
            {"suite": "libero_object", "task_id": 4, "init_id": 0},
            {"suite": "libero_object", "task_id": 4, "init_id": 1},
        ]
        tasks = {"libero_object/4": {"instruction": "original", "max_steps": 280}}
        good = backend.BackendEpisode(
            report=fake_report([[0] * 7], [[0] * 8], True),
            trace_frames=[], environment_actions=[np.zeros(7)],
            policy_calls=[{"policy_seed": 708}],
        )
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "harness", cases)
            patches = self.patches(tasks, [RuntimeError("service"), good])
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                result = cli.execute(args)
            self.assertEqual(result["completed"], 2)
            self.assertEqual(result["errors"], 1)
            self.assertEqual(result["cases"][0]["error_type"], "RuntimeError")
            saved = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(saved["completed"], 2)
            self.assertTrue((args.output / "episodes/000_libero_object_4_0_r00.json").is_file())
            self.assertTrue((args.output / "episodes/001_libero_object_4_1_r00.json").is_file())

    def test_parity_rejects_nonzero_replicate_before_execution(self):
        cases = [
            {"suite": "libero_object", "task_id": 4, "init_id": 0, "replicate_id": 1}
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "parity", cases)
            with self.assertRaisesRegex(ValueError, "replicate_id=0"):
                cli.execute(args)
            self.assertFalse(args.output.exists())

    def test_main_returns_nonzero_for_parity_mismatch(self):
        with mock.patch.object(cli, "parse_args", return_value=object()), mock.patch.object(
            cli, "execute", return_value={"mode": "parity", "errors": 0, "parity_equal": False}
        ):
            self.assertEqual(cli.main([]), 2)

    def test_parity_mode_wires_same_registry_harness_and_persists_both_traces(self):
        cases = [{"suite": "libero_object", "task_id": 4, "init_id": 0}]
        tasks = {"libero_object/4": {"instruction": "original", "max_steps": 280}}
        action, state = [0.0] * 7, [0.0] * 8
        old = {
            "status": "success", "success": True, "steps": 1, "inference_calls": 1,
            "trace": [{"action7": action, "proprio8": state, "policy_seed": 707}],
            "frames": [], "service_metadata": {"policy_id": "object"},
            "init_asset": {"count": 50},
        }
        new = backend.BackendEpisode(
            report=fake_report([action], [state], True),
            trace_frames=[], environment_actions=[np.asarray(action)],
            policy_calls=[{"policy_seed": 707}],
        )
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "parity", cases)
            patches = self.patches(tasks, [new])
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], mock.patch.object(
                cli, "_run_old_loop", return_value=old
            ):
                result = cli.execute(args)
            self.assertTrue(result["parity_equal"])
            evidence = json.loads((args.output / "episodes/000_libero_object_4_0_r00.json").read_text())
            self.assertIn("old", evidence)
            self.assertIn("runner", evidence)
            self.assertEqual(evidence["harness"]["stages"][0]["instruction"], "original")

    def test_legacy_mode_persists_original_loop_backend(self):
        cases = [{"suite": "libero_object", "task_id": 4, "init_id": 1,
                  "replicate_id": 9}]
        tasks = {"libero_object/4": {"instruction": "original", "max_steps": 280}}
        old = {
            "status": "failure", "success": False, "steps": 280, "inference_calls": 56,
            "trace": [], "frames": [], "service_metadata": {"policy_id": "object"},
            "init_asset": {"count": 50}, "payload_hashes": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "legacy", cases)
            patches = self.patches(tasks, [])
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], mock.patch.object(
                cli, "_run_old_loop", return_value=old
            ) as run_old:
                result = cli.execute(args)
            self.assertEqual(result["errors"], 0)
            self.assertEqual(result["cases"][0]["execution_backend"], "legacy")
            self.assertEqual(result["cases"][0]["status"], "failure")
            self.assertEqual(run_old.call_args.args[-2:], ("original", 280))
            evidence_path = args.output / "episodes/000_libero_object_4_1_r09.json"
            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["execution_backend"], "legacy")


class PayloadHashTransportTest(unittest.TestCase):
    def test_records_real_payload_bytes_seed_prompt_and_response_actions(self):
        class Inner:
            metadata = {"policy_id": "object"}

            def infer(self, payload):
                return {"actions": np.arange(35, dtype=np.float32).reshape(5, 7)}

            def close(self):
                self.closed = True

        records = []
        inner = Inner()
        transport = cli.PayloadHashTransport(inner, records)
        payload = {
            "observation/image": np.zeros((2, 2, 3), np.uint8),
            "observation/wrist_image": np.ones((2, 2, 3), np.uint8),
            "observation/state": np.arange(8, dtype=np.float64),
            "prompt": "original",
            "policy_id": "object",
            "policy_seed": 9_000_000_771,
        }
        response = transport.infer(payload)
        self.assertEqual(records[0]["policy_seed"], payload["policy_seed"])
        self.assertEqual(records[0]["prompt"], "original")
        self.assertEqual(records[0]["status"], "ok")
        self.assertEqual(
            records[0]["observation_image"]["sha256"],
            cli.hashlib.sha256(payload["observation/image"].tobytes()).hexdigest(),
        )
        self.assertEqual(
            records[0]["response_actions"]["sha256"],
            cli.hashlib.sha256(response["actions"].tobytes()).hexdigest(),
        )
        self.assertIs(transport.metadata, inner.metadata)


if __name__ == "__main__":
    unittest.main()
