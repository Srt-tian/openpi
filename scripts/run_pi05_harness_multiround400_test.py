#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import run_pi05_harness_multiround400 as target


COMMIT = "1" * 40
PHYSICAL = "2" * 40
MANIFEST = "3" * 64
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(root: Path) -> tuple[Path, dict]:
    source = root / "source"; source.mkdir()
    baseline_tasks, candidate_tasks, catalog, plan = {}, {}, [], []
    joint = 0
    for suite in SUITES:
        for task in range(10):
            key = f"{suite}/{task}"; rel = f"tasks/{suite}/{task}.json"
            baseline_tasks[key] = candidate_tasks[key] = rel
            value = {"schema": 1, "key": key, "harness": {"name": "pi05-base"}}
            dump(source / "baseline" / rel, value)
            if key == "libero_goal/3": value = {**value, "pi05_control": {"enabled": True}}
            dump(source / "candidate" / rel, value)
            catalog.append({"key": key, "suite": suite, "task_index": task,
                            "instruction": key, "max_steps": target.OFFICIAL_CAPS[suite]})
            for init in range(10):
                plan.append({"suite": suite, "task_id": task, "init_id": init,
                    "replicate_id": 0, "joint_task_number": joint,
                    "official_cap": target.OFFICIAL_CAPS[suite]})
            joint += 1
    dump(source / "baseline/registry.json", {"schema": 1, "tasks": baseline_tasks})
    dump(source / "candidate/registry.json", {"schema": 1, "tasks": candidate_tasks})
    dump(source / "case_plan.json", {"schema": "pi05_harness_candidate400_cases.v1",
                                     "cases": plan})
    dump(source / "catalog.json", {"schema": 1, "tasks": catalog})
    dump(source / "routes.json", {"tasks": {row["key"]: "base" for row in catalog},
        "identities": {"base": {"checkpoint_sha256": MANIFEST, "adapter_sha256": None}}})
    (source / "aggregate.py").write_text("# aggregate\n")
    (source / "export.py").write_text("# export\n")
    for worker in range(4):
        cases = [{key: row[key] for key in target.WORKER_CASE_FIELDS}
                 for row in plan[worker::4]]
        dump(source / f"cases{worker}.json", {"schema": "pi05_harness_cases.v1", "cases": cases})
        common = {"cases": f"cases{worker}.json", "routes": "routes.json", "mode": "harness"}
        dump(source / f"job{worker}.json", {"schema": "pi05_harness_worker.v1", "batches": [
            {**common, "name": f"worker{worker}_control", "registry": "baseline/registry.json"},
            {**common, "name": f"worker{worker}_candidate", "registry": "candidate/registry.json"}]})
    spec = {"schema": target.SCHEMA, "round_id": "round01",
        "round_dir": str(root / "round"),
        "canonical": {"repository": "https://github.com/example/openpi.git",
                      "branch": "feature/candidate", "commit": COMMIT},
        "candidate_registry": "candidate/registry.json",
        "baseline_registry": "baseline/registry.json",
        "changed_tasks": ["libero_goal/3"], "case_plan": "case_plan.json",
        "task_catalog": "catalog.json", "routes": "routes.json",
        "aggregate_script": "aggregate.py", "export_script": "export.py",
        "weights": {"base_checkpoint": "/weights/base", "plugin_checkpoint": "/weights/plugins",
                    "plugin_manifest_sha256": MANIFEST, "policy_id": "base", "adapter_sha256": None},
        "runtime": {"physicalrsi_root": "/code/r69", "physicalrsi_commit": PHYSICAL,
                    "runtime_root": "/runtime/libero"},
        "workers": [{"worker_id": i, "host": f"worker{i}", "user": "magiclab",
                     "control_path": f"/tmp/worker{i}.sock", "checkout": "/code/candidate",
                     "job": f"job{i}.json", "output": f"/outputs/round01_worker{i}",
                     "policy_python": "/venv/bin/python", "gpu_id": 0, "port": 18501}
                    for i in range(4)],
        "collection": {"collector_worker_id": 0, "relay_dir": str(root / "relay"),
                       "collection_root": "/outputs/round01_collection",
                       "max_compressed_bytes": 500_000_000}}
    spec_path = root / "spec.json"; dump(spec_path, spec)
    return source, spec


def validate(path: Path, source: Path):
    def git(_root, *args):
        values = {("rev-parse", "HEAD"): COMMIT, ("rev-parse", "@{upstream}"): COMMIT,
            ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"):
                "origin/feature/candidate",
            ("branch", "--show-current"): "feature/candidate",
            ("remote", "get-url", "origin"): "https://github.com/example/openpi.git",
            ("status", "--porcelain"): ""}
        return values[args]
    with mock.patch.object(target, "git_value", side_effect=git):
        return target.load_and_validate(path, source)


class MultiRound400Test(unittest.TestCase):
    def test_valid_exact_400_and_four_matched_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, spec = fixture(root)
            value = validate(root / "spec.json", source)
            self.assertEqual([w["worker_id"] for w in value["workers"]], list(range(4)))
            self.assertTrue(all(len(w["job_sha256"]) == 64 for w in value["workers"]))
            self.assertEqual(value["changed_tasks"], ["libero_goal/3"])

    def test_real_checked_in_candidate400_jobs_and_four_field_cases(self):
        source = Path(os.environ.get("PI05_MULTIRound400_SOURCE",
                          Path(__file__).resolve().parents[1])).resolve()
        required = source / "configs/pi05_harness_candidate400/job_worker0.json"
        if not required.is_file():
            self.skipTest("real checked-in candidate400 fixtures are not in this patch-only tree")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"schema": target.SCHEMA, "round_id": "real-config-integration",
                "round_dir": str(root / "round"),
                "canonical": {"repository": "https://github.com/example/openpi.git",
                    "branch": "feature/candidate", "commit": COMMIT},
                "candidate_registry": "configs/pi05_harness_candidate400/registry.json",
                "baseline_registry": "configs/pi05_harness/registry.json",
                "changed_tasks": ["libero_goal/3", "libero_10/8"],
                "case_plan": "configs/pi05_harness_candidate400/case_plan.json",
                "task_catalog": "configs/pi05_harness_candidate400/task_catalog.json",
                "routes": "configs/pi05_harness/routes_base.json",
                "aggregate_script": "scripts/aggregate_pi05_harness_candidate400.py",
                "export_script": "scripts/export_pi05_harness_artifacts.py",
                "weights": {"base_checkpoint": "/weights/base",
                    "plugin_checkpoint": "/weights/plugins",
                    "plugin_manifest_sha256": MANIFEST, "policy_id": "base",
                    "adapter_sha256": None},
                "runtime": {"physicalrsi_root": "/code/r69",
                    "physicalrsi_commit": PHYSICAL, "runtime_root": "/runtime/libero"},
                "workers": [{"worker_id": i, "host": f"worker{i}", "user": "magiclab",
                    "control_path": f"/tmp/worker{i}.sock", "checkout": "/code/candidate",
                    "job": f"configs/pi05_harness_candidate400/job_worker{i}.json",
                    "output": f"/outputs/round_worker{i}",
                    "policy_python": "/venv/bin/python", "gpu_id": 0, "port": 18501}
                    for i in range(4)],
                "collection": {"collector_worker_id": 0, "relay_dir": str(root / "relay"),
                    "collection_root": "/outputs/round_collection",
                    "max_compressed_bytes": 500_000_000}}
            spec_path = root / "spec.json"; dump(spec_path, spec)
            value = validate(spec_path, source)
            self.assertEqual(len(value["workers"]), 4)
            real_cases = json.loads((source /
                "configs/pi05_harness_candidate400/cases_worker0.json").read_text())
            self.assertEqual(set(real_cases["cases"][0]), target.WORKER_CASE_FIELDS)

    def test_unchanged_task_must_be_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, _ = fixture(root)
            path = source / "candidate/tasks/libero_spatial/0.json"
            value = json.loads(path.read_text()); value["changed"] = True; dump(path, value)
            with self.assertRaisesRegex(ValueError, "unchanged task"):
                validate(root / "spec.json", source)

    def test_case_plan_rejects_duplicate_wild_and_seed_selection(self):
        for mutation in ("duplicate", "wild", "replicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source, _ = fixture(root)
                path = source / "case_plan.json"; value = json.loads(path.read_text())
                if mutation == "duplicate": value["cases"][-1] = copy.deepcopy(value["cases"][0])
                elif mutation == "wild": value["cases"][0]["seed"] = 9
                else: value["cases"][0]["replicate_id"] = 1
                dump(path, value)
                with self.assertRaises(ValueError): validate(root / "spec.json", source)

    def test_plan_joint_and_official_cap_are_derived_from_catalog(self):
        for field in ("joint_task_number", "official_cap"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source, _ = fixture(root)
                path = source / "case_plan.json"; value = json.loads(path.read_text())
                value["cases"][0][field] += 1; dump(path, value)
                with self.assertRaisesRegex(ValueError, "disagrees with catalog"):
                    validate(root / "spec.json", source)

    def test_job_overlap_or_candidate_registry_mismatch_rejected(self):
        for mutation in ("overlap", "registry"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source, _ = fixture(root)
                if mutation == "overlap":
                    (source / "cases1.json").write_bytes((source / "cases0.json").read_bytes())
                else:
                    p = source / "job1.json"; value = json.loads(p.read_text())
                    value["batches"][1]["registry"] = "baseline/registry.json"; dump(p, value)
                with self.assertRaises(ValueError): validate(root / "spec.json", source)

    def test_unknown_training_retry_or_selector_fields_rejected(self):
        for field in ("train", "retries", "selected_seeds"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source, spec = fixture(root)
                spec[field] = True; dump(root / "spec.json", spec)
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    validate(root / "spec.json", source)

    def test_worker_command_reuses_checked_entrypoint_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, _ = fixture(root)
            spec = validate(root / "spec.json", source)
            command = target.worker_command(spec, spec["workers"][0])
            self.assertTrue(command[1].endswith("scripts/run_pi05_harness_worker.py"))
            self.assertNotIn("retry", " ".join(command).lower())
            self.assertIn("--expected-commit", command)

    def test_failure_and_decision_are_factual_and_never_promote(self):
        aggregate = {"score_complete": True, "exact_prefix_attribution_complete": False,
            "score": {"control_successes": 390, "candidate_successes": 391}, "pairs": [
                {"suite": "libero_goal", "task_id": 3, "init_id": 4,
                 "outcome": "recovered", "candidate_success": True, "causal_gate_pass": True},
                {"suite": "libero_10", "task_id": 8, "init_id": 8,
                 "outcome": "regressed", "candidate_success": False,
                 "causal_gate_pass": False, "causal_confounds": ["different"]}]}
        spec = {"round_id": "r", "changed_tasks": ["libero_goal/3", "libero_10/8"]}
        failures, decision = target.failure_decision(aggregate, spec)
        self.assertEqual(len(failures["recoveries"]), 1)
        self.assertEqual(len(failures["regressions"]), 1)
        self.assertEqual(len(failures["candidate_failures"]), 1)
        self.assertFalse(decision["eligible_for_promotion"])
        self.assertFalse(decision["promotion_performed"])
        self.assertEqual(decision["decision"], "manual_review_required")
        self.assertEqual(decision["review_authority"], "root_model_review")

    def test_wait_repolls_same_launch_after_transient_network_error(self):
        with tempfile.TemporaryDirectory() as directory:
            round_dir = Path(directory)
            spec = {"round_dir": str(round_dir)}
            complete = [{"worker_id": i, "alive": False,
                         "controller_status": "complete", "episodes": 200}
                        for i in range(4)]
            transient = subprocess.CalledProcessError(255, ["ssh"])
            with mock.patch.object(target, "status", side_effect=[transient, complete]) as status_mock, \
                    mock.patch.object(target.time, "sleep") as sleep_mock:
                rows = target.wait_terminal(spec, mock.Mock(), 10)
            self.assertEqual(rows, complete)
            self.assertEqual(status_mock.call_count, 2)
            sleep_mock.assert_called_once_with(10)
            self.assertTrue((round_dir / "terminal.json").is_file())

    def test_create_only_round_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            target.save_create_only(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                target.save_create_only(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
