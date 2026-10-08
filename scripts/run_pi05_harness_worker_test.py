#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock


if "run_pi05_shard" not in sys.modules:
    shard = ModuleType("run_pi05_shard")
    shard.ensure_gpu_idle = lambda gpu: None
    shard.gpu_locks = lambda root, gpu: []
    shard.save = lambda path, value: path.write_text(json.dumps(value))
    shard.stop = lambda process: None
    sys.modules["run_pi05_shard"] = shard

import run_pi05_harness_worker as worker


def write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class FakeProcess:
    next_pid = 100

    def __init__(self, command, **kwargs):
        self.command, self.returncode = command, None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 9
        return self.returncode


class WorkerTest(unittest.TestCase):
    def test_summary_requires_exact_coverage_identity_and_parity(self):
        cases = [{"suite": "libero_object", "task_id": 4, "init_id": 2, "replicate_id": 7}]
        row = {**cases[0], "policy_id": "object", "status": "success"}
        summary = {"cases": [row], "complete": True, "errors": 0,
                   "planned": 1, "completed": 1, "parity_equal": True}
        worker.validate_summary(summary, cases, "object", "parity")
        mutations = [
            {**summary, "cases": []},
            {**summary, "cases": [{**row, "replicate_id": 8}]},
            {**summary, "cases": [{**row, "policy_id": "base"}]},
            {**summary, "errors": 1},
            {**summary, "parity_equal": False},
        ]
        for invalid in mutations:
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                worker.validate_summary(invalid, cases, "object", "parity")

    def test_checked_path_rejects_escape_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            inside = write(root / "cases.json", {})
            outside = write(Path(directory) / "outside.json", {})
            self.assertEqual(worker.checked_path(root, Path("cases.json")), inside)
            for path in (outside, Path("missing.json")):
                with self.assertRaisesRegex(ValueError, "checked-in file"):
                    worker.checked_path(root, path)

    def test_checkout_requires_exact_head_and_clean_tree(self):
        with mock.patch.object(worker.subprocess, "check_output",
                               side_effect=["abc\n", ""]):
            worker.check_checkout(Path("/checkout"), "abc")
        with mock.patch.object(worker.subprocess, "check_output",
                               side_effect=["wrong\n"]):
            with self.assertRaisesRegex(ValueError, "frozen commit"):
                worker.check_checkout(Path("/checkout"), "abc")
        with mock.patch.object(worker.subprocess, "check_output",
                               side_effect=["abc\n", " M file\n"]):
            with self.assertRaisesRegex(ValueError, "not clean"):
                worker.check_checkout(Path("/checkout"), "abc")

    def fixture(self, directory: str, checkpoint_override=None):
        root = Path(directory) / "code"
        plugins = Path(directory) / "plugins"
        root.mkdir(); plugins.mkdir()
        manifest = write(plugins / "manifest.json", {"banks": {
            "object": {"adapter_sha256": "a" * 64}}})
        checkpoint = hashlib.sha256(manifest.read_bytes()).hexdigest()
        routes = write(root / "routes.json", {
            "schema": "pi05_harness_routes.v1",
            "tasks": {"libero_object/4": "object"},
            "identities": {"object": {"checkpoint_sha256": checkpoint_override or checkpoint,
                                        "base_graph": "pi05_lora", "adapter_sha256": "a" * 64}}})
        cases = write(root / "cases.json", {"schema": "pi05_harness_cases.v1", "cases": [
            {"suite": "libero_object", "task_id": 4, "init_id": 0, "replicate_id": 7}]})
        registry = write(root / "registry.json", {"schema": 1})
        job = write(root / "job.json", {"schema": "pi05_harness_worker.v1", "batches": [{
            "name": "object-probe", "mode": "harness", "registry": registry.name,
            "routes": routes.name, "cases": cases.name}]})
        args = argparse.Namespace(code=root, job=job, base=Path(directory) / "base",
            plugins=plugins, output=Path(directory) / "output",
            physicalrsi_root=Path(directory) / "physicalrsi", runtime_root=Path(directory) / "runtime",
            policy_python=Path(directory) / "venv/bin/python", expected_commit="c" * 40,
            expected_physicalrsi_commit="d" * 40, gpu_id=0, port=self.free_port())
        return args

    @staticmethod
    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def test_checkpoint_mismatch_fails_before_output_or_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory, "f" * 64)
            with mock.patch.object(argparse.ArgumentParser, "parse_args", return_value=args), \
                 mock.patch.object(worker, "check_checkout"), \
                 mock.patch.object(worker, "ensure_gpu_idle"), \
                 mock.patch.object(worker, "gpu_locks", return_value=[]), \
                 mock.patch.object(worker.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    worker.main()
            popen.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_eval_error_stops_batch_and_preserves_exact_input_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            stopped, processes = [], []
            def launch(command, **kwargs):
                process = FakeProcess(command, **kwargs)
                processes.append(process)
                return process
            with mock.patch.object(argparse.ArgumentParser, "parse_args", return_value=args), \
                 mock.patch.object(worker, "check_checkout"), \
                 mock.patch.object(worker, "ensure_gpu_idle"), \
                 mock.patch.object(worker, "gpu_locks", return_value=[object()]), \
                 mock.patch.object(worker.socket, "create_connection"), \
                 mock.patch.object(worker.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(worker, "stop", side_effect=stopped.append):
                with self.assertRaisesRegex(RuntimeError, "without retry"):
                    worker.main()
            self.assertEqual(len(processes), 2)
            eval_command = processes[1].command
            for flag, filename in (("--registry", "registry.json"), ("--routes", "routes.json"),
                                   ("--cases", "cases.json")):
                self.assertEqual(Path(eval_command[eval_command.index(flag) + 1]).name, filename)
            self.assertIn(processes[0], stopped)
            self.assertIn(processes[1], stopped)
            status = json.loads((args.output / "controller.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["retry_policy"], "none")
            self.assertEqual(status["services_per_gpu"], 1)
            self.assertEqual(status["simulators_per_gpu"], 1)


if __name__ == "__main__":
    unittest.main()
