#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import build_pi05_harness_configs as baseline_builder
import build_pi05_instruction_lease_plan as lease
import pi05_harness_backend as backend


SOURCE = Path(os.environ.get("PI05_TEST_PHYSICALRSI_ROOT", str(lease.SOURCE)))


class InstructionLeasePlanTest(unittest.TestCase):
    def test_two_complete_registries_and_57_episode_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"banks": {name: {"adapter_sha256": char * 64}
                for name, char in zip(("spatial", "object", "goal", "long"), "abcd")}}))
            baseline = baseline_builder.build(root / "baseline", manifest, SOURCE)
            output = lease.build(root / "lease", baseline, SOURCE)
            base_registry = json.loads((baseline / "registry.json").read_text())
            catalog = json.loads((baseline / "task_catalog.json").read_text())
            rows = {row["key"]: row for row in catalog["tasks"]}
            api = backend.import_roborsi(SOURCE)
            tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                     for key, row in rows.items()}
            for variant, steps in lease.LEASES.items():
                registry = json.loads((output / variant / "registry.json").read_text())
                for key, relative in registry["tasks"].items():
                    old = json.loads((baseline / base_registry["tasks"][key]).read_text())
                    new = json.loads((output / variant / relative).read_text())
                    if key == "libero_goal/3":
                        expected = dict(old, pi05_control=lease.control(steps))
                        self.assertEqual(new, expected)
                        self.assertEqual(new["harness"], old["harness"])
                        self.assertEqual(new["parent_sha256"], old["parent_sha256"])
                    else:
                        self.assertEqual(new, old)
                self.assertEqual(registry["metadata"]["evidence"]["hard_case"], "8/10")
                self.assertEqual(registry["metadata"]["evidence"]["unconditional_regressions"], 3)
                proposal = api.TaskHarnessRegistry(
                    output / variant / "registry.json", {"pi05"}).materialize(tasks)
                self.assertEqual(proposal["task_config_sha256"],
                                 registry["metadata"]["task_config_sha256"])
            episodes = 0
            for worker in range(5):
                job = json.loads((output / f"job_worker{worker}.json").read_text())
                self.assertEqual([b["name"].split(f"worker{worker}_", 1)[1]
                                  for b in job["batches"]], ["control", "lease40", "lease80"])
                self.assertTrue(all(b["mode"] == "harness" and
                                    b["routes"].endswith("routes_base.json") for b in job["batches"]))
                cases = json.loads((lease.PATCH / job["batches"][0]["cases"]).read_text())["cases"]
                episodes += len(cases) * len(job["batches"])
            self.assertEqual(episodes, 57)

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                lease.build(output, Path(directory) / "unused", SOURCE)


if __name__ == "__main__":
    unittest.main()
