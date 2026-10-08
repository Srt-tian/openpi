#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import build_pi05_closed_dwell_lift_plan as plan
import build_pi05_harness_configs as baseline_builder
import pi05_harness_backend as backend


SOURCE = Path(os.environ.get("PI05_TEST_PHYSICALRSI_ROOT", str(plan.SOURCE)))


class ClosedDwellLiftPlanTest(unittest.TestCase):
    def test_complete_registry_route_and_110_episode_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"banks": {name: {"adapter_sha256": char * 64}
                for name, char in zip(("spatial", "object", "goal", "long"), "abcd")}}))
            baseline = baseline_builder.build(root / "baseline", manifest, SOURCE)
            output = plan.build(root / "plan", baseline, SOURCE)
            old_registry = json.loads((baseline / "registry.json").read_text())
            registry = json.loads((output / "registry.json").read_text())
            catalog = json.loads((output / "task_catalog.json").read_text())
            rows = {row["key"]: row for row in catalog["tasks"]}
            for key, relative in registry["tasks"].items():
                old = json.loads((baseline / old_registry["tasks"][key]).read_text())
                new = json.loads((output / relative).read_text())
                self.assertEqual(new, dict(old, **({"pi05_control": plan.CONTROL}
                                                   if key == "libero_10/8" else {})))
            self.assertFalse((output / "routes_long8_plugin.json").exists())
            self.assertEqual(registry["metadata"]["evidence"]["current_native_reference"],
                             {"base": "26/50", "hard_init_2_base": "1/10"})
            seen, episodes = set(), 0
            for worker in range(5):
                cases = json.loads((output / f"cases_worker{worker}.json").read_text())["cases"]
                job = json.loads((output / f"job_worker{worker}.json").read_text())
                self.assertEqual(len(cases), 11)
                self.assertEqual([case["init_id"] for case in cases[:10]], [plan.PRIMARY[worker]] * 10)
                self.assertEqual(cases[-1]["init_id"], plan.OTHER[worker])
                seen.update((case["init_id"], case["replicate_id"]) for case in cases)
                self.assertEqual([b["name"].split(f"worker{worker}_", 1)[1]
                                  for b in job["batches"]], ["control", "assist"])
                self.assertEqual({b["routes"] for b in job["batches"]},
                                 {"configs/pi05_harness/routes_base.json"})
                episodes += len(cases) * len(job["batches"])
            self.assertEqual((len(seen), episodes), (55, 110))
            api = backend.import_roborsi(SOURCE)
            tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                     for key, row in rows.items()}
            proposal = api.TaskHarnessRegistry(output / "registry.json", {"pi05"}).materialize(tasks)
            self.assertEqual(proposal["task_config_sha256"], registry["metadata"]["task_config_sha256"])

    def test_worker_bounds_and_create_only(self):
        for worker in (-1, 5, 1.0):
            with self.assertRaises(ValueError):
                plan.worker_cases(worker)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                plan.build(output, Path(directory) / "unused", SOURCE)
        with self.assertRaises(ValueError):
            plan.build(Path("unused"), Path("unused"), SOURCE, policy_id="object")


if __name__ == "__main__":
    unittest.main()
