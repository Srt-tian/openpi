#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import build_pi05_harness_configs as baseline_builder
import build_pi05_strong_response_probe_plan as probe_builder
import pi05_harness_backend as backend


class StrongResponseProbePlanTest(unittest.TestCase):
    def test_builds_task_wide_control_and_exact_19_pair_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"banks": {name: {"adapter_sha256": char * 64}
                for name, char in zip(("spatial", "object", "goal", "long"), "abcd")}}))
            baseline = baseline_builder.build(root / "baseline", manifest)
            output = probe_builder.build(root / "probe", baseline)
            old_registry = json.loads((baseline / "registry.json").read_text())
            registry = json.loads((output / "registry.json").read_text())
            catalog = json.loads((output / "task_catalog.json").read_text())
            rows = {row["key"]: row for row in catalog["tasks"]}
            controls, pair_count, observed = [], 0, set()
            for key, relative in registry["tasks"].items():
                old = json.loads((baseline / old_registry["tasks"][key]).read_text())
                new = json.loads((output / relative).read_text())
                if "pi05_control" in new:
                    controls.append(key)
                    self.assertEqual(new["pi05_control"], probe_builder.CONTROL)
                    self.assertEqual(new["parent_sha256"], old["parent_sha256"])
                    self.assertEqual(new["harness"], old["harness"])
                    self.assertIn("forbidden_oracles", new["evidence"])
                else:
                    self.assertEqual(new, old)
            self.assertEqual(controls, ["libero_goal/3"])
            expected_inits = {0: [4] * 5, 1: [4] * 5,
                              2: [0, 1, 2, 3, 5], 3: [6, 7, 8, 9]}
            for index in range(4):
                cases = json.loads((output / f"cases_worker{index}.json").read_text())["cases"]
                job = json.loads((output / f"job_worker{index}.json").read_text())
                pair_count += len(cases)
                observed.update((c["init_id"], c["replicate_id"]) for c in cases)
                self.assertEqual([b["name"] for b in job["batches"]],
                                 [f"worker{index}_control", f"worker{index}_probe"])
                self.assertTrue(all(b["routes"].endswith("routes_base.json") for b in job["batches"]))
                self.assertEqual([case["init_id"] for case in cases], expected_inits[index])
            self.assertEqual(pair_count, 19)
            self.assertEqual(len(observed), 19)
            self.assertEqual({rep for init, rep in observed if init == 4}, set(range(10)))
            self.assertIsNone(json.loads((baseline / "routes_base.json").read_text())
                              ["identities"]["base"]["adapter_sha256"])
            api = backend.import_roborsi(probe_builder.SOURCE)
            tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                     for key, row in rows.items()}
            proposal = api.TaskHarnessRegistry(output / "registry.json", {"pi05"}).materialize(tasks)
            self.assertEqual(proposal["task_config_sha256"], registry["metadata"]["task_config_sha256"])

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                probe_builder.build(existing, Path(directory) / "unused")


if __name__ == "__main__":
    unittest.main()
