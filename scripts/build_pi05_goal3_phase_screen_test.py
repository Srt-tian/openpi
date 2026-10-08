#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import build_pi05_goal3_phase_screen as screen
import build_pi05_harness_configs as baseline_builder
import pi05_harness_backend as backend


SOURCE = Path(os.environ.get("PI05_TEST_PHYSICALRSI_ROOT", str(screen.SOURCE)))


class Goal3PhaseScreenTest(unittest.TestCase):
    def test_three_complete_registries_and_five_worker_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"banks": {name: {"adapter_sha256": char * 64}
                for name, char in zip(("spatial", "object", "goal", "long"), "abcd")}}))
            baseline = baseline_builder.build(root / "baseline", manifest, SOURCE)
            output = screen.build(root / "screen", baseline, SOURCE)
            base_registry = json.loads((baseline / "registry.json").read_text())
            catalog = json.loads((baseline / "task_catalog.json").read_text())
            rows = {row["key"]: row for row in catalog["tasks"]}
            api = backend.import_roborsi(SOURCE)
            tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                     for key, row in rows.items()}
            for variant, stages in screen.VARIANTS.items():
                registry = json.loads((output / variant / "registry.json").read_text())
                changed = []
                for key, relative in registry["tasks"].items():
                    old = json.loads((baseline / base_registry["tasks"][key]).read_text())
                    new = json.loads((output / variant / relative).read_text())
                    self.assertNotIn("pi05_control", new)
                    if key == "libero_goal/3":
                        changed.append(key)
                        self.assertEqual(new["parent_sha256"], old["parent_sha256"])
                        self.assertEqual([(s["instruction"], s["max_steps"], s["on_timeout"])
                                          for s in new["harness"]["stages"]], stages)
                        self.assertEqual(new["harness"]["remember"], [])
                    else:
                        self.assertEqual(new, old)
                self.assertEqual(changed, ["libero_goal/3"])
                proposal = api.TaskHarnessRegistry(
                    output / variant / "registry.json", {"pi05"}).materialize(tasks)
                self.assertEqual(proposal["task_config_sha256"],
                                 registry["metadata"]["task_config_sha256"])
            pairs = 0
            expected = ["control", "probe", "prefix100", "prefix150", "sequential"]
            for worker in range(5):
                job = json.loads((output / f"job_worker{worker}.json").read_text())
                self.assertEqual([b["name"].split(f"worker{worker}_", 1)[1]
                                  for b in job["batches"]], expected)
                self.assertTrue(all(b["mode"] == "harness" and
                                    b["routes"].endswith("routes_base.json") for b in job["batches"]))
                cases = json.loads((screen.PATCH / job["batches"][0]["cases"]).read_text())["cases"]
                pairs += len(cases)
            self.assertEqual(pairs, 19)
            self.assertEqual(pairs * 5, 95)

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                screen.build(output, Path(directory) / "unused", SOURCE)

    def test_early_prefix_variants_boundaries_and_plan(self):
        for invalid in ([], [0], [300], [25, 25], [1.5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                screen.prefix_variants(invalid)
        variants = screen.prefix_variants([25, 50, 75, 100])
        self.assertEqual(variants["prefix25"], [
            (screen.PREFIX, 25, "next"), (screen.ORIGINAL, 275, "abort")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"banks": {name: {"adapter_sha256": char * 64}
                for name, char in zip(("spatial", "object", "goal", "long"), "abcd")}}))
            baseline = baseline_builder.build(root / "baseline", manifest, SOURCE)
            output = screen.build(root / "early", baseline, SOURCE, variants=variants,
                                  namespace="pi05_goal3_early_prefix", include_probe=False)
            episodes = 0
            for worker in range(5):
                job = json.loads((output / f"job_worker{worker}.json").read_text())
                self.assertEqual([b["name"].split(f"worker{worker}_", 1)[1]
                                  for b in job["batches"]],
                                 ["control", "prefix25", "prefix50", "prefix75", "prefix100"])
                self.assertTrue(all("configs/pi05_goal3_early_prefix/" in b["registry"]
                                    for b in job["batches"][1:]))
                cases = json.loads((screen.PATCH / job["batches"][0]["cases"]).read_text())["cases"]
                episodes += len(cases) * len(job["batches"])
            self.assertEqual(episodes, 95)
            config = json.loads((output / "prefix25/tasks/libero_goal/3.json").read_text())
            self.assertEqual(config["evidence"]["prior_phase_screen"],
                             {"hard_cases": "8/10", "unconditional_regressions": 3})
            self.assertIn("zero_net_gain", config["evidence"]["later_instruction_lease"])


if __name__ == "__main__":
    unittest.main()
