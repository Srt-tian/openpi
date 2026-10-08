#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import build_pi05_harness_configs as baseline_builder
import build_pi05_early_strong_response_probe_plan as candidate_builder
import build_pi05_goal3_early_probe_validation_plan as target


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Goal3EarlyProbeValidationPlanTest(unittest.TestCase):
    def test_exact_candidate_and_100_pair_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"banks": {
                name: {"adapter_sha256": char * 64}
                for name, char in zip(("spatial", "object", "goal", "long"), "abcd")}}))
            baseline = baseline_builder.build(root / "baseline", manifest)
            validated = candidate_builder.build(root / "validated", baseline)
            output = target.build(root / "validation", validated)

            source_registry = json.loads((validated / "registry.json").read_text())
            copied_registry = json.loads((output / "registry.json").read_text())
            self.assertEqual(digest(validated / "registry.json"), digest(output / "registry.json"))
            self.assertEqual(digest(validated / "task_catalog.json"),
                             digest(output / "task_catalog.json"))
            self.assertEqual(source_registry, copied_registry)
            for relative in source_registry["tasks"].values():
                self.assertEqual(digest(validated / relative), digest(output / relative))

            plan = json.loads((output / "case_plan.json").read_text())
            self.assertEqual(set(plan), {"schema", "task", "cases"})
            self.assertEqual(plan["schema"], "pi05_response_probe_case_plan.v1")
            self.assertEqual(len(plan["cases"]), 100)
            keys = {(row["init_id"], row["replicate_id"]) for row in plan["cases"]}
            self.assertEqual(keys, set(target.ALL_100))
            self.assertEqual(sum(row["cohort"] == "known_19_reproduction"
                                 for row in plan["cases"]), 19)
            self.assertEqual(sum(row["cohort"] == "noise_expansion_81"
                                 for row in plan["cases"]), 81)

            partition = []
            for index in range(4):
                cases_doc = json.loads((output / f"cases_worker{index}.json").read_text())
                self.assertEqual(set(cases_doc), {"schema", "cases"})
                self.assertEqual(len(cases_doc["cases"]), 25)
                self.assertTrue(all(set(row) == {
                    "suite", "task_id", "init_id", "replicate_id"}
                    for row in cases_doc["cases"]))
                partition.extend((row["init_id"], row["replicate_id"])
                                 for row in cases_doc["cases"])
                job = json.loads((output / f"job_worker{index}.json").read_text())
                self.assertEqual([row["name"] for row in job["batches"]],
                                 [f"worker{index}_control", f"worker{index}_probe"])
                self.assertEqual(job["batches"][0]["cases"], job["batches"][1]["cases"])
                self.assertTrue(all(row["routes"].endswith("routes_base.json")
                                    for row in job["batches"]))
            self.assertEqual(len(partition), 100)
            self.assertEqual(set(partition), set(target.ALL_100))

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                target.build(existing, Path(directory) / "unused")


if __name__ == "__main__":
    unittest.main()
