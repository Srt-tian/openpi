from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import build_pi05_repeat_probe as builder


class BuildRepeatProbeTest(unittest.TestCase):
    def test_exact_frozen_grid_and_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "repeat"
            builder.build(output)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertTrue(metadata["exact_2x14x10_coverage"])
            self.assertEqual((metadata["base_episodes"], metadata["plugin_episodes"],
                              metadata["episodes"]), (140, 140, 280))
            expected_batch_counts = {0: 2, 1: 2, 2: 2, 3: 3, 4: 3}
            for worker, count in expected_batch_counts.items():
                job = json.loads((output / f"job_worker{worker}.json").read_text())
                self.assertEqual(job["schema"], "pi05_harness_worker.v1")
                self.assertEqual(len(job["batches"]), count)
                self.assertTrue(all(set(row) == {"name", "mode", "registry", "routes", "cases"}
                                    and row["mode"] == "legacy" for row in job["batches"]))
            all_cases = []
            for worker in range(5):
                value = json.loads((output / f"cases_worker{worker}_all.json").read_text())
                self.assertEqual(value["schema"], "pi05_harness_cases.v1")
                self.assertTrue(all(set(row) == {"suite", "task_id", "init_id", "replicate_id"}
                                    for row in value["cases"]))
                all_cases.extend(value["cases"])
            keys = {(r["suite"], r["task_id"], r["init_id"], r["replicate_id"])
                    for r in all_cases}
            self.assertEqual(len(all_cases), len(keys))
            self.assertEqual(len(keys), 140)
            self.assertEqual(builder.seed("libero_10", 8, 2, 9),
                             7 + 38 * 50 + 2 + 9 * 1000000007)

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                builder.build(output)


if __name__ == "__main__":
    unittest.main()
