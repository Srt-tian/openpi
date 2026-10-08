#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import build_pi05_harness_configs as builder
import pi05_harness_backend as backend


class BuildPi05HarnessConfigsTest(unittest.TestCase):
    @staticmethod
    def manifest(directory: str) -> Path:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps({"banks": {name: {"adapter_sha256": char * 64}
                    for name, char in zip(("spatial", "object", "goal", "long"), "abcd")}}))
        return path

    def test_builds_valid_explicit_libero40_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pi05_harness"
            builder.build(output, self.manifest(directory))
            registry = json.loads((output / "registry.json").read_text())
            catalog = json.loads((output / "task_catalog.json").read_text())
            rows = {row["key"]: row for row in catalog["tasks"]}
            self.assertEqual(len(registry["tasks"]), 40)
            api = backend.import_roborsi(builder.SOURCE)
            tasks = {key: {"instruction": row["instruction"], "max_steps": row["max_steps"]}
                     for key, row in rows.items()}
            proposal = api.TaskHarnessRegistry(output / "registry.json", {"pi05"}).materialize(tasks)
            self.assertEqual(proposal["task_config_sha256"], registry["metadata"]["task_config_sha256"])
            for key, relative in registry["tasks"].items():
                config = json.loads((output / relative).read_text())
                row, stage = rows[key], config["harness"]["stages"][0]
                self.assertEqual(config["parent_sha256"], proposal["parent_sha256_by_task"][key])
                self.assertEqual((stage["skill"], stage["instruction"], stage["max_steps"]),
                                 ("pi05", row["instruction"], builder.CAPS[row["suite"]]))
                self.assertEqual(config["evidence"]["routing_scope"], "task_only_no_init_image_or_state")
            selected = json.loads((output / "routes_selected.json").read_text())
            self.assertEqual(selected["schema"], "pi05_harness_routes.v1")
            self.assertEqual(set(selected["identities"]), {"base", "spatial", "object", "goal", "long"})
            self.assertIsNone(selected["identities"]["base"]["adapter_sha256"])
            self.assertEqual(Counter(selected["tasks"].values()),
                             Counter({"base": 35, "spatial": 2, "object": 1, "long": 2}))
            self.assertEqual({k for k, v in selected["tasks"].items() if v != "base"},
                             set(builder.SELECTED))

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                builder.build(output, self.manifest(directory))


if __name__ == "__main__":
    unittest.main()
