#!/usr/bin/env python3
"""Create and validate the explicit PI0.5 LIBERO-40 harness configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pi05_harness_backend as backend


SOURCE = Path("/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007")
CATALOG = SOURCE / "configs/task_catalog.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "configs/pi05_harness"
CAPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SELECTED = {
    "libero_spatial/3": "spatial", "libero_spatial/7": "spatial",
    "libero_object/4": "object", "libero_10/8": "long", "libero_10/9": "long",
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def identities(manifest_path: Path) -> dict:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    banks = manifest.get("banks")
    if not isinstance(banks, dict) or set(banks) != {"spatial", "object", "goal", "long"}:
        raise ValueError("plugin manifest must contain exactly spatial/object/goal/long banks")
    checksum = hashlib.sha256(raw).hexdigest()
    result = {"base": {"checkpoint_sha256": checksum,
                       "base_graph": "original_pi05_libero", "adapter_sha256": None}}
    for policy in sorted(banks):
        adapter = banks[policy].get("adapter_sha256")
        if not isinstance(adapter, str) or len(adapter) != 64:
            raise ValueError(f"invalid adapter_sha256 for {policy}")
        result[policy] = {"checkpoint_sha256": checksum, "base_graph": "pi05_lora",
                          "adapter_sha256": adapter}
    return result


def route(name: str, tasks: dict[str, str], basis: str, policy_identities: dict) -> dict:
    return {
        "schema": "pi05_harness_routes.v1", "name": name, "tasks": tasks,
        "identities": policy_identities,
        "selection_basis": basis, "validation_status": "unvalidated",
        "routing_inputs": ["suite", "task_index"],
    }


def build(output: Path, plugin_manifest: Path, source: Path = SOURCE) -> Path:
    output, source = output.resolve(), source.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    catalog = json.loads((source / "configs/task_catalog.json").read_text())
    policy_identities = identities(plugin_manifest.resolve())
    rows = catalog.get("tasks")
    expected = {f"{suite}/{i}" for suite in CAPS for i in range(10)}
    if not isinstance(rows, list) or len(rows) != 40 or {r.get("key") for r in rows} != expected:
        raise ValueError("task catalog must contain exactly the known LIBERO-40 tasks")
    if any(r["max_steps"] != CAPS[r["suite"]] or r["task_index"] not in range(10) for r in rows):
        raise ValueError("task catalog does not use official suite caps and task indices")

    api = backend.import_roborsi(source)
    from roborsi.self_harness.core import digest
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pi05_harness.", dir=parent))
    try:
        tasks, filenames, config_digests = {}, {}, {}
        for row in sorted(rows, key=lambda r: r["key"]):
            key, suite, index = row["key"], row["suite"], row["task_index"]
            harness = api.initial_harness(row["instruction"], row["max_steps"], "pi05")
            config = {
                "schema": 1, "task": key, "parent_sha256": digest(harness),
                "harness": harness,
                "reasoning": "Explicit unchanged PI0.5 single-stage baseline at the official task cap.",
                "evidence": {"phase": "phase1dev400", "status": "unvalidated",
                             "routing_scope": "task_only_no_init_image_or_state"},
                "proposal": {"candidate": SELECTED.get(key, "base"),
                             "rationale": "Developer-set per-task candidate; not an oracle or per-init rule."},
                "status": "unvalidated",
            }
            relative = f"tasks/{suite}/{index}.json"
            dump(temporary / relative, config)
            filenames[key] = relative
            tasks[key] = {"instruction": row["instruction"], "max_steps": row["max_steps"]}
            config_digests[key] = digest(config)
        registry = {"schema": 1, "name": "pi05-libero40-explicit-baseline", "default_skill": "pi05",
                    "tasks": filenames, "metadata": {"task_config_sha256": config_digests}}
        dump(temporary / "registry.json", registry)
        dump(temporary / "task_catalog.json", catalog)
        keys = sorted(expected)
        dump(temporary / "routes_base.json", route("pi05-base", {k: "base" for k in keys},
                                                   "fixed_baseline", policy_identities))
        plugins = {k: ({"libero_spatial": "spatial", "libero_object": "object",
                        "libero_goal": "goal", "libero_10": "long"}[k.split("/")[0]]) for k in keys}
        dump(temporary / "routes_plugins.json", route(
            "pi05-plugin-candidates", plugins, "phase1dev400", policy_identities))
        dump(temporary / "routes_selected.json", route(
            "pi05-selected", {k: SELECTED.get(k, "base") for k in keys},
            "phase1dev400", policy_identities))
        proposal = api.TaskHarnessRegistry(temporary / "registry.json", {"pi05"}).materialize(tasks)
        if proposal["task_config_sha256"] != config_digests or len(proposal["harnesses"]) != 40:
            raise AssertionError("registry validation did not reproduce all task config digests")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--plugin-manifest", type=Path, required=True)
    arguments = parser.parse_args()
    print(build(arguments.output_root, arguments.plugin_manifest, arguments.source_root))
