from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

import numpy as np

from openpi.training import plugin_rollouts


class RolloutManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _sample(self, name: str, *, actions: bool) -> tuple[str, str]:
        path = self.root / name
        arrays = {
            "observation/image": np.zeros((12, 16, 3), dtype=np.uint8),
            "observation/wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
            "observation/state": np.arange(8, dtype=np.float32),
        }
        if actions:
            arrays["actions"] = np.arange(70, dtype=np.float32).reshape(10, 7)
        np.savez(path, **arrays)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()

    def _handoff(self, record_id: str = "handoff-1", *, split: str = "train") -> dict:
        sample_path, sample_hash = self._sample(f"{record_id}.npz", actions=True)
        return {
            "kind": "handoff",
            "record_id": record_id,
            "root_episode_id": f"episode-{record_id}",
            "split": split,
            "source_policy_id": "base",
            "target_policy_id": "spatial",
            "policy_bundle_id": "bundle-0001",
            "budget_steps": 30,
            "provenance": "simulator_rollout",
            "rollout_id": "rollout-0001",
            "source_step": 4,
            "sample_path": sample_path,
            "sample_sha256": sample_hash,
            "prompt": "pick up the black bowl",
            "action_schema": "libero_raw_delta7",
            "action_reference": "current_observation",
            "continuation_success": True,
            "continuation_target_source": "successful_rollout",
        }

    def _call(self, record_id: str = "call-1", *, split: str = "validation") -> dict:
        sample_path, sample_hash = self._sample(f"{record_id}.npz", actions=False)
        return {
            "kind": "call",
            "record_id": record_id,
            "root_episode_id": f"episode-{record_id}",
            "split": split,
            "source_policy_id": "base",
            "target_policy_id": "goal",
            "policy_bundle_id": "bundle-0001",
            "budget_steps": 20,
            "provenance": "simulator_rollout",
            "rollout_id": "rollout-0002",
            "source_step": 2,
            "sample_path": sample_path,
            "sample_sha256": sample_hash,
            "prompt": "place the mug on the plate",
            "attempted_policy_id": "goal",
            "success": False,
            "executed_steps": 7,
            "termination": "terminal_failure",
            "execution_scope": "single_policy_until_terminal_or_budget",
            "policy_switches": 0,
        }

    def _manifest(self, records: list[dict]) -> dict:
        return {
            "schema_version": 1,
            "source": "real_rollouts",
            "suite_order": ["spatial", "object", "goal", "long"],
            "feature_schema": "libero_rgb_state_v1",
            "source_base_inventory_sha256": "1" * 64,
            "norm_stats_sha256": "2" * 64,
            "metadata": {
                "excluded_call_counts": {"interrupted": 2, "infra_error": 1},
                "records": records,
            },
        }

    def _write_manifest(self, manifest: dict, name: str = "manifest.json") -> pathlib.Path:
        path = self.root / name
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_load_query_raw_samples_summary_and_deterministic_batch(self) -> None:
        handoff = self._handoff()
        handoff2 = self._handoff("handoff-2")
        call = self._call()
        path = self._write_manifest(self._manifest([handoff, handoff2, call]))

        store = plugin_rollouts.load_rollout_manifest(path)

        handoff_records = store.handoff_records("spatial", "train")
        self.assertEqual([record.record_id for record in handoff_records], ["handoff-1", "handoff-2"])
        self.assertEqual([record.record_id for record in store.call_records("validation")], ["call-1"])
        raw_handoff = store.raw_sample(handoff_records[0])
        self.assertEqual(raw_handoff["prompt"], "pick up the black bowl")
        self.assertEqual(raw_handoff["actions"].shape, (10, 7))
        raw_call = store.raw_sample(store.call_records("validation")[0])
        self.assertNotIn("actions", raw_call)
        labels, observed = store.call_targets(store.call_records("validation"))
        self.assertEqual(labels.dtype, np.float32)
        self.assertEqual(observed.dtype, np.bool_)
        attempted_column = plugin_rollouts.POLICY_IDS.index("goal")
        self.assertEqual(observed.sum(), 1)
        self.assertTrue(observed[0, attempted_column])
        self.assertEqual(labels[0, attempted_column], 0.0)
        first = store.select_batch(handoff_records, 5, seed=9, step=3)
        second = store.select_batch(handoff_records, 5, seed=9, step=3)
        self.assertEqual(first, second)
        summary = store.summary()
        self.assertEqual(summary["record_counts"]["total"], 3)
        self.assertEqual(summary["call_termination_counts"]["terminal_failure"], 1)
        self.assertEqual(summary["excluded_call_counts"], {"interrupted": 2, "infra_error": 1})
        self.assertEqual(summary["policy_bundle_ids"], ["bundle-0001"])
        self.assertRegex(summary["manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_official_eval_and_episode_split_leakage(self) -> None:
        official_eval = self._call(split="official_eval")
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "official_eval"):
            plugin_rollouts.load_rollout_manifest(
                self._write_manifest(self._manifest([official_eval]), "official.json")
            )

        train = self._handoff("train")
        validation = self._call("validation")
        validation["root_episode_id"] = train["root_episode_id"]
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "crosses train/validation"):
            plugin_rollouts.load_rollout_manifest(
                self._write_manifest(self._manifest([train, validation]), "leak.json")
            )

    def test_rejects_duplicate_record_id_across_kinds(self) -> None:
        handoff = self._handoff("same")
        call = self._call("other")
        call["record_id"] = "same"
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "must be unique"):
            plugin_rollouts.load_rollout_manifest(
                self._write_manifest(self._manifest([handoff, call]))
            )

    def test_rejects_path_escape_and_checksum_mismatch(self) -> None:
        outside = pathlib.Path(self.temp_dir.name).parent / "outside-rollout.npz"
        np.savez(
            outside,
            **{
                "observation/image": np.zeros((2, 2, 3), np.uint8),
                "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
                "observation/state": np.zeros(8, np.float32),
                "actions": np.zeros((10, 7), np.float32),
            },
        )
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        escaped = self._handoff("escape")
        escaped["sample_path"] = f"../{outside.name}"
        escaped["sample_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "escapes"):
            plugin_rollouts.load_rollout_manifest(
                self._write_manifest(self._manifest([escaped]), "escape.json")
            )

        mismatched = self._handoff("mismatch")
        mismatched["sample_sha256"] = "f" * 64
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "does not match"):
            plugin_rollouts.load_rollout_manifest(
                self._write_manifest(self._manifest([mismatched]), "mismatch.json")
            )

    def test_rejects_actions_in_call_sample(self) -> None:
        call = self._call()
        sample_path, sample_hash = self._sample("bad-call.npz", actions=True)
        call["sample_path"] = sample_path
        call["sample_sha256"] = sample_hash
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "NPZ keys differ"):
            plugin_rollouts.load_rollout_manifest(self._write_manifest(self._manifest([call])))

    def test_rejects_object_array_and_nonfinite_or_wrong_shape(self) -> None:
        handoff = self._handoff("object")
        object_path = self.root / "object-payload.npz"
        np.savez(
            object_path,
            **{
                "observation/image": np.array([object()], dtype=object),
                "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
                "observation/state": np.zeros(8, np.float32),
                "actions": np.zeros((10, 7), np.float32),
            },
        )
        handoff["sample_path"] = object_path.name
        handoff["sample_sha256"] = hashlib.sha256(object_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "pickled/object"):
            plugin_rollouts.load_rollout_manifest(
                self._write_manifest(self._manifest([handoff]), "object.json")
            )

        invalid = self._handoff("nan")
        invalid_path = self.root / "nan-payload.npz"
        actions = np.zeros((10, 7), np.float32)
        actions[0, 0] = np.nan
        np.savez(
            invalid_path,
            **{
                "observation/image": np.zeros((2, 2, 3), np.uint8),
                "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
                "observation/state": np.zeros(8, np.float32),
                "actions": actions,
            },
        )
        invalid["sample_path"] = invalid_path.name
        invalid["sample_sha256"] = hashlib.sha256(invalid_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "non-finite"):
            plugin_rollouts.load_rollout_manifest(
                self._write_manifest(self._manifest([invalid]), "nan.json")
            )

    def test_enforces_successful_adapter_handoff(self) -> None:
        handoff = self._handoff()
        for field, bad_value in (
            ("continuation_success", False),
            ("continuation_target_source", "ground_truth_future"),
            ("target_policy_id", "base"),
            ("action_schema", "normalized_actions"),
            ("action_reference", "future_observation"),
        ):
            with self.subTest(field=field):
                invalid = copy.deepcopy(handoff)
                invalid[field] = bad_value
                with self.assertRaises(plugin_rollouts.RolloutManifestError):
                    plugin_rollouts.load_rollout_manifest(
                        self._write_manifest(self._manifest([invalid]), f"bad-{field}.json")
                    )

    def test_enforces_call_outcome_logic_and_censoring(self) -> None:
        base = self._call()
        invalid_records = []
        success_mismatch = copy.deepcopy(base)
        success_mismatch["success"] = True
        invalid_records.append(success_mismatch)
        exhausted = copy.deepcopy(base)
        exhausted["termination"] = "budget_exhausted"
        exhausted["executed_steps"] = exhausted["budget_steps"] - 1
        invalid_records.append(exhausted)
        interrupted = copy.deepcopy(base)
        interrupted["termination"] = "interrupted"
        invalid_records.append(interrupted)
        too_long = copy.deepcopy(base)
        too_long["executed_steps"] = too_long["budget_steps"] + 1
        invalid_records.append(too_long)
        for index, record in enumerate(invalid_records):
            with self.subTest(index=index), self.assertRaises(plugin_rollouts.RolloutManifestError):
                plugin_rollouts.load_rollout_manifest(
                    self._write_manifest(self._manifest([record]), f"bad-call-{index}.json")
                )

    def test_call_requires_single_policy_attribution(self) -> None:
        for field, value in (
            ("execution_scope", "fallback_chain"),
            ("policy_switches", 1),
            ("policy_switches", False),
        ):
            with self.subTest(field=field, value=value):
                call = self._call(f"attribution-{field}-{value}")
                call[field] = value
                with self.assertRaises(plugin_rollouts.RolloutManifestError):
                    plugin_rollouts.load_rollout_manifest(
                        self._write_manifest(self._manifest([call]), f"bad-{field}-{value}.json")
                    )

    def test_exclusion_counts_cannot_be_negative(self) -> None:
        manifest = self._manifest([])
        manifest["metadata"]["excluded_call_counts"]["infra_error"] = -1
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "non-negative"):
            plugin_rollouts.load_rollout_manifest(self._write_manifest(manifest))

    def test_explicit_missing_or_malformed_manifest_never_falls_back(self) -> None:
        with self.assertRaises(FileNotFoundError):
            plugin_rollouts.load_rollout_manifest(self.root / "absent.json")
        malformed = self.root / "malformed.json"
        malformed.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(plugin_rollouts.RolloutManifestError, "invalid"):
            plugin_rollouts.load_rollout_manifest(malformed)


if __name__ == "__main__":
    unittest.main()
