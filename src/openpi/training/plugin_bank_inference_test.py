import copy
import json
import tempfile
from pathlib import Path
import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import plugin_bank


class _TinyModel(nnx.Module):
    def __init__(self):
        self.base = nnx.Param(jnp.array([[2.0], [-1.0]]))
        self.expert_lora_a = nnx.Param(jnp.array([[0.2], [0.3]]))
        self.expert_lora_b = nnx.Param(jnp.zeros((1, 1)))


def _is_lora(path, value):
    del value
    return any("lora" in str(part) for part in path)


class PluginBankInferenceTest(unittest.TestCase):
    def _saved_bank(self, directory):
        _, _, adapter = plugin_bank.partition_model(_TinyModel(), _is_lora)
        adapters = {suite: copy.deepcopy(adapter) for suite in plugin_bank.SUITES}
        tx = optax.adam(1e-2)
        path = plugin_bank.save_bank(
            Path(directory) / "bank",
            adapters,
            plugin_bank.initialize_optimizer_states(tx, adapters),
            {suite: 4 for suite in plugin_bank.SUITES},
            base_checkpoint_path="/checkpoint/params",
            norm_stats_hash="norm-hash",
            base_manifest_hash="source-manifest-hash",
        )
        return path, adapter

    def test_load_one_adapter_does_not_read_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path, adapter = self._saved_bank(directory)
            (path / "bank_00.optimizer.msgpack").write_bytes(b"corrupt optimizer")
            restored, manifest = plugin_bank.load_adapter(
                path,
                "spatial",
                adapter,
                expected_base_checkpoint_path="/checkpoint/params",
                expected_norm_stats_hash="norm-hash",
                expected_base_manifest_hash="source-manifest-hash",
            )
            jax.tree.map(
                np.testing.assert_array_equal,
                restored.to_pure_dict(),
                adapter.to_pure_dict(),
            )
            self.assertEqual(manifest["global_update_count"], 16)

    def test_adapter_and_binding_checks_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, adapter = self._saved_bank(directory)
            with self.assertRaisesRegex(ValueError, "norm stats hash mismatch"):
                plugin_bank.load_adapter(
                    path, "spatial", adapter, expected_norm_stats_hash="wrong"
                )
            with (path / "bank_03.adapter.msgpack").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "adapter checksum mismatch for long"):
                plugin_bank.load_adapter(path, "spatial", adapter)

    def test_inference_manifest_validation_does_not_require_optimizer_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._saved_bank(directory)
            for item in json.loads((path / "manifest.json").read_text())["banks"].values():
                (path / item["optimizer_file"]).unlink()
            manifest = plugin_bank.verify_adapter_bank(
                path,
                expected_base_checkpoint_path="/checkpoint/params",
                expected_norm_stats_hash="norm-hash",
                expected_base_manifest_hash="source-manifest-hash",
            )
            self.assertEqual(manifest["global_update_count"], 16)


if __name__ == "__main__":
    unittest.main()
