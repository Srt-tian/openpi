import unittest
from unittest import mock

from scripts import serve_plugin_policy


class ServePluginPolicyTest(unittest.TestCase):
    def test_service_passes_one_fixed_explicit_policy_id(self):
        args = serve_plugin_policy.Args(
            base_checkpoint="/base",
            plugin_checkpoint="/plugins/step_00004000",
            policy_id="object",
            default_prompt="pick up the object",
            host="127.0.0.1",
            port=8000,
            num_fsdp_devices=1,
            allow_verified_base_relocation=True,
        )
        sentinel = mock.Mock()
        sentinel.metadata = {"checkpoint_sha256": "a" * 64}
        with mock.patch.object(
            serve_plugin_policy.plugin_policy,
            "create_plugin_policy",
            return_value=sentinel,
        ) as create:
            wrapped = serve_plugin_policy.create_policy(args)
            self.assertIs(wrapped._policy, sentinel)
        create.assert_called_once_with(
            "/base",
            "/plugins/step_00004000",
            "object",
            default_prompt="pick up the object",
            num_fsdp_devices=1,
            allow_verified_base_relocation=True,
        )

    def test_default_bind_is_loopback(self):
        self.assertEqual(serve_plugin_policy.Args("/base", "/plugins", "base").host, "127.0.0.1")

    def test_invalid_service_policy_ids_fail_closed_before_model_loading(self):
        for policy_id in ("auto", "call", "grasp"):
            with self.assertRaisesRegex(ValueError, "automatic routing is not available"):
                serve_plugin_policy.plugin_policy._require_policy_id(policy_id)


if __name__ == "__main__":
    unittest.main()
