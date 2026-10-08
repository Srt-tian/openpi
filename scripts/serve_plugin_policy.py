"""Serve one explicitly selected PI0.5 LIBERO base or suite policy."""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from openpi.policies import plugin_policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass(frozen=True)
class Args:
    base_checkpoint: str
    plugin_checkpoint: str
    policy_id: str
    default_prompt: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    num_fsdp_devices: int | None = None
    allow_verified_base_relocation: bool = False


def create_policy(args: Args):
    if args.policy_id == "base":
        policy = plugin_policy.create_original_base_policy(
            args.base_checkpoint,
            args.plugin_checkpoint,
            default_prompt=args.default_prompt,
            allow_verified_base_relocation=args.allow_verified_base_relocation,
        )
    else:
        policy = plugin_policy.create_plugin_policy(
            args.base_checkpoint,
            args.plugin_checkpoint,
            args.policy_id,
            default_prompt=args.default_prompt,
            num_fsdp_devices=args.num_fsdp_devices,
            allow_verified_base_relocation=args.allow_verified_base_relocation,
        )
    return plugin_policy.FixedPolicyService(policy, args.policy_id)


def main(args: Args) -> None:
    policy = create_policy(args)
    hostname = socket.gethostname()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info(
        "Serving fixed PI05 policy_id=%s on host=%s port=%d",
        args.policy_id,
        f"{hostname}/{args.host}",
        args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
