"""Compare zero-B and original pi05_libero inference on identical finite fixtures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.policies import plugin_policy, policy_config
from openpi.training import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--plugins", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate = plugin_policy.create_plugin_policy(
        args.base, args.plugins, "base", allow_verified_base_relocation=True)
    reference = policy_config.create_trained_policy(config.get_config("pi05_libero"), args.base)
    rng = np.random.default_rng(12345)
    observations, noises = [], []
    for prompt in ("pick up the black bowl", "pick up the ketchup and place it in the basket", "put both moka pots on the stove"):
        observations.append({
            "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.array([0.45, 0.0, 0.25, 3.0, 0.0, 0.0, 0.02, -0.02]),
            "prompt": prompt,
        })
        noises.append(rng.standard_normal((10, 32), dtype=np.float32))
    result = plugin_policy.measure_same_noise_base_alignment(candidate, reference, observations, noises)
    result.update({"fixture_kind": "synthetic images/state; numerical regression only, not policy competence",
        "base_path": args.base, "plugin_path": args.plugins, "fixture_seed": 12345,
        "noise_sha256": hashlib.sha256(b"".join(x.tobytes() for x in noises)).hexdigest(),
        "scope": "3 same-noise forward passes; not an exhaustive equivalence proof or official evaluation"})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["all_close"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
