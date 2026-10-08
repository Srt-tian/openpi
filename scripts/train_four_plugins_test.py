"""CPU-only regressions for the four-plugin entrypoint."""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
from train_four_plugins import (
    DeterministicBatchSampler,
    numpy_collate,
    resolve_loss_status,
    sample_summary,
    validate_call_supervision,
    validation_loss_records,
)


class _FakeRolloutStore:
    def __init__(self):
        self.handoffs = {"spatial": [object()], "object": [], "goal": [object()], "long": [object()]}
        # The store contract has already excluded two censored records; only these
        # three real terminal outcomes are eligible for the runner.
        self.calls = [
            SimpleNamespace(attempted_policy_id="base"),
            SimpleNamespace(attempted_policy_id="spatial"),
            SimpleNamespace(attempted_policy_id="goal"),
        ]
        self.queries = []

    def handoff_records(self, target_policy_id, split):
        self.queries.append(("handoff", target_policy_id, split))
        return self.handoffs[target_policy_id] if split == "train" else []

    def call_records(self, split):
        self.queries.append(("call", split))
        return self.calls if split == "train" else []


class RunnerTests(unittest.TestCase):
    def test_collates_numpy_boolean_masks(self):
        sample = {"image_mask": {"camera": np.True_}, "state": np.zeros(8, np.float32)}
        batch = numpy_collate([sample, sample])
        self.assertEqual(batch["image_mask"]["camera"].dtype, np.bool_)
        self.assertEqual(batch["image_mask"]["camera"].tolist(), [True, True])
        self.assertEqual(batch["state"].shape, (2, 8))

    def test_sample_summary_accepts_array_root(self):
        self.assertEqual(
            sample_summary(np.zeros((2, 5), dtype=np.float32)),
            {"shape": [2, 5], "dtype": "float32"},
        )

    def test_resume_matches_across_epoch_boundary(self):
        full = DeterministicBatchSampler(19, 4, 42)
        resumed = DeterministicBatchSampler(19, 4, 42, start_batch=7)
        expected = list(itertools.islice(full, 7, 12))
        self.assertEqual(expected, list(itertools.islice(resumed, 5)))

    def test_epoch_has_no_duplicate_indices(self):
        batches = list(itertools.islice(DeterministicBatchSampler(19, 4, 42), 4))
        indices = list(itertools.chain.from_iterable(batches))
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(set(indices)), 16)
        self.assertTrue(all(0 <= i < 19 for i in indices))

    def test_joint_loss_gate_is_per_target_and_counts_only_eligible_calls(self):
        args = SimpleNamespace(handoff_weight=0.3, call_weight=0.1, require_joint_losses=False)
        status, _, calls = resolve_loss_status(args, _FakeRolloutStore())
        self.assertTrue(status["handoff"]["spatial"]["active"])
        self.assertFalse(status["handoff"]["object"]["active"])
        self.assertEqual(
            status["handoff"]["object"]["disabled_reason"],
            "no_real_train_handoff_labels_for_target",
        )
        self.assertEqual(status["call"]["count"], 3)
        self.assertEqual(status["call"]["counts_by_policy"]["base"], 1)
        self.assertEqual(status["call"]["counts_by_policy"]["object"], 0)
        self.assertFalse(status["call"]["full_policy_coverage"])
        self.assertEqual(status["mode"], "partial_experimental_joint_losses")
        self.assertEqual(len(calls), 3)

        args.require_joint_losses = True
        with self.assertRaisesRegex(RuntimeError, "handoff:object"):
            resolve_loss_status(args, _FakeRolloutStore())

        full_store = _FakeRolloutStore()
        full_store.handoffs["object"] = [object()]
        with self.assertRaisesRegex(RuntimeError, "call:object"):
            resolve_loss_status(args, full_store)
        full_store.calls = [SimpleNamespace(attempted_policy_id=policy) for policy in
                            ("base", "spatial", "object", "goal", "long")]
        args.require_joint_losses = False
        full_status, _, _ = resolve_loss_status(args, full_store)
        self.assertEqual(full_status["mode"], "full_experimental_joint_losses")

    def test_call_targets_mask_exactly_one_attempted_policy(self):
        targets = np.asarray([[1, 0, 0, 0, 0], [1, 1, 1, 0, 1]], np.float32)
        observed = np.asarray([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], np.bool_)
        checked_targets, checked_mask = validate_call_supervision(targets, observed, batch_size=2)
        self.assertEqual(checked_targets[checked_mask].tolist(), [1.0, 0.0])
        # Values for all unattempted policies are irrelevant; the mask, not a
        # synthetic zero target, defines supervision.
        self.assertEqual(int(checked_mask.sum()), 2)

        bad_mask = observed.copy()
        bad_mask[0, 1] = True
        with self.assertRaisesRegex(ValueError, "exactly its one attempted policy"):
            validate_call_supervision(targets, bad_mask, batch_size=2)

    def test_validation_records_query_each_real_label_branch(self):
        store = _FakeRolloutStore()
        handoff, calls = validation_loss_records(store)
        self.assertEqual(handoff, {suite: [] for suite in ("spatial", "object", "goal", "long")})
        self.assertEqual(calls, [])
        self.assertEqual(
            store.queries,
            [
                ("handoff", "spatial", "validation"),
                ("handoff", "object", "validation"),
                ("handoff", "goal", "validation"),
                ("handoff", "long", "validation"),
                ("call", "validation"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
