"""CPU-only regressions for the four-plugin entrypoint."""

import itertools
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from train_four_plugins import (
    DeterministicBatchSampler,
    execution_commit_provenance,
    numpy_collate,
    parse_args,
    resolve_loss_status,
    sample_summary,
    validate_call_supervision,
    validate_loss_args,
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
    def test_execution_commit_provenance_is_strict_and_explicit(self):
        with mock.patch.dict(os.environ, {"PI05_ALLOW_UNPUBLISHED_COMMIT": "0"}):
            self.assertEqual(
                execution_commit_provenance(),
                {
                    "execution_commit_policy": "canonical_upstream_required",
                    "canonical_publication_verified": True,
                },
            )
        with mock.patch.dict(os.environ, {"PI05_ALLOW_UNPUBLISHED_COMMIT": "1"}):
            self.assertEqual(
                execution_commit_provenance(),
                {
                    "execution_commit_policy": "user_approved_local_commit",
                    "canonical_publication_verified": False,
                },
            )
        for invalid in ("", "true", "yes", "2", "01"):
            with self.subTest(invalid=invalid), mock.patch.dict(
                os.environ, {"PI05_ALLOW_UNPUBLISHED_COMMIT": invalid}
            ), self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                execution_commit_provenance()

    def test_explicit_fm_only_zeros_auxiliary_weights_and_records_selection(self):
        args = parse_args(["--fm-only", "--handoff-weight", "0.7", "--call-weight", "0.9"])
        validate_loss_args(args)
        self.assertEqual(args.handoff_weight, 0.0)
        self.assertEqual(args.call_weight, 0.0)
        status, handoffs, calls = resolve_loss_status(args, None)
        self.assertEqual(status["mode"], "stage_a_fm_only")
        self.assertEqual(status["stage_selection"], "user_explicit_stage_a_fm_only")
        self.assertEqual(status["weights"], {"handoff": 0.0, "call": 0.0})
        self.assertTrue(all(
            item["disabled_reason"] == "user_explicit_fm_only"
            for item in status["handoff"].values()
        ))
        self.assertEqual(status["call"]["disabled_reason"], "user_explicit_fm_only")
        self.assertEqual(handoffs, {suite: [] for suite in ("spatial", "object", "goal", "long")})
        self.assertEqual(calls, [])

    def test_fm_only_rejects_every_joint_loss_input_mode(self):
        with self.assertRaisesRegex(ValueError, "rollout-manifest"):
            validate_loss_args(parse_args(["--fm-only", "--rollout-manifest", "labels.json"]))
        with self.assertRaisesRegex(ValueError, "require-joint-losses"):
            validate_loss_args(parse_args(["--fm-only", "--require-joint-losses"]))

    def test_generic_runner_without_new_flag_remains_backward_compatible(self):
        args = parse_args([])
        validate_loss_args(args)
        status, _, _ = resolve_loss_status(args, None)
        self.assertEqual(status["mode"], "stage_a_fm_only")
        self.assertEqual(status["stage_selection"], "resolved_from_joint_loss_inputs")
        self.assertEqual(status["weights"], {"handoff": 0.3, "call": 0.1})

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


class LauncherArgumentTests(unittest.TestCase):
    launcher = Path(__file__).with_name("launch_pi05_libero_test.sh")
    helper = """
source "$1"
shift
classify_launch_args "$@"
validate_launch_mode
build_runner_argv "$@"
printf '<%s>\\n' "${PI05_RUNNER_ARGS[@]}"
"""

    def run_helper(self, *args):
        return subprocess.run(
            ["bash", "-c", self.helper, "stage-a-test", str(self.launcher), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_launcher_preserves_explicit_fm_only_argv_exactly(self):
        result = self.run_helper("--fm-only", "--output-dir", "/tmp/run,with,commas", "--batch-size=32")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["<--fm-only>", "<--output-dir>", "</tmp/run,with,commas>", "<--batch-size=32>"],
        )

    def test_launcher_joint_mode_preserves_argv_and_appends_strict_gate(self):
        result = self.run_helper(
            "--output-dir=/tmp/run,with,commas", "--rollout-manifest", "/tmp/labels.json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "<--output-dir=/tmp/run,with,commas>",
                "<--rollout-manifest>",
                "</tmp/labels.json>",
                "<--require-joint-losses>",
            ],
        )

    def test_launcher_rejects_fm_only_conflicts_and_implicit_fallback(self):
        for args, message in (
            (("--fm-only", "--rollout-manifest=x.json"), "rollout-manifest"),
            (("--fm-only", "--require-joint-losses"), "require-joint-losses"),
            (("--batch-size", "32"), "requires --rollout-manifest"),
        ):
            with self.subTest(args=args):
                result = self.run_helper(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)


class LauncherGitGateTests(unittest.TestCase):
    launcher = Path(__file__).with_name("launch_pi05_libero_test.sh")
    helper = """
source "$1"
verify_git_checkout "$2" "$3"
printf '%s|%s\n' "${PI05_EXECUTION_COMMIT_POLICY}" "${PI05_CANONICAL_PUBLICATION_VERIFIED}"
"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Stage A Test")
        self.git("config", "user.email", "stage-a-test@example.invalid")
        (self.repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "-qm", "fixture commit")
        self.head = self.git("rev-parse", "HEAD").stdout.strip()

    def tearDown(self):
        self.temp_dir.cleanup()

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            text=True,
            capture_output=True,
            check=True,
        )

    def run_gate(self, override, expected=None):
        env = os.environ.copy()
        if override is None:
            env.pop("PI05_ALLOW_UNPUBLISHED_COMMIT", None)
        else:
            env["PI05_ALLOW_UNPUBLISHED_COMMIT"] = override
        return subprocess.run(
            [
                "bash", "-c", self.helper, "git-gate-test", str(self.launcher),
                str(self.repo), expected or self.head,
            ],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )

    def test_no_upstream_is_rejected_by_default_but_explicit_one_allows_it(self):
        default = self.run_gate(None)
        self.assertEqual(default.returncode, 2)
        self.assertIn("no upstream", default.stderr)

        approved = self.run_gate("1")
        self.assertEqual(approved.returncode, 0, approved.stderr)
        self.assertEqual(approved.stdout.strip(), "user_approved_local_commit|false")

    def test_override_accepts_only_literal_zero_or_one(self):
        for invalid in ("", "true", "yes", "2", "01"):
            with self.subTest(invalid=invalid):
                result = self.run_gate(invalid)
                self.assertEqual(result.returncode, 2)
                self.assertIn("exactly 0 or 1", result.stderr)

    def test_default_requires_upstream_at_the_exact_expected_commit(self):
        self.git("branch", "published", self.head)
        self.git("branch", "--set-upstream-to=published")
        canonical = self.run_gate("0")
        self.assertEqual(canonical.returncode, 0, canonical.stderr)
        self.assertEqual(canonical.stdout.strip(), "canonical_upstream_required|true")

        (self.repo / "tracked.txt").write_text("new commit\n", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "-qm", "local commit beyond upstream")
        new_head = self.git("rev-parse", "HEAD").stdout.strip()
        mismatched = self.run_gate("0", expected=new_head)
        self.assertEqual(mismatched.returncode, 2)
        self.assertIn("upstream commit does not match", mismatched.stderr)

    def test_wrong_head_and_dirty_tree_are_rejected_even_when_approved(self):
        wrong_head = self.run_gate("1", expected="0" * 40)
        self.assertEqual(wrong_head.returncode, 2)
        self.assertIn("HEAD does not match", wrong_head.stderr)

        (self.repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        dirty = self.run_gate("1")
        self.assertEqual(dirty.returncode, 2)
        self.assertIn("not clean", dirty.stderr)


if __name__ == "__main__":
    unittest.main()
