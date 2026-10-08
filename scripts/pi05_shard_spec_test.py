import json
from pathlib import Path
import tempfile
import unittest

import pi05_shard_spec as spec


def reused_row(case, arm):
    return {
        **case,
        "case_id": case["id"],
        "arm": arm,
        "policy_id": "base" if arm == "base" else spec.PLUGIN_IDS[case["suite"]],
        "success": False,
        "status": "failure",
    }


class ShardSpecTest(unittest.TestCase):
    def test_round_robin_is_exact_7_7_6_6_6_and_disjoint(self):
        all_ids = set()
        for worker_id, task_count in enumerate(spec.WORKER_TASK_COUNTS):
            cases = spec.worker_cases(worker_id)
            self.assertEqual(len(cases), task_count * 10)
            ids = {case["id"] for case in cases}
            self.assertFalse(ids & all_ids)
            all_ids |= ids
        self.assertEqual(len(all_ids), 320)
        self.assertFalse(all_ids & set(spec.reused_case_ids()))

    def test_case_rederives_identity_and_rejects_control_injection(self):
        case = spec.case_for("libero_goal", 7, 9)
        self.assertEqual(spec.validate_case(case), case)
        with self.assertRaises(ValueError):
            spec.validate_case({**case, "arm": "base"})
        with self.assertRaises(ValueError):
            spec.validate_case({**case, "policy_seed": case["policy_seed"] + 1})

    def test_manifest_requires_complete_prior_outcomes_not_success_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir = root / "pilot" / "episodes"
            episode_dir.mkdir(parents=True)
            cases = [spec.case_for(suite, task, init_id)
                     for suite in spec.SUITES for task in spec.REUSED_TASKS[suite] for init_id in range(10)]
            for index, case in enumerate(cases):
                for arm in ("base", "plugin"):
                    row = reused_row(case, arm)
                    # Freeze the known pilot02 outcome totals without selecting
                    # which cases are retained: all 80 cases and both arms stay.
                    cutoff = 74 if arm == "base" else 77
                    row["success"] = index < cutoff
                    row["status"] = "success" if row["success"] else "failure"
                    (episode_dir / f"{index:03d}_{arm}.json").write_text(json.dumps(row))
            manifest = spec.build_manifest(root)
            self.assertEqual(spec.validate_manifest(manifest), manifest)
            self.assertEqual(manifest["reused_episode_count"], 160)
            (episode_dir / "000_base.json").unlink()
            with self.assertRaisesRegex(ValueError, "both arms"):
                spec.build_manifest(root)


if __name__ == "__main__":
    unittest.main()
