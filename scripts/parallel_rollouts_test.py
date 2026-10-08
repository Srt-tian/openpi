from pathlib import Path
import tempfile
import unittest

from parallel_rollouts import rollout_workers_arg, run_episodes_ordered


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "parallel_rollouts_fixture.py"


def item(index, fixture=None):
    value = {"id": f"libero_goal/3/{index}", "arm": "plugin", "policy_id": "goal",
             "suite": "libero_goal", "task_id": 3, "init_id": index}
    if fixture:
        value["fixture"] = fixture
    return value


class ParallelRolloutsTest(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(rollout_workers_arg("1"), 1)
        self.assertEqual(rollout_workers_arg("3"), 3)
        for invalid in ("0", "4", "bad"):
            with self.assertRaises(Exception):
                rollout_workers_arg(invalid)

    def test_order_and_uncaught_failure_keep_denominator(self):
        plan = [item(0, "slow"), item(1, "raise"), item(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episodes").mkdir()
            (root / "videos").mkdir()
            rows = run_episodes_ordered(FIXTURE, plan, {"fixture_setting": "kept"},
                                        root / "episodes", root / "videos", 3)
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["plan_index"] for row in rows], [0, 1, 2])
            self.assertEqual(rows[1]["status"], "error")
            self.assertTrue((root / "episodes/0001_plugin_libero_goal_3_1.json").is_file())


if __name__ == "__main__":
    unittest.main()
