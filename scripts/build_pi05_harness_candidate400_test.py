from __future__ import annotations
import json, os, tempfile, unittest
from pathlib import Path
import build_pi05_harness_candidate400 as target
import pi05_harness_backend as backend

SOURCE=Path(os.environ.get("PI05_TEST_PHYSICALRSI_ROOT",str(target.SOURCE)))

class Candidate400PlanTest(unittest.TestCase):
    def test_exact_registry_controls_hashes_and_400_pair_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            out=target.build(Path(directory)/"out",source=SOURCE)
            reg=json.loads((out/"registry.json").read_text())
            base_reg=json.loads((target.BASELINE/"registry.json").read_text())
            changed=[]
            for key,rel in reg["tasks"].items():
                actual=(out/rel).read_bytes(); baseline=(target.BASELINE/base_reg["tasks"][key]).read_bytes()
                if actual != baseline: changed.append(key)
                if key=="libero_goal/3": self.assertEqual(json.loads(actual)["pi05_control"],target.GOAL_CONTROL)
                if key=="libero_10/8": self.assertEqual(json.loads(actual)["pi05_control"],target.LONG_CONTROL)
            self.assertEqual(changed,["libero_10/8","libero_goal/3"])
            self.assertEqual(reg["metadata"]["oracle_inputs"],[])
            self.assertIn("pending",reg["metadata"]["status"])
            seen=set();episodes=0
            for worker in range(4):
                cases=json.loads((out/f"cases_worker{worker}.json").read_text())["cases"]
                job=json.loads((out/f"job_worker{worker}.json").read_text())
                self.assertEqual(len(cases),100);episodes += 2*len(cases)
                seen|={(x["suite"],x["task_id"],x["init_id"],x["replicate_id"]) for x in cases}
                self.assertEqual([b["name"].rsplit("_",1)[1] for b in job["batches"]],["control","candidate"])
                self.assertEqual({b["routes"] for b in job["batches"]},{"configs/pi05_harness/routes_base.json"})
            self.assertEqual((len(seen),episodes),(400,800))
            plan=json.loads((out/"case_plan.json").read_text())["cases"]
            self.assertEqual(len(plan),400);self.assertTrue(all(x["replicate_id"]==0 for x in plan))
            catalog=json.loads((out/"task_catalog.json").read_text()); rows={x["key"]:x for x in catalog["tasks"]}
            self.assertTrue(all(x["official_cap"]==rows[f'{x["suite"]}/{x["task_id"]}']["max_steps"] for x in plan))
            api=backend.import_roborsi(SOURCE);inputs={k:{"instruction":v["instruction"],"max_steps":v["max_steps"]} for k,v in rows.items()}
            proposal=api.TaskHarnessRegistry(out/"registry.json",{"pi05"}).materialize(inputs)
            self.assertEqual(proposal["task_config_sha256"],reg["metadata"]["task_config_sha256"])

    def test_bounds_and_create_only(self):
        for value in (-1,4,True):
            with self.assertRaises(ValueError): target.worker_cases(value)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"exists";path.mkdir()
            with self.assertRaises(FileExistsError): target.build(path,source=SOURCE)

if __name__=="__main__": unittest.main()
