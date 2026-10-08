from __future__ import annotations
import copy,json,tempfile,unittest
from pathlib import Path
import aggregate_pi05_harness_candidate400 as target

class Candidate400AggregateTest(unittest.TestCase):
    def test_case_plan_is_exact_and_caps_are_authoritative(self):
        catalog=json.loads((Path(__file__).resolve().parents[1]/"configs/pi05_harness/task_catalog.json").read_text())
        rows={r["key"]:r for r in catalog["tasks"]}; cases=[]
        for joint,suite in enumerate(target.SUITES):
            for task in range(10):
                for init in range(10):cases.append({"suite":suite,"task_id":task,"init_id":init,
                    "replicate_id":0,"joint_task_number":joint*10+task,"official_cap":rows[f"{suite}/{task}"]["max_steps"]})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cp=root/"cases.json";tc=root/"catalog.json"
            cp.write_text(json.dumps({"schema":"pi05_harness_candidate400_cases.v1","cases":cases}));tc.write_text(json.dumps(catalog))
            self.assertEqual(len(target.load_plan(cp,tc)),400)
            cases[0]["official_cap"]+=1;cp.write_text(json.dumps({"schema":"pi05_harness_candidate400_cases.v1","cases":cases}))
            with self.assertRaises(ValueError):target.load_plan(cp,tc)

    def test_unchanged_task_exactness_is_attribution_not_score_filter(self):
        runner={"report":{"trace":[{"step":0}],"success":True,"status":"task_success"},
                "policy_calls":[{"x":1}],"payload_hashes":[{"y":2}]}
        control={"runner":copy.deepcopy(runner)};candidate={"runner":copy.deepcopy(runner)}
        audit,passed,confounds,triggered=target.pair_gate(control,candidate,("libero_object",0,0))
        self.assertEqual((audit,passed,confounds,triggered),([],True,[],False))
        candidate["runner"]["report"]["trace"][0]["state"]=[1]
        _,passed,confounds,_=target.pair_gate(control,candidate,("libero_object",0,0))
        self.assertFalse(passed);self.assertTrue(confounds)
        summary=target.summarize([{"outcome":"both","control_success":True,"candidate_success":True}])
        self.assertEqual((summary["pairs"],summary["candidate_successes"]),(1,1))

if __name__=="__main__":unittest.main()
