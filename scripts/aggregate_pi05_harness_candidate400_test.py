from __future__ import annotations
import copy,json,tempfile,unittest
from pathlib import Path
import aggregate_pi05_harness_candidate400 as target

class Candidate400AggregateTest(unittest.TestCase):
    def _synthetic_800(self,root):
        catalog=json.loads((Path(__file__).resolve().parents[1]/"configs/pi05_harness/task_catalog.json").read_text());rows={r["key"]:r for r in catalog["tasks"]}
        catalog_path=root/"catalog.json";catalog_path.write_text(json.dumps(catalog));plan=[];identity={"policy_id":"base","base_graph":"original_pi05_libero","adapter_sha256":None,"checkpoint_sha256":"c"*64}
        assignments=[[] for _ in range(4)]
        for suite_index,suite in enumerate(target.SUITES):
            for task in range(10):
                joint=suite_index*10+task;cap=rows[f"{suite}/{task}"]["max_steps"]
                for init in range(10):
                    item={"suite":suite,"task_id":task,"init_id":init,"replicate_id":0,"joint_task_number":joint,"official_cap":cap};plan.append(item);assignments[joint%4].append(item)
        plan_path=root/"plan.json";plan_path.write_text(json.dumps({"schema":"pi05_harness_candidate400_cases.v1","cases":plan}));receipts=[];roots=[]
        for worker,cases in enumerate(assignments):
            wr=root/f"worker{worker}";wr.mkdir();roots.append(wr);batches=[]
            for arm in target.ARMS:
                name=f"worker{worker}_{arm}";batch=wr/name;(batch/"episodes").mkdir(parents=True);summary=[]
                (batch/"manifest.json").write_text(json.dumps({"fixed_policy_id":"base","verified_service_identity":identity}))
                for index,item in enumerate(cases):
                    suite,task,init=item["suite"],item["task_id"],item["init_id"];joint=item["joint_task_number"];seed=7+joint*50+init;key=f"{suite}/{task}";state=[0.0]*8;action=[0.0]*7
                    case={"suite":suite,"task_id":task,"init_id":init,"replicate_id":0,"joint_task_number":joint,"ambient_seed":seed,"policy_seed":seed,"policy_id":"base"}
                    receipt={"inference_call":0,"policy_seed":seed,"status":"ok","prompt":rows[key]["instruction"],"observation_image":{"shape":[224,224,3],"dtype":"uint8","sha256":"1"*64},"observation_wrist_image":{"shape":[224,224,3],"dtype":"uint8","sha256":"2"*64},"observation_state":{"shape":[8],"dtype":"float64","sha256":"3"*64}}
                    call={"inference_call":0,"policy_seed":seed,"metadata":identity,"response_valid":True}
                    skill={"kind":"fake"};control={"kind":"response_probe_v1","enabled":False}
                    if arm=="candidate" and key=="libero_goal/3":
                        control=target.GOAL;er={"emission_index":0,"kind":"native","stage_index":0,"expected_actual_step":0,"action":action,"executed":True,"post_state8":state}
                        skill={"kind":"pi05_response_probe_v1","execution_reconciled":True,"manual_actions_emitted":0,"events":[],"emitted_rows":[copy.deepcopy(er)],"executed_rows":[er],"parameters":{"lift_z_command":.2,"max_lift_steps":20,"lift_target_m":.025,"native_reserve_steps":80,"minimum_actual":100,"max_manual_actions":26,"trigger_remaining_steps":111}}
                    elif arm=="candidate" and key=="libero_10/8":
                        control=target.LONG;er={"emission_index":0,"kind":"native","expected_actual_step":0,"cache_chunk_inference_index":0,"raw_action":action,"executed_action":action,"modified":False,"executed":True,"pre_state8":state}
                        skill={"kind":"closed_dwell_lift_v1","execution_reconciled":True,"native_reserve":80,"mechanical_proxy_only":True,"grasp_or_task_success_certificate":False,"oracle_inputs":[],"emitted_rows":[copy.deepcopy(er)],"executed_rows":[er],"chunks":[{"chunk_index":0,"delegate_call_index":0,"inference_actual_step":0,"emitted_rows":1,"executed_rows":1}],"modification_count":0,"emitted_modification_count":0,"assist_slots_executed":0,"first_changed_action_step":None,"attempted":False,"veto_native_upward_intent":True,"veto_reason":None,"native_incoming_z":None}
                    episode={"case":case,"pi05_control":control,"harness":{"remember":[],"stages":[{"skill":"pi05","instruction":rows[key]["instruction"],"max_steps":item["official_cap"],"until":None,"on_timeout":"abort"}]},"video":{"written":True,"path":f"{index:03d}.mp4"},"runner":{"policy_calls":[call],"payload_hashes":[receipt],"report":{"steps":1,"trace":[{"step":0,"stage":0,"skill":"pi05","state":state,"action":action}],"success":False,"status":"budget_exhausted","skills":{"pi05":skill},"environment":{"kind":"pi05_official_libero","step_cap":item["official_cap"],"success_decision_source":"official_libero_env.step.done","init_asset":{"sha256":"a"*64}}}}}
                    rel=f"episodes/{index:03d}.json";path=batch/rel;path.write_text(json.dumps(episode));summary.append({**case,"episode":rel,"success":False,"steps":1})
                (batch/"summary.json").write_text(json.dumps({"mode":"harness","complete":True,"errors":0,"planned":100,"completed":100,"cases":summary}));batches.append({"name":name,"server_pid":7,"service_reused_from_previous_batch":arm=="candidate"})
            controller={"status":"complete","batches":batches};(wr/"controller.json").write_text(json.dumps(controller));artifacts=[]
            for ep in sorted(wr.glob("*/episodes/*.json")):
                value=json.loads(ep.read_text());video=ep.parent.parent/"videos"/value["video"]["path"]
                artifacts.append({"episode_path":ep.relative_to(wr).as_posix(),"episode_sha256":target.sha256(ep),"video_path":video.relative_to(wr).as_posix(),"video_sha256":"b"*64,"video_size":123})
            receipt_path=root/f"receipt{worker}.json";receipt_path.write_text(json.dumps({"schema":"pi05_harness_artifact_inventory.v1","worker_output_name":wr.name,"controller_sha256":target.sha256(wr/"controller.json"),"artifacts":artifacts}));receipts.append(receipt_path)
        return roots,receipts,plan_path,catalog_path

    def test_complete_synthetic_800_and_strict_receipt_identity_seed_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);roots,receipts,plan,catalog=self._synthetic_800(root)
            result=target.aggregate(roots,plan,catalog,receipts);self.assertTrue(result["score_complete"]);self.assertTrue(result["exact_prefix_attribution_complete"]);self.assertEqual(result["coverage"]["episodes"],800)
            receipt=json.loads(receipts[0].read_text());removed=receipt["artifacts"].pop();receipts[0].write_text(json.dumps(receipt));self.assertFalse(target.aggregate(roots,plan,catalog,receipts)["score_complete"])
            receipt["artifacts"].append(dict(removed,episode_sha256="d"*64));receipts[0].write_text(json.dumps(receipt));self.assertFalse(target.aggregate(roots,plan,catalog,receipts)["score_complete"])
            receipt["artifacts"][-1]=removed;receipts[0].write_text(json.dumps(receipt));manifest=next(roots[0].glob("*/manifest.json"));bad=json.loads(manifest.read_text());bad["verified_service_identity"]["checkpoint_sha256"]="d"*64;manifest.write_text(json.dumps(bad));self.assertFalse(target.aggregate(roots,plan,catalog,receipts)["score_complete"])
            manifest.write_text(json.dumps({"fixed_policy_id":"base","verified_service_identity":{"policy_id":"base","base_graph":"original_pi05_libero","adapter_sha256":None,"checkpoint_sha256":"c"*64}}));summary=next(roots[0].glob("*/summary.json"));bad=json.loads(summary.read_text());bad["cases"][0]["policy_seed"]+=1;summary.write_text(json.dumps(bad));self.assertFalse(target.aggregate(roots,plan,catalog,receipts)["score_complete"])

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
            expected=target.load_plan(cp,tc);self.assertEqual(len(expected),400)
            valid={"suite":"libero_spatial","task_id":0,"init_id":0,"replicate_id":0,
                   "joint_task_number":0,"ambient_seed":7,"policy_seed":7,"policy_id":"base"}
            self.assertEqual(target.case_key(valid,expected),("libero_spatial",0,0))
            for field in ("task_id","init_id","replicate_id","joint_task_number","ambient_seed","policy_seed"):
                bad=dict(valid);bad[field]=False
                with self.assertRaises(ValueError):target.case_key(bad,expected)
            bad=dict(valid);bad["policy_id"]="long"
            with self.assertRaises(ValueError):target.case_key(bad,expected)
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
