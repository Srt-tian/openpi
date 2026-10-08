from __future__ import annotations
import copy, unittest
import aggregate_pi05_closed_dwell_lift as target

IDENTITY={"policy_id":"base","base_graph":"original_pi05_libero","adapter_sha256":None}

def episode(changed=True):
    n=130; trace=[{"step":i,"stage":0,"skill":"pi05","state":[0.0]*8,"action":[0.0]*7} for i in range(n)]
    calls=[]; hashes=[]
    for i in range(26):
        calls.append({"inference_call":i,"policy_seed":1909+i*1000003,"metadata":IDENTITY,"response_valid":True})
        hashes.append({"inference_call":i,"policy_seed":1909+i*1000003,"status":"ok"})
    rows=[]
    for i in range(n):
        raw=[0.0]*7
        if changed and i==120: raw[6]=1.0
        action=list(raw); kind="native"
        if changed and i==120: action[2]=.2; kind="assist"; trace[i]["action"]=action
        rows.append({"emission_index":i,"kind":kind,"expected_actual_step":i,"cache_chunk_inference_index":i//5,
          "raw_action":raw,"executed_action":action,"modified":action!=raw,"executed":True,"pre_state8":[0.0]*8})
    chunks=[{"chunk_index":i,"delegate_call_index":i,"inference_actual_step":i*5,"emitted_rows":5,"executed_rows":5} for i in range(26)]
    prov={"kind":"closed_dwell_lift_v1","execution_reconciled":True,"native_reserve":80,"mechanical_proxy_only":True,
      "grasp_or_task_success_certificate":False,"oracle_inputs":[],"emitted_rows":copy.deepcopy(rows),"executed_rows":rows,
      "chunks":chunks,"modification_count":int(changed),"emitted_modification_count":int(changed),"assist_slots_executed":int(changed),
      "first_changed_action_step":120 if changed else None,"attempted":changed,
      "cue":{"window":60,"closed_command_rows":57,"xyz_ptp_m":[.01]*3,"aperture_m":.02,
             "context":{"actual_executed":120,"remaining_episode":400,"stage_executed":120,
                        "remaining_stage":400,"stage_index":0}} if changed else None}
    return {"case":{"policy_seed":1909},"harness":{"remember":[],"stages":[{"skill":"pi05","max_steps":520,"until":None,"on_timeout":"abort"}]},
      "runner":{"report":{"steps":n,"trace":trace,"skills":{"pi05":{"provenance":prov}}},"policy_calls":calls,"payload_hashes":hashes}}

class TestAudit(unittest.TestCase):
    def test_valid_assist(self): self.assertEqual(target.audit_episode(episode(),"assist",IDENTITY),[])
    def test_valid_intent_veto_is_unmodified_and_causally_identical(self):
        c, a = episode(False), episode(False)
        for value in (c, a):
            value["runner"]["report"]["trace"][120]["action"][2] = .1
        p = a["runner"]["report"]["skills"]["pi05"]["provenance"]
        for rows in (p["emitted_rows"], p["executed_rows"]):
            rows[120]["raw_action"][2] = .1
            rows[120]["executed_action"][2] = .1
        p.update(veto_native_upward_intent=True,
                 veto_reason="incoming_native_upward_intent", native_incoming_z=.1,
                 attempted=True, cue={"window":60,"closed_command_rows":57,
                 "xyz_ptp_m":[.01]*3,"aperture_m":.02,
                 "context":{"actual_executed":120,"remaining_episode":400,
                 "stage_executed":120,"remaining_stage":400,"stage_index":0}})
        self.assertEqual(target.audit_episode(a, "assist", IDENTITY), [])
        self.assertEqual(target.causal(c, a), [])
    def test_non_z_change_fails(self):
        x=episode(); p=x["runner"]["report"]["skills"]["pi05"]["provenance"]
        p["executed_rows"][120]["executed_action"][0]=1;p["emitted_rows"][120]["executed_action"][0]=1;x["runner"]["report"]["trace"][120]["action"][0]=1
        self.assertIn("assist is not exact z-only max(raw_z,.2)",target.audit_episode(x,"assist",IDENTITY))
    def test_prechange_difference_confounds(self):
        c,a=episode(False),episode();a["runner"]["report"]["trace"][10]["state"][0]=1
        self.assertIn("pre-change trace differs",target.causal(c,a))
    def test_unchanged_must_match(self):
        c,a=episode(False),episode(False);a["runner"]["report"]["trace"][-1]["state"][0]=1
        self.assertIn("unchanged whole-episode evidence differs",target.causal(c,a))
    def test_case_protocol(self):
        c={"suite":"libero_10","task_id":8,"joint_task_number":38,"init_id":2,"replicate_id":3,"ambient_seed":1909,"policy_seed":3000001930}
        self.assertEqual(target.case_key(c),(2,3));c["joint_task_number"]=37
        with self.assertRaises(ValueError): target.case_key(c)

if __name__=="__main__": unittest.main()
