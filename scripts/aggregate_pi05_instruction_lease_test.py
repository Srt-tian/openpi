from __future__ import annotations

import copy
import unittest

import aggregate_pi05_instruction_lease as target

IDENTITY={"policy_id":"base","checkpoint_sha256":"a"*64,"base_graph":"original_pi05_libero","adapter_sha256":None}


def evidence(which="control", steps=140):
    trace=[{"step":i,"stage":0,"skill":"pi05","state":[float(i)]*8,"action":[float(i)]*7} for i in range(steps)]
    calls=[];receipts=[];prompt_records=[];emitted=[];executed=[]
    planned=40 if which=="lease40" else 80
    for call in range((steps+4)//5):
        leased=which!="control" and call>=25; prompt=target.LEASED if leased else target.ORIGINAL; seed=1157+call*1000003
        calls.append({"inference_call":call,"policy_seed":seed,"metadata":IDENTITY,"response_valid":True})
        desc=lambda char,shape,dtype:{"sha256":char*64,"shape":shape,"dtype":dtype}
        receipts.append({"inference_call":call,"policy_seed":seed,"prompt":prompt,"status":"ok",
          "observation_image":desc("b",[224,224,3],"uint8"),"observation_wrist_image":desc("c",[224,224,3],"uint8"),"observation_state":desc("d",[8],"float64")})
        if which!="control":
            prompt_records.append({"prompt_call_index":call,"delegate_call_index":call,"actual_step":call*5,
              "planned_prompt":prompt,"used_prompt":prompt,"original_prompt":target.ORIGINAL,
              "leased":leased,"emitted_steps":5,"executed_steps":min(5,max(0,steps-call*5))})
            for off in range(5):
                idx=call*5+off; row={"emission_index":idx,"kind":"leased" if leased else "native",
                  "prompt_call_index":call,"stage_index":0,"expected_actual_step":idx,"action":[float(idx)]*7,
                  "executed":idx<steps}
                if idx<steps: executed.append(copy.deepcopy(row))
                else: row["truncated_before_execution"]=True
                emitted.append(row)
    skills={}
    if which!="control":
        actual=max(0,steps-125)
        skills={"pi05":{"kind":"instruction_lease_v1","configured_instruction":target.LEASED,
          "planned_lease_steps":planned,"actual_leased_steps":actual,"original_native_reserve":80,
          "first_changed_prompt_step":125,"attempted":True,"events":[{"status":"terminal_truncated_during_lease",
            "cue":{"context":{"actual_executed":120,"remaining_episode":180,"stage_executed":120,"remaining_stage":180,"stage_index":0}},
            "recheck":{"context":{"actual_executed":125,"remaining_episode":175,"stage_executed":125,"remaining_stage":175,"stage_index":0}}}],
          "prompt_call_records":prompt_records,"emitted_rows":emitted,"executed_rows":executed,
          "execution_reconciled":True,"actions_transformed":False,"oracle_inputs":[]}}
    return {"case":{"policy_seed":1157},"harness":{"remember":[],"stages":[{"skill":"pi05","instruction":target.ORIGINAL,"max_steps":300,"until":None,"on_timeout":"abort"}]},
      "runner":{"report":{"steps":steps,"trace":trace,"success":False,"status":"program_completed","skills":skills},
                "policy_calls":calls,"payload_hashes":receipts}}


class InstructionLeaseAggregateTest(unittest.TestCase):
    def test_valid_terminal_truncated_lease(self):
        self.assertEqual(target.audit_episode(evidence("lease40"),"lease40",IDENTITY),[])

    def test_prelease_difference_is_confounded(self):
        control,candidate=evidence(),evidence("lease40")
        candidate["runner"]["report"]["trace"][124]["state"][0]+=1e-12
        self.assertIn("pre-lease trace/calls/payload hashes differ",target.causal(control,candidate))

    def test_action_edit_and_bound_fail(self):
        value=evidence("lease40");prov=target.lease_provenance(value)
        prov["actual_leased_steps"]=41
        prov["executed_rows"][125]["action"][0]=99.0
        errors=target.audit_episode(value,"lease40",IDENTITY)
        self.assertIn("actual leased-step bound violated",errors)
        self.assertIn("executed emission differs from trace",errors)

    def test_bad_prompt_receipt_and_call_count_fail(self):
        value=evidence("lease80");value["runner"]["payload_hashes"][25]["prompt"]=target.ORIGINAL
        value["runner"]["policy_calls"].pop()
        errors=target.audit_episode(value,"lease80",IDENTITY)
        self.assertIn("call/hash count is not ceil(steps/5)",errors)
        self.assertIn("prompt call does not match actual payload",errors)

    def test_untriggered_must_match_whole_episode(self):
        control=evidence(); candidate=evidence("lease40");prov=target.lease_provenance(candidate)
        prov.update(first_changed_prompt_step=None,actual_leased_steps=0,prompt_call_records=[])
        prov["emitted_rows"]=[];prov["executed_rows"]=[]
        candidate["runner"]["report"]["success"]=True
        errors=target.causal(control,candidate)
        self.assertIn("untriggered outcome differs",errors)

    def test_short_nonterminal_lease_and_call_count_fail(self):
        value=evidence("lease40",180);prov=target.lease_provenance(value)
        prov["prompt_call_records"][30]["leased"]=False
        prov["prompt_call_records"][30]["planned_prompt"]=target.ORIGINAL
        prov["prompt_call_records"][30]["used_prompt"]=target.ORIGINAL
        prov["actual_leased_steps"]-=5
        errors=target.audit_episode(value,"lease40",IDENTITY)
        self.assertIn("changed prompts are not one continuous full-or-terminal-truncated lease",errors)
        value=evidence("lease40");target.lease_provenance(value)["prompt_call_records"][25]["executed_steps"]=4
        self.assertIn("prompt-call executed_steps differs from executed rows",
                      target.audit_episode(value,"lease40",IDENTITY))

    def test_bad_event_grace_budget_fails(self):
        value=evidence("lease80");event=target.lease_provenance(value)["events"][-1]
        event["cue"]["context"]["actual_executed"]=121
        self.assertIn("event cue/recheck does not prove five-step grace and fixed 300 budget",
                      target.audit_episode(value,"lease80",IDENTITY))


if __name__=="__main__":unittest.main()
