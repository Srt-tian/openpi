#!/usr/bin/env python3
import copy,unittest
import aggregate_pi05_transport_release_semantic_resume as target

IDENTITY={"policy_id":"base","checkpoint_sha256":"a"*64,"base_graph":"original_pi05_libero","adapter_sha256":None,"policy_seed_protocol":target.PROTOCOL}
ORIGINAL="put both moka pots on the stove"

def episode():
 case={"suite":"libero_10","task_id":8,"init_id":0,"replicate_id":0,"joint_task_number":38,"ambient_seed":1907,"policy_seed":1907,"policy_id":"base"}
 trace=[]; emitted=[]; executed=[]
 for i in range(25):
  action=[float(i),0,0,0,0,0,.6]; state=[float(i+1),0,1,0,0,0,.02,-.02]
  trace.append({"step":i,"action":action,"state":state,"stage":0,"skill":"pi05"})
  row={"emission_index":i,"expected_actual_step":i,"stage_index":0,"action":action,"executed":True,"execution_index":i,"post_state8":([float(i+2),0,1,0,0,0,.02,-.02] if i<24 else state)}
  emitted.append(copy.deepcopy(row)); executed.append(row)
 calls=[]; receipts=[]; prompts=[]
 for i in range(5):
  prompt=ORIGINAL if i<4 else target.RESUME["libero_10/8"]
  calls.append({"inference_call":i,"policy_seed":1907+i*1000003,"metadata":IDENTITY,"response_valid":True})
  receipts.append({"inference_call":i,"policy_seed":1907+i*1000003,"policy_id":"base","prompt":prompt,"status":"ok"})
  prompts.append({"inference_call":i,"actual_step":i*5,"prompt":prompt,"prompt_changed":i>=4})
 p={"kind":"transport_release_semantic_resume_v1","parameters":target.PARAMS,"mechanical_proxy_only":True,"object_or_task_completion_certificate":False,"execution_reconciled":True,"witness_actual_step":17,"first_prompt_change_actual_step":20,"first_prompt_change_call":4,"call_prompt_receipts":prompts,"emitted_rows":emitted,"executed_rows":executed}
 return {"case":case,"harness":{"remember":[],"stages":[{"instruction":ORIGINAL,"skill":"pi05","max_steps":520,"until":None,"on_timeout":"abort"}]},"runner":{"policy_calls":calls,"payload_hashes":receipts,"report":{"steps":25,"trace":trace,"skills":{"pi05":p}}}}

class TestAudit(unittest.TestCase):
 def test_valid_and_strict_prompt_timing(self):
  value=episode(); self.assertEqual(target.audit(value,"semantic_resume",IDENTITY),[])
  for mutate in (lambda v:v["runner"]["payload_hashes"][4].update(prompt="wrong"),lambda v:v["runner"]["report"]["skills"]["pi05"].update(first_prompt_change_actual_step=19),lambda v:v["runner"]["report"]["skills"]["pi05"]["executed_rows"][3].update(action=[0]*7)):
   bad=copy.deepcopy(value); mutate(bad); self.assertTrue(target.audit(bad,"semantic_resume",IDENTITY))
 def test_control_never_changes_prompt(self):
  value=episode(); value["runner"]["payload_hashes"][4]["prompt"]=ORIGINAL
  self.assertEqual(target.audit(value,"control",IDENTITY),[])
  value["runner"]["payload_hashes"][3]["prompt"]="residual"; self.assertIn("control prompt changed",target.audit(value,"control",IDENTITY))
 def test_inventory_exact(self): self.assertEqual(len(target.EXPECTED),92)

if __name__=="__main__": unittest.main()
