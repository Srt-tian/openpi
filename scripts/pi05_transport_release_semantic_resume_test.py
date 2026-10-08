#!/usr/bin/env python3
import copy,unittest
from types import MappingProxyType
import numpy as np
import pi05_transport_release_semantic_resume as target

class Delegate:
 def __init__(self,chunks): self.chunks=[np.asarray(x,float) for x in chunks]; self.calls=0; self.call_records=[]; self.prompts=[]; self.provenance={}
 def begin_episode(self): pass
 def reset(self): self.resets=getattr(self,"resets",0)+1
 def act(self,o,instruction,m):
  self.prompts.append(instruction); self.call_records.append({"inference_call":self.calls}); out=self.chunks[self.calls]; self.calls+=1; return out.copy()
 def finalize_episode(self,t=()): self.provenance={"final":True}

def chunk(grips): return np.asarray([[i+.1,i+.2,i+.3,0,0,0,g] for i,g in enumerate(grips)])
def state(x=0,ap=.04): return np.asarray([x,0,1,0,0,0,ap/2,-ap/2])
def context(step,remaining=520): return MappingProxyType({"actual_executed":step,"remaining_episode":remaining-step,"stage_executed":step,"remaining_stage":remaining-step,"stage_index":0})

class TestSemanticResume(unittest.TestCase):
 def run_call(self,skill,delegate,step,states,instruction="full goal"):
  skill.set_execution_context(context(step)); out=skill.act(object(),instruction,{})
  for action,post in zip(out[:5],states): skill.on_execution(action,post)
  return out
 def test_exact_witness_switch_and_no_extra_calls(self):
  chunks=[chunk([.6]*5),chunk([.6]*5),chunk([.6]+[-.6]*4),chunk([-.6]*5),chunk([0]*5)]
  d=Delegate(chunks); s=target.Pi05TransportReleaseSemanticResumeSkill(d,resume_instruction="full goal; residual")
  s.begin_episode(); s.on_reset(state())
  before=[]; before.append(self.run_call(s,d,0,[state()]*5)); before.append(self.run_call(s,d,5,[state()]*5))
  before.append(self.run_call(s,d,10,[state(.11)]+[state(.11,.08)]*4))
  before.append(self.run_call(s,d,15,[state(.11,.08)]*5))
  self.run_call(s,d,20,[state(.11,.08)]*5)
  self.assertEqual(d.prompts,["full goal"]*4+["full goal; residual"])
  self.assertEqual(d.calls,5); self.assertTrue(all(np.array_equal(x,chunks[i]) for i,x in enumerate(before)))
  s.finalize_episode(tuple((i,0) for i in range(25)))
  p=s.provenance; self.assertEqual((p["witness_actual_step"],p["first_prompt_change_actual_step"],p["first_prompt_change_call"]),(16,20,4))
  self.assertTrue(p["mechanical_proxy_only"]); self.assertFalse(p["object_or_task_completion_certificate"])
 def test_hold_and_release_boundaries(self):
  d=Delegate([chunk([.6]*5)]*2+[chunk([.6,-.6,-.6,-.6,-.6]),chunk([-.6]*5),chunk([0]*5)])
  s=target.Pi05TransportReleaseSemanticResumeSkill(d,resume_instruction="residual"); s.begin_episode(); s.on_reset(state())
  self.run_call(s,d,0,[state()]*5); self.run_call(s,d,5,[state()]*5)
  self.run_call(s,d,10,[state(.11)]+[state(.11,.08)]*4); self.run_call(s,d,15,[state(.11,.08)]*5)
  self.assertIsNone(s._resume_step); self.run_call(s,d,20,[state()]*5); self.assertEqual(s._resume_step,20)
 def test_anchor_resets_only_once_before_transport(self):
  chunks=[chunk([.6]*5)]*2+[chunk([-1]*5)]+[chunk([.6]*5)]*2+[chunk([-1]*5)]
  d=Delegate(chunks); s=target.Pi05TransportReleaseSemanticResumeSkill(d,resume_instruction="residual"); s.begin_episode(); s.on_reset(state())
  self.run_call(s,d,0,[state()]*5); self.run_call(s,d,5,[state()]*5); self.run_call(s,d,10,[state(ap=.08)]*5)
  self.run_call(s,d,15,[state()]*5); self.run_call(s,d,20,[state()]*5); self.run_call(s,d,25,[state(ap=.08)]*5)
  self.assertTrue(s._anchor_reset_used); self.assertTrue(s._detection_vetoed); self.assertIsNone(s._witness_step)
 def test_reserve_blocks_switch_and_terminal_truncation(self):
  d=Delegate([chunk([0]*5)]); s=target.Pi05TransportReleaseSemanticResumeSkill(d,resume_instruction="residual"); s.begin_episode(); s.on_reset(state()); s._witness_step=1
  s.set_execution_context(context(441)); s.act(object(),"full goal",{}); s.on_execution(d.chunks[0][0],state())
  s.finalize_episode(((441,0),)); self.assertIsNone(s.provenance["first_prompt_change_actual_step"]); self.assertEqual(d.prompts,["full goal"]); self.assertEqual(sum(r.get("truncated_before_execution") is True for r in s.provenance["emitted_rows"]),4)
 def test_admitted_resume_is_latched_through_last_step(self):
  d=Delegate([chunk([0]*5)]*3); s=target.Pi05TransportReleaseSemanticResumeSkill(d,resume_instruction="full goal; residual"); s.begin_episode(); s.on_reset(state()); s._witness_step=419
  self.run_call(s,d,420,[state()]*5)
  self.run_call(s,d,445,[state()]*5)
  s.set_execution_context(context(519)); s.act(object(),"full goal",{}); s.on_execution(d.chunks[2][0],state())
  self.assertEqual(d.prompts,["full goal; residual"]*3); self.assertEqual(d.calls,3); self.assertEqual(s._resume_step,420)
  s.finalize_episode(tuple((i,0) for i in list(range(420,425))+list(range(445,450))+[519]))
  self.assertEqual(sum(r.get("truncated_before_execution") is True for r in s.provenance["emitted_rows"]),4)
 def test_context_and_instruction_validation(self):
  with self.assertRaises(ValueError): target.Pi05TransportReleaseSemanticResumeSkill(Delegate([]),resume_instruction=" ")
  s=target.Pi05TransportReleaseSemanticResumeSkill(Delegate([chunk([0]*5)]),resume_instruction="residual"); s.begin_episode(); s.on_reset(state())
  with self.assertRaises(ValueError): s.set_execution_context(dict(context(0)))
 def test_stage_reset_truncates_pending_and_requires_fresh_context(self):
  d=Delegate([chunk([0]*5)]); s=target.Pi05TransportReleaseSemanticResumeSkill(d,resume_instruction="residual"); s.begin_episode(); s.on_reset(state()); s.set_execution_context(context(0)); s.act(object(),"full goal",{}); s.reset()
  self.assertEqual(d.resets,1); self.assertTrue(all(r.get("truncated_before_execution") is True for r in s._emitted))
  with self.assertRaises(RuntimeError): s.act(object(),"full goal",{})

if __name__=="__main__": unittest.main()
