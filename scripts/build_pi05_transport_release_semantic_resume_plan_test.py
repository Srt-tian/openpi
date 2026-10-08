#!/usr/bin/env python3
import json,os,tempfile,unittest
from pathlib import Path
import build_pi05_transport_release_semantic_resume_plan as target

class TestPlan(unittest.TestCase):
 def test_inventory(self):
  rows=target.all_cases(); self.assertEqual(len(rows),92); self.assertEqual([len(target.worker_cases(i)) for i in range(4)],[23]*4)
  self.assertEqual(sum(r["task_id"]==8 for r in rows),73); self.assertEqual(sum(r["task_id"]==9 for r in rows),19)
  self.assertTrue(all(any(r["task_id"]==t and r["init_id"]==i and r["replicate_id"]==0 for r in rows) for t in (8,9) for i in range(10)))
 def test_complete_registry_and_exact_unchanged_38(self):
  source=os.environ.get("PI05_TEST_PHYSICALRSI_ROOT")
  if not source: self.skipTest("PI05_TEST_PHYSICALRSI_ROOT required")
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/"plan"; target.build(out,target.BASELINE,Path(source))
   reg=json.loads((out/"registry.json").read_text()); base=json.loads((target.BASELINE/"registry.json").read_text())
   self.assertEqual(set(reg["tasks"]),set(base["tasks"])); changed=[]
   for key,rel in base["tasks"].items():
    if (out/rel).read_bytes()!=(target.BASELINE/rel).read_bytes(): changed.append(key)
   self.assertEqual(changed,["libero_10/8","libero_10/9"])
   for key,text in target.RESUME.items():
    cfg=json.loads((out/reg["tasks"][key]).read_text()); self.assertEqual(cfg["pi05_control"],{"kind":"transport_release_semantic_resume_v1","enabled":True,"resume_instruction":text})
    original=json.loads((target.BASELINE/base["tasks"][key]).read_text()); self.assertEqual(cfg["harness"],original["harness"]); self.assertEqual(cfg.get("parent_sha256"),original.get("parent_sha256"))
   for w in range(4):
    job=json.loads((out/f"job_worker{w}.json").read_text()); self.assertEqual([b["name"].rsplit("_",1)[-1] for b in job["batches"]],["control","resume"])
    self.assertTrue(all(b["routes"]=="configs/pi05_harness/routes_base.json" and b["mode"]=="harness" for b in job["batches"]))
   with self.assertRaises(FileExistsError): target.build(out,target.BASELINE,Path(source))

if __name__=="__main__": unittest.main()
