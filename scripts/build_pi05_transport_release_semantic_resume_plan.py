#!/usr/bin/env python3
"""Create the paired transport-release semantic-resume study."""
from __future__ import annotations
import argparse,copy,json,shutil,tempfile
from pathlib import Path
import pi05_harness_backend as backend

PATCH=Path(__file__).resolve().parents[1]
SOURCE=Path("/home/user/tian_ws/eip_training_runs/full2000_public_transport_20261007")
BASELINE=PATCH/"configs/pi05_harness"; NAMESPACE="pi05_transport_release_semantic_resume"
OUTPUT=PATCH/"configs"/NAMESPACE
RESUME={
 "libero_10/8":"put both moka pots on the stove; place the remaining moka pot on the stove and leave the already placed pot there",
 "libero_10/9":"put both the yellow mug and the white mug in the microwave; put the remaining mug inside before closing the microwave door",
}

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def all_cases():
 rows=[]
 for init in (0,2,3,4,5,7,8):
  rows += [{"suite":"libero_10","task_id":8,"init_id":init,"replicate_id":rep} for rep in range(10)]
 rows += [{"suite":"libero_10","task_id":8,"init_id":init,"replicate_id":0} for init in (1,6,9)]
 rows += [{"suite":"libero_10","task_id":9,"init_id":3,"replicate_id":rep} for rep in range(10)]
 rows += [{"suite":"libero_10","task_id":9,"init_id":init,"replicate_id":0} for init in (0,1,2,4,5,6,7,8,9)]
 if len(rows)!=92 or len({tuple(r.values()) for r in rows})!=92: raise AssertionError("case inventory")
 return rows

def worker_cases(worker):
 if type(worker) is not int or not 0<=worker<4: raise ValueError("worker must be 0..3")
 return all_cases()[worker::4]

def build(output=OUTPUT,baseline=BASELINE,source=SOURCE):
 output,baseline,source=map(Path,(output,baseline,source))
 if output.exists(): raise FileExistsError(f"create-only output already exists: {output}")
 registry=json.loads((baseline/"registry.json").read_text()); catalog=json.loads((baseline/"task_catalog.json").read_text())
 rows={r["key"]:r for r in catalog["tasks"]}
 if len(rows)!=40 or set(registry["tasks"])!=set(rows): raise ValueError("complete LIBERO-40 baseline required")
 api=backend.import_roborsi(source); from roborsi.self_harness.core import digest
 output.parent.mkdir(parents=True,exist_ok=True); tmp=Path(tempfile.mkdtemp(prefix=".pi05_semantic_resume.",dir=output.parent))
 try:
  digests={}
  for key,relative in sorted(registry["tasks"].items()):
   src=baseline/relative; dst=tmp/relative; dst.parent.mkdir(parents=True,exist_ok=True)
   if key not in RESUME:
    shutil.copy2(src,dst); config=json.loads(src.read_text())
   else:
    config=json.loads(src.read_text())
    if "pi05_control" in config: raise ValueError("baseline contains control")
    config["pi05_control"]={"kind":"transport_release_semantic_resume_v1","enabled":True,
                            "resume_instruction":RESUME[key]}
    dump(dst,config)
   digests[key]=digest(config)
  proposal=copy.deepcopy(registry); proposal["name"]="pi05-transport-release-semantic-resume"
  proposal["metadata"]={"task_config_sha256":digests,"candidate_status":"posthoc_unvalidated_research_candidate",
   "control_scope":"task_wide_long8_long9_no_init_or_seed_routing",
   "evidence":{"source":"prior_native_control_20_episode_fixed_proxy_scan",
    "fixed_proxy_coverage":{"failures":"5/6","successes":"14/14"},
    "parameters":{"hold_rows":10,"hold_aperture_m":[.008,.070],"transport_xy_m":.10,
                  "release_rows":5,"release_aperture_m":.075,"native_reserve_steps":80},
    "interpretation":"physical_transport_release_proxy_not_object_or_task_completion_certificate",
    "weights_changed":False}}
  dump(tmp/"registry.json",proposal); shutil.copy2(baseline/"task_catalog.json",tmp/"task_catalog.json")
  tasks={k:{"instruction":v["instruction"],"max_steps":v["max_steps"]} for k,v in rows.items()}
  materialized=api.TaskHarnessRegistry(tmp/"registry.json",{"pi05"}).materialize(tasks)
  if materialized["task_config_sha256"]!=digests or len(materialized["harnesses"])!=40: raise AssertionError("materialization")
  for worker in range(4):
   cases=worker_cases(worker); dump(tmp/f"cases_worker{worker}.json",{"schema":"pi05_harness_cases.v1","cases":cases})
   common={"routes":"configs/pi05_harness/routes_base.json","cases":f"configs/{NAMESPACE}/cases_worker{worker}.json","mode":"harness"}
   dump(tmp/f"job_worker{worker}.json",{"schema":"pi05_harness_worker.v1","batches":[
    {**common,"name":f"worker{worker}_control","registry":"configs/pi05_harness/registry.json"},
    {**common,"name":f"worker{worker}_semantic_resume","registry":f"configs/{NAMESPACE}/registry.json"}]})
  tmp.rename(output)
 except BaseException:
  shutil.rmtree(tmp,ignore_errors=True); raise
 return output

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,default=OUTPUT); p.add_argument("--baseline-root",type=Path,default=BASELINE); p.add_argument("--source-root",type=Path,default=SOURCE); a=p.parse_args(); print(build(a.output_root,a.baseline_root,a.source_root))
