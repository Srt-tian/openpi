#!/usr/bin/env python3
"""Strict paired audit for transport-release semantic resume."""
from __future__ import annotations
import argparse,copy,json
from collections import Counter
from pathlib import Path
from aggregate_pi05_goal3_phase_screen import initial_receipt,valid_initial
from build_pi05_transport_release_semantic_resume_plan import RESUME,all_cases

ARMS=("control","semantic_resume"); CAP=520
PROTOCOL="paired_episode_plus_call_1000003_numpy_pcg64_noise_10x32_f32_v1"
EXPECTED={(r["suite"],r["task_id"],r["init_id"],r["replicate_id"]) for r in all_cases()}
PARAMS={"hold_rows":10,"hold_aperture_m":[.008,.070],"transport_xy_m":.10,
        "release_rows":5,"release_aperture_m":.075,"native_reserve_steps":80}

def arm(name):
 hits=[a for a in ARMS if a in name]
 if len(hits)!=1: raise ValueError(f"invalid arm name {name}")
 return hits[0]

def key(case):
 vals=tuple(case.get(k) for k in ("suite","task_id","init_id","replicate_id"))
 if vals not in EXPECTED or any(type(v) is not int for v in vals[1:]): raise ValueError("case outside frozen inventory")
 joint=30+vals[1]; ambient=7+joint*50+vals[2]
 if (case.get("joint_task_number")!=joint or case.get("ambient_seed")!=ambient
         or case.get("policy_seed")!=ambient+vals[3]*1000000007 or case.get("policy_id")!="base"):
  raise ValueError("case identity/seed/protocol invalid")
 return vals

def prov(v):
 raw=v.get("runner",{}).get("report",{}).get("skills",{}).get("pi05",{})
 return raw.get("provenance",raw)

def audit(v,which,identity):
 e=[]; report=v.get("runner",{}).get("report",{}); trace=report.get("trace",[]); calls=v.get("runner",{}).get("policy_calls",[]); receipts=v.get("runner",{}).get("payload_hashes",[])
 task=f"{v['case']['suite']}/{v['case']['task_id']}"; original=v["harness"]["stages"][0]["instruction"]
 if report.get("steps")!=len(trace) or not 0<len(trace)<=CAP or [r.get("step") for r in trace]!=list(range(len(trace))): e.append("trace invalid")
 if len(calls)!=len(receipts) or len(calls)!=(len(trace)+4)//5: e.append("call count invalid")
 for i,(c,r) in enumerate(zip(calls,receipts)):
  seed=v["case"]["policy_seed"]+i*1000003
  if (c.get("inference_call")!=i or r.get("inference_call")!=i or c.get("policy_seed")!=seed or r.get("policy_seed")!=seed
      or c.get("metadata")!=identity or c.get("response_valid") is not True or r.get("status")!="ok" or r.get("policy_id")!="base"):
   e.append("call/seed/identity invalid"); break
 if which=="control":
  if any(r.get("prompt")!=original for r in receipts): e.append("control prompt changed")
  return e
 p=prov(v); emitted=p.get("emitted_rows",[]); executed=p.get("executed_rows",[]); prompts=p.get("call_prompt_receipts",[])
 if (p.get("kind")!="transport_release_semantic_resume_v1" or p.get("parameters")!=PARAMS
     or p.get("mechanical_proxy_only") is not True or p.get("object_or_task_completion_certificate") is not False
     or p.get("execution_reconciled") is not True or len(executed)!=len(trace) or len(prompts)!=len(calls)):
  e.append("semantic provenance invalid")
 for i,(pr,receipt) in enumerate(zip(prompts,receipts)):
  expected_prompt=RESUME[task] if pr.get("prompt_changed") else original
  if pr!={"inference_call":i,"actual_step":i*5,"prompt":expected_prompt,"prompt_changed":pr.get("prompt_changed")} or receipt.get("prompt")!=expected_prompt:
   e.append("call prompt receipt invalid"); break
 change=p.get("first_prompt_change_actual_step"); change_call=p.get("first_prompt_change_call"); witness=p.get("witness_actual_step")
 changed=[i for i,r in enumerate(prompts) if r.get("prompt_changed") is True]
 if changed:
  first=changed[0]
  if (changed!=list(range(first,len(prompts))) or change_call!=first or change!=first*5
      or type(witness) is not int or not witness<=change<=witness+4 or CAP-change<80): e.append("prompt-change timing invalid")
 else:
  if change is not None or change_call is not None: e.append("change recorded without changed prompt")
 for i,row in enumerate(executed):
  if (row.get("execution_index")!=i or row.get("expected_actual_step")!=i or row.get("executed") is not True
      or row.get("action")!=trace[i].get("action")): e.append("executed/native trace mismatch"); break
  if i+1<len(trace) and row.get("post_state8")!=trace[i+1].get("state"): e.append("post-state alignment mismatch"); break
 by={r.get("emission_index"):r for r in executed}
 for row in emitted:
  if row.get("emission_index") not in by and row.get("truncated_before_execution") is not True: e.append("truncation unrecorded"); break
 return e

def input_receipt(r):
 return {k:copy.deepcopy(r.get(k)) for k in ("inference_call","observation_image","observation_wrist_image","observation_state","policy_id","policy_seed","status")}

def aggregate(roots,routes):
 errors=[]; episodes={}; route=json.loads(Path(routes).read_text()); expected_identity={"policy_id":"base",**route["identities"]["base"],"policy_seed_protocol":PROTOCOL}
 if len(roots)!=4: errors.append("expected four workers")
 for root in roots:
  ctl=json.loads((root/"controller.json").read_text()); batches=ctl.get("batches",[])
  if ctl.get("status")!="complete" or len(batches)!=2 or [arm(x.get("name","")) for x in batches]!=list(ARMS) or len({x.get("server_pid") for x in batches})!=1 or batches[1].get("service_reused_from_previous_batch") is not True: errors.append(f"{root}: controller invalid")
  for sp in sorted(root.glob("*/summary.json")):
   which=arm(sp.parent.name); summary=json.loads(sp.read_text()); manifest=json.loads((sp.parent/"manifest.json").read_text()); identity=manifest.get("verified_service_identity")
   if identity!=expected_identity or summary.get("complete") is not True or summary.get("errors")!=0 or summary.get("planned")!=summary.get("completed") or summary.get("planned")!=23: errors.append(f"{sp}: summary/identity invalid")
   for row in summary.get("cases",[]):
    ep=json.loads((sp.parent/row["episode"]).read_text()); k=key(ep["case"])
    if k!=key(row) or row.get("success") is not ep["runner"]["report"].get("success"): errors.append(f"{sp}: row mismatch")
    if (which,k) in episodes: raise ValueError("duplicate episode")
    errors += [f"{k}/{which}: {x}" for x in audit(ep,which,identity)]; episodes[(which,k)]=ep
 if set(episodes)!={(a,k) for a in ARMS for k in EXPECTED}: errors.append("coverage is not exact 92 x 2")
 stats=Counter(); pairs=[]
 for k in sorted(EXPECTED):
  if any((a,k) not in episodes for a in ARMS): continue
  c,a=episodes[("control",k)],episodes[("semantic_resume",k)]; pc=prov(a); change=pc.get("first_prompt_change_actual_step"); call=pc.get("first_prompt_change_call")
  ci,ai=initial_receipt(c),initial_receipt(a)
  try: valid=valid_initial(ci) and valid_initial(ai)
  except (TypeError,AttributeError): valid=False
  if not valid or ci!=ai: errors.append(f"{k}: initial observations differ")
  if c["runner"]["report"]["trace"][:change] != a["runner"]["report"]["trace"][:change]: errors.append(f"{k}: pre-change trace differs")
  cr,ar=c["runner"]["payload_hashes"],a["runner"]["payload_hashes"]
  if cr[:call]!=ar[:call] or (call is not None and input_receipt(cr[call])!=input_receipt(ar[call])): errors.append(f"{k}: pre-change call inputs differ")
  if change is None and (c["runner"]["report"]["trace"]!=a["runner"]["report"]["trace"] or cr!=ar): errors.append(f"{k}: no-change pair differs")
  cs=bool(c["runner"]["report"]["success"]); ass=bool(a["runner"]["report"]["success"]); label="both" if cs and ass else "recovered" if ass else "regressed" if cs else "neither"; stats[label]+=1
  pairs.append({"case":k,"control_success":cs,"semantic_resume_success":ass,"outcome":label,"first_prompt_change_actual_step":change})
 return {"schema":"pi05_transport_release_semantic_resume.aggregate.v1","complete":not errors,"coverage":{"expected_episodes":184,"observed_episodes":len(episodes),"errors":errors},"paired_outcomes":{k:stats[k] for k in ("both","recovered","regressed","neither")},"pairs":pairs,"evidence_limit":"physical proxy only; not an object or task completion certificate"}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--worker-output",action="append",type=Path,required=True); p.add_argument("--routes",type=Path,default=Path("configs/pi05_harness/routes_base.json")); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 if a.output.exists(): raise FileExistsError("output is create-only")
 result=aggregate(a.worker_output,a.routes); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return 0 if result["complete"] else 1
if __name__=="__main__": raise SystemExit(main())
