#!/usr/bin/env python3
"""Strict score and causal-attribution aggregation for paired LIBERO-400."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

import aggregate_pi05_closed_dwell_lift as dwell
import aggregate_pi05_response_probe as response
from aggregate_pi05_goal3_phase_screen import initial_receipt, valid_initial

ARMS=("control","candidate")
SUITES=("libero_spatial","libero_object","libero_goal","libero_10")
GOAL={"kind":"response_probe_v1","enabled":True,"lift_z_command":.2,
      "max_lift_steps":20,"lift_target_m":.025,"native_reserve_steps":80,
      "minimum_actual":100}
LONG={"kind":"closed_dwell_lift_v1","enabled":True,"veto_native_upward_intent":True}
HEX=set("0123456789abcdef")
NOISE_PROTOCOL="paired_episode_plus_call_1000003_numpy_pcg64_noise_10x32_f32_v1"
DEFAULT_ROUTES=Path(__file__).resolve().parents[1]/"configs/pi05_harness/routes_base.json"

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def valid_hash(value):return isinstance(value,str) and len(value)==64 and set(value)<=HEX

def expected_identity(routes:Path):
    document=json.loads(routes.read_text());base=document.get("identities",{}).get("base")
    if (not isinstance(base,dict) or base.get("base_graph")!="original_pi05_libero"
            or base.get("adapter_sha256") is not None or not valid_hash(base.get("checkpoint_sha256"))):
        raise ValueError("frozen base route identity invalid")
    return {**base,"policy_id":"base","policy_seed_protocol":NOISE_PROTOCOL}

def valid_identity(value,expected):return isinstance(value,dict) and value==expected

def load_receipt(path:Path,root:Path):
    value=json.loads(path.read_text())
    if (value.get("schema")!="pi05_harness_artifact_inventory.v1"
            or value.get("worker_output_name")!=root.name
            or value.get("controller_sha256")!=sha256(root/"controller.json")):
        raise ValueError("artifact receipt worker/controller binding invalid")
    artifacts={}
    for row in value.get("artifacts",[]):
        episode=row.get("episode_path");video=row.get("video_path")
        if (not isinstance(episode,str) or episode in artifacts or not isinstance(video,str)
                or Path(episode).is_absolute() or Path(video).is_absolute()
                or ".." in Path(episode).parts or ".." in Path(video).parts
                or not valid_hash(row.get("episode_sha256")) or not valid_hash(row.get("video_sha256"))
                or type(row.get("video_size")) is not int or row["video_size"]<=0):
            raise ValueError("artifact receipt row invalid")
        artifacts[episode]=row
    return artifacts

def load_plan(path:Path,catalog_path:Path):
    plan=json.loads(path.read_text()); catalog=json.loads(catalog_path.read_text())
    rows={r["key"]:r for r in catalog.get("tasks",[])}; expected={}
    if len(rows)!=40 or plan.get("schema")!="pi05_harness_candidate400_cases.v1":
        raise ValueError("invalid catalog/case-plan schema")
    for row in plan.get("cases",[]):
        key=(row.get("suite"),row.get("task_id"),row.get("init_id"))
        task=f"{key[0]}/{key[1]}"; joint=SUITES.index(key[0])*10+key[1] if key[0] in SUITES and type(key[1]) is int else -1
        if (key in expected or task not in rows or type(key[2]) is not int or not 0<=key[2]<10
                or row.get("replicate_id")!=0 or row.get("joint_task_number")!=joint
                or row.get("official_cap")!=rows[task]["max_steps"]):
            raise ValueError("case plan differs from official LIBERO-400 grid")
        expected[key]={"joint":joint,"cap":rows[task]["max_steps"],"instruction":rows[task]["instruction"]}
    if len(expected)!=400: raise ValueError("case plan is not exact LIBERO-400")
    return expected

def arm(name):
    hits=[x for x in ARMS if x in name]
    if len(hits)!=1: raise ValueError(f"batch name has no unique arm: {name}")
    return hits[0]

def provenance(value):
    raw=value.get("runner",{}).get("report",{}).get("skills",{}).get("pi05",{})
    return raw.get("provenance",raw)

def case_key(case,expected):
    key=(case.get("suite"),case.get("task_id"),case.get("init_id")); spec=expected.get(key)
    ambient=7+spec["joint"]*50+key[2] if spec else None
    integer_fields=(case.get("task_id"),case.get("init_id"),case.get("replicate_id"),
                    case.get("joint_task_number"),case.get("ambient_seed"),case.get("policy_seed"))
    if (any(type(value) is not int for value in integer_fields) or spec is None
            or case.get("replicate_id")!=0 or case.get("joint_task_number")!=spec["joint"]
            or case.get("ambient_seed")!=ambient or case.get("policy_seed")!=ambient
            or case.get("policy_id")!="base"):
        raise ValueError("case identity/seed differs from frozen plan")
    return key

def audit_common(value,which,spec,identity,batch:Path,worker:Path,artifact=None):
    errors=[]; report=value.get("runner",{}).get("report",{}); trace=report.get("trace",[])
    stages=value.get("harness",{}).get("stages",[]); env=report.get("environment",{})
    cap_valid=(type(report.get("steps")) is int and report["steps"]==len(trace)
        and 0<len(trace)<=spec["cap"] and [r.get("step") for r in trace]==list(range(len(trace))))
    if not cap_valid: errors.append("trace/cap invalid")
    harness_valid=(value.get("harness",{}).get("remember")==[] and len(stages)==1
        and (stages[0].get("skill"),stages[0].get("instruction"),stages[0].get("max_steps"),
             stages[0].get("until"),stages[0].get("on_timeout")) == ("pi05",spec["instruction"],spec["cap"],None,"abort")
        and not any(r.get("stage")!=0 or r.get("skill")!="pi05" for r in trace))
    if not harness_valid: errors.append("original harness invalid")
    asset=env.get("init_asset",{})
    source_valid=(env.get("kind")=="pi05_official_libero" and env.get("step_cap")==spec["cap"]
            and env.get("success_decision_source")=="official_libero_env.step.done"
            and isinstance(asset.get("sha256"),str) and len(asset["sha256"])==64)
    if not source_valid:
        errors.append("environment source invalid")
    video=value.get("video",{}); name=video.get("path");video_path=batch/"videos"/name if isinstance(name,str) else batch/"videos"/"invalid"
    expected_relative=video_path.relative_to(worker).as_posix()
    local_present=video_path.is_file()
    if local_present:
        size=video_path.stat().st_size;digest=sha256(video_path)
        receipt_matches=(artifact is None or (artifact.get("video_path")==expected_relative
            and artifact.get("video_size")==size and artifact.get("video_sha256")==digest))
        video_valid=size>0 and receipt_matches
    else:
        size=artifact.get("video_size") if isinstance(artifact,dict) else None
        digest=artifact.get("video_sha256") if isinstance(artifact,dict) else None
        video_valid=(isinstance(artifact,dict) and artifact.get("video_path")==expected_relative
                     and type(size) is int and size>0 and valid_hash(digest))
    video_valid=(video.get("written") is True and isinstance(name,str)
                 and Path(name).name==name and video_valid)
    video_status={"written":video.get("written"),"path":name,"origin_checked":video_valid,
                  "retained_size":size,"retained_sha256":digest}
    if not video_valid: errors.append("video receipt/file invalid")
    calls=value.get("runner",{}).get("policy_calls",[]); receipts=value.get("runner",{}).get("payload_hashes",[])
    if not calls or len(calls)!=len(receipts): errors.append("policy call/hash coverage invalid")
    seed=value.get("case",{}).get("policy_seed")
    for i,(call,receipt) in enumerate(zip(calls,receipts)):
        if (call.get("inference_call")!=i or receipt.get("inference_call")!=i
                or call.get("policy_seed")!=seed+i*1000003 or receipt.get("policy_seed")!=seed+i*1000003
                or call.get("metadata")!=identity or call.get("response_valid") is not True
                or receipt.get("status")!="ok"):
            errors.append("policy source/seed/receipt invalid");break
    task=f'{value.get("case",{}).get("suite")}/{value.get("case",{}).get("task_id")}'
    expected_control=GOAL if which=="candidate" and task=="libero_goal/3" else LONG if which=="candidate" and task=="libero_10/8" else {"kind":"response_probe_v1","enabled":False}
    if value.get("pi05_control")!=expected_control: errors.append("task control snapshot invalid")
    return errors,video_status,{"cap_valid":cap_valid,"harness_valid":harness_valid,
                                "source_valid":source_valid,"video_valid":video_valid}

def pair_gate(control,candidate,key,identity=None):
    task=f"{key[0]}/{key[1]}"
    if task=="libero_goal/3":
        _,_,_,_,audit_errors=response.manual_info(candidate)
        gate=response.causal_pair(control,candidate)
        return audit_errors,gate["causal_gate_pass"],gate["confounds"],gate.get("triggered")
    if task=="libero_10/8":
        confounds=dwell.causal(control,candidate)
        audit=[] if identity is None else dwell.audit_episode(candidate,"assist",identity)
        return audit,not confounds,confounds,provenance(candidate).get("first_changed_action_step") is not None
    cr,ar=control["runner"],candidate["runner"]
    equal=(cr.get("report",{}).get("trace")==ar.get("report",{}).get("trace")
           and cr.get("policy_calls")==ar.get("policy_calls") and cr.get("payload_hashes")==ar.get("payload_hashes"))
    outcomes=(cr.get("report",{}).get("success"),cr.get("report",{}).get("status"))==(ar.get("report",{}).get("success"),ar.get("report",{}).get("status"))
    confounds=[] if equal and outcomes else ["unchanged-task full evidence/outcome differs"]
    return [],not confounds,confounds,False

def summarize(rows):
    outcomes=Counter(r["outcome"] for r in rows)
    return {"pairs":len(rows),"control_successes":sum(r["control_success"] for r in rows),
            "candidate_successes":sum(r["candidate_success"] for r in rows),"paired_outcomes":dict(outcomes)}

def aggregate(roots,case_plan,catalog,artifact_receipts=None,routes=DEFAULT_ROUTES):
    expected=load_plan(case_plan,catalog); episodes={}; structural=[]; summaries=0;manifest_identities=[]
    frozen_identity=expected_identity(Path(routes))
    receipt_maps={};receipt_used={}
    for path in artifact_receipts or []:
        value=json.loads(path.read_text());name=value.get("worker_output_name")
        root=next((r for r in roots if r.name==name),None)
        if root is None or name in receipt_maps:raise ValueError("artifact receipt has unknown/duplicate worker")
        receipt_maps[name]=load_receipt(path,root);receipt_used[name]=set()
    if artifact_receipts is not None and len(receipt_maps)!=len(roots):
        structural.append("artifact receipt coverage is not one per worker")
    if len(roots)!=4: structural.append("expected exactly four workers")
    for root in roots:
        controller=json.loads((root/"controller.json").read_text()); batches=controller.get("batches",[])
        if (controller.get("status")!="complete" or len(batches)!=2 or [arm(x.get("name","")) for x in batches]!=list(ARMS)
                or len({x.get("server_pid") for x in batches})!=1 or batches[1].get("service_reused_from_previous_batch") is not True): structural.append(f"{root}: controller/service invalid")
        for sp in sorted(root.glob("*/summary.json")):
            summaries+=1;which=arm(sp.parent.name);summary=json.loads(sp.read_text());manifest=json.loads((sp.parent/"manifest.json").read_text());identity=manifest.get("verified_service_identity")
            manifest_identities.append(identity)
            if (summary.get("mode")!="harness" or summary.get("complete") is not True or summary.get("errors")!=0 or summary.get("planned")!=summary.get("completed")
                    or manifest.get("fixed_policy_id")!="base" or not valid_identity(identity,frozen_identity)):
                structural.append(f"{sp}: summary/identity invalid")
            for row in summary.get("cases",[]):
                try:key=case_key(row,expected)
                except ValueError as error:structural.append(f"{sp}: {error}");continue
                ep=(sp.parent/row["episode"]).resolve()
                if not ep.is_relative_to(sp.parent.resolve()):structural.append(f"{sp}: path escape");continue
                value=json.loads(ep.read_text())
                try:actual=case_key(value.get("case",{}),expected)
                except ValueError as error:structural.append(f"{ep}: {error}");continue
                if actual!=key or row.get("success") is not value["runner"]["report"].get("success") or row.get("steps")!=value["runner"]["report"].get("steps"):structural.append(f"{ep}: summary mismatch")
                relative=ep.relative_to(root.resolve()).as_posix();artifact=receipt_maps.get(root.name,{}).get(relative)
                if artifact is not None:
                    receipt_used[root.name].add(relative)
                    if artifact.get("episode_sha256")!=sha256(ep):structural.append(f"{ep}: artifact episode hash mismatch")
                errs,video,flags=audit_common(value,which,expected[key],identity,sp.parent,root,artifact)
                structural += [f"{ep}: {e}" for e in errs]
                slot=(which,key)
                if slot in episodes:raise ValueError(f"duplicate {slot}")
                episodes[slot]={"value":value,"identity":identity,"video":video,"flags":flags}
    if summaries!=8:structural.append("expected eight summaries")
    if (not manifest_identities or any(item!=manifest_identities[0] for item in manifest_identities)
            or not valid_identity(manifest_identities[0],frozen_identity)):
        structural.append("verified service identities differ across batches/workers")
    for name,artifacts in receipt_maps.items():
        if receipt_used.get(name,set())!=set(artifacts):structural.append(f"{name}: artifact receipt exact coverage mismatch")
    wanted={(a,k) for a in ARMS for k in expected}
    if set(episodes)!=wanted:structural.append("coverage is not exact 400 pairs x 2")
    pairs=[]
    for key in sorted(expected):
        if any((a,key) not in episodes for a in ARMS):continue
        c,a=episodes["control",key],episodes["candidate",key];cv,av=c["value"],a["value"]
        ci,ai=initial_receipt(cv),initial_receipt(av);obs_ok=valid_initial(ci) and valid_initial(ai) and ci==ai
        identity_ok=c["identity"]==a["identity"]
        seed_ok=(cv["case"]["policy_seed"],cv["case"]["ambient_seed"])==(av["case"]["policy_seed"],av["case"]["ambient_seed"])
        if not identity_ok:structural.append(f"{key}: paired service identity differs")
        if not seed_ok:structural.append(f"{key}: paired seeds differ")
        audit_errors,causal_ok,confounds,triggered=pair_gate(cv,av,key,a["identity"]);structural += [f"{key}: {e}" for e in audit_errors]
        cs,vs=bool(cv["runner"]["report"]["success"]),bool(av["runner"]["report"]["success"])
        label="both" if cs and vs else "recovered" if vs else "regressed" if cs else "neither"
        pairs.append({"suite":key[0],"task_id":key[1],"init_id":key[2],"control_success":cs,
            "candidate_success":vs,"outcome":label,"triggered":triggered,
            "source_identity_equal":identity_ok,"paired_seeds_equal":seed_ok,
            "initial_observation_hashes_equal":obs_ok,
            "paired_noise_protocol_and_seeds_valid":seed_ok and identity_ok
                and c["identity"].get("policy_seed_protocol")==NOISE_PROTOCOL,
            "cap_valid":c["flags"]["cap_valid"] and a["flags"]["cap_valid"],
            "harness_valid":c["flags"]["harness_valid"] and a["flags"]["harness_valid"],
            "source_receipts_valid":c["flags"]["source_valid"] and a["flags"]["source_valid"],
            "video_status":{"control":{**c["video"],"valid":c["flags"]["video_valid"]},
                            "candidate":{**a["video"],"valid":a["flags"]["video_valid"]}},
            "causal_gate_pass":causal_ok and identity_ok and seed_ok and obs_ok,
            "causal_confounds":confounds + ([] if obs_ok else ["initial observation hashes differ"])})
    score_complete=not structural and len(pairs)==400
    exact=score_complete and all(x["causal_gate_pass"] for x in pairs)
    by_suite={s:summarize([x for x in pairs if x["suite"]==s]) for s in SUITES}
    by_task={f"{s}/{t}":summarize([x for x in pairs if x["suite"]==s and x["task_id"]==t]) for s in SUITES for t in range(10)}
    return {"schema":"pi05_harness_candidate400.aggregate.v1","score_complete":score_complete,
        "exact_prefix_attribution_complete":exact,"coverage":{"expected_pairs":400,"observed_pairs":len(pairs),"episodes":len(episodes),"structural_errors":structural},
        "score":summarize(pairs),"by_suite":by_suite,"by_task":by_task,
        "causal_flags":{"passing":sum(x["causal_gate_pass"] for x in pairs),"failing":sum(not x["causal_gate_pass"] for x in pairs)},"pairs":pairs}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker-output",action="append",type=Path,required=True);p.add_argument("--artifact-receipt",action="append",type=Path);p.add_argument("--case-plan",type=Path,required=True);p.add_argument("--task-catalog",type=Path,required=True);p.add_argument("--routes",type=Path,default=DEFAULT_ROUTES);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError("output is create-only")
    result=aggregate(a.worker_output,a.case_plan,a.task_catalog,a.artifact_receipt,a.routes);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");raise SystemExit(0 if result["score_complete"] else 1)
