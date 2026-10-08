#!/usr/bin/env python3
"""Train four frozen-base PI0.5 temporal physical-residual heads."""
from __future__ import annotations

import argparse, hashlib, json, os, signal
from pathlib import Path
import sys
import numpy as np

from scripts import train_four_plugins as legacy
from openpi.training.physical_residual import balanced_task_batch
from openpi.training import physical_residual_bank as bank

TASK_NAME="PI05-LIBERO-TEST-V2";TOTAL_UPDATES=40_000;UPDATES_PER_HEAD=10_000
SAVE_EVERY=20_000;EVAL_EVERY=2_000;SMOKE_SAVE_STEP=4;BATCH_SIZE=40;SEED=42

def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--base-checkpoint",type=Path,default=legacy.DEFAULT_BASE)
    p.add_argument("--data-root",type=Path,default=legacy.DEFAULT_DATA);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--preflight-only",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--from-checkpoint",type=Path)
    p.add_argument("--seed",type=int,default=SEED);p.add_argument("--batch-size",type=int,default=BATCH_SIZE);p.add_argument("--num-workers",type=int,default=2)
    return p.parse_args(argv)

def intervals_by_task(dataset):
    result={};start=0
    for end,episode in zip(dataset._ends.tolist(),dataset.episodes,strict=True):
        result.setdefault(episode.task,[]).append((start,int(end)-start));start=int(end)
    if len(result)!=10:return (_ for _ in ()).throw(ValueError("suite dataset must contain exactly ten tasks"))
    return result

class BalancedSampler:
    def __init__(self,dataset,seed,start_step=0):self.intervals=intervals_by_task(dataset);self.seed=seed;self.step=start_step
    def __iter__(self):
        while True:
            indices,_=balanced_task_batch(self.intervals,seed=self.seed,step=self.step);self.step+=1;yield indices.tolist()
    def __len__(self):return sys.maxsize

class TaggedTransformedDataset:
    def __init__(self,dataset,transform):self.dataset,self.transform=dataset,transform;self.tasks=sorted(intervals_by_task(dataset))
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        pos=int(np.searchsorted(self.dataset._ends,index,side="right"));episode=self.dataset.episodes[pos];task=episode.task
        start=0 if pos==0 else int(self.dataset._ends[pos-1]);frame=index-start
        value=self.transform(dict(self.dataset[index]));value["v2_task_id"]=np.int32(self.tasks.index(task))
        value["v2_valid_horizon"]=(np.arange(10)<episode.length-frame).astype(np.float32);return value

def make_loaders(args,datasets,transform,steps,train):
    from torch.utils.data import DataLoader
    out={}
    for i,suite in enumerate(bank.SUITES):
        raw=datasets[suite];ds=TaggedTransformedDataset(raw,transform)
        sampler=(BalancedSampler(raw,args.seed+i,steps[suite]) if train
                 else legacy.FixedIndexBatchSampler(raw.diagnostic_indices(80)))
        workers=args.num_workers if train else 0
        out[suite]=DataLoader(ds,batch_sampler=sampler,num_workers=workers,collate_fn=legacy.numpy_collate,
          multiprocessing_context="spawn" if workers else None,persistent_workers=bool(workers),
          worker_init_fn=legacy.worker_init_cpu_only if workers else None)
    return out

def batch_parts(batch):
    batch=dict(batch);tasks=batch.pop("v2_task_id");valid=batch.pop("v2_valid_horizon")
    obs,actions=legacy.observation_and_actions(batch);return obs,actions,tasks,valid

def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

def main():
    args=parse_args();
    git_sha=legacy.current_git_sha();commit_provenance=legacy.execution_commit_provenance()
    if args.batch_size!=40:raise ValueError("V2 requires global batch 40 (10 tasks x 4)")
    if args.seed!=42:raise ValueError("V2 frozen protocol requires seed 42")
    if args.resume != bool(args.from_checkpoint):raise ValueError("--resume and --from-checkpoint must be paired")
    norm_path,_,norm_hash,base_hash=legacy.require_inputs(args)
    train,val,dataset_manifest=legacy.load_datasets(args)
    if dataset_manifest.get("seed")!=42 or dataset_manifest.get("holdout_per_task")!=2:raise ValueError("dataset split must remain seed42/two demos per task")
    transform=legacy.make_transform(norm_path);zero={s:0 for s in bank.SUITES}
    if args.preflight_only:
        loaders=make_loaders(args,train,transform,zero,True)
        report={"task_name":TASK_NAME,"git_sha":git_sha,**commit_provenance,"dataset_manifest":dataset_manifest,"batches":{}}
        for suite in bank.SUITES:
            _,actions,tasks,valid=batch_parts(next(iter(loaders[suite])));counts=np.bincount(tasks,minlength=10)
            report["batches"][suite]={"actions":list(actions.shape),"task_counts":counts.tolist(),"valid_rows":int(valid.sum())}
            if actions.shape!=(40,10,32) or counts.tolist()!=[4]*10:raise ValueError("balanced transformed batch contract failed")
        print(json.dumps(report,sort_keys=True));return
    if os.environ.get("JAX_PROCESS_COUNT","1")!="1":raise RuntimeError("single-host process required")
    if "WANDB_API_KEY" not in os.environ:raise RuntimeError("WANDB_API_KEY must be injected")
    if args.output_dir.exists() and not args.resume:raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    import jax, optax, wandb
    from flax import nnx
    from openpi.training import sharding
    if len(jax.devices())!=8:raise RuntimeError("V2 requires exactly 8 devices")
    mesh=sharding.make_mesh(8);params_path=args.base_checkpoint/"params"
    base_graphdef,frozen=bank.initialize_native_base(str(params_path),mesh)
    base_model=nnx.merge(base_graphdef,frozen)
    feature_dim=int(base_model.action_out_proj.in_features)
    head_config={"feature_dim":feature_dim,"state_dim":8,"width":256,"horizon":10,"heads":4,"ffn_dim":1024,"residual_bound":1.0}
    head_graphdef,heads=bank.initialize_head_bank(feature_dim,args.seed)
    schedule=optax.warmup_cosine_decay_schedule(0,1e-4,500,UPDATES_PER_HEAD,1e-5)
    tx=optax.chain(optax.clip_by_global_norm(1),optax.adamw(
      schedule,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-4))
    opts=bank.initialize_optimizer_states(tx,heads);steps={s:0 for s in bank.SUITES}
    manifest={"schema":"pi05_residual_train.v1","task_name":TASK_NAME,"git_sha":git_sha,**commit_provenance,
      "seed":42,"total_updates":TOTAL_UPDATES,
      "updates_per_head":UPDATES_PER_HEAD,"batch_size":40,"holdout_every":EVAL_EVERY,"checkpoint_every":SAVE_EVERY,
      "smoke_checkpoint":{"global_step":SMOKE_SAVE_STEP,"directory":"checkpoints_smoke","not_regular_checkpoint":True},
      "optimizer":{"name":"adamw","peak_lr":1e-4,"end_lr":1e-5,"warmup_per_head":500,"b1":.9,"b2":.95,"eps":1e-8,"weight_decay":1e-4,"clip":1,"ema":False},
      "head_config":head_config,"dataset_manifest":dataset_manifest,"base_inventory_sha256":base_hash,"norm_stats_sha256":norm_hash,
      "gate_semantics":"FM surrogate; not rollout success"};manifest["manifest_sha256"]=canonical_hash(manifest)
    manifest_path=args.output_dir/"run_manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text())!=manifest:raise ValueError("resume manifest mismatch")
        heads,opts,steps,saved=bank.load_head_bank(args.from_checkpoint,heads,opts,expected_base_checkpoint_path=str(params_path),
          expected_norm_stats_sha256=norm_hash,expected_base_manifest_sha256=base_hash,expected_head_config=head_config)
        if saved.get("metadata",{}).get("run_manifest_sha256")!=manifest["manifest_sha256"]:raise ValueError("resume checkpoint/run manifest mismatch")
        expected_steps={suite:(sum(steps.values())+3-index)//4 for index,suite in enumerate(bank.SUITES)}
        if steps!=expected_steps:raise ValueError("resume head counters violate deterministic round-robin")
    else:manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    heads=jax.device_put(heads,sharding.fsdp_sharding(heads,mesh));opts=jax.device_put(opts,sharding.fsdp_sharding(opts,mesh))
    loaders=make_loaders(args,train,transform,steps,True);iters={s:iter(loaders[s]) for s in bank.SUITES}
    val_loaders=make_loaders(args,val,transform,zero,False);val_batches={s:batch_parts(next(iter(val_loaders[s]))) for s in bank.SUITES}
    step_fn=bank.make_sharded_head_step(base_graphdef,head_graphdef,tx,mesh)
    eval_fn=bank.make_sharded_head_eval(base_graphdef,head_graphdef,mesh)
    run=wandb.init(project="physicalrsi",name=TASK_NAME,config=manifest);stop=False
    def request_stop(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    global_step=sum(steps.values())
    while global_step<TOTAL_UPDATES:
        suite=bank.SUITES[global_step%4];obs,actions,tasks,valid=batch_parts(next(iters[suite]));obs,actions,tasks,valid=legacy.put_batch_on_mesh((obs,actions,tasks,valid),mesh)
        rng=jax.random.fold_in(jax.random.key(args.seed),global_step)
        heads[suite],opts[suite],metrics=step_fn(frozen,heads[suite],opts[suite],obs,actions,tasks,valid,rng,steps[suite])
        steps[suite]+=1;global_step+=1;run.log({f"train/{suite}/{k}":float(np.asarray(jax.device_get(v))) for k,v in metrics.items()},step=global_step)
        if global_step%EVAL_EVERY==0:
            # Holdout is diagnostic only; no optimizer update and no success claim.
            logs={}
            for name in bank.SUITES:
                vo,va,vt,vv=legacy.put_batch_on_mesh(val_batches[name],mesh)
                value,metric=eval_fn(frozen,heads[name],vo,va,vt,vv,
                  jax.random.fold_in(jax.random.key(args.seed),20_000+global_step),steps[name])
                logs[f"holdout/{name}/loss"]=float(np.asarray(jax.device_get(value)))
            run.log(logs,step=global_step)
        if global_step==SMOKE_SAVE_STEP or global_step%SAVE_EVERY==0 or stop:
            checkpoint_group="checkpoints_smoke" if global_step==SMOKE_SAVE_STEP else "checkpoints"
            bank.save_head_bank(args.output_dir/checkpoint_group/f"step_{global_step:08d}",heads,opts,steps,
              base_checkpoint_path=str(params_path),norm_stats_sha256=norm_hash,base_manifest_sha256=base_hash,
              head_config=head_config,metadata={"run_manifest_sha256":manifest["manifest_sha256"]})
        if stop:break
    run.finish()

if __name__=="__main__":main()
