#!/usr/bin/env python3
"""Fit demo_support_cfn_v1 from a safe cache; no rollout/success labels."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from openpi.training.demo_support_cfn import DemoSupportCFN

def parse(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-source',required=True);p.add_argument('--base-sha256',required=True);p.add_argument('--norm-sha256',required=True);p.add_argument('--dataset-manifest-sha256',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=2000);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);return p.parse_args(argv)
def load_cache(path):
    m=json.loads((path/'manifest.json').read_text());raw=(path/'features.npz').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=m['features_sha256']: raise ValueError('cache hash mismatch')
    z=np.load(path/'features.npz',allow_pickle=False)
    required={'schema','source_commit','base_sha256','norm_sha256','dataset_manifest_sha256','dataset_meta','feature_schema','selection','parent_target','coverage','features_sha256','arrays','frame_keys'}
    if set(m)!=required or m['schema']!='demo_support_cache.v1':raise ValueError('cache manifest schema mismatch')
    if any(not isinstance(m[k],str) or len(m[k])!=64 for k in ('base_sha256','norm_sha256','dataset_manifest_sha256','features_sha256')):raise ValueError('cache digest mismatch')
    episodes={}
    for split in ('train','val'):
        x=z[split];keys=z[f'{split}_keys'];tasks=z[f'{split}_tasks']
        rows=m['frame_keys'][split];episodes[split]={int(r['episode']) for r in rows}
        if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=1024 or len(x)<1 or list(x.shape)!=m['arrays'][split] or len(x)!=len(keys) or len(x)!=len(tasks) or [str(k) for k in keys]!=[r['key'] for r in rows] or not np.isfinite(x).all(): raise ValueError('invalid cache arrays')
    if set(z['train_keys'])&set(z['val_keys']): raise ValueError('train/val keys overlap')
    if episodes['train']&episodes['val']:raise ValueError('train/val episodes overlap')
    task_ids=sorted(set(map(int,z['train_tasks'])));expected=m['coverage']['tasks']
    if len(task_ids)!=10 or task_ids!=expected or sorted(set(map(int,z['val_tasks'])))!=expected:raise ValueError('task coverage mismatch')
    fs=m['feature_schema']
    if fs.get('schema')!='pi05-demo-support-feature-probe.v1' or fs.get('base_sha256')!=m['base_sha256'] or fs.get('norm_sha256')!=m['norm_sha256'] or fs.get('feature_width')!=1024:raise ValueError('feature schema binding mismatch')
    return m,z
def main(argv=None):
    a=parse(argv)
    if (a.updates,a.batch_size,a.seed,a.lr,a.weight_decay)!=(2000,256,42,1e-4,1e-4): raise ValueError('pilot config is frozen')
    if a.output.exists(): raise FileExistsError(a.output)
    m,z=load_cache(a.cache)
    if (m['source_commit'],m['base_sha256'],m['norm_sha256'],m['dataset_manifest_sha256'])!=(a.expected_source,a.base_sha256,a.norm_sha256,a.dataset_manifest_sha256):raise ValueError('cache runtime binding mismatch')
    status={"status":"ready" if a.execute else "dry_run_only","train":len(z['train']),"val":len(z['val'])}
    if not a.execute: print(json.dumps(status,sort_keys=True));return
    if a.device!='cuda:0' or not torch.cuda.is_available() or torch.cuda.device_count()<1:raise RuntimeError('authorized execution requires cuda:0; no CPU fallback')
    torch.manual_seed(42);model=DemoSupportCFN();train_cpu=torch.from_numpy(z['train']);model.calibrate(train_cpu)
    model=model.to(a.device);train=train_cpu.to(a.device);val_tensor=torch.from_numpy(z['val']).to(a.device)
    opt=torch.optim.AdamW(model.learned.parameters(),lr=1e-4,weight_decay=1e-4);rng=np.random.default_rng(42)
    task_ids=sorted(set(map(int,z['train_tasks'])));bytask={t:np.flatnonzero(z['train_tasks']==t) for t in task_ids}
    a.output.mkdir();metrics=[]
    for step in range(1,2001):
        ids=np.concatenate([rng.choice(bytask[t],26 if i<6 else 25,replace=True) for i,t in enumerate(task_ids)])[:256];rng.shuffle(ids)
        loss=model.loss(train[ids],[str(x) for x in z['train_keys'][ids]]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.learned.parameters(),1);opt.step()
        if step%500==0:
            checkpoint=a.output/f'step_{step:04d}.safetensors'
            save_file({k:v.detach().cpu() for k,v in model.state_dict().items()},str(checkpoint))
            with torch.no_grad(): val=float(model.loss(val_tensor,[str(x) for x in z['val_keys']]))
            metrics.append({"step":step,"train_mse":float(loss),"val_mse_supportive_only":val,
              "checkpoint":checkpoint.name,"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    out={"schema":"demo_support_cfn_train.v1","cache_manifest_sha256":hashlib.sha256((a.cache/'manifest.json').read_bytes()).hexdigest(),
      "source_commit":subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),
      "selection":"fixed_final_step_2000_not_best_of_validation","optimizer":{"name":"AdamW","lr":1e-4,"weight_decay":1e-4,"batch":256,"updates":2000,"clip":1,"seed":42},
      "labels":"deterministic_Rademacher64_by_logical_sample_key","prior_calibration":"train_features_only","task_id_runtime_input":False,
      "claim":"validation MSE is support diagnostic only, not success or advantage","metrics":metrics}
    (a.output/'manifest.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
