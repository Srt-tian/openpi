#!/usr/bin/env python3
"""Memory adapter pilot from hash-bound offline features; no backbone updates."""
import argparse,json,os,subprocess
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from safetensors.torch import save_file
from openpi.training.execution_memory_flow import MemoryFlowAdapter,memory_flow_loss
from openpi.training.memory_feature_cache import CacheReader,sha


def batch(reader,split,indices,times,device):
    items=[reader.item(split,int(i),int(t)) for i,t in zip(indices,times,strict=True)]
    length=max(len(x['history']) for x in items)
    history=np.zeros((len(items),length,23),np.float32);mask=np.zeros((len(items),length),np.bool_)
    for i,item in enumerate(items):
        n=len(item['history']);history[i,:n]=item['history'];mask[i,:n]=True
    values={k:torch.from_numpy(np.stack([x[k] for x in items])).to(device) for k in ('hidden','base_velocity','target_velocity7')}
    return {'frozen_hidden':values['hidden'],'base_velocity':values['base_velocity'],'target_velocity7':values['target_velocity7'],
        'time':torch.tensor([x['time'] for x in items],device=device),
        'history':torch.from_numpy(history).to(device),'history_mask':torch.from_numpy(mask).to(device)}


def task_groups(rows):
    groups={}
    for i,row in enumerate(rows):groups.setdefault(row['task'],[]).append(i)
    return groups


def balanced_indices(rows,rng,step,count=16,groups=None):
    groups=task_groups(rows) if groups is None else groups
    tasks=sorted(groups);cycle=np.roll(tasks,-((step-1)*count)%len(tasks))
    return [int(rng.choice(groups[int(t)])) for t in np.resize(cycle,count)]


def main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-source','base-sha256','norm-sha256','split-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    reader=CacheReader(args.cache);m=reader.manifest
    if (m['source_commit'],m['base_sha256'],m['norm_sha256'],m['split_manifest_sha256'])!=(args.expected_source,args.base_sha256,args.norm_sha256,args.split_sha256):
        raise ValueError('memory cache training binding mismatch')
    root=Path(__file__).resolve().parents[1]
    source=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if source!=args.expected_source:raise ValueError('source does not match cache')
    if args.output.exists():raise FileExistsError(args.output)
    print(json.dumps({'mode':'execute' if args.execute else 'preflight_only','train_rows':len(reader.rows['train']),
        'val_rows':len(reader.rows['val']),'updates':3000,'batch':16,'seed':42,'lr':1e-4},sort_keys=True),flush=True)
    if not args.execute:return
    if os.environ.get('PI05_MEMORY_AUTHORIZATION')!='CONFIRMED':raise ValueError('specific training confirmation required')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('requires exactly one visible CUDA GPU')
    torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    model=MemoryFlowAdapter().to('cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(42);vrng=np.random.default_rng(1042)
    train_groups=task_groups(reader.rows['train'])
    valid_ids=balanced_indices(reader.rows['val'],vrng,1,32);valid_times=vrng.integers(0,4,32)
    validation=batch(reader,'val',valid_ids,valid_times,'cuda:0')
    args.output.mkdir(parents=True);receipt={'schema':'pi05-memory-training.v1','status':'running','source_commit':source,
        'cache_manifest_sha256':sha(args.cache/'manifest.json'),'backbone':'frozen_offline_features_only',
        'optimizer':{'name':'AdamW','lr':1e-4,'weight_decay':1e-4,'clip':1,'batch':16,'updates':3000,'seed':42},
        'selection':'fixed_final_step3000_not_best_validation','metrics':[]}
    status=args.output/'manifest.json'
    try:
        for step in range(1,3001):
            ids=balanced_indices(reader.rows['train'],rng,step,groups=train_groups)
            inputs=batch(reader,'train',ids,rng.integers(0,4,len(ids)),'cuda:0')
            model.train();loss,_=memory_flow_loss(model,**inputs)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            if step%500==0:
                checkpoint=args.output/f'step_{step:05d}.safetensors'
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
                model.eval()
                with torch.no_grad():val,_=memory_flow_loss(model,**validation)
                if not torch.isfinite(val):raise RuntimeError('nonfinite validation loss')
                receipt['metrics'].append({'step':step,'train_mse_regularized':float(loss),
                    'fixed32_validation_loss_not_success_rate':float(val),'checkpoint':checkpoint.name,'sha256':sha(checkpoint)})
                status.write_text(json.dumps(receipt,indent=2)+'\n')
        receipt['status']='complete'
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__);raise
    finally:status.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
