#!/usr/bin/env python3
"""Authorized synthetic GPU gradient smoke; never writes trained weights."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from openpi.training.execution_memory_flow import MemoryFlowAdapter,memory_flow_loss


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    if not a.execute:print('preflight_only: synthetic GPU smoke not executed');return
    if os.environ.get('PI05_MEMORY_AUTHORIZATION')!='CONFIRMED':raise ValueError('memory GPU execution not confirmed')
    if a.output.exists():raise FileExistsError(a.output)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one actual CUDA GPU required')
    torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    device='cuda:0';model=MemoryFlowAdapter().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    history=torch.randn(16,32,23,device=device);lengths=torch.tensor([0,1,8,32]*4,device=device)
    inputs={'frozen_hidden':torch.randn(16,10,1024,device=device),'time':torch.tensor([1.,.9,.8,.7]*4,device=device),
        'history':history,'history_mask':torch.arange(32,device=device)[None]<lengths[:,None],
        'base_velocity':torch.randn(16,10,32,device=device),'target_velocity7':torch.randn(16,10,7,device=device)}
    losses=[]
    for _ in range(2):
        loss,_=memory_flow_loss(model,**inputs)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite smoke loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
        losses.append(float(loss))
    gradient=model.memory.gru.weight_hh_l0.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum()==0:
        raise RuntimeError('GPU recurrent memory gradient missing')
    a.output.write_text(json.dumps({'status':'passed','fixture':'synthetic_not_demonstrations','weights_saved':False,
        'cuda_device':torch.cuda.get_device_name(0),'losses':losses,'history_gradient_finite_nonzero':True,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')


if __name__=='__main__':main()
