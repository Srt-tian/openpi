import pytest
import torch
from openpi.training.execution_memory_flow import (
    ExecutionMemoryEncoder, MemoryFlowAdapter, apply_early_velocity, memory_flow_loss)


def test_batch_history_matches_streamed_execution_and_empty_history():
    encoder=ExecutionMemoryEncoder(16)
    history=torch.randn(2,5,23);mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
    batched=encoder(history,mask)
    for i,length in enumerate([5,3]):
        h=torch.zeros(1,16)
        for step in range(length):h=encoder.advance(history[i:i+1,step],h)
        torch.testing.assert_close(h[0],batched[i])
    assert torch.equal(encoder(torch.zeros(2,0,23),torch.zeros(2,0,dtype=torch.bool)),torch.zeros(2,16))
    changed=history.clone();changed[1,3:]=1000
    torch.testing.assert_close(encoder(changed,mask),batched)


def test_history_rejects_holes_and_unbounded_or_nonfinite_inputs():
    model=ExecutionMemoryEncoder(16)
    with pytest.raises(ValueError,match='padding'):
        model(torch.zeros(1,3,23),torch.tensor([[True,False,True]]))
    with pytest.raises(ValueError):model(torch.zeros(1,521,23),torch.ones(1,521,dtype=torch.bool))
    with pytest.raises(ValueError):model(torch.full((1,1,23),float('nan')),torch.ones(1,1,dtype=torch.bool))


def test_zero_initialization_and_actual_first_four_handoff():
    adapter=MemoryFlowAdapter(feature_width=8,width=16,memory_hidden=16)
    hidden=torch.randn(2,10,8);time=torch.tensor([1.,.7]);history=torch.randn(2,3,23);mask=torch.ones(2,3,dtype=torch.bool)
    delta=adapter(hidden,time,history,mask)
    assert torch.equal(delta,torch.zeros(2,10,7))
    base=torch.randn(2,10,32)
    for index in range(10):
        actual=apply_early_velocity(base,delta,denoise_index=index)
        assert torch.equal(actual,base)
        if index>=4:assert actual is base
    assert apply_early_velocity(base,None,denoise_index=None,enabled=False) is base
    active=apply_early_velocity(base,torch.full((2,10,7),.2),denoise_index=0)
    assert torch.equal(active[...,7:],base[...,7:])
    assert apply_early_velocity(base,torch.full((2,10,7),.2),denoise_index=4) is base


def test_training_has_history_gradients_but_no_backbone_or_target_gradients():
    adapter=MemoryFlowAdapter(feature_width=8,width=16,memory_hidden=16)
    # Open zero output projection to test gradients after the first update.
    torch.nn.init.normal_(adapter.output.weight,std=.1)
    hidden=torch.randn(2,10,8,requires_grad=True);base=torch.randn(2,10,32,requires_grad=True)
    target=torch.randn(2,10,7,requires_grad=True);history=torch.randn(2,4,23,requires_grad=True)
    loss,metrics=memory_flow_loss(adapter,frozen_hidden=hidden,time=torch.tensor([1.,.8]),
        history=history,history_mask=torch.ones(2,4,dtype=torch.bool),base_velocity=base,target_velocity7=target)
    loss.backward()
    assert torch.isfinite(loss) and hidden.grad is None and base.grad is None and target.grad is None
    assert history.grad is not None and torch.isfinite(history.grad).all()
    assert history.grad.abs().sum()>0
    assert adapter.memory.cell.weight_hh.grad.abs().sum()>0
    assert set(metrics)=={'flow_mse','base_mse','paired_regret','correction_mse'}


def test_repeat_last_tail_remains_in_loss():
    adapter=MemoryFlowAdapter(feature_width=8,width=16,memory_hidden=16)
    kwargs=dict(frozen_hidden=torch.zeros(1,10,8),time=torch.ones(1),
        history=torch.zeros(1,0,23),history_mask=torch.zeros(1,0,dtype=torch.bool),base_velocity=torch.zeros(1,10,32))
    target=torch.zeros(1,10,7);target[:,9]=1
    loss,_=memory_flow_loss(adapter,target_velocity7=target,**kwargs)
    # Zero correction has exactly base error, so paired regret is zero.
    torch.testing.assert_close(loss,torch.tensor(.1))
    with pytest.raises(ValueError,match='early'):
        memory_flow_loss(adapter,target_velocity7=target,**{**kwargs,'time':torch.tensor([.1])})
