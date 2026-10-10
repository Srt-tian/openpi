import copy
import numpy as np
import pytest
import torch

from openpi.training import demo_support_cfn as cfn


def schema():
    return {"base_sha256":"a"*64,"input_dim":8,"width":16,"output_dim":64,
      "pooling":"provided","layer":"provided","time":"provided","noise":"provided",
      "action_normalization":"base_quantile_no_clip_physical7","executed_horizon":5}


def test_coin_labels_deterministic_64_and_no_global_rng_mutation():
    assert torch.equal(cfn.coin_label('episode:3/frame:7'),cfn.coin_label('episode:3/frame:7'))
    assert cfn.coin_label('x').shape==(64,) and set(cfn.coin_label('x').tolist())=={-1,1}
    torch.manual_seed(91);before=torch.random.get_rng_state();cfn.DemoSupportCFN(8,16);after=torch.random.get_rng_state()
    assert torch.equal(before,after)


def test_prior_frozen_learned_gradient_finite_and_inference_immutable():
    model=cfn.DemoSupportCFN(8,16);x=torch.randn(4,8)
    with pytest.raises(RuntimeError,match='calibration'): model.loss(x,['a','b','c','d'])
    model.calibrate(x)
    loss=model.loss(x,['a','b','c','d']);loss.backward()
    assert all(p.grad is None for p in model.prior.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.learned.parameters())
    before=copy.deepcopy(model.state_dict())
    a=model.support_proxy(x);b=model.support_proxy(x)
    assert torch.equal(a,b) and all(torch.equal(before[k],model.state_dict()[k]) for k in before)
    manual=model.learned(x)+(model.prior(x)-model.calibration.mean)/torch.sqrt(model.calibration.variance())
    torch.testing.assert_close(model(x),manual)
    torch.testing.assert_close(a,-torch.mean(manual**2,dim=-1))


def test_harness_candidate_contract_with_provided_features():
    from pi05_capability_harness import CandidateProtocol, select_candidate
    model=cfn.DemoSupportCFN(8,16);model.calibrate(torch.randn(4,8)); protocol=CandidateProtocol(3);seeds=protocol.seeds(7,2)
    chunks=[np.zeros((5,7)) for _ in range(3)]
    chunks[1][0,0]=1;chunks[2][0,0]=2
    scorer=cfn.provided_feature_scorer(model,np.ones((3,8)),chunks)
    index,chosen=select_candidate(chunks,seeds,base_seed=7,call_index=2,protocol=protocol,
      observables={"state8":np.zeros(8)},scorer=scorer)
    assert 0<=index<3 and chosen.shape==(5,7)
    with pytest.raises(ValueError): cfn.provided_feature_scorer(model,np.ones((2,8)),chunks)
    with pytest.raises(ValueError,match='bound'): scorer(tuple(reversed(chunks)),{})


def test_schema_checkpoint_and_unavailable_fail_closed():
    model=cfn.DemoSupportCFN(8,16);model.calibrate(torch.randn(4,8));state=copy.deepcopy(model.state_dict());expected=schema()
    cfn.load_safe_state_dict(model,state,expected,expected)
    bad=dict(expected);bad['time']='guessed'
    with pytest.raises(ValueError,match='schema'): cfn.load_safe_state_dict(model,state,bad,expected)
    broken=dict(state);key=next(iter(broken));broken[key]=broken[key].to(torch.float64)
    with pytest.raises(ValueError,match='tensor'): cfn.load_safe_state_dict(model,broken,expected,expected)
    bad_count=copy.deepcopy(state);bad_count['calibration.count']=torch.zeros((),dtype=torch.int64)
    with pytest.raises(ValueError,match='calibration'): cfn.load_safe_state_dict(model,bad_count,expected,expected)
    nonfinite=copy.deepcopy(state);nonfinite['calibration.mean'][0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'): cfn.load_safe_state_dict(model,nonfinite,expected,expected)
    assert cfn.unavailable_bypass(False) is None
    with pytest.raises(RuntimeError,match='unavailable'): cfn.unavailable_bypass(True)
