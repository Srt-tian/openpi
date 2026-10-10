import json
import numpy as np
import pytest
from openpi.training.memory_feature_cache import CacheWriter,CacheReader,bound_path
from train_pi05_execution_memory import batch,balanced_indices


def make_cache(path):
    writer=CacheWriter(path,source='c'*40,base_sha='a'*64,norm_sha='b'*64,split_sha='d'*64,dataset_meta={},shard_size=3)
    features={'hidden':np.zeros((4,10,1024),np.float32),'base_velocity':np.zeros((4,10,32),np.float32),
        'target_velocity7':np.ones((4,10,7),np.float32),'time':np.array([1.,.9,.8,.7],np.float32)}
    for split,offset in [('train',0),('val',10)]:
        for task in range(10):
            episode=task+offset
            writer.episode(split,episode,task,np.arange(4*23,dtype=np.float32).reshape(4,23))
            writer.append({'split':split,'episode':episode,'frame':2,'task':task,
                'key':f'long/{split}/episode:{episode}/frame:2'},features)
    writer.finish({'train':10,'val':10})
    return CacheReader(path)


def test_sharded_roundtrip_and_no_future_history_access(tmp_path):
    reader=make_cache(tmp_path/'cache')
    before=reader.item('train',0,0)
    assert before['hidden'].shape==(10,1024) and before['history'].shape==(2,23)
    reader.histories['train:0'][2:]=999
    after=reader.item('train',0,0)
    np.testing.assert_array_equal(before['history'],after['history'])
    before['history'][:]=888
    assert not (reader.item('train',0,0)['history']==888).all()


def test_hash_and_incomplete_cache_rejection(tmp_path):
    path=tmp_path/'cache';reader=make_cache(path)
    shard=next(iter(reader.manifest['shards']))
    with (path/shard).open('ab') as stream:stream.write(b'corrupted')
    with pytest.raises(ValueError,match='hash'):CacheReader(path)
    manifest=reader.manifest;manifest['status']='building'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incomplete'):CacheReader(path)


def test_duplicate_positions_and_split_overlap_rejected(tmp_path):
    path=tmp_path/'cache';reader=make_cache(path);manifest=reader.manifest
    manifest['rows']['train'][1]['shard']=manifest['rows']['train'][0]['shard']
    manifest['rows']['train'][1]['offset']=manifest['rows']['train'][0]['offset']
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='duplicate'):CacheReader(path)
    with pytest.raises(ValueError,match='escapes'):bound_path(path,'../outside.npz')


def test_rotation_balances_all_ten_tasks_and_cpu_batch(tmp_path):
    reader=make_cache(tmp_path/'cache');rng=np.random.default_rng(42);counts=np.zeros(10,int)
    for step in range(1,11):
        ids=balanced_indices(reader.rows['train'],rng,step)
        for i in ids:counts[reader.rows['train'][i]['task']]+=1
    assert np.array_equal(counts,np.full(10,16))
    values=batch(reader,'train',[0,1],[0,3],'cpu')
    assert tuple(values['frozen_hidden'].shape)==(2,10,1024)
    assert tuple(values['history'].shape)==(2,2,23)
    assert values['history_mask'].all()
    np.testing.assert_array_equal(values['time'].numpy(),np.array([1.,.7],np.float32))


def test_fixture_cache_to_loss_optimizer_and_safe_weight_roundtrip(tmp_path):
    import torch
    from safetensors.torch import save_file,load_file
    from openpi.training.execution_memory_flow import MemoryFlowAdapter,memory_flow_loss
    reader=make_cache(tmp_path/'cache');inputs=batch(reader,'train',[0,1],[0,3],'cpu')
    torch.manual_seed(42);model=MemoryFlowAdapter(width=16,memory_hidden=16)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss,_=memory_flow_loss(model,**inputs)
    optimizer.zero_grad();loss.backward();optimizer.step()
    after,_=memory_flow_loss(model,**inputs)
    assert after<loss
    optimizer.zero_grad();after.backward()
    assert model.memory.gru.weight_ih_l0.grad.abs().sum()>0
    assert np.array_equal(reader.features[reader.rows['train'][0]['shard']]['hidden'],np.zeros((3,4,10,1024),np.float32))
    checkpoint=tmp_path/'fixture.safetensors'
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
    restored=MemoryFlowAdapter(width=16,memory_hidden=16);restored.load_state_dict(load_file(str(checkpoint)),strict=True)
    model.eval();restored.eval()
    torch.testing.assert_close(model(inputs['frozen_hidden'],inputs['time'],inputs['history'],inputs['history_mask']),
        restored(inputs['frozen_hidden'],inputs['time'],inputs['history'],inputs['history_mask']))
