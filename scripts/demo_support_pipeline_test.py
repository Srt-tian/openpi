import json,hashlib
import numpy as np
import pytest
import build_pi05_demo_support_cache as cache
import train_demo_support_cfn as trainer
from serve_pi05_feature_probe import FeaturePolicy

def test_compiled_feature_forward_matches_eager_on_cpu():
    import jax.numpy as jnp
    from flax import nnx
    from serve_pi05_feature_probe import CompiledFeatureModel
    class Model(nnx.Module):
        def flow_features(self,obs,x,t):
            hidden=jnp.broadcast_to(x[:,:,:1]+t[:,None,None],(len(x),10,1024))
            return x,hidden
    model=Model();wrapped=CompiledFeatureModel(model)
    x=jnp.ones((1,10,32));t=jnp.array([.1])
    eager=model.flow_features({},x,t)
    for _ in range(2):
        compiled=wrapped.flow_features({},x,t)
        for a,b in zip(eager,compiled,strict=True):np.testing.assert_array_equal(a,b)

def test_feature_service_main_binds_loopback_only(monkeypatch,tmp_path):
    import sys
    import serve_pi05_feature_probe as service
    monkeypatch.setattr(sys,'argv',['probe','--base',str(tmp_path),'--base-sha256','a'*64,
        '--norm-sha256','b'*64,'--expected-source-commit','c'*40,'--port','18671'])
    monkeypatch.setattr(service,'validate_preload',lambda a:(None,'c'*40,tmp_path/'norm_stats.json','a'*64,'b'*64))
    monkeypatch.setattr(service.sharding,'make_mesh',lambda n:None)
    monkeypatch.setattr(service.bank,'initialize_native_base',lambda *a:(None,None))
    class Model:
        def eval(self):pass
    monkeypatch.setattr(service.nnx,'merge',lambda *a:Model())
    monkeypatch.setattr(service,'make_transform',lambda *a,**k:None)
    monkeypatch.setattr(service.normalize,'load',lambda *a:{'actions':None})
    monkeypatch.setattr(service.fp,'schema_record',lambda **k:{})
    captured={}
    class Server:
        def __init__(self,policy,**kwargs):captured.update(kwargs)
        def serve_forever(self):captured['started']=True
    monkeypatch.setattr(service,'WebsocketPolicyServer',Server)
    service.main()
    assert captured['host']=='127.0.0.1' and captured['port']==18671
    assert captured['metadata']['role']=='pi05_feature_probe_only.v1'
    assert captured['started'] is True

def test_frame_plan_every_five_plus_last_and_disjoint():
    class E: pass
    e=E();e.index=3;e.length=12;e.task=6
    class D: pass
    d=D();d._ends=np.array([12]);d.episodes=[e]
    rows=cache.frame_plan(d,'train');assert [x['frame'] for x in rows]==[0,5,10,11]
    with pytest.raises(ValueError,match='overlap'):cache.validate_plan(rows,[{**rows[0],'split':'val'}])

def test_safe_cache_and_dryrun_args(tmp_path,capsys):
    root=tmp_path/'cache';root.mkdir();x=np.ones((10,1024),np.float32);tasks=np.arange(30,40,dtype=np.int32)
    train_keys=np.array([f'long/train/episode:{i}/frame:0' for i in range(10)]);val_keys=np.array([f'long/val/episode:{i+10}/frame:0' for i in range(10)])
    np.savez(root/'features.npz',train=x,val=x,train_keys=train_keys,val_keys=val_keys,train_tasks=tasks,val_tasks=tasks)
    raw=(root/'features.npz').read_bytes();frames={split:[{'key':str(k),'episode':i+(10 if split=='val' else 0)} for i,k in enumerate(keys)] for split,keys in [('train',train_keys),('val',val_keys)]}
    manifest={'schema':'demo_support_cache.v1','source_commit':'s','base_sha256':'a'*64,'norm_sha256':'b'*64,
      'dataset_manifest_sha256':'c'*64,'dataset_meta':{},'feature_schema':{'schema':'pi05-demo-support-feature-probe.v1','base_sha256':'a'*64,'norm_sha256':'b'*64,'feature_width':1024},
      'selection':'x','parent_target':'x','coverage':{'tasks':list(range(30,40))},'features_sha256':hashlib.sha256(raw).hexdigest(),
      'arrays':{'train':[10,1024],'val':[10,1024]},'frame_keys':frames}
    (root/'manifest.json').write_text(json.dumps(manifest))
    trainer.main(['--cache',str(root),'--output',str(tmp_path/'out'),'--expected-source','s','--base-sha256','a'*64,
      '--norm-sha256','b'*64,'--dataset-manifest-sha256','c'*64]);assert 'dry_run_only' in capsys.readouterr().out
    broken=json.loads((root/'manifest.json').read_text());broken['features_sha256']='0'*64;(root/'manifest.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='hash'):trainer.load_cache(root)

def test_prepare_output_creates_missing_parents_before_transport(tmp_path):
    output=tmp_path/'missing'/'cache';status=cache.prepare_output(output)
    assert status.exists() and json.loads(status.read_text())['status']=='extracting'

def test_feature_service_only_returns_features(monkeypatch):
    class Transform:
        def __call__(self,x):return {'state':x['observation/state']}
    class Stats: q01=np.zeros(7);q99=np.ones(7)
    policy=FeaturePolicy(object(),Transform(),Stats())
    monkeypatch.setattr('serve_pi05_feature_probe.model_api.Observation.from_dict',lambda x:x)
    monkeypatch.setattr('serve_pi05_feature_probe.model_api.preprocess_observation',lambda *a,**k:a[1])
    monkeypatch.setattr('serve_pi05_feature_probe.fp.probe_features',lambda *a,**k:np.ones((1,1024)))
    payload={'observation/image':np.zeros((2,2,3),np.uint8),'observation/wrist_image':np.zeros((2,2,3),np.uint8),
      'observation/state':np.zeros(8),'prompt':'p','chunks7':np.zeros((1,10,7)),'noise32':np.zeros((10,32))}
    local=policy.infer(payload)['feature'];assert local.shape==(1,1024)
    remote={'feature':local.copy()};np.testing.assert_array_equal(local,remote['feature'])
    expected={'role':'pi05_feature_probe_only.v1','base_sha256':'a'}
    assert cache.validate_service_metadata(expected,expected)
    for bad in ({**expected,'role':'policy'},{**expected,'base_sha256':'b'}):
        with pytest.raises(ValueError,match='identity'):cache.validate_service_metadata(bad,expected)
    with pytest.raises(ValueError):
        feature=np.ones((1,63));
        if feature.shape!=(1,1024):raise ValueError('invalid remote feature response')
