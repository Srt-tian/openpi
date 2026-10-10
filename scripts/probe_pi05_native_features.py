#!/usr/bin/env python3
"""One-shot feature smoke; generates parity arrays but never executes actions."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
import jax
from openpi.models import model as model_api
from openpi.shared import normalize
from openpi.training import native_feature_probe as probe, physical_residual_bank as bank, sharding
from scripts.train_four_plugins import make_transform
from scripts.train_four_residual_heads import params_content_hash
from flax import nnx

def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--base-sha256',required=True)
    p.add_argument('--norm-sha256',required=True);p.add_argument('--expected-source-commit',required=True)
    p.add_argument('--prompt',default='put both moka pots on the stove');p.add_argument('--seed',type=int,default=7);a=p.parse_args(argv)
    return a

def validate_preload(a):
    root=Path(__file__).resolve().parents[1];source=subprocess.check_output(
      ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if source!=a.expected_source_commit: raise ValueError('source commit mismatch')
    norm=a.base/'assets/physical-intelligence/libero/norm_stats.json'
    actual=params_content_hash(a.base/'params')
    if actual!=a.base_sha256: raise ValueError('base content digest mismatch')
    norm_sha=hashlib.sha256(norm.read_bytes()).hexdigest()
    if norm_sha!=a.norm_sha256: raise ValueError('norm stats digest mismatch')
    if jax.default_backend()!='gpu' or len(jax.devices('gpu'))!=1:
        raise RuntimeError('feature smoke requires exactly one actual JAX GPU')
    return root,source,norm,actual,norm_sha

def main(argv=None):
    a=parse_args(argv);root,source,norm,actual,norm_sha=validate_preload(a)
    started=time.monotonic();mesh=sharding.make_mesh(1);graph,state=bank.initialize_native_base(str(a.base/'params'),mesh)
    model=nnx.merge(graph,state);model.eval()
    raw={'observation/image':np.zeros((224,224,3),np.uint8),'observation/wrist_image':np.zeros((224,224,3),np.uint8),
      'observation/state':np.zeros(8,np.float32),'prompt':a.prompt}
    transformed=make_transform(norm,require_actions=False)(raw)
    obs=model_api.Observation.from_dict(jax.tree.map(
      lambda x:jax.numpy.broadcast_to(jax.numpy.asarray(x),(2,)+np.asarray(x).shape),transformed))
    obs=model_api.preprocess_observation(None,obs,train=False)
    chunks=np.stack([np.zeros((10,7),np.float32),np.full((10,7),.01,np.float32)])
    stats=normalize.load(norm.parent)['actions'];noise=probe.common_noise(probe.feature_probe_seed(a.seed,0))
    batched_noise=jax.numpy.broadcast_to(jax.numpy.asarray(noise),(2,10,32));key=jax.random.key(a.seed)
    native_before=model.sample_actions(key,obs,noise=batched_noise)
    first=probe.probe_features(model,obs,chunks,stats,noise);second=probe.probe_features(model,obs,chunks,stats,noise)
    if not np.array_equal(np.asarray(first),np.asarray(second)): raise AssertionError('probe is not repeatable')
    native_after=model.sample_actions(key,obs,noise=batched_noise)
    if not np.array_equal(np.asarray(native_before),np.asarray(native_after)): raise AssertionError('native action path changed')
    print(json.dumps({'feature':{'shape':list(first.shape),'dtype':str(first.dtype),'finite':bool(np.isfinite(first).all())},
      'repeat_equal':True,'native_same_noise_before_after_equal':True,'elapsed_seconds':time.monotonic()-started,
      'source_commit':source,'jax_backend':jax.default_backend(),'devices':[str(x) for x in jax.devices()],
      'schema':probe.schema_record(base_sha256=actual,norm_sha256=norm_sha)},sort_keys=True))
if __name__=='__main__':main()
