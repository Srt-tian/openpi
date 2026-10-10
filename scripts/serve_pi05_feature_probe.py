#!/usr/bin/env python3
"""Feature-only websocket endpoint; returns hidden features, never actions."""
import argparse, json
from pathlib import Path
import jax
import numpy as np
from flax import nnx
from openpi.models import model as model_api
from openpi.shared import normalize
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import native_feature_probe as fp, physical_residual_bank as bank, sharding
from probe_pi05_native_features import validate_preload
from train_four_plugins import make_transform

ROLE='pi05_feature_probe_only.v1'
class CompiledFeatureModel:
    """Cache the frozen model forward compilation across demonstration frames."""
    def __init__(self,model):
        self.model=model
        self.compiled=nnx.jit(lambda m,o,x,t:m.flow_features(o,x,t))
    def flow_features(self,obs,x,t):
        return self.compiled(self.model,obs,x,t)

class FeaturePolicy:
    def __init__(self,model,transform,stats):
        self.model,self.transform,self.stats=CompiledFeatureModel(model),transform,stats
    def infer(self,p):
        required={'observation/image','observation/wrist_image','observation/state','prompt','chunks7','noise32'}
        if type(p) is not dict or set(p)!=required:raise ValueError('feature request schema mismatch')
        chunks=np.asarray(p.pop('chunks7'),np.float32);noise=np.asarray(p.pop('noise32'),np.float32);n=len(chunks)
        transformed=self.transform(p)
        obs=model_api.Observation.from_dict(jax.tree.map(lambda x:jax.numpy.broadcast_to(jax.numpy.asarray(x),(n,)+np.asarray(x).shape),transformed))
        obs=model_api.preprocess_observation(None,obs,train=False)
        feature=np.asarray(fp.probe_features(self.model,obs,chunks,self.stats,noise),np.float32)
        return {'feature':feature}
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--base-sha256',required=True)
    p.add_argument('--norm-sha256',required=True);p.add_argument('--expected-source-commit',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    _,source,norm,base_hash,norm_hash=validate_preload(a)
    graph,state=bank.initialize_native_base(str(a.base/'params'),sharding.make_mesh(1));model=nnx.merge(graph,state);model.eval()
    policy=FeaturePolicy(model,make_transform(norm,require_actions=False),normalize.load(norm.parent)['actions'])
    metadata={'role':ROLE,'source_commit':source,'base_sha256':base_hash,'norm_sha256':norm_hash,
      'feature_schema':fp.schema_record(base_sha256=base_hash,norm_sha256=norm_hash)}
    # Feature inputs include private demonstration images. Expose this endpoint
    # only through the explicitly configured SSH tunnel, never on the LAN.
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
if __name__=='__main__':main()
