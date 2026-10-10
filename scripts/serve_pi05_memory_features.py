#!/usr/bin/env python3
"""Private training-only early-flow feature endpoint; no action execution."""
import argparse
from pathlib import Path
import jax
import numpy as np
from flax import nnx
from openpi.models import model as model_api
from openpi.shared import normalize
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import native_memory_features as feature,physical_residual_bank as bank,sharding
from probe_pi05_native_features import validate_preload
from serve_pi05_feature_probe import CompiledFeatureModel
from train_four_plugins import make_transform

ROLE='pi05_memory_training_features_only.v1'


class MemoryFeaturePolicy:
    def __init__(self,model,transform,stats):
        self.model,self.transform,self.stats=CompiledFeatureModel(model),transform,stats

    def infer(self,payload):
        required={'observation/image','observation/wrist_image','observation/state','prompt','actions7','noise32'}
        if type(payload) is not dict or set(payload)!=required:
            raise ValueError('training feature request schema mismatch')
        request=dict(payload)
        actions,noise=request.pop('actions7'),request.pop('noise32')
        transformed=self.transform(request)
        obs=model_api.Observation.from_dict(jax.tree.map(
            lambda x:jax.numpy.broadcast_to(jax.numpy.asarray(x),(4,)+np.asarray(x).shape),transformed))
        obs=model_api.preprocess_observation(None,obs,train=False)
        return feature.extract_training_features(self.model,obs,actions,self.stats,noise)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    for flag in ('base-sha256','norm-sha256','expected-source-commit'):
        parser.add_argument('--'+flag,required=True)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    _,source,norm,base_hash,norm_hash=validate_preload(args)
    graph,state=bank.initialize_native_base(str(args.base/'params'),sharding.make_mesh(1))
    model=nnx.merge(graph,state);model.eval()
    policy=MemoryFeaturePolicy(model,make_transform(norm,require_actions=False),normalize.load(norm.parent)['actions'])
    metadata={'role':ROLE,'source_commit':source,'feature_schema':feature.schema(base_hash,norm_hash)}
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,metadata=metadata).serve_forever()


if __name__=='__main__':main()
