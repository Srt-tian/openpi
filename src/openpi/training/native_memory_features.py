"""Frozen native early-flow training features, distinct from pooled CFN cache."""
import hashlib
import jax.numpy as jnp
import numpy as np
from openpi.training.native_feature_probe import canonical_physical_chunks

SCHEMA='pi05-native-memory-training-features.v1'
TIMES=(1.,.9,.8,.7)


def schema(base_sha256,norm_sha256):
    if any(len(x)!=64 for x in (base_sha256,norm_sha256)):
        raise ValueError('exact base and norm identities required')
    return {'schema':SCHEMA,'base_sha256':base_sha256,'norm_sha256':norm_sha256,
        'times':list(TIMES),'hidden_shape':[4,10,1024],'velocity_shape':[4,10,32],
        'target':'explicit IID noise minus official quantile normalized H10 actions',
        'action_normalization':'physical7_quantile_no_clip_then_pad32',
        'history':'past_only_normalized_prestate8_action7_statedelta8',
        'use':'training_cache_only_not_runtime_candidate_scoring'}


def cache_noise(logical_key):
    if not isinstance(logical_key,str) or not logical_key:
        raise ValueError('logical training sample key required')
    seed=int.from_bytes(hashlib.sha256((SCHEMA+'|'+logical_key).encode()).digest()[:8],'big')
    return np.random.default_rng(seed).standard_normal((10,32),dtype=np.float32)


def extract_training_features(model,observation,actions7,action_stats,noise32):
    """Observation is already preprocessed and broadcast to the four times."""
    actions=canonical_physical_chunks(np.asarray(actions7)[None],action_stats)
    noise=np.asarray(noise32,dtype=np.float32)
    if noise.shape!=(10,32) or not np.isfinite(noise).all():
        raise ValueError('explicit finite IID noise [10,32] required')
    times=jnp.asarray(TIMES,jnp.float32)
    actions=jnp.broadcast_to(actions,(4,10,32))
    shared=jnp.broadcast_to(jnp.asarray(noise),(4,10,32))
    latent=times[:,None,None]*shared+(1-times[:,None,None])*actions
    velocity,hidden=model.flow_features(observation,latent,times)
    result={'hidden':np.asarray(hidden,np.float32),'base_velocity':np.asarray(velocity,np.float32),
        'target_velocity7':np.asarray((shared-actions)[...,:7],np.float32),'time':np.asarray(times,np.float32)}
    expected={'hidden':(4,10,1024),'base_velocity':(4,10,32),'target_velocity7':(4,10,7),'time':(4,)}
    if any(result[k].shape!=shape or not np.isfinite(result[k]).all() for k,shape in expected.items()):
        raise ValueError('native early-flow feature response is invalid')
    return result


def quantile(value,stats,width):
    value=np.asarray(value,np.float32)
    lo,hi=(np.asarray(getattr(stats,k),np.float32)[:width] for k in ('q01','q99'))
    if value.shape[-1:]!=(width,) or lo.shape!=(width,) or hi.shape!=(width,) or not all(np.isfinite(x).all() for x in (value,lo,hi)):
        raise ValueError('invalid history values or normalization assets')
    return (value-lo)/(hi-lo+1e-6)*2-1


def episode_history(table,frame,action_stats,state_stats,max_history=520):
    if type(frame) is not int or not 0<=frame<len(table.state) or type(max_history) is not int or not 0<=max_history<=520:
        raise ValueError('invalid current frame or history budget')
    first=max(0,frame-max_history)
    pre=quantile(table.state[first:frame],state_stats,8)
    post=quantile(table.state[first+1:frame+1],state_stats,8)
    action=quantile(table.action[first:frame],action_stats,7)
    if not len(pre)==len(post)==len(action):
        raise ValueError('unaligned past transitions')
    return np.concatenate([pre,action,post-pre],axis=-1).astype(np.float32),{
        'first':first,'current_frame':frame,'omitted_past':first,'burn_in_executed':0,
        'source':'expert_demonstration_not_policy_rollout'}
