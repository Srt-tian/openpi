from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import copy
import flax.struct
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.training import physical_residual_bank as bank
from openpi.models import model as model_api
from openpi.training import sharding


@flax.struct.dataclass
class _Obs:
    state: jax.Array


class _TinyBase(nnx.Module):
    def __init__(self): self.scale=nnx.Param(jnp.asarray(1.0))
    def flow_features(self,observation,x_t,time):
        del observation,time
        hidden=jnp.ones((*x_t.shape[:2],4))*self.scale
        return jnp.zeros_like(x_t)+self.scale,hidden


def test_selected_only_update_and_four_independent_heads(monkeypatch):
    graphdef, states = bank.initialize_head_bank(4, 7, width=8, horizon=2, ffn_dim=16)
    tx = optax.adam(1e-2); opts = bank.initialize_optimizer_states(tx, states)
    before_states, before_opts = dict(states), dict(opts)
    def fake_loss(base, head, rng, observation, actions, task_ids, **kwargs):
        del base,rng,observation,actions,task_ids,kwargs
        leaves=jax.tree.leaves(nnx.state(head)); value=sum(jnp.sum(x*x) for x in leaves)
        return value,{"physical_fm":value}
    monkeypatch.setattr(bank,"residual_training_loss",fake_loss)
    states,opts,suite,metrics=bank.update_selected_head(2,graphdef,object(),states,opts,tx,
        jax.random.key(0),object(),object(),jnp.arange(4),num_tasks=10)
    assert suite=="goal" and jnp.isfinite(metrics["loss"])
    for name in bank.SUITES:
        if name!="goal": assert states[name] is before_states[name] and opts[name] is before_opts[name]
    assert states["goal"] is not before_states["goal"]


def test_compiled_step_has_explicit_base_state_and_updates_head_only(monkeypatch):
    monkeypatch.setattr(model_api,"preprocess_observation",lambda key,obs,train:obs)
    base_graphdef,base_state=nnx.split(_TinyBase())
    head_graphdef,states=bank.initialize_head_bank(4,8,width=8,horizon=2,ffn_dim=16)
    tx=optax.adam(1e-3);opts=bank.initialize_optimizer_states(tx,states);mesh=sharding.make_mesh(1)
    fn=bank.make_sharded_head_step(base_graphdef,head_graphdef,tx,mesh)
    obs=_Obs(jnp.ones((10,32)));actions=jnp.ones((10,2,32));tasks=jnp.arange(10);valid=jnp.ones((10,2))
    before=np.asarray(base_state["scale"]).copy()
    new_state,new_opt,metrics=fn(base_state,states["spatial"],opts["spatial"],obs,actions,tasks,valid,jax.random.key(0),0)
    assert jnp.isfinite(metrics["loss"]) and new_state is not states["spatial"] and new_opt is not opts["spatial"]
    np.testing.assert_array_equal(np.asarray(base_state["scale"]),before)


def test_checkpoint_roundtrip_binding_checksum_and_serving_smoke(tmp_path):
    graphdef,states=bank.initialize_head_bank(4,3,width=8,horizon=2,ffn_dim=16);tx=optax.adam(1e-3)
    opts=bank.initialize_optimizer_states(tx,states);steps=dict(zip(bank.SUITES,(2,2,1,1),strict=True))
    config={"feature_dim":4,"state_dim":8,"width":8,"horizon":2,"heads":4,"ffn_dim":16,"residual_bound":1.0}
    path=bank.save_head_bank(tmp_path/"step",states,opts,steps,base_checkpoint_path="/frozen/base",
        norm_stats_sha256="a"*64,base_manifest_sha256="b"*64,head_config=config,
        metadata={"gate_semantics":"surrogate_not_success"})
    restored,restored_opts,restored_steps,manifest=bank.load_head_bank(path,states,opts,
        expected_base_checkpoint_path="/frozen/base",expected_norm_stats_sha256="a"*64,
        expected_base_manifest_sha256="b"*64,expected_head_config=config)
    assert restored_steps==steps and set(restored_opts)==set(bank.SUITES)
    head=bank.restore_suite_head(graphdef,restored,"long")
    out=head(jnp.ones((1,2,4)),jnp.ones((1,32)),jnp.ones((1,2,32)),jnp.ones((1,)))
    np.testing.assert_array_equal(out["residual32"][...,7:],0)
    assert manifest["metadata"]["gate_semantics"]=="surrogate_not_success"
    with pytest.raises(ValueError): bank.load_head_bank(path,states,opts,
        expected_base_checkpoint_path="/wrong",expected_norm_stats_sha256="a"*64,
        expected_base_manifest_sha256="b"*64,expected_head_config=config)
    payload=path/manifest["banks"]["spatial"]["head_file"]
    payload.write_bytes(payload.read_bytes()+b"bad")
    with pytest.raises(ValueError,match="checksum"): bank.load_head_bank(path,states,opts,
        expected_base_checkpoint_path="/frozen/base",expected_norm_stats_sha256="a"*64,
        expected_base_manifest_sha256="b"*64,expected_head_config=config)


def test_restore_rejects_changed_shape_and_non_roundrobin_steps(tmp_path):
    _,states=bank.initialize_head_bank(4,4,width=8,horizon=2,ffn_dim=16);tx=optax.adam(1e-3);opts=bank.initialize_optimizer_states(tx,states)
    config={"feature_dim":4};path=bank.save_head_bank(tmp_path/"shape",states,opts,
      dict(zip(bank.SUITES,(1,1,1,0),strict=True)),base_checkpoint_path="/base",norm_stats_sha256="n",base_manifest_sha256="m",head_config=config)
    _,wrong=bank.initialize_head_bank(5,4,width=8,horizon=2,ffn_dim=16);wrong_opts=bank.initialize_optimizer_states(tx,wrong)
    with pytest.raises((ValueError,TypeError)):
      bank.load_head_bank(path,wrong,wrong_opts,expected_base_checkpoint_path="/base",expected_norm_stats_sha256="n",expected_base_manifest_sha256="m",expected_head_config=config)
    manifest_path=path/"manifest.json";manifest=__import__("json").loads(manifest_path.read_text());manifest["banks"]["spatial"]["step"]=0;manifest["global_update_count"]=2;manifest_path.write_text(__import__("json").dumps(manifest))
    with pytest.raises(ValueError,match="round-robin"):
      bank.load_head_bank(path,states,opts,expected_base_checkpoint_path="/base",expected_norm_stats_sha256="n",expected_base_manifest_sha256="m",expected_head_config=config)


def test_checkpoint_create_only_and_invalid_suite(tmp_path):
    graphdef,states=bank.initialize_head_bank(4,1,width=8,horizon=2,ffn_dim=16);tx=optax.sgd(1e-2)
    opts=bank.initialize_optimizer_states(tx,states);steps={s:0 for s in bank.SUITES};config={"feature_dim":4}
    path=bank.save_head_bank(tmp_path/"bank",states,opts,steps,base_checkpoint_path="/base",
        norm_stats_sha256="a",base_manifest_sha256="b",head_config=config)
    with pytest.raises(FileExistsError): bank.save_head_bank(path,states,opts,steps,
        base_checkpoint_path="/base",norm_stats_sha256="a",base_manifest_sha256="b",head_config=config)
    with pytest.raises(ValueError): bank.restore_suite_head(graphdef,states,"unknown")
