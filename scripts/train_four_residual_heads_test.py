import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from types import SimpleNamespace
from scripts import train_four_residual_heads as target
from openpi.models import model as model_api
from openpi.training import physical_residual_bank as bank, sharding
from openpi.training.physical_residual import physical_residual_loss

class E:
 def __init__(self,task,length,index=None):self.task,self.length,self.index=task,length,task if index is None else index
class D:
 def __init__(self):self.episodes=tuple(E(t,80+t) for t in range(10));self._ends=np.cumsum([e.length for e in self.episodes])
 def diagnostic_indices(self,n):return list(range(n))

class ChunkD:
 def __init__(self):self.episodes=tuple(E(t,10) for t in range(10));self._ends=np.cumsum([10]*10)
 def __len__(self):return 100
 def __getitem__(self,index):
  start=(index//10)*10;frame=index-start
  return {"actions":np.minimum(frame+np.arange(10),9)}

def test_fixed_protocol_and_resumable_balanced_sampler():
 assert (target.TOTAL_UPDATES,target.UPDATES_PER_HEAD,target.SAVE_EVERY,target.EVAL_EVERY,target.BATCH_SIZE,target.SEED)==(40000,10000,20000,2000,40,42)
 a=target.BalancedSampler(D(),45,3);b=target.BalancedSampler(D(),45,3)
 assert next(iter(a))==next(iter(b)) and len(next(iter(a)))==40
 fresh=target.BalancedSampler(D(),45,0);it=iter(fresh)
 for _ in range(3):next(it)
 assert next(it)==next(iter(target.BalancedSampler(D(),45,3)))

def test_cli_requires_output_and_fixed_defaults(tmp_path):
 args=target.parse_args(["--output-dir",str(tmp_path),"--base-params-sha256","a"*64])
 assert (args.batch_size,args.seed,args.chunk_sampling,args.padding_supervision)==(40,42,"all_observations","official_repeat_last")

def test_parameter_content_hash_binds_names_sizes_and_bytes(tmp_path):
 (tmp_path/"a").write_bytes(b"one");(tmp_path/"b").write_bytes(b"two")
 first=target.params_content_hash(tmp_path);(tmp_path/"b").write_bytes(b"too")
 assert first!=target.params_content_hash(tmp_path)

def test_first_real_optimizer_update_changes_head_and_is_finite():
 params={"weight":jnp.array([0.0,0.0],dtype=jnp.float32)}
 grads={"weight":jnp.array([1.0,-2.0],dtype=jnp.float32)}
 tx=target.make_optimizer();state=tx.init(params)
 updates,_=tx.update(grads,state,params);changed=optax.apply_updates(params,updates)
 assert np.isfinite(np.asarray(updates["weight"])).all()
 assert np.isfinite(np.linalg.norm(np.asarray(updates["weight"])))
 assert not np.array_equal(np.asarray(changed["weight"]),np.asarray(params["weight"]))

def test_all_observations_admits_tail_with_official_repeat_last_supervision():
 ds=ChunkD();intervals=target.intervals_by_task(ds)
 assert intervals[0]==[(0,10)] and intervals[9]==[(90,10)]
 tagged=target.TaggedTransformedDataset(ds,lambda x:x)
 value=tagged[0];assert value["actions"][-1]==9 and np.array_equal(value["v2_valid_horizon"],np.ones(10))
 for frame in range(1,10):
  value=tagged[frame]
  assert int(value["v2_valid_horizon"].sum())==10
  assert int(value["v2_episode_valid_horizon"].sum())==10-frame
  assert value["actions"][-1]==9
 assert int(tagged[9]["v2_valid_horizon"].sum())==10
 assert int(tagged[9]["v2_episode_valid_horizon"].sum())==1
 prefix=target.TaggedTransformedDataset(ds,lambda x:x,padding_supervision="valid_prefix")
 assert int(prefix[9]["v2_valid_horizon"].sum())==1

def test_full_chunk_mode_isolated_and_short_episode_retained_by_default():
 ds=ChunkD();assert target.intervals_by_task(ds,"full_chunks")[0]==[(0,1)]
 tagged=target.TaggedTransformedDataset(ds,lambda x:x,"full_chunks")
 assert tagged[0]["v2_valid_horizon"].sum()==10
 try:tagged[1]
 except IndexError:pass
 else:raise AssertionError("tail start leaked into full_chunks ablation")
 ds=D();ds.episodes=(E(0,9),*ds.episodes[1:]);ds._ends=np.cumsum([e.length for e in ds.episodes])
 assert target.intervals_by_task(ds)[0]==[(0,9)]

def test_tail_holdout_supports_global_suite_task_ids_and_stays_in_frame_bounds():
 for offset in (0,10,20,30):
  episodes=[]
  for task in range(offset,offset+10):episodes.extend((E(task,3,task*2),E(task,8,task*2+1)))
  ds=type("SuiteD",(),{})();ds.episodes=tuple(episodes);ds._ends=np.cumsum([e.length for e in episodes])
  indices,metadata=target.fixed_tail_indices(ds)
  assert len(indices)==40 and {item["task"] for item in metadata}==set(range(offset,offset+10))
  index_tasks=[ds.episodes[int(np.searchsorted(ds._ends,index,side="right"))].task-offset for index in indices]
  assert np.bincount(index_tasks,minlength=10).tolist()==[4]*10
  for index in indices:
   pos=int(np.searchsorted(ds._ends,index,side="right"));start=0 if pos==0 else int(ds._ends[pos-1])
   assert start<=index<int(ds._ends[pos])

def test_fixed_general_and_tail_holdouts_are_task_balanced():
 ds=D();indices=target.fixed_holdout_indices(ds,42);assert len(indices)==80
 tasks=[]
 for index in indices:tasks.append(int(np.searchsorted(ds._ends,index,side="right")))
 assert np.bincount(tasks,minlength=10).tolist()==[8]*10
 episodes=[]
 for task in range(10):episodes.extend((E(task,3,task*2),E(task,8,task*2+1)))
 ds.episodes=tuple(episodes);ds._ends=np.cumsum([e.length for e in episodes])
 indices,metadata=target.fixed_tail_indices(ds)
 tasks=[ds.episodes[int(np.searchsorted(ds._ends,index,side="right"))].task for index in indices]
 assert len(indices)==40 and np.bincount(tasks,minlength=10).tolist()==[4]*10
 assert len(metadata)==20 and sum(item["clamped_or_repeated"] for item in metadata)==10
 valid_counts=[count for item in metadata for count in item["valid_counts"]]
 assert valid_counts.count(1)==20 and valid_counts.count(3)==10 and valid_counts.count(5)==10

def test_holdout_rng_is_fixed_per_suite():
 a=target.fixed_eval_rng(42,"spatial");b=target.fixed_eval_rng(42,"spatial");c=target.fixed_eval_rng(42,"object")
 assert np.array_equal(np.asarray(jax.random.key_data(a)),np.asarray(jax.random.key_data(b)))
 assert not np.array_equal(np.asarray(jax.random.key_data(a)),np.asarray(jax.random.key_data(c)))

def test_invalid_gt_suffix_has_zero_loss_and_gradient():
 base=jnp.zeros((1,10,32));residual=jnp.zeros((1,10,7));gate=jnp.ones((1,10,1));mask=jnp.array([[1]+[0]*9])
 def loss(target):
  return physical_residual_loss(base_velocity=base,residual7=residual,gate=gate,target_velocity=target,
    task_ids=jnp.array([0]),num_tasks=1,valid_horizon=mask,correction_weight=0,gate_weight=0)[0]
 clean=jnp.zeros((1,10,32));dirty=clean.at[:,1:,:7].set(123)
 np.testing.assert_allclose(loss(clean),loss(dirty))
 np.testing.assert_array_equal(np.asarray(jax.grad(loss)(dirty)[:,1:,:7]),0)


class PipelineD:
 def __init__(self,offset=20):
  self.episodes=tuple(E(task,length,index) for task in range(offset,offset+10)
   for index,length in ((task*2,6),(task*2+1,10)))
  self._ends=np.cumsum([e.length for e in self.episodes])
 def __len__(self):return int(self._ends[-1])
 def __getitem__(self,index):
  pos=int(np.searchsorted(self._ends,index,side="right"));episode=self.episodes[pos]
  start=0 if pos==0 else int(self._ends[pos-1]);frame=index-start
  action=np.minimum(frame+np.arange(10),episode.length-1).astype(np.float32)
  return {"image":{"cam":np.zeros((2,2,3),np.float32)},"image_mask":{"cam":np.bool_(True)},
    "state":np.ones(32,np.float32),"actions":np.broadcast_to(action[:,None],(10,32)).copy()}


class PipelineBase(nnx.Module):
 def __init__(self):self.scale=nnx.Param(jnp.asarray(1.0))
 def flow_features(self,observation,x_t,time):
  del observation,time
  hidden=jnp.ones((*x_t.shape[:2],4))*self.scale
  return jnp.zeros_like(x_t)+self.scale,hidden


def test_fake_dataloader_batch_step_eval_and_checkpoint_roundtrip(tmp_path,monkeypatch):
 monkeypatch.setattr(model_api,"preprocess_observation",lambda key,obs,train:obs)
 ds=PipelineD();datasets={suite:ds for suite in bank.SUITES};steps={suite:0 for suite in bank.SUITES}
 args=SimpleNamespace(seed=42,num_workers=0,chunk_sampling="all_observations",padding_supervision="official_repeat_last")
 train=target.make_loaders(args,datasets,lambda x:x,steps,True)
 general=target.make_loaders(args,datasets,lambda x:x,steps,False,"general")
 tail=target.make_loaders(args,datasets,lambda x:x,steps,False,"tail")
 obs,actions,tasks,mask,episode_mask=target.batch_parts(next(iter(train["goal"])))
 assert actions.shape==(40,10,32) and set(np.asarray(tasks))==set(range(10))
 np.testing.assert_array_equal(mask,1);assert np.any(episode_mask==0)
 prefix_args=SimpleNamespace(**{**vars(args),"padding_supervision":"valid_prefix"})
 prefix_tail=target.make_loaders(prefix_args,datasets,lambda x:x,steps,False,"tail")
 _,_,_,prefix_mask,prefix_episode=target.batch_parts(next(iter(prefix_tail["goal"])))
 np.testing.assert_array_equal(prefix_mask,prefix_episode);assert {1,5}.issubset(set(prefix_mask.sum(1).astype(int)))
 base_graph,base_state=nnx.split(PipelineBase());head_graph,heads=bank.initialize_head_bank(4,7,width=8,horizon=10,ffn_dim=16)
 tx=optax.adam(1e-3);opts=bank.initialize_optimizer_states(tx,heads);mesh=sharding.make_mesh(1)
 step_fn=bank.make_sharded_head_step(base_graph,head_graph,tx,mesh);eval_fn=bank.make_sharded_head_eval(base_graph,head_graph,mesh)
 obs,actions,tasks,mask=target.legacy.put_batch_on_mesh((obs,actions,tasks,mask),mesh)
 heads["spatial"],opts["spatial"],metrics=step_fn(base_state,heads["spatial"],opts["spatial"],obs,actions,tasks,mask,jax.random.key(1),0)
 assert all(jnp.isfinite(value) for value in metrics.values());steps["spatial"]=1
 for loader in (general,tail):
  eo,ea,et,em,_=target.batch_parts(next(iter(loader["goal"])))
  eo,ea,et,em=target.legacy.put_batch_on_mesh((eo,ea,et,em),mesh)
  value,_=eval_fn(base_state,heads["goal"],eo,ea,et,em,jax.random.key(2),0);assert jnp.isfinite(value)
 config={"feature_dim":4};path=bank.save_head_bank(tmp_path/"step4",heads,opts,steps,
  base_checkpoint_path="/base",norm_stats_sha256="n",base_manifest_sha256="m",head_config=config)
 _,_,restored,_=bank.load_head_bank(path,heads,opts,expected_base_checkpoint_path="/base",
  expected_norm_stats_sha256="n",expected_base_manifest_sha256="m",expected_head_config=config)
 assert restored==steps
