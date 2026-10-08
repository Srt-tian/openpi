import numpy as np
import jax.numpy as jnp
import optax
from scripts import train_four_residual_heads as target
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
 assert args.batch_size==40 and args.seed==42 and not args.preflight_only and args.chunk_sampling=="all_observations"

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

def test_all_observations_admits_tail_with_native_padding_and_prefix_masks():
 ds=ChunkD();intervals=target.intervals_by_task(ds)
 assert intervals[0]==[(0,10)] and intervals[9]==[(90,10)]
 tagged=target.TaggedTransformedDataset(ds,lambda x:x)
 value=tagged[0];assert value["actions"][-1]==9 and np.array_equal(value["v2_valid_horizon"],np.ones(10))
 for frame in range(1,10):
  value=tagged[frame]
  assert int(value["v2_valid_horizon"].sum())==10-frame
  assert value["actions"][-1]==9
 assert int(tagged[9]["v2_valid_horizon"].sum())==1

def test_full_chunk_mode_isolated_and_short_episode_retained_by_default():
 ds=ChunkD();assert target.intervals_by_task(ds,"full_chunks")[0]==[(0,1)]
 tagged=target.TaggedTransformedDataset(ds,lambda x:x,"full_chunks")
 assert tagged[0]["v2_valid_horizon"].sum()==10
 try:tagged[1]
 except IndexError:pass
 else:raise AssertionError("tail start leaked into full_chunks ablation")
 ds=D();ds.episodes=(E(0,9),*ds.episodes[1:]);ds._ends=np.cumsum([e.length for e in ds.episodes])
 assert target.intervals_by_task(ds)[0]==[(0,9)]

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
 import jax
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
