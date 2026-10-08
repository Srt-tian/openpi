import numpy as np
import jax.numpy as jnp
import optax
from scripts import train_four_residual_heads as target

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
 assert args.batch_size==40 and args.seed==42 and not args.preflight_only

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

def test_full_chunk_intervals_cover_terminal_action_and_masks_are_all_valid():
 ds=ChunkD();intervals=target.intervals_by_task(ds)
 assert intervals[0]==[(0,1)] and intervals[9]==[(90,1)]
 tagged=target.TaggedTransformedDataset(ds,lambda x:x)
 value=tagged[0];assert value["actions"][-1]==9 and np.array_equal(value["v2_valid_horizon"],np.ones(10))
 try:tagged[1]
 except IndexError:pass
 else:raise AssertionError("tail-padded start was admitted")

def test_short_episode_fails_and_fixed_holdout_is_balanced():
 ds=D();ds.episodes=(E(0,9),*ds.episodes[1:]);ds._ends=np.cumsum([e.length for e in ds.episodes])
 try:target.intervals_by_task(ds)
 except ValueError as error:assert "shorter than horizon" in str(error)
 else:raise AssertionError("short episode silently dropped")
 ds=D();indices=target.fixed_holdout_indices(ds,42);assert len(indices)==80
 tasks=[]
 for index in indices:tasks.append(int(np.searchsorted(ds._ends,index,side="right")))
 assert np.bincount(tasks,minlength=10).tolist()==[8]*10

def test_holdout_rng_is_fixed_per_suite():
 import jax
 a=target.fixed_eval_rng(42,"spatial");b=target.fixed_eval_rng(42,"spatial");c=target.fixed_eval_rng(42,"object")
 assert np.array_equal(np.asarray(a),np.asarray(b)) and not np.array_equal(np.asarray(a),np.asarray(c))
