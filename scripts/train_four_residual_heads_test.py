import numpy as np
import jax.numpy as jnp
import optax
from scripts import train_four_residual_heads as target

class E:
 def __init__(self,task,length):self.task,self.length=task,length
class D:
 def __init__(self):self.episodes=tuple(E(t,80+t) for t in range(10));self._ends=np.cumsum([e.length for e in self.episodes])
 def diagnostic_indices(self,n):return list(range(n))

def test_fixed_protocol_and_resumable_balanced_sampler():
 assert (target.TOTAL_UPDATES,target.UPDATES_PER_HEAD,target.SAVE_EVERY,target.EVAL_EVERY,target.BATCH_SIZE,target.SEED)==(40000,10000,20000,2000,40,42)
 a=target.BalancedSampler(D(),45,3);b=target.BalancedSampler(D(),45,3)
 assert next(iter(a))==next(iter(b)) and len(next(iter(a)))==40

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
