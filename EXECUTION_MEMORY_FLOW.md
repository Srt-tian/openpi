# Execution-memory flow adapter — model prototype, not trained

Authoritative branch: `feature/pi05-memory-train`, checkout
`/pfs/user/code/openpi_pi05_memory_train`. The separate CFN submission checkout
is unchanged. No training task was started for this prototype.

This is our own bounded GRU-memory flow adapter inspired by ActMem's causal
action-history and early/late handoff. It is **not** Mamba-2, not the original
PAE, and not a high-fidelity ActMem reproduction. Benefit remains unproven.

## Inputs and integration boundary

For each j<t, concatenate normalized pre-state8, executed action7, and
normalized post-minus-pre state8 into one 23D observed transition. All state
normalization must use the frozen base assets; this module assumes its inputs
have already been transformed. Expert histories are provided by the existing
causal-history reader; runtime histories must come from confirmed environment
execution, not the unexecuted suffix of a predicted action chunk. Current
state is allowed as the last observed post-state. No future observation,
success, task index, initial-state index or policy seed enters the network.

The batched packed-sequence GRU retains gradients through all supplied valid past
steps, ignores suffix padding, and permits empty history. Maximum history is
520 transitions. Training queries beyond that window must retain the reader's
omitted-past/burn-in metadata; do not label them full-prefix recurrent training.
The interface includes single-transition advancement for later episode-local
runtime integration. Episode identity and reset are the harness's obligation;
the model alone is not a validated episode-state service.

The native PI05 supplies final action hidden features [B,10,1024] at the
current noisy action x_t and time t. These are read-only, detached conditioning.
Memory128 produces four tokens256; a two-layer Transformer mixes them with the
ten projected current action tokens and time. The zero-initialized output
produces 0.5*tanh(delta) in physical7 normalized velocity space. Remaining25
velocity components remain untouched. This bound is **not** a physical-safety
or no-regression guarantee.

## Exact training/deployment time convention

Native source `pi0.py` uses x_t=t*noise+(1-t)*action and target u=noise-action.
The official ten-step sampler starts at 1 and decreases by .1. Proposed
adaptation only applies at indices0–3 (times1,.9,.8,.7); indices4–9 return the
original velocity untouched. Disabled mode returns the original object.

Therefore the CFN pooled t=.1 features cannot train this module. Build a new
cache containing full H10 native hidden/base velocity at the four early times,
explicitly seeded IID noise, normalized target u7, and causal demonstration
histories. GT actions may construct training flow points/labels, but are never
available at runtime; runtime uses the actual current sampler latent x_t.

## Loss

Let v0 denote frozen native physical7 velocity, d the learned correction,
and u the flow target. Each e is the mean squared error over all H10 positions
and seven physical dimensions, including official repeat-last tail positions.

$$e=\operatorname{mean}_{h,d}(v_0+d-u)^2,\qquad e_0=\operatorname{mean}_{h,d}(v_0-u)^2$$

$$\mathcal{L}=\operatorname{mean}(e)+0.1\operatorname{mean}\max(0,e-e_0)+10^{-4}\operatorname{mean}(d^2)$$

Base velocity, hidden features, time and target labels are detached from
optimization; memory and adapter receive gradients. The paired error penalty
is a supervised flow-error regularizer, not causal success benefit or a
harmlessness guarantee. No success labels, privileged states or candidate
outcome oracle are needed for this initial supervised model.

## Remaining work before a training submission

1. Verify raw-to-normalized history conversion and training/runtime convention.
2. The early-time feature extractor and loopback training-only service are
   implemented with CPU fixtures. Durable sharded cache storage with
   base/norm/source/split hashes is still pending; deduplicate per-episode
   history rather than store it per query. No real GPU cache exists yet.
3. Implement the deterministic task-balanced sequence trainer and safe
   checkpoints; fix optimizer, steps, validation and checkpoint choice.
4. Integrate the adapter into the real ten-step sampler with explicit
   episode reset/executed-action updates and disabled/native GPU parity.
5. Validate dependencies in the pinned image, measure a small authorized GPU
   workload, resolve resource/cost and obtain specific training confirmation.

Evaluation controls: native, same-capacity no-history adapter, memory adapter,
and chunk-reset history. Only then consider combination with CFN. CPU fixture
tests prove only implemented numerical contracts, not LIBERO success gains.
