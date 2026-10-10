# Unified capability training task (preparation, not submitted)

## Image evidence

Previous successful CLOUD Task 13585 used
`magiclab-cn-shanghai.cr.volces.com/magiclab/openpi_server:latest` with Python
`/.venv/bin/python`, and an existing AV overlay under
`/pfs/user/data/physicalrsi_pi05/runtime_deps/av15_overlay`.
This is a historical reference, not an immutable image digest and not proof
that today's image contains the new dependencies. Do not call it validated for
Memory/PAE. No image was built/pushed and no task was submitted by this change.

Before submission: resolve immutable image identity; run imports, CPU tests,
then a separately authorized GPU forward/backward smoke in that exact image.
The current CFN path needs JAX/Flax/OpenPI and PyTorch/safetensors; the dataset
reader also needs pyarrow/AV. Memory dependencies are not finalized; Mamba
should not be installed until architecture and compatible CUDA build are fixed.
Do not install packages from the internet in the training task.

## One entrypoint

```bash
REPO_DIR=/pfs/user/code/openpi_pi05_capability_harness \
PI05_PYTHON=/path/to/validated/python \
bash scripts/launch_pi05_capability_bundle.sh
```

Default mode checks **all** listed trainable modules, prints every missing
prerequisite, and exits 2 while anything is blocked. It neither trains the one
ready model silently nor reports the full batch complete. `--modules` is an
explicit subset selection whose omissions are always reported. No GPU work
occurs during the default preflight.

The CFN execution path is implemented: start a loopback-only frozen feature
service, extract the deterministic Long demonstration feature cache, stop only
the owned service, fit the CFN, and verify its final checkpoint receipt. One
GPU is exposed and stages run sequentially. No simulator or benchmark runs in
the training task. Independent stage logs/cache/checkpoints plus a bundle
receipt are stored under a fresh output directory. Failure exits nonzero;
there is no automatic retry, overwrite, best-checkpoint selection or resume.
The configured 7200-second wall-time cap is a proposed safety cap, **not** a
measured runtime estimate or billing quote; it includes extraction and fitting.

Execution requires the approved exact commit, a clean canonical Git checkout
whose upstream matches it, `PI05_BUNDLE_AUTHORIZATION=CONFIRMED`, and
`--execute`. The environment flag is an execution guard, not a substitute for
the user's specific approval of the resolved training submission.

## Pending modules

| Module | Current training status | Missing input |
|---|---|---|
| Demo-support CFN | Trainer implemented, no real fit | Native feature cache and validated image runtime |
| Execution Memory + PAE | Not implemented | Sequence model, frozen-base integration, flow loss/trainer |
| Action alignment verifier | Not implemented | LIBERO candidate/contrastive data and trainer |
| Intervention gain gate | Not implemented | Paired intervention benefit labels and calibration split |
| Value/latent guidance | Not implemented | Reward/transition data and critic/RL trainer |

RTS retrieval-library construction and RTC execution adaptation are separately
tracked non-training activities. TraceVLA is a source of observation-feature
design ideas, not a ready frozen-PI05 plugin. No survey entry is automatically
treated as an implementable training job. The old four trained residual heads
are frozen evaluation controls; this batch does not retrain them.

Next gate: implement and validate the selected pending trainers/datasets,
resolve image/resource/optimizer/loss/budget, publish canonical code, present
the whole batch, and obtain explicit approval for **one** submission.
