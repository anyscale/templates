# kMoL Serving Environment — validated recipe & findings

This captures the **exact, tested** steps to make kMoL importable and servable, plus
the environment findings that matter for productionization. Everything here was
verified on an Anyscale workspace (CPU head node, conda/mamba available).

## What was proven ✅

- kMoL's conda env **builds cleanly** from `kmol/environment.yml` (348 pkgs, ~5 GB).
- kMoL **fully imports on CPU** with two tiny stubs and **no compiled extensions**.
- The full inference path works end-to-end: **5 checkpoints load once**, then
  `featurize → collate (one PyG Batch) → EnsembleNetwork forward (5 models + mean)`
  returns `logits (N, n_targets)` + ensemble `variance`. (Plan Stages 1 & 2.)

## The recipe (CPU, verified)

```bash
# 0. kMoL source (pinned)
git clone --depth 1 https://github.com/elix-tech/kmol.git   # commit c7f8833

# 1. Conda env (py3.9, torch 1.13.1). ~5 GB, a few minutes.
mamba env create -f kmol/environment.yml                    # creates env "kmol"
PY=$(conda info --base)/envs/kmol/bin/python
PIP=$(conda info --base)/envs/kmol/bin/pip

# 2. kMoL pip deps NOT in environment.yml (light ones only; skip deepspeed/openfold).
$PIP install "disklist==0.4.0" "filelock==3.12.4" "boxsdk[jwt]==3.6.1" \
             "torch-lr-finder==0.2.1" "ml-collections==0.1.0" "proDy" "biopython==1.81"

# 3. Ray Serve. IMPORTANT: py3.9 caps Ray at 2.51.2 (2.56 has NO py3.9 wheel).
$PIP install "ray[serve]==2.51.2" requests

# 4. Stubs on PYTHONPATH so the eagerly-imported protein/graphormer paths load
#    without compiling CUDA/Cython (molecule serving never calls them).
#    -> stubs/attn_core_inplace_cuda.py, stubs/algos.py (shipped in this template)

export PYTHONPATH="$PWD/stubs:$PWD:$PWD/kmol/src"
$PY -c "from kmol.model.executors import Predictor; print('IMPORT_OK')"
```

## Critical findings for productionization

### 1. Python 3.9 is the hard constraint — and it ceilings Ray
kMoL pins **Python 3.9 / PyTorch 1.13.1 / CUDA 11.7** (conda). Consequences:
- **Newest Ray on py3.9 is 2.51.2.** The Anyscale base image uses Ray **2.56** on
  py3.11 — so kMoL **cannot** run in the workspace's managed Ray in-process, and a
  managed **Service image must pin Ray ≤ 2.51** (with a matching runtime) **or** kMoL
  must be ported to py3.11 + modern torch/PyG. This is the single biggest decision.

### 2. Import-time coupling forces two stubs (not code edits)
`architectures/__init__.py` and `featurizers.py` unconditionally import the
AlphaFold/openfold + graphormer paths, which need compiled `attn_core_inplace_cuda`
(CUDA, GPU-at-build) and `algos` (Cython). The molecule path never *calls* them, so
we provide stub modules on PYTHONPATH. `deepspeed`/`flash_attn` are already guarded
by openfold (`importlib.util.find_spec`) and can be omitted.

### 3. Running kMoL's Ray inside an Anyscale workspace needs isolation
The workspace injects `RAY_RUNTIME_ENV_HOOK=cgroup_runtime_plugin` and a default
`RAY_ADDRESS` for the managed 2.56 cluster. To run the kMoL (2.51.2) Serve app
locally you must `unset RAY_RUNTIME_ENV_HOOK RAY_ADDRESS` and start an **isolated**
head on non-default ports (`--port 6399 --temp-dir /tmp/kmolray`).

> ⚠️ **NEVER run `ray stop` in a shared workspace** — it kills the Anyscale-managed
> cluster and forces a workspace recovery. Rely on process cleanup instead.

The clean production path avoids all of this: build a dedicated **Service image**
(Ray 2.51.2 + kMoL env + stubs) and deploy it as an Anyscale Service, where the
image *is* the runtime and there is no host-Ray to collide with.
