# kMoL Serving Environment — validated recipe & findings

This captures the **exact, tested** steps to make kMoL importable and servable, plus
the environment findings that matter for productionization. Everything here was
verified on an Anyscale workspace (CPU head node, conda/mamba available).

## What was proven ✅ (live in an Anyscale workspace, CPU)

- kMoL's conda env **builds cleanly** from `kmol/environment.yml` (348 pkgs, ~5 GB).
- kMoL **fully imports on CPU** with two tiny stubs and **no compiled extensions**.
- The full inference path works: `featurize → collate (one PyG Batch) →
  EnsembleNetwork forward (5 models + mean)` → `logits (N, n_targets)` + `variance`.
- **A live Ray Serve endpoint** (`scripts/serve_local.py`, Ray 2.51.2 / py3.9) serves
  single **and** batched molecule requests over HTTP. Verified:
  - **[REC 1] load-once:** exactly 5 "Restoring from Checkpoint" in the replica log —
    all checkpoints loaded once in `__init__`, never per request.
  - **[REC 2] batching:** a 4-molecule request returns 4 predictions; the same SMILES
    yields identical logits in single vs batch calls (correct, deterministic).
  - **[REC 6] warm-up:** `warm-up forward complete` before the replica is healthy.
  - **throughput:** ~80 molecules/sec on a **single CPU replica** with dynamic
    batching (one core, `set_num_threads(1)`). GPU + fractional packing scales up.

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
# 3b. kMoL's conda grpcio (1.54.2) segfaults Ray 2.51's runtime_env_agent → replace it.
$PIP install "grpcio==1.66.2"

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

### 3. grpcio conflict crashes Ray's runtime_env_agent
kMoL's conda `grpcio=1.54.2` segfaults Ray 2.51's `runtime_env_agent` (the raylet
fate-shares and the whole cluster fails to start). Fix: `pip install grpcio==1.66.2`
(the gRPC/federated path is unused by molecule serving).

### 4. Running kMoL's Ray inside an Anyscale workspace needs isolation
The workspace injects **two** runtime-env vars — `RAY_RUNTIME_ENV_HOOK=cgroup_runtime_plugin._hook`
AND `RAY_RUNTIME_ENV_PLUGINS=[{"class":"cgroup_runtime_plugin.CgroupV2Plugin",...}]`
— plus a default `RAY_ADDRESS` for the managed 2.56 cluster. The plugin var is loaded
by the runtime_env_agent, so unsetting only the hook is not enough. Because the shell
profile re-sources them per invocation, the reliable fix is to clear them **in-process
before `import ray`** (see `scripts/serve_local.py`) and start a fresh local cluster
in its own `_temp_dir`.

> ⚠️ **NEVER run `ray stop` in a shared workspace** — it kills the Anyscale-managed
> cluster and forces a full workspace recovery (node replacement, ~15-20 min, and the
> conda env — which lives outside `/home/ray/default` — is lost). Rely on process
> cleanup / cgroup teardown instead.

The clean production path avoids all of #3-#4: build a dedicated **Service image**
(Ray 2.51.2 + kMoL env + stubs + grpcio 1.66.2) and deploy it as an Anyscale Service,
where the image *is* the runtime and there is no host-Ray to collide with.
