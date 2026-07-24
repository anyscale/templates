# kMoL Ensemble Serving — Plan & Handoff

**Goal:** Serve Takeda's kMoL 5-model GNN ensemble on Ray Serve / Anyscale and hit the
throughput recommendations from Geoff's email — load the ensemble once, dynamically
batch forward passes, and get real **GPU** throughput (~4× the current per-GPU number,
then linear scaling). The email's premise: the model is a *small* GNN, so the old
"~5 ms/molecule" figure is per-call overhead at batch size 1; batching should blow past it.

**This doc is a fresh-start handoff.** Read the post-mortem, then start at "The plan → P0".
You do **not** need the previous session's context beyond what's written here.

---

## TL;DR status

- ✅ Serving **logic** is proven correct (load-once, dynamic PyG batching, ensemble
  averaging + variance) — verified live on **CPU**.
- ❌ **No GPU numbers yet.** That is the whole remaining job.
- 🔀 **The approach is changing.** Stop trying to run kMoL's frozen py3.9 stack next to
  Anyscale's modern Ray. **Port the molecule path to modern PyTorch and run it as normal
  Ray Serve actors.** Details below.

---

## Post-mortem — where the last attempt went wrong (so you don't repeat it)

**Root mistake:** I optimized for "don't modify kMoL / don't build a container" instead of
for the goal (GPU throughput). That made kMoL's **Python 3.9** pin feel immovable, which
cascaded into a dead end:

1. kMoL pins **py3.9 / torch 1.13 / CUDA 11.7**. The newest Ray with a py3.9 wheel is
   **2.51.2**. Anyscale's workspace runs **Ray 2.56 / py3.11**. Ray needs one version
   cluster-wide, so kMoL **cannot run as actors on the workspace's managed cluster.**
2. To get *anything* working I stood up a **second, isolated Ray 2.51.2 cluster** inside
   kMoL's env. It served on CPU — but an isolated cluster **can't use the workspace's GPU
   autoscaler** (those L4 workers belong to the managed 2.56 cluster).
3. So I tried to stage kMoL's **11 GB conda env** onto shared `/mnt` to run it on a GPU
   node. This workspace's NFS is far too slow for that (a conda-clone ran 26 min+ without
   finishing; `tar` blew the command budget). **Dead end.**

**The clean answer we should have taken:** the molecule GNN path uses **only standard
PyTorch + PyTorch-Geometric ops — no custom CUDA kernels** (the only custom kernel is
OpenFold's, protein-only, which the molecule path never calls). The checkpoints are plain
`state_dict`s. So **port the molecule path to modern torch/py3.11**, and it runs as ordinary
Ray Serve actors on the managed cluster with native GPU autoscaling — no separate Ray, no
NFS, no container gymnastics.

**Two hard-won gotchas — keep these:**
- ⚠️ **Never run `ray stop` in the shared workspace.** It killed the managed cluster and
  forced a ~20-min recovery (node replaced, node-local conda env wiped).
- Node-local installs (`/home/ray/anaconda3/...`) do **not** survive node replacement;
  `/home/ray/default` and `/mnt/*` persist.

**Also likely true (and why the port isn't optional):** kMoL's **torch 1.13 / cu117 probably
can't even run on the L4 GPUs** (L4 is Ada / sm_89, newer than torch 1.13). Never got to
test it — but porting to modern torch removes this risk entirely. So the port isn't just
cleaner; it's plausibly *required* to use these GPUs at all.

---

## The plan

### P0 — Port spike (DO THIS FIRST; it's the make-or-break)
Get the kMoL **molecule** ensemble running on **Python 3.11 + current PyTorch + PyG**,
loading the existing 5 checkpoints, with outputs that **match the py3.9 version numerically.**

Steps:
1. In the workspace (py3.11 base, which has modern torch), `pip install` current
   `torch-geometric`, `rdkit`, and whatever else the molecule path needs.
2. Make kMoL's molecule model + graph featurizer importable on py3.11. **Try importing
   kMoL's classes first**; if `openbabel`/`prody` (imported at the top of `config.py` and
   `data/resources.py`) block the py3.11 install, **extract the minimal path** instead —
   the `graph_convolutional` architecture + the rdkit graph featurizer + the ensemble
   mean. Both are small and well-defined (see "Technical reference").
3. Load the 5 `state_dict`s into the modern model.
4. **Parity check** against ground truth: the previous session left a *working py3.9 kMoL*
   in this workspace (see "Ground truth"). Run the same molecules through both and confirm
   logits match to tolerance. This validates the port AND proves torch works on the L4.

**Success = ported model produces matching logits on a GPU.** Everything after is easy.

### P1 — GPU throughput on Ray Serve (native, no hacks)
Reuse the existing serve wrapper logic (`src/kmol_ensemble.py`) but with the **ported**
model, deployed as a normal Ray Serve app on the **managed** cluster. Request fractional
GPU (`num_gpus: 0.25`) so several replicas pack one L4 (REC 7) and GPU workers autoscale.
Measure batched throughput vs batch size (reuse `scripts/gpu_microbench.py` logic).
**Success = molecules/sec/GPU that clears the 4× bar; report the batch-size curve.**

### P2 — Locust load test (REC 4)
`scripts/locustfile.py` exists. Drive it **open-loop from a separate process** at high
concurrency; watch GPU utilization at peak (low util ⇒ client/featurization-bound, not Ray).
**Success = sustained throughput + p50/p99 under load, GPU util high at peak.**

### P3 — Containerize → Anyscale Service
Now trivial: modern base image (`anyscale/ray:<ver>-py311-cu12x`) + `pip install` the ported
deps. Deploy `service.yaml` as an Anyscale Service on GPU workers. **Do this LAST**, only
after P1/P2 prove out. (Geoff's ordering; he approves GPU spend / compute changes.)

---

## What's genuinely reusable (don't rebuild these)

| Asset | Status |
|---|---|
| `src/kmol_ensemble.py` — Serve deployment (load-once, `@serve.batch`+PyG collate, warm-up, fractional GPU) | **Logic is correct & reusable** — just swap the model to the ported one and run on the managed cluster's Ray (drop the isolated-cluster bits). |
| `scripts/gpu_microbench.py` | Batched throughput harness (pure torch, no Ray) — reuse. |
| `scripts/locustfile.py` | Load test — reuse. |
| `scripts/make_synthetic_checkpoints.py` | Generate weights for testing without the real ones. |
| 7-recommendation → kMoL-class mapping | Validated (see README.md). |
| `ENVIRONMENT.md` | The py3.9 recipe — keep ONLY as a fallback if the port proves infeasible and we must containerize the old stack. |
| `stubs/` | Only needed if you import *all* of kMoL (protein path). A minimal port won't need them. |

---

## Technical reference (what the next engineer needs)

- **kMoL:** github.com/elix-tech/kmol @ `c7f8833`. Molecule model type `graph_convolutional`
  (`GraphConvolutionalNetwork`): `torch_geometric.nn.LEConv` ×7 layers, `BatchNorm`, `ReLU`,
  hidden 96, `in_features` 45, `molecule_features` 17, `out_features` = #targets (12 for the
  tox21 example). **~264K params per sub-model** — tiny.
- **Ensemble:** `EnsembleNetwork` holds 5 sub-models in one `ModuleList`; `forward` returns
  `torch.mean` of the 5 (+ `torch.var` as `logits_var`). Averaging is native — do NOT shard.
- **Checkpoints:** `torch.load(path)["model"]` is a plain `state_dict`; loaded via
  `load_state_dict(strict=False)`. Version-agnostic → this is why the port works.
- **Featurizer:** `graph` featurizer = `rdkit.Chem.MolFromSmiles` → atom/bond features →
  `torch_geometric.data.Data`. `GeneralCollater` batches a `list[Data]` into one PyG `Batch`.
- **Config shape:** see `configs/ensemble_serve.example.json` (model `type: ensemble` with 5
  `model_configs`; `checkpoint_path` is a **list** of 5).

**Workspace:** `expwrk_9e9qajmqr7w6astmetm8v9tv9s`
(project `prj_cz951f43jjdybtzkx1s5sjgz99`, cloud `cld_kvedZWag2qA8i5BjxUevf5i7`).
Head `m5.2xlarge` (CPU); GPU workers `g6.2xlarge` (**L4**, autoscale 0-4) on the managed
cluster; image `anyscale/ray:2.56.0-py311-cu121`. Drive it with
`anyscale workspace_v2 run_command --id <id> '...'` (kills its process tree on exit — run
long work detached + poll; the GPU playbook is in the `geoff/fm_recs_and_fraud` branch's
`fintech_transaction_fm/claude-anyscale/`).

**Ground truth for parity (left in the workspace):**
- Working py3.9 kMoL env: `/home/ray/anaconda3/envs/kmol` (may be wiped on next node
  replacement — rebuildable via `scripts/setup_env.sh`).
- Code + 5 synthetic checkpoints: `/home/ray/default/kmol_serving/` and
  `/mnt/cluster_storage/kmol_serving/`.
- To get reference logits: run `scripts/gpu_microbench.py` / the standalone path with that
  env and record outputs for a fixed SMILES set, then match the ported model against them.

---

## Measurements so far (CPU only — NOT the goal, just proof the plumbing works)

- Live Ray Serve endpoint: single + batched requests → 12 logits + labels + ensemble variance.
- **[REC 1] load-once:** exactly **5** checkpoint restores in the replica log.
- **[REC 2] batching:** same SMILES → identical logits single vs batched (deterministic).
- **[REC 6] warm-up:** confirmed before healthy.
- **Throughput: ~80 molecules/sec on ONE CPU core** (single replica, `set_num_threads(1)`)
  with dynamic batching. This is a per-core plumbing baseline, **not** comparable to the
  team's ~150-170/s (a whole-box number) and **not** the GPU answer. GPU + batching is the win.

---

## Open risks / unknowns for the port
- `openbabel` / `prody` are imported at kMoL module load and are conda-friendly / py3.11-wheel-
  iffy → may force the "extract minimal molecule path" route rather than importing all of kMoL.
- PyG API drift (2.3 → current): check `LEConv`, `Data`, `Batch`, collate. Mostly stable.
- Numerical parity: verify against the py3.9 ground truth before trusting the port.
- Decide early: import full kMoL on modern env vs. reimplement the ~264K-param GCN + featurizer
  faithfully and load weights. The parity check makes either safe.

---

## Progress log
- [x] Wrapper/serve logic designed & CPU-proven (reusable)
- [x] 7-rec → kMoL-class mapping validated
- [x] **P0 — port spike DONE.** Minimal port `port/kmolport/` (torch/PyG/rdkit only, no
  openbabel/prody/openfold/torch_scatter). Parity: **0.000e+00** on CPU, **1.14e-05** on
  an **L4** (torch 2.5.1+cu124) vs py3.9 kMoL. Throughput: **60,101 mol/s forward-only
  (353× the 170/s baseline)**; end-to-end **~561 mol/s (3.3×)**, featurization-bound (GPU
  has ~100× headroom). Confirmed torch runs on the L4. See `port/README.md` + `port/gpu_results.json`.
- [x] **P1 — served on Ray Serve DONE.** 6 fractional-GPU replicas (`num_gpus=0.16`)
  packed on ONE autoscaled L4, native on the managed cluster (runtime_env installed
  CUDA torch, no container). Bulk/screening workload: **1,697 mol/s = 10.0× baseline**
  (`port/serve_bulk_results.json`). CPU-featurization-bound, not GPU-bound. Naive
  1-SMILES-per-request tops out ~244/s = the client/RPC limit, not the service (REC 4).
  Scripts: `port/scripts/serve_bulk.py` (+ `serve_run.py`).
- [x] **P1b — two-stage pipeline DONE (the real throughput lever).** 12 CPU featurizer
  actors → 1 GPU forward actor (pure Ray, no Serve overhead): **3,904 mol/s = 23.0×
  baseline** (`port/pipeline_results.json`), within 4% of the featurize-only rate
  (4,050/s) — so the L4 is ~7% utilized; throughput scales linearly with featurizer
  cores toward the 60k/s GPU ceiling. `port/scripts/scaled_pipeline.py`.
- [ ] P2 — locust load test (independent multi-process HTTP confirmation). Optional now;
  three methods already agree the GPU story holds.
- [ ] P3 — containerize → Anyscale **Service** (modern `anyscale/ray:*-py311-cu12x` +
  ported deps). **Deploy is gated on Geoff** (persistent GPU spend). Files can be
  prepared; the actual `anyscale service deploy` waits for approval.

**Bottom line: single-GPU story proven three ways** — 60k mol/s forward (353×), 1,697/s
served (10×), 3,904/s pipeline (23×). All featurization-bound with GPU headroom; the 4×
target is cleared by a wide margin.
