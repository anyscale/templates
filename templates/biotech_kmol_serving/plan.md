# kMoL Ensemble Serving — Execution Plan

Turns the review email into staged, verifiable work. Each stage has a **goal**, an
**action**, and a **verify** (the check that lets us move on). Stages map 1:1 to the
email's recommendations (`REC n`).

**Target outcome:** ~4× the current per-GPU throughput, then linear scaling as GPUs
are added.

**Workspace:** `expwrk_9e9qajmqr7w6astmetm8v9tv9s`
(project `prj_cz951f43jjdybtzkx1s5sjgz99`, cloud `cld_kvedZWag2qA8i5BjxUevf5i7`).

---

## Prerequisites / open inputs

| Need | Status | Notes |
|------|--------|-------|
| kMoL importable on the cluster | ⛔ blocker | Conda/CUDA-11.7/py3.9 stack; build image (Dockerfile) or install env. |
| 5 trained checkpoints | ⛔ needed for accuracy | For **throughput** work we can use synthetic checkpoints of the example architecture — mechanics are identical. Swap real ones in later. |
| Real sub-model architecture | ⛔ needed for accuracy | Example uses tox21 `graph_convolutional`. |
| GPU nodes in workspace compute | ❓ verify | L4 (g6) targeted; fractional-GPU packing needs GPUs present. |

> **Strategy:** stages 0–7 (the throughput mechanics the email is about) can be
> proven with **synthetic checkpoints** on the example architecture. Real weights are
> a drop-in swap and only affect prediction values, not throughput.

---

## Stage 0 — Environment

**Goal:** `import kmol` and `import ray` both succeed on the workspace, GPU visible.
**Action:**
- Build the image (`Dockerfile`: `ray[serve]` on kMoL's prebuilt image), OR create
  kMoL's conda env in the workspace and `pip install ray[serve]` into it.
- Confirm a GPU is attached to the workspace compute.
**Verify:**
```bash
python -c "import kmol, ray, torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
**Artifact:** working image URI / env.

---

## Stage 1 — Load the ensemble once (REC 1)

**Goal:** Checkpoints load exactly once per replica, never per request.
**Action:** Deploy `serve_app:app` (the `KmolEnsemble` deployment). All 5 checkpoints
load in `__init__` via `Predictor` → `EnsembleNetwork.load_checkpoint`.
**Verify:**
- Replica logs show "Restoring from Checkpoint" 5× at startup, **zero** on subsequent
  requests.
- 2nd request latency ≪ 1st (no reload cost).
**Artifact:** `serve run` up; single prediction returns.

---

## Stage 2 — Dynamic batching (REC 2)

**Goal:** Many concurrent requests coalesce into ONE `torch_geometric.data.Batch`,
one forward per model.
**Action:** `@serve.batch` (already wired) + `GeneralCollater.apply`. Tune
`KMOL_MAX_BATCH_SIZE`, `KMOL_BATCH_WAIT_S`.
**Verify:** at concurrency ≫ 1, throughput (molecules/sec) rises sharply vs Stage 1;
batch sizes > 1 visible in logs/metrics.
**Artifact:** throughput-vs-concurrency curve.

---

## Stage 3 — Re-measure GPU vs CPU (REC 3)

**Goal:** A fair GPU throughput number *with batching on* — replaces the batch-1
5 ms/molecule figure.
**Action:** `client.py --mode bench` at increasing concurrency; record molecules/sec
per GPU. Repeat CPU-only if desired.
**Verify:** batched GPU molecules/sec recorded; compare to the 4× cost bar. Hold the
CPU/GPU hybrid decision until this exists.
**Artifact:** benchmark table (concurrency × batch × molecules/sec × p50/p99).

---

## Stage 4 — Validate the load generator (REC 4)

**Goal:** Confirm scaling tests measure the cluster, not the client.
**Action:** Drive `bench` **open-loop from a separate node** at high concurrency;
watch GPU + CPU utilization at peak.
**Verify:** GPU util is high at peak (if low → client/featurization bound, not Ray).
The 150→120 regression when adding a node is explained (client-side vs cluster-side).
**Artifact:** utilization screenshots + notes.

---

## Stage 5 — Thread / CPU hygiene (REC 5)

**Goal:** Multiple replicas on one box don't oversubscribe cores.
**Action:** `torch.set_num_threads(1)` (in `__init__`) + `OMP_NUM_THREADS` (service
env). Test several replicas per node.
**Verify:** throughput does not regress as replicas-per-node increases; no CPU
thrash.
**Artifact:** replicas-per-node sweep.

---

## Stage 6 — Warm start + autoscaling (REC 6)

**Goal:** No cold-start cost on the request path; scales on load.
**Action:** `min_replicas: 1`, `_warmup()` real forward, autoscaling bounds.
**Verify:** first request after a fresh deploy is fast (warm); replicas scale up
under `bench` load and back down after.
**Artifact:** autoscaling event log.

---

## Stage 7 — Consolidate ensemble + fractional GPU (REC 7)

**Goal:** One replica holds all 5 models; pack replicas per card; linear GPU scaling.
**Action:** Keep single-replica ensemble (native `torch.mean`); set `num_gpus`
fractional (e.g. 0.25). Add GPU nodes and re-bench.
**Verify:** N replicas per card healthy; aggregate throughput scales ~linearly with
GPU count.
**Artifact:** throughput-vs-GPU-count curve (the linear-scaling proof).

---

## Stage 8 — (stretch) Featurization off the GPU replica

**Goal:** If featurization (RDKit) caps GPU util, move it to CPU replicas.
**Action:** Two-stage Serve composition: CPU `Featurizer` deployment → GPU
`collate+forward` deployment. Only if Stage 4 shows featurization-bound.
**Verify:** GPU util rises; molecules/sec/GPU improves over Stage 7.
**Artifact:** hybrid topology + before/after numbers.

---

## Progress log

- [ ] Stage 0 — environment
- [ ] Stage 1 — load once
- [ ] Stage 2 — batching
- [ ] Stage 3 — GPU/CPU re-measure
- [ ] Stage 4 — load-gen validation
- [ ] Stage 5 — thread hygiene
- [ ] Stage 6 — warm start + autoscale
- [ ] Stage 7 — consolidate + fractional GPU
- [ ] Stage 8 — featurization split (stretch)
