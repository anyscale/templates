# Serving a kMoL GNN Ensemble on Ray Serve / Anyscale

Wraps a trained [kMoL](https://github.com/elix-tech/kmol) **5-model GNN ensemble**
(PyTorch Geometric) as a batched, autoscaling Ray Serve endpoint on Anyscale.

**We do not modify kMoL.** This template imports kMoL's own primitives — `Config`,
`Predictor` / `EnsembleNetwork`, the preprocessor (featurizers + transformers), and
`GeneralCollater` — and wraps them in a Ray Serve deployment. kMoL is a dependency
(installed into the image), never vendored or edited.

---

## Why this exists — the throughput problem

kMoL's offline `predict` path (`src/kmol/run.py` → `_collect_predictions`) does two
expensive things **on every call**:

1. Builds a fresh `Predictor`, which reloads **all 5 checkpoints from disk**.
2. Builds a fresh `GeneralStreamer`, which **reloads and splits the whole dataset**.

At batch size ~1 that overhead dwarfs the actual GPU forward of a small GNN. This
template moves all of that to replica startup and batches the forwards, targeting
**~4× the current per-GPU throughput and linear scaling as GPUs are added.**

---

## How the design maps to the review recommendations

| # | Recommendation | Where it lives |
|---|----------------|----------------|
| **1** | Load the ensemble once, not per predict | `KmolEnsemble.__init__` builds `Predictor` once → `EnsembleNetwork.load_checkpoint` loads all 5 checkpoints a single time. |
| **2** | Batch forward passes into one PyG `Batch` | `@serve.batch` collects requests; `GeneralCollater.apply(list[DataPoint])` builds **one** `torch_geometric.data.Batch`; one forward per model. |
| **3** | Re-run GPU-vs-CPU after 1+2 | `client.py --mode bench` gives batched numbers. Hold the CPU/GPU hybrid until these are in. |
| **4** | Validate the load generator | Run `bench` from a **separate node** at high concurrency; watch GPU util at peak (low util ⇒ you're benchmarking the client, not the cluster). |
| **5** | `torch.set_num_threads(1)` on CPU replicas | Called in `__init__`; `OMP_NUM_THREADS` also set in `service.yaml`. |
| **6** | `min_replicas=1` + warm-up forward | `autoscaling_config.min_replicas: 1`; `_warmup()` runs a real featurize→collate→5-model forward before the replica is healthy. |
| **7** | Keep all 5 checkpoints in one replica; fractional GPUs | `EnsembleNetwork.forward` averages the 5 sub-models in-graph (`torch.mean`) — no fan-out/fan-in. `num_gpus: 0.25` packs 4 replicas per L4. |

### Key correctness facts (verified in the kMoL source)

- **Averaging is native.** `EnsembleNetwork.forward` runs the 5 sub-models and returns
  `torch.mean(...)` plus `logits_var`. Sharding the ensemble across replicas would
  fight the framework — one replica, one forward, one mean is the intended path.
- **No dataset needed at serve time.** We skip `GeneralStreamer` and build the
  preprocessor directly. kMoL's transformers (`LogNormalize`, `MinMaxNormalize`,
  `FixedNormalize`) take their parameters from **config, not from fitted data**, so
  this reproduces the offline preprocessing exactly.
- **Output semantics match the offline `predictions.csv`.** We return exactly what
  `Predictor.run().logits` produces for your config (pre-threshold). Apply the same
  threshold/sigmoid the offline pipeline does on the client side.

---

## Repository layout

```
biotech_kmol_serving/
├── serve_app.py                       # entrypoint: app = build_app(KMOL_CONFIG_PATH)
├── src/kmol_ensemble.py               # the Ray Serve deployment (the wrapper)
├── configs/ensemble_serve.example.json# example 5-model ensemble config
├── client.py                          # single / batch / bench client
├── service.yaml                       # Anyscale Service config
├── Dockerfile                         # ray[serve] layered on kMoL's image
├── setup_local.sh                     # clone kMoL for local dev (gitignored)
└── requirements.txt                   # serving-layer overlay only
```

---

## Environment — the one hard part

kMoL is a **conda / CUDA 11.7 / Python 3.9 / PyTorch 1.13.1** stack. It does **not**
drop into a stock Ray `py311/cu12` image, and its install compiles a CUDA extension
that needs a GPU at *build* time.

**Recommended:** base the image on the kMoL image Takeda already builds
(`make build-docker` → `elix-kmol:<version>`) and layer Ray on top — see `Dockerfile`.
This sidesteps the whole build-time GPU requirement.

**Two coordination points to confirm with the team:**
1. The Ray version must support Python 3.9 **and** be a version your Anyscale cloud
   supports (2.9.x fits both). Longer term, aligning kMoL's env to py3.11 / CUDA 12
   would let you use stock Anyscale images.
2. The molecule-serving path never imports the OpenFold CUDA kernel (only the
   protein/MSA path does), so runtime needs no GPU-compiled extension.

---

## Run it

### 1. Provide your artifacts
Edit `configs/ensemble_serve.example.json`:
- Replace the 5 `model_configs` with your real sub-model architecture.
- Point the 5 `checkpoint_path` entries at your trained checkpoints.
- Set `loader.input_column_names` / `target_column_names` to match your data.

### 2. Local smoke test (inside kMoL's conda env)
```bash
./setup_local.sh
export PYTHONPATH="$PWD/kmol/src:$PWD"
serve run serve_app:app
python client.py --mode single
```

### 3. Deploy on Anyscale
```bash
# Build & push the image, set image_uri in service.yaml, then:
anyscale service deploy -f service.yaml
anyscale service status --name biotech-kmol-ensemble-serving
```

### 4. Get real numbers (recommendations 3 & 4)
```bash
# From a SEPARATE node, open-loop, high concurrency:
python client.py --url <service-url> --token <token> --mode bench --n 5000 --concurrency 128
```
Watch GPU utilization at peak. If it's low, you're bottlenecked on the client or on
featurization — not on Ray.

---

## Tuning knobs (env vars, no code changes)

| Var | Default | Meaning |
|-----|---------|---------|
| `KMOL_MAX_BATCH_SIZE` | 64 | Max requests coalesced per forward. |
| `KMOL_BATCH_WAIT_S` | 0.02 | Max wait to fill a batch (keep under your latency SLO). |
| `KMOL_NUM_GPUS` | 0.25 | Fraction of a GPU per replica (0.25 ⇒ 4 replicas/card). |
| `KMOL_MIN_REPLICAS` / `KMOL_MAX_REPLICAS` | 1 / 8 | Autoscaling bounds. |
| `KMOL_TARGET_ONGOING` | 32 | Target in-flight requests per replica before scaling up. |

---

## Moving featurization off the GPU replica (planned next step)

RDKit featurization (`GraphFeaturizer._process`: `Chem.MolFromSmiles` → PyG graph) is
the real per-request **CPU** cost and currently runs inline on the GPU replica. It is
deliberately isolated in `KmolEnsemble._featurize_one`. To split it later (recommendation
5's hybrid): make a CPU-only `Featurizer` deployment that returns `DataPoint`s and have
the GPU deployment only collate + forward — a two-stage Serve composition. Do this
**after** the batched GPU numbers are in, per recommendation 3; batching alone may
already clear the 4× bar.
