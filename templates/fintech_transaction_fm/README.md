# Transaction Foundation Model — pretraining to serving on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/fintech_transaction_fm"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/fintech_transaction_fm" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 45 min

### Anyscale Technical Demo — Ray Data + Ray Train + Ray Serve

---

## Business Context

Banks and fintechs are converging on **transaction foundation models** (TFMs): a single self-supervised transformer pretrained on raw transaction sequences, producing a reusable **customer embedding** that powers fraud, churn, credit, and personalization — replacing dozens of hand-built feature pipelines. Stripe, Visa (TREASURE), Nubank, and Revolut (PRAGMA) have all published variants of this recipe.

The model itself is small and not the hard part. The hard parts are **engineering at scale**: tokenizing petabytes of transactions, pretraining across many GPUs, and re-embedding every customer on a schedule — then serving those embeddings both in batch and in real time.

**This template** builds the whole pipeline on Ray, with one upgrade over the standard NVIDIA blueprint: a **static/dynamic field split** in the tokenizer and model (the idea behind Visa TREASURE and FATA-Trans), which is cheaper and a stronger inductive bias than flattening every field into the token stream.

---

## Architecture

```
                       ┌─────────────────────── Anyscale ───────────────────────┐
 raw transactions ──►  │ [Ray Data]   static/dynamic tokenization (map_groups)   │
   (Parquet, S3)       │ [Ray Train]  masked-feature pretraining (PyTorch + DDP)│
                       │ [Ray Data]   batch embedding extraction (CPU read+GPU)  │
                       │ [XGBoost]    downstream fraud: raw vs FM vs fusion       │
                       │ [Ray Serve]  online embedding + fraud score (cached)     │
                       └─────────────────────────────────────────────────────────┘
```

Every stage is the **same code** from laptop to multi-node cluster — you change `ScalingConfig`, not your program.

---

**Walkthrough:** this notebook runs end-to-end at `smoke` scale (a few thousand cards, a 2-layer model) so it completes in minutes on a small cluster. Flip `SCALE` to `small`/`full` and `USE_GPU=True` for the real distributed story.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/fintech_transaction_fm
```

## Step 1: Connect to the Ray Cluster

In an Anyscale Workspace, Ray is **pre-initialized** — no cluster setup, no Spark context. The install cell pulls the template's dependencies; on the GPU base image PyTorch is already present.

> In production you'd install from the generated `python_depset.lock`. Here we install from `requirements.txt` for portability.


```python
!pip install -q -r requirements.txt
```


```python
import sys, os
DEMO_ROOT = os.path.abspath(os.getcwd())
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

import ray
ray.init(ignore_reinit_error=True, runtime_env={"working_dir": DEMO_ROOT})

from src.utils import print_cluster_resources
print_cluster_resources()
```

## Step 2: Load & inspect transaction data

By default we use the **real IBM TabFormer dataset** (Padhi et al., ICASSP 2021 — the public benchmark for transaction foundation models: 24.4M transactions, ~6.1k cards, 0.12% fraud), downloaded once (~266MB) and sampled down to the scale's card budget. Each *card* has **static** fields (issuer, card type, BIN region, home state — derived where TabFormer lacks them) plus a time-ordered stream of transactions with **dynamic** fields (amount, merchant, MCC, hour, day-of-week) and a fraud label.

The loader also writes `splits.json` with **temporal 80/10/10 cutoffs** (train on the past, test on the most recent 10%) — the same evaluation protocol as NVIDIA's transaction-FM blueprint, so downstream numbers are comparable.

> Fully offline alternative: `python scripts/01_generate_data.py --source synthetic` generates schema-identical synthetic data.



```python
import pandas as pd
from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir
from src.tabformer import prepare_tabformer

SCALE = "small"        # "small" / "full" for the distributed story
USE_GPU = True        # set True on a GPU cluster for train + embed

BASE_DIR = get_demo_base_dir()
paths = artifact_paths(BASE_DIR, SCALE)

if not (os.path.exists(paths["raw"]) and os.path.exists(paths["splits"])):
    prepare_tabformer(
        paths["raw"], paths["splits"],
        num_cards=SCALE_MAP[SCALE], source_dir=paths["source"],
    )

df = pd.read_parquet(paths["raw"])
print(f"{len(df):,} transactions / {df['card_id'].nunique():,} cards / fraud {df['is_fraud'].mean()*100:.3f}%")
df.head(4)
```

## Step 3: Tokenize with Ray Data — the static/dynamic split

This is the core idea. NVIDIA's blueprint flattens every transaction into ~12 tokens in one shared vocabulary, so a sequence of *N* transactions costs ~12*N* positions. We instead:

- embed **static** card-level fields **once** and broadcast them to every position (they never spend sequence length), and
- give each **dynamic** field its own embedding table, so each transaction is **one** position whose vector is the sum of its field embeddings.

The vocabulary is fully deterministic (fixed amount buckets + merchant hashing), so tokenization is a **stateless `map_groups`** with no global shuffle — exactly what Ray Data is built for. NVIDIA's RAPIDS path is single-GPU; this scales across the cluster.

Two representation choices live here. **Positions are time-aware**: alongside the ordinal position we embed the log-bucketed *gap since the previous transaction*, because for transactions *when* matters more than ordinal slot. And **amount uses bucketing** (`AMOUNT_MODE`), with an optional `"soft"` mode that blends the two adjacent bin embeddings so $86.99 and $87.01 don't land on unrelated vectors.

> **What you'll see while this runs** (and why it's fine):
> - **"Cluster does not have any available CPUs / job may hang"** — on a fresh or idle cluster the GPU worker is still launching; Ray Data waits and the warnings clear once it lands (~2 min). The head node intentionally schedules no work.
> - **Hash-shuffle aggregator warnings** — Ray's shuffle defaults assume a much bigger cluster; we right-size it per scale (`shuffle_partitions` + `max_hash_shuffle_aggregators` in `scripts/02_tokenize.py`).
> - **"Numba isn't available"** — `numba` (in `requirements.txt`) lets RayTurbo JIT the hash-partitioning kernel of this groupby; without it the shuffle falls back to slower Python.
> - **"Object store is configured to use only 28% of memory"** — set at cluster start on Anyscale; at this dataset size (~3GB) the default is plenty.



```python
import json
from ray.data.expressions import col
from src.tokenizer import SEQ_LEN_BY_SCALE, tokenize_dataset, write_vocab

seq_len = SEQ_LEN_BY_SCALE[SCALE]
with open(paths["splits"]) as f:
    splits = json.load(f)

ds = ray.data.read_parquet(paths["raw"])
tokenized = tokenize_dataset(
    ds, seq_len,
    train_end=splits["train_end"],   # temporal cutoffs -> pretrain never sees val/test
    val_end=splits["val_end"],
    normal_keep=0.005,               # downsample normal eval samples (all frauds kept)
    max_pretrain_windows=8,
).materialize()

PRETRAIN_DROP = ["kind", "split", "label", "weight",
                 "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts"]
tokenized.filter(expr=col("kind") == "pretrain").drop_columns(PRETRAIN_DROP) \
    .write_parquet(paths["tokenized_pretrain"])
tokenized.filter(expr=col("kind") == "eval").drop_columns(["kind"]) \
    .write_parquet(paths["tokenized_eval"])
write_vocab(paths["vocab"])

tok = pd.read_parquet(paths["tokenized_eval"])
print(f"{len(tok):,} eval samples ({int((tok['label'] == 1).sum()):,} fraud), seq_len={seq_len}")
row = tok.iloc[0]
print("  static:", {c: int(row[c]) for c in tok.columns if c.startswith("s_")})
print("  amount-bucket tokens:", list(row["d_amount_bucket"])[-8:])
print("  attention mask:", list(row["attention_mask"])[-8:])
```

## Step 4: Pretrain with Ray Train (masked-feature modeling, DDP)

We pretrain by **masking dynamic-field tokens and predicting them** (the Stripe / Open-Banking objective — bidirectional context beats next-token for the fixed-window tasks fintech cares about). There's **one classification head per dynamic field**, and because those heads differ wildly in difficulty (merchant is ~2000-way, day-of-week is 9-way) we balance them with **uncertainty weighting** (Kendall & Gal) so the big head doesn't dominate.

The training loop is plain PyTorch; **Ray Train** handles worker setup, dataset sharding, **DDP** wrapping, checkpointing, and fault tolerance. The model fits one GPU at every scale, so we use DDP (data-parallel) and scale by adding workers — go from 1 CPU worker (here) to N GPU workers by changing only `num_workers` and `use_gpu`. (`use_fsdp` is available for when the model itself outgrows a GPU.)


```python
from src.pretrain import pretrain

metrics = pretrain(
    tokenized_path=paths["tokenized_pretrain"],
    vocab_path=paths["vocab"],
    checkpoint_out=paths["checkpoint"],
    size=SCALE,
    max_len=seq_len,
    epochs=2,
    batch_size=64,
    num_workers=1,          # bump to 4-8 GPU workers at scale
    use_gpu=USE_GPU,
    use_fsdp=False,         # True for sharded multi-GPU training
    storage_base=BASE_DIR,  # shared storage — workers run on other nodes
)
print("final MLM loss:", round(metrics["mlm_loss"], 4))
```

## Step 5: Batch embedding extraction with Ray Data

The recurring production job: score every customer to a fresh embedding. This is a heterogeneous **CPU-read + GPU-infer** workload that streams through one Ray Data pipeline — the model loads once per replica, batches stream through, output is written idempotently. This is the stage with no clean public reference, and where Ray clearly earns its keep.


```python
from src.embed import extract_embeddings

extract_embeddings(
    tokenized_path=paths["tokenized_eval"],
    checkpoint_dir=paths["checkpoint"],
    output_path=paths["embeddings"],
    num_workers=1,          # scale out across GPU replicas at `full`
    use_gpu=USE_GPU,
)
emb = pd.read_parquet(paths["embeddings"])
print(f"{len(emb):,} transaction-window embeddings, dim={len(emb['embedding'].iloc[0])}")
```

## Step 6: Downstream fraud — raw vs FM vs fusion

The headline result, evaluated with the **NVIDIA transaction-FM blueprint protocol**: temporal 80/10/10 split, per-transaction last-event fraud labels, AUC-ROC + PR-AUC at natural fraud prevalence (downsampled normals are importance-weighted back). Same XGBoost recipe, three feature sets:

1. **raw** — the target transaction's tabular fields (what you have today)
2. **fm** — the FM embedding of the history window only
3. **fusion** — embedding ++ raw features (Nubank's joint fusion)

The lift of (2) and (3) over (1) is the case for a transaction FM. *(At `smoke` scale — 2 CPU epochs, a 2-layer model — expect fusion ≈ raw; the gap opens with the `small`/`full` GPU pretrain.)*



```python
from src.downstream import run_downstream, print_summary

summary = run_downstream(paths["embeddings"], paths["downstream"])
print_summary(summary)
```

## Step 7: Online serving with Ray Serve

The default production path is batch (Step 5) → feature store → XGBoost. But fraud also needs a real-time path, so we ship a **Ray Serve** deployment that mirrors the two-tier pattern real shops use: it **caches static (card-level) embeddings** and runs the transformer only over the recent dynamic window, returning an embedding + fraud score in one call. Autoscales 1→4 replicas.


```python
import requests
from ray import serve
from src.serve import build_app
from src.utils import sample_serve_payload

serve.run(build_app(paths["checkpoint"]), name="txn-fm")
payload = sample_serve_payload(paths["tokenized_eval"])
resp = requests.post("http://localhost:8000/", json=payload, timeout=30).json()
print("card_id   :", resp["card_id"])
print("embedding :", [round(x, 3) for x in resp["embedding"][:6]], "...")
print("fraud_score:", round(resp["fraud_score"], 4))
serve.shutdown()
```

## Step 8: Observability & fault tolerance

**Observability** (built into Anyscale)
- **Ray Dashboard** — watch data stream through tokenize/embed stages, per-worker throughput, GPU utilization.
- **Ray Train reports** — per-epoch MLM loss, checkpoint lineage.
- **Serve metrics** — latency, replica autoscaling, ongoing requests.

**Fault tolerance** (built into Ray)
- Ray Data retries failed batches per-partition — no full restart.
- Ray Train checkpoints let pretraining resume after a node loss.
- Streaming + backpressure keeps memory bounded across stages.

## Step 9: Path to production

The same code scales up by changing config, and runs as a scheduled **Anyscale Job**. `scripts/run_pipeline.py` wraps all six stages (data -> tokenize -> pretrain -> embed -> downstream -> validate) in one command:

```bash
# Full pipeline as a Job (GPU workers, autoscaling):
anyscale job submit --config-file job_config.yaml

# Larger scale:
anyscale job submit --config-file job_config.yaml \
  --override-entrypoint 'python scripts/run_pipeline.py --scale full'
```

| Stage | Ray primitive | Scale knob |
|-------|---------------|-----------|
| Tokenize | Ray Data `map_groups` | partitions / cluster size |
| Pretrain | Ray Train + DDP | `num_workers`, `use_gpu` |
| Batch embed | Ray Data `map_batches` | `num_workers`, `num_gpus` |
| Online serve | Ray Serve | replica autoscaling |


## Validate

A quick end-to-end check that every stage produced sane artifacts.


```python
from scripts.validate_results import validate_pipeline, print_report
print_report(validate_pipeline(paths))
```
