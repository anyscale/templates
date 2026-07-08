# Transaction Foundation Model — pretraining to serving on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/fintech_transaction_fm"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/fintech_transaction_fm" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 45 min (notebook walkthrough) / ~5 h of unattended jobs (full reproduction)

### Anyscale Technical Demo — Ray Data + Ray Train + Ray Serve

---

## What this is

One self-supervised transformer pretrained on raw transaction sequences, producing a per-card **embedding** that beats the **NVIDIA transaction-FM blueprint** on its own TabFormer fraud protocol — with a plain linear head on the embedding, no fusion with hand-built features. The trick is the tokenizer: **1 position per transaction** (static/dynamic field split) instead of their ~12 tokens per transaction, so a 512-position window covers **512 transactions of history** vs ~315 in their 4096-token context — and scales to 1024/2048.

## Reproduce the headline — three commands

Each command is an Anyscale Job against durable storage (`/mnt/user_storage`); run them in order from this directory:

```bash
# 1. THE GATE — reproduce NVIDIA's published baseline through this pipeline (CPU)
anyscale job submit -f job_baseline.yaml     # -> Test ROC-AUC ~0.9875 / AP ~0.1421

# 2. THE HEADLINE — pretrain the FM from scratch + print the fraud table (4-8x A10G)
anyscale job submit -f job_full.yaml

# 3. THE SCALING STORY — same recipe at 1024 transactions of context
anyscale job submit -f job_xl.yaml
```

Expected table from step 2 (seq 512, TabFormer, test split at natural ~0.1% fraud):

| model            | what it is                                    | ROC-AUC | AP        |
|------------------|-----------------------------------------------|---------|-----------|
| baseline         | their 13 raw features + their XGBoost         | ~0.987  | ~0.14     |
| embed_pca64_xgb  | their notebook-05 protocol on our embedding   | ~0.98   | ~0.16     |
| embed_logistic   | our embedding, no PCA, linear head            | ~0.98   | **~0.23+**|
| embed_xgb        | our embedding, no PCA, XGBoost                | ~0.98   | **~0.23+**|
| *NVIDIA published* | *their baseline / their fusion*             | *0.9885 / 0.9925* | *0.1238 / 0.1755* |

**Hardware**: the job yamls autoscale 0→8 `g5.xlarge` (A10G) GPU workers + CPU workers on `aws-public-us-west-2`; edit `compute_config` for your cloud. Pretraining at `full` is ~4 h on 4x A10G.

---

## Business context

Banks and fintechs are converging on **transaction foundation models** (TFMs): a single self-supervised transformer pretrained on raw transaction sequences, producing a reusable **customer embedding** that powers fraud, churn, credit, and personalization — replacing dozens of hand-built feature pipelines. Stripe, Visa (TREASURE), Nubank, and Revolut (PRAGMA) have all published variants of this recipe.

The model itself is small and not the hard part. The hard parts are **engineering at scale**: tokenizing petabytes of transactions, building a vocabulary over 100M+ merchants, pretraining across many GPUs, and re-embedding every customer on a schedule — then serving those embeddings both in batch and in real time. The model follows **FATA-Trans** (static/dynamic field split + time-aware positions + masked-feature pretraining) and adopts **Visa TREASURE's** InfoNCE with shared negative sampling so the real ~100k-merchant vocabulary is tractable — the same embedding also ranks next-merchant recommendations as a template extra.

---

## Architecture

```
                       ┌─────────────────────── Anyscale ───────────────────────┐
 raw transactions ──►  │ [Ray Data]   distributed merchant vocab (freq + top-K)  │
   (Parquet, S3)       │ [Ray Data]   static/dynamic tokenization (map_groups)   │
                       │ [Ray Train]  masked pretrain + InfoNCE (PyTorch + DDP)  │
                       │ [Ray Data]   batch embedding extraction (CPU read+GPU)  │
                       │ ├─ [XGBoost] fraud: raw baseline vs FM embedding        │
                       │ └─ [rank]    recommendation: next-merchant (HR@K/NDCG@K)│
                       │ [Ray Serve]  online embedding + fraud score (cached)     │
                       └─────────────────────────────────────────────────────────┘
```

Every stage is the **same code** from laptop to multi-node cluster — you change `ScalingConfig`, not your program.

---

**Walkthrough:** the rest of this notebook runs the same pipeline end-to-end at `small` scale on a modest GPU cluster (1-2 GPU workers, ~45 min). Flip `SCALE` to `full` for the real distributed story, or drop to `smoke` + `USE_GPU=False` for a CPU-only run in minutes.

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

By default we use the **real IBM TabFormer dataset** (Padhi et al., ICASSP 2021 — the public benchmark for transaction foundation models: 24.4M transactions, ~6.1k cards, 0.12% fraud), downloaded once (~266MB) and sampled down to the scale's card budget. Each *card* has **static** fields (issuer, card type, BIN region, home state — derived where TabFormer lacks them) plus a time-ordered stream of transactions with **dynamic** fields (amount, merchant, MCC, hour, day-of-week, channel), a **payment-network signal** (decline/response code — an output target, never a model input), and a fraud label.

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

import numpy as np
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
    num_partitions=32,            # right-size the shuffle (Ray defaults to 200)
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
print("  amount-bucket tokens:", np.asarray(row["d_amount_bucket"])[-8:].tolist())
print("  attention mask:", np.asarray(row["attention_mask"])[-8:].tolist())
```

## Step 4: Pretrain with Ray Train (masked-feature modeling, DDP)

We pretrain by **masking dynamic-field tokens and predicting them** (the FATA-Trans / Stripe / Open-Banking objective — bidirectional context beats next-token for the fixed-window tasks fintech cares about, and masked-item prediction is also a strong recommendation objective à la BERT4Rec). Each dynamic field has its own head, balanced with **uncertainty weighting** (Kendall & Gal) so the big head doesn't dominate. On the **learned merchant-vocab path** (`small`/`full`), merchant is a real ~100k-way field, so its head uses **InfoNCE with shared negative sampling** (TREASURE Alg. 1) instead of a full softmax — only the positive plus a batch-shared pool of negatives are scored, keeping a 100k+ vocab tractable. The smoke/CI path hash-buckets merchant and stays plain cross-entropy.

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

## Step 6: Downstream fraud — the headline table

Evaluated with the **NVIDIA transaction-FM blueprint protocol** (their notebook-01 split and sampling: temporal 80/10/10, 1M balanced train, 100k stratified val/test, their Optuna'd XGBoost params). Stage 05 prints one table:

1. **baseline** — their 13 raw transaction features (the gate: reproduces their published numbers through this pipeline)
2. **embed_pca64_xgb** — *their* notebook-05 protocol (PCA→64d + XGBoost) applied to *our* embedding
3. **embed_logistic / embed_xgb** — our readout: the raw pooled-last embedding, no PCA, into a linear head / XGBoost

At `full` scale the no-PCA readouts are the headline — AP well above their published fusion. *(At `smoke` scale — 2 CPU epochs, a 2-layer model, synthetic fallback eval — expect embedding ≈ raw; the gap opens with the `small`/`full` GPU pretrain on TabFormer.)*

**Template extra — next-merchant recommendation.** On the learned-vocab path the *same* backbone ranks the next merchant a card will transact with (BERT4Rec-style: mask the target, score it against the InfoNCE merchant table), reported as HR@K / NDCG@K via `scripts/06_recommend.py`. It is off the headline path — fraud is the story; reco shows the one-backbone-many-consumers shape.


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

The same code scales up by changing config, and runs as scheduled **Anyscale Jobs** — the three yamls from the top of this README are the production path:

```bash
anyscale job submit -f job_baseline.yaml   # gate: their baseline through this pipeline
anyscale job submit -f job_full.yaml       # pretrain + extract + headline table (seq 512)
anyscale job submit -f job_xl.yaml         # the same at 1024 transactions of context
```

| Stage | Ray primitive | Scale knob |
|-------|---------------|-----------|
| Merchant vocab | Ray Data `groupby().count()` | `merchant_top_k`, `merchant_aggregate` |
| Tokenize | Ray Data `map_groups` | partitions / cluster size |
| Pretrain | Ray Train + DDP | `num_workers`, `use_gpu`, `infonce_negatives` |
| Batch embed | Ray Data `map_batches` | `num_workers`, `num_gpus` |
| Recommend | Ray Data `map_batches` | `num_workers`, `num_gpus` |
| Online serve | Ray Serve | replica autoscaling |

## Bring your own data

Stage 01 ingests the raw TabFormer CSV schema. To run on your own transactions, provide a CSV with these columns (or adapt `src/tabformer._normalize`, ~40 lines):

| column | example | notes |
|--------|---------|-------|
| `User`, `Card` | `29`, `1` | together they form the card id |
| `Year`, `Month`, `Day`, `Time` | `2019`, `3`, `7`, `13:42` | transaction timestamp (minute resolution) |
| `Amount` | `$57.20` | dollar string; sign carries refunds |
| `Use Chip` | `Swipe Transaction` | channel |
| `Merchant Name` | `-34...` (int64 id) | any stable merchant id |
| `Merchant City` / `Merchant State` / `Zip` | `La Verne` / `CA` / `91750` | blank state = online |
| `MCC` | `5912` | merchant category code |
| `Errors?` | `Insufficient Balance` | payment-network signal (output-only head) |
| `Is Fraud?` | `Yes` / `No` | the label |

The load-bearing columns are card id, timestamp, amount, merchant id, MCC, and the label — the rest can be constant placeholders if you don't track them.

## Validate

A quick end-to-end check that every stage produced sane artifacts.


```python
from scripts.validate_results import validate_pipeline, print_report
print_report(validate_pipeline(paths))
```
