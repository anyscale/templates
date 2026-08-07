# Visual Product Search — built on Ray

**The business problem.** A customer sees a product on Instagram. They open our app, upload the photo, and expect the closest matches from our catalog — across hundreds of thousands of SKUs, in under a second. That's the feature. Now the engineering problem behind it.

**What it takes — and the scale wall at each step:**

| # | We need to … | Breaks at scale because … | Ray library |
|---|--------------|---------------------------|-------------|
| 1 | Ingest the product catalog and clean text | A `for` loop OOMs at 100M rows | **Ray Data** |
| 2 | Fine-tune an embedding model on our taxonomy | Single-process training stalls and offers no fault tolerance | **Ray Train** |
| 3 | Embed every product in the catalog | Loading the model per process and looping serially is hours of wall-clock | **Ray Data** (actor pool) |
| 4 | Serve `image → caption → top-K` over HTTP | A monolithic server lets the slowest model throttle the fastest | **Ray Serve** |

```
┌──────────────────────────────────────────────────────────────────────┐
│  ▼  Stage 1 — Ray Data    streaming ingest + clean                   │
│  ▼  Stage 2 — Ray Train   distributed fine-tune, auto-checkpointing  │
│  ▼  Stage 3 — Ray Data    actor-pool batch embedding                 │
│  ▼  Stage 4 — Ray Serve   multi-deployment composition (HTTP API)    │
└──────────────────────────────────────────────────────────────────────┘
```

This notebook runs the whole pipeline on **one CPU node with 1,000 products**. The same code runs on 100M products on a multi-GPU Anyscale cluster — only `ScalingConfig` and `ActorPoolStrategy(size=…)` numbers change. We'll call out which knob at each stage.

> Production configs (Anyscale Job + Service YAMLs, optional Streamlit UI) live in `setup/` — the closing section walks through them.


## Stage 0 — Set up dependencies, connect to Ray, inspect the catalog

Everything the demo needs is pinned in `python_depset.lock` (compiled from `requirements.txt`). The next cell installs that lock on the **driver**; `init_ray()` (idempotent) forwards the same lock to every **worker** via `runtime_env`, so Ray Data actors and Ray Train workers run the identical pinned environment. Stage 4 passes it to the Ray Serve replicas explicitly.

The catalog plot shows one product image per category — the input shape the rest of the pipeline operates on.



```python
# Install this template's pinned dependencies on the driver. Ray workers
# receive the same lock through runtime_env — see utils.init_ray().
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match

```


```python
from utils import CATEGORIES, PRODUCTS, get_product_image, init_ray
from utils.viz import plot_category_samples

init_ray()
plot_category_samples(PRODUCTS, CATEGORIES, get_product_image)

```

---
## Stage 1 — Ray Data: ingest + clean the catalog

**Objective.** Load every product, normalise the text we'll embed.

**Why Ray Data.** `Dataset.map_batches` streams — only the active batch lives in memory at any moment. Same `batch_size=64` call runs over 1,000 rows here or 100M rows on a cluster, with no chunking logic to write. Scale knob: none required for this stage; Ray Data picks block sizes from the dataset.

APIs: `ray.data.from_items`, `Dataset.map_batches`, `Dataset.schema`.



```python
import ray

from utils import clean_text, expand_catalog, generate_catalog


def clean_text_batch(batch: dict) -> dict:
    # Runs inside a Ray Data task. Whatever batch_size we choose, Ray streams
    # exactly that many rows through this function — never the whole catalog.
    batch["text_clean"] = [clean_text(t) for t in batch["training_text"]]
    return batch


products = expand_catalog(PRODUCTS, target_size=1000)
ds = (
    ray.data.from_items(generate_catalog(products=products))
    .map_batches(clean_text_batch, batch_size=64)
    .materialize()
)

print(f"Ray Dataset    : {ds.count()} rows across {ds.num_blocks()} block(s)")
print(f"Schema         : {[(f.name, str(f.type)) for f in ds.schema().base_schema]}")
print(f"Categories     : {sorted({p['category'] for p in products})}")

# Materialise once so downstream stages reuse the same rows without rerunning
# the ingest pipeline.
catalog_records = ds.take_all()

```

---
## Stage 2 — Ray Train: fine-tune the embedding model

**Objective.** Push same-category products closer in embedding space and other-category products further apart. Pretrained `all-MiniLM-L6-v2` was trained on general web text and doesn't know our taxonomy — a "Wireless Headphones" query still pulls in Books titles that share words. We fix that with a small contrastive fine-tune (same-category pair → cosine 1, cross-category pair → cosine -1).

**Why fine-tune at all — instead of a bigger model or a hosted embedding API?**

* **Better.** A small model that has *seen our catalog* beats a generic billion-parameter embedder that hasn't. Fine-tuning bends the embedding space around *our* taxonomy, so "wireless headphones" lands next to other Electronics instead of next to any text that happens to share words. On a domain task, domain fit beats raw size.
* **Cheaper to train.** This is a contrastive nudge over a few hundred pairs on top of an existing checkpoint — not pretraining from scratch. Minutes on CPU here; a short GPU job at catalog scale. We adapt a model instead of paying to build one.
* **Cheaper to serve — and that's the cost that compounds.** The tuned MiniLM is 22M params and runs on CPU. A hosted embedding API bills per token on *every* product at index time **and** *every* query at serve time, plus a network hop on the latency path. Owning a small model collapses both to local compute. You pay training once; you pay inference on every request forever — optimize the one that repeats.

**Why Ray Train (vs. a plain PyTorch loop):**

* `report(metrics, checkpoint=...)` — every worker streams metrics + artifacts to the driver each epoch.
* `CheckpointConfig(num_to_keep=2, ...)` — Ray automatically retains the best-N checkpoints by our chosen metric, prunes the rest. No bookkeeping.
* `FailureConfig(max_failures=1)` — auto-retries a crashed worker before failing the run.
* `get_checkpoint()` — a re-spawned worker resumes from the last good checkpoint mid-run; no lost epochs.
* **Scale knob:** `ScalingConfig(num_workers=1)` → `ScalingConfig(num_workers=8, use_gpu=True)`. The `train_loop_per_worker` code is unchanged.



```python
import os
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ray.train import Checkpoint, get_checkpoint, get_context, report

from utils import resolve_artifact_paths, sample_per_category
from utils.training import ContrastivePairDataset, forward_embeddings

paths = resolve_artifact_paths()  # picks /mnt/cluster_storage if available
train_records = sample_per_category(catalog_records, n_per_category=20, seed=42)


def train_loop_per_worker(config: dict) -> None:
    """Runs on every Ray Train worker. Ray Train APIs in use:
       • get_context().get_world_rank()  — which worker am I in the group?
       • get_checkpoint()                — resume after a restart
       • report(metrics, checkpoint=...) — stream metrics + artifacts to driver
       • Checkpoint.from_directory(path) — wrap a folder as a Ray Checkpoint
    """
    from sentence_transformers import SentenceTransformer

    rank   = get_context().get_world_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def collate(batch):
        a, b, lab = zip(*batch)
        return list(a), list(b), torch.stack(lab)

    loader = DataLoader(
        ContrastivePairDataset(config["records"], seed=config["seed"]),
        batch_size=config["batch_size"], shuffle=True, collate_fn=collate,
    )
    model     = SentenceTransformer(config["base_model"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    # Resume from the last good checkpoint if Ray Train restarted this worker.
    start_epoch = 0
    if (ckpt := get_checkpoint()) is not None:
        with ckpt.as_directory() as d:
            start_epoch = torch.load(os.path.join(d, "meta.pt"), weights_only=False)["epoch"] + 1
            model = SentenceTransformer(d, device=device)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        loss_sum, n = 0.0, 0
        for a, b, labels in loader:
            ea = forward_embeddings(model, a, device)
            eb = forward_embeddings(model, b, device)
            loss = F.mse_loss(F.cosine_similarity(ea, eb), labels.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item(); n += 1

        # Rank 0 packages the latest weights as a Ray Checkpoint; every worker
        # calls report() so Ray treats the step as finished and persists the
        # checkpoint+metric per the trainer's CheckpointConfig.
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_out = None
            if rank == 0:
                model.save(tmp)
                torch.save({"epoch": epoch}, os.path.join(tmp, "meta.pt"))
                ckpt_out = Checkpoint.from_directory(tmp)
            report({"epoch": epoch, "train_loss": loss_sum / max(1, n)}, checkpoint=ckpt_out)

```


```python
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "epochs": 2, "batch_size": 8, "lr": 2e-5, "seed": 42,
        "records": train_records,
    },
    scaling_config=ScalingConfig(num_workers=1, use_gpu=torch.cuda.is_available()),
    run_config=RunConfig(
        name="ecomm_embedding_finetune",
        storage_path=os.path.abspath(paths["train_result_dir"]),
        # Keep the best-2 checkpoints by lowest train_loss; Ray prunes the rest.
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="train_loss",
            checkpoint_score_order="min",
        ),
        # On worker crash: retry once before failing the run.
        failure_config=FailureConfig(max_failures=1),
    ),
)

result = trainer.fit()
print(result.metrics_dataframe[["epoch", "train_loss"]].to_string(index=False))

```


```python
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Ray Train sorts `best_checkpoints` by our CheckpointConfig metric (min
# train_loss), so [0][0] is the best checkpoint. `.as_directory()` materialises
# it locally even if Ray persisted it to remote storage like S3.
best_ckpt = result.best_checkpoints[0][0]
Path(paths["model_dir"]).mkdir(parents=True, exist_ok=True)
with best_ckpt.as_directory() as ckpt_dir:
    SentenceTransformer(ckpt_dir).save(paths["model_dir"])
print(f"Best model → {paths['model_dir']}")

```


```python
from utils.viz import plot_checkpoint_history, plot_training_loss

plot_training_loss(result.metrics_dataframe)
# Ray Train's CheckpointConfig kept only the best-N — visualise which survived.
plot_checkpoint_history(result)

```

---
## Stage 3 — Ray Data: actor-pool batch embedding

**Objective.** Encode every product in the catalog into an embedding vector and persist the matrix for online lookup. The naive approach — load the model in one process, loop — costs hours of wall-clock on a real catalog and won't fit in memory.

**Why Ray Data actor pools:**

* `ActorPoolStrategy(size=N)` — N replicas of `ProductEmbedder` each load the model **once** at construction and reuse it across every batch. Stateful, not re-loaded per task.
* **Streaming execution** — only batches in flight live in memory; the 100M-row catalog never materialises.
* **Scale knob:** `size=2` → `size=32`. Nothing else changes.

We also tag each row with the actor id that processed it — so the plot below *shows* the parallelism instead of just claiming it.



```python
import numpy as np
import ray
from sentence_transformers import SentenceTransformer


class ProductEmbedder:
    """Stateful Ray Data actor — model loads once, then encodes many batches."""

    def __init__(self, model_dir: str):
        self.model = SentenceTransformer(model_dir)
        # Capture the actor id so we can visualise the work distribution.
        self.actor_id = ray.get_runtime_context().get_actor_id()

    def __call__(self, batch: dict) -> dict:
        embs = self.model.encode(
            batch["text_clean"].tolist(),
            convert_to_numpy=True, show_progress_bar=False,
        )
        return {
            "product_id":   batch["product_id"],
            "name":         batch["name"],
            "category":     batch["category"],
            "embedding":    list(embs),
            "actor_id":     [self.actor_id] * len(embs),
        }


POOL_SIZE = 2

ds = ray.data.from_items(catalog_records).select_columns(
    ["product_id", "name", "category", "text_clean"]
)

# Bumping `size=` is the only knob needed to scale this from 1k → 100M rows.
rows = ds.map_batches(
    ProductEmbedder,
    fn_constructor_kwargs={"model_dir": paths["model_dir"]},
    batch_size=8,
    num_cpus=1,
    compute=ray.data.ActorPoolStrategy(size=POOL_SIZE),
    batch_format="numpy",
).take_all()

sample = rows[0]
print(f"Rows embedded : {len(rows)}    pool size : {POOL_SIZE}")
print(f"Embedding shape (per row): {np.asarray(sample['embedding']).shape}")
print(f"Sample        : {sample['name']!r}  [{sample['category']}]")

```


```python
from utils.embedding import save_embeddings_and_metadata

embeddings, metadata = save_embeddings_and_metadata(
    rows, paths["embeddings_path"], paths["metadata_path"],
)
print(f"Embeddings : {embeddings.shape}  →  {paths['embeddings_path']}")
print(f"Metadata   : {len(metadata)} products  →  {paths['metadata_path']}")

```


```python
from utils.viz import plot_actor_pool_throughput

# How did Ray Data's actor pool distribute the 1,000 rows across N replicas?
# Bars of similar height = healthy load balancing.
plot_actor_pool_throughput([r["actor_id"] for r in rows], pool_size=POOL_SIZE)

```


```python
# Did fine-tuning actually reshape the embedding space?
# Three views: (a) raw similarity heatmap; (b) t-SNE base-vs-tuned; (c) numbers.
from utils.viz import (
    plot_similarity_heatmap,
    plot_tsne_comparison,
    print_embedding_quality_report,
)

plot_similarity_heatmap(embeddings, [m["name"] for m in metadata])
base_embs, ft_embs, aligned_meta = plot_tsne_comparison(
    metadata, embeddings, catalog_records=catalog_records,
)
print_embedding_quality_report(base_embs, ft_embs, aligned_meta)

```

---
## Stage 4 — Ray Serve: caption + recommend behind one HTTP endpoint

**Objective.** `POST /recommend {image_base64}` returns the top-K matching products. The pipeline is heterogeneous: BLIP (224M params, slow) captions the image; the fine-tuned MiniLM (22M params, fast) embeds the caption and does cosine top-K against the index.

**Why Ray Serve (vs. one FastAPI process):**

* **Independent deployments** — `ImageToText` and `ProductRecommender` run as separate actors with their own replica pools. A spike in image uploads adds BLIP replicas; query-traffic spikes add MiniLM replicas. They never throttle each other.
* **Independent rollouts** — swap the BLIP model without redeploying MiniLM. The async ingress wires them with `DeploymentHandle`.
* **One graph, one deploy** — `serve.run(app)` brings up all three deployments.
* **Scale knob:** each `@serve.deployment(num_replicas=N)` (or `autoscaling_config=...`) scales independently.



```python
from ray import serve

from serve_app import build_app
from utils.viz import print_serve_topology

# Pass paths through bind() — each replica gets them on construction, so we
# don't have to mutate the driver's os.environ. The lock rides along as the
# replicas' runtime_env: Serve replicas don't inherit the driver's
# ray.init(runtime_env=...), so without it they'd miss torch/transformers.
app = build_app(
    embedding_model_dir=paths["model_dir"],
    embeddings_path=paths["embeddings_path"],
    metadata_path=paths["metadata_path"],
    runtime_env={"pip": os.path.abspath("python_depset.lock")},
)
serve.run(app)
print("Service running at http://localhost:8000\n")

print_serve_topology([
    {"name": "RecommendationService", "role": "HTTP ingress",       "replicas": 1, "num_cpus": 1},
    {"name": "ImageToText",           "role": "BLIP captioning",    "replicas": 1, "num_cpus": 2},
    {"name": "ProductRecommender",    "role": "MiniLM + cosine top-k", "replicas": 1, "num_cpus": 2},
])

```


```python
# Smoke-test the running /recommend endpoint with a known Electronics product.
# A healthy fine-tune should fill the top results with Electronics, not Books.
from utils import post_recommend

product = PRODUCTS[0]
result  = post_recommend(get_product_image(product))

print(f"Caption : {result['caption']!r}")
print(f"\nTop-{len(result['recommendations'])} recommendations:")
for r in result["recommendations"]:
    print(f"  {r['rank']}. [{r['category']:14s}] {r['name']:35s}  sim={r['similarity']:.3f}")

```


```python
# One image per category — does the pipeline generalise beyond headphones?
print(f"{'Category':<16} {'Caption':<45} {'Top recommendation'}")
print("-" * 85)
for cat in CATEGORIES:
    prod = next(p for p in PRODUCTS if p["category"] == cat)
    r = post_recommend(get_product_image(prod))
    top = r["recommendations"][0]
    print(f"{cat:<16} {r['caption'][:44]:<45} {top['name'][:20]} ({top['similarity']:.2f})")

```

---
## Scaling this from 1k products to 100M

Same code path. Three numbers change — and Ray handles the rest.

```python
# Stage 2 — Ray Train: distribute the fine-tune across GPUs
ScalingConfig(num_workers=1, use_gpu=False)          →  ScalingConfig(num_workers=8, use_gpu=True)

# Stage 3 — Ray Data: more embedder replicas, same streaming
ActorPoolStrategy(size=2)                            →  ActorPoolStrategy(size=32)

# Stage 4 — Ray Serve: scale each model on its own signal
@serve.deployment(num_replicas=1)                    →  @serve.deployment(autoscaling_config={"min_replicas": 2, "max_replicas": 20})
```

**What you get without rewriting anything:**

* **Streaming all the way through.** Ray Data never materialises the catalog; `map_batches` and `ActorPoolStrategy` stream batches through stateful replicas.
* **Fault tolerance for free.** `FailureConfig` retries crashed Ray Train workers; `get_checkpoint()` resumes mid-run; Ray Serve restarts dead replicas.
* **Independent scaling per model.** BLIP, MiniLM, and the ingress each react to their own load — one slow model never throttles another.
* **One control plane.** `ray status`, `serve status`, and the Anyscale dashboard cover ingest, training, embedding, and serving.

---
## From notebook to production — Jobs & Services

The workspace ran every stage interactively. In production those stages map onto Anyscale's two execution modes — and the **same code** backs both. The deployment graph and `train_loop_per_worker` don't change; only *how they're launched* does.

**First, the runtime: image + compute config.** Both Jobs and Services run on a container image and a compute config (instance types, autoscaling bounds). This workspace already runs on both — the template pinned them for you — and the `setup/` YAMLs below reference the same image, so what you deploy matches what you just ran. Swap in your own image or instance types there when you productionize.

**Offline pipeline → an Anyscale Job.** Ingest → fine-tune → embed runs to completion and exits — the textbook shape for a **Job**. A Job runs your entrypoint on its own ephemeral cluster: it spins up, executes, writes its artifacts (the tuned model + embedding matrix to blob storage), and tears the cluster down — so you never pay for idle GPUs between runs. `setup/job.yaml` packages exactly that — it runs this notebook top to bottom with papermill on a fresh cluster; schedule it nightly to re-embed a refreshed catalog.

```bash
# Run the offline stages as a managed job on an ephemeral cluster
anyscale job submit -f setup/job.yaml
```

**Online endpoint → an Anyscale Service.** `serve.run(app)` is perfect inside the workspace, but a production endpoint needs more than a process that dies with the notebook. An Anyscale **Service** takes the *same* `serve_app:app` and adds what an always-on API requires: a stable HTTPS URL with a token, health checks with auto-restart, autoscaling per deployment, and zero-downtime rolling upgrades when you ship a new model.

```bash
# Deploy the exact same app as a managed, autoscaling service
anyscale service deploy -f setup/service.yaml
anyscale service status --name ecomm-recommender-service
```

That's the payoff of building on Ray: debug interactively here, then promote the identical code to a scheduled **Job** or an always-on **Service** without a rewrite. Config for both lives in `setup/`.

```python
# Optional teardown. In a workspace you can leave the service up to keep
# testing; serve.shutdown() stops it and frees the replicas.
# serve.shutdown()
```

> Want a clickable demo UI on top of the service? See `setup/STREAMLIT_UI.md`.

