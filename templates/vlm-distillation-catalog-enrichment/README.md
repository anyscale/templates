# VLM teacher-student distillation for ecommerce catalogs

<a href="https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=vlm-distillation-catalog-enrichment&redirectTo=/v2/template-preview/vlm-distillation-catalog-enrichment">
<img src="https://raw.githubusercontent.com/ray-project/ray/c34b74c22a9390aa89baf80815ede59397786d2e/doc/source/_static/img/run-on-anyscale.svg" alt="Run on Anyscale">
</a>
<br></br>
<div align="left">
<a href="https://github.com/anyscale/templates" role="button"><img src="https://img.shields.io/static/v1?label=&amp;message=View%20On%20GitHub&amp;color=586069&amp;logo=github&amp;labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 45 min

This tutorial distills a Qwen2.5-VL-7B teacher into a Qwen2.5-VL-3B student so a product catalog can be enriched and embedded at 3B inference cost without sacrificing teacher-quality structured outputs. The full pipeline runs on Ray and Anyscale across three stages, all on the same 4× L4 GPU node:

1. **Teacher batch labeling** — Run Qwen2.5-VL-7B over a catalog subset with [`ray.data.llm`](https://docs.ray.io/en/latest/data/working-with-llms.html) to produce `{category, attributes, search_tags, description}` JSON per product.
2. **Distillation SFT** — Fine-tune Qwen2.5-VL-3B on the teacher labels with Ray Train + FSDP + LoRA. Only the language model gets adapters; the vision tower stays frozen.
3. **Enrichment and embeddings** — Run the LoRA-adapted student to emit catalog JSON *and* SigLIP-2 image and text embeddings in a single streaming Ray Data graph.

Ray is particularly powerful for this workload because it:
- **Schedules CPU and GPU stages in one cluster** so image fetching, VLM inference, and embedding extraction share a single resource pool with no idle hardware between stages.
- **Streams data between stages** without intermediate disk writes — SigLIP embeddings begin computing as soon as the first VLM-enriched batch is ready.
- **Hot-swaps LoRA adapters** at inference via vLLM's [multi-LoRA support](https://docs.vllm.ai/en/stable/features/multi_lora.html), so the distilled student loads from the same base model as the un-tuned variant.
- **Promotes any notebook to a scheduled production job** with `anyscale job submit` — no rewrite, same cluster shape.

## Architecture

```
Stage 1                    Stage 2                    Stage 3
─────────────              ──────────────             ─────────────────
7B teacher        ───►     3B student SFT  ───►       3B distilled enrich
batch labeling             (FSDP + LoRA)              + SigLIP embeddings
                                                       (one Ray Data graph)

ray.data.llm               Ray Train + FSDP            ray.data.llm + Ray Data
   ▼                          ▼                          ▼
teacher.parquet            LoRA adapter               enriched_with_embeddings.parquet
(JSON labels)              (~100 MB)                  (JSON + 1152-dim vectors)
```

Stages 1 and 2 each take 15–30 min on the smoke configuration (N=20 rows); Stage 3 takes ~5 min. Production runs at N=10,000 take roughly 1.5 hours on the same cluster.

The deep-dive notebooks under [`notebooks/`](notebooks/) walk through each stage cell by cell — useful when you want to swap models, change the catalog, or tune the SFT hyperparameters.

## Set up

The pipeline pulls model weights and the [Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset from Hugging Face. Make sure `HF_TOKEN` is available in your environment before running the cells below.


```python
import os
import subprocess
import sys

# Cache HF downloads on cluster storage so the three stages share weights.
os.environ.setdefault("HF_HOME", "/mnt/cluster_storage/hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Smoke knobs — overridden by tests.sh during CI.
# Set these higher for a production run (N_ROWS=10000 is the default in each script).
N_ROWS = int(os.environ.get("N_ROWS", "20"))
TEACHER_N_ROWS = int(os.environ.get("TEACHER_N_ROWS", str(N_ROWS)))
CATEGORY = os.environ.get("CATEGORY", "Electronics")

os.environ["N_ROWS"] = str(N_ROWS)
os.environ["TEACHER_N_ROWS"] = str(TEACHER_N_ROWS)
os.environ["CATEGORY"] = CATEGORY

print(f"Running with N_ROWS={N_ROWS} TEACHER_N_ROWS={TEACHER_N_ROWS} CATEGORY={CATEGORY}")
```


```python
!pip install -q -r requirements.txt
```

## Stage 1 — Teacher batch labeling

Qwen2.5-VL-7B reads a product image and merchant-supplied title and returns a four-key JSON object (`category`, `attributes`, `search_tags`, `description`). [`ray.data.llm`](https://docs.ray.io/en/latest/data/working-with-llms.html#multimodal) wraps the vLLM engine — preprocessing, batching, and postprocessing all happen as `map_batches` stages over a Ray Dataset. With `concurrency=4`, one 7B replica runs on each of the four L4 GPUs.

The output parquet becomes Stage 2's supervised training data.

The deep dive: [`notebooks/01_teacher_batch_label.ipynb`](notebooks/01_teacher_batch_label.ipynb).


```python
!python scripts/run_teacher_batch_label.py
```


```python
# Inspect a sample row of the teacher output.
import pyarrow.parquet as pq

teacher_path = f"/mnt/cluster_storage/vlm-distillation-catalog-enrichment/teacher_7b_enriched_{TEACHER_N_ROWS}.parquet"
tbl = pq.read_table(teacher_path)
print(f"rows: {tbl.num_rows}")
print(f"schema: {tbl.schema.names}")
row = tbl.slice(0, 1).to_pylist()[0]
print(f"\nsample title: {row['title'][:100]}")
print(f"sample teacher output (raw_output):\n{row['raw_output']}")
```

## Stage 2 — Distill the 3B student with FSDP + LoRA

Ray Train + FSDP fine-tunes Qwen2.5-VL-3B on the teacher parquet using LoRA adapters. The vision tower stays frozen (standard [LLaVA recipe](https://arxiv.org/abs/2304.08485)); only the language model gets adapters, so trainable parameters drop to ~1% of the model. FSDP `FULL_SHARD` distributes the optimizer state across the four GPUs at bf16 mixed precision.

The output is a small (~100 MB) adapter directory that drops into Stage 3 as a model swap — same base weights, fine-tuned head.

The deep dive: [`notebooks/02_distill_student_lora.ipynb`](notebooks/02_distill_student_lora.ipynb).


```python
!python scripts/run_distill_student_lora.py
```


```python
# Inspect the adapter.
from pathlib import Path

adapter_dir = Path(f"/mnt/cluster_storage/vlm-distillation-catalog-enrichment/qwen25vl_3b_enrichment_lora_{N_ROWS}")
print(f"adapter dir: {adapter_dir}")
for p in sorted(adapter_dir.rglob("*"))[:10]:
    print(f"  {p.relative_to(adapter_dir)}")

# Point Stage 3 at the freshly trained adapter.
os.environ["QWEN_LORA_ADAPTER_DIR"] = str(adapter_dir)
```

## Stage 3 — Enrichment and embeddings in one Ray Data graph

The third stage chains together the LoRA-adapted 3B student (via `ray.data.llm`) and the SigLIP-2 dual-tower encoder (via a Ray Data actor pool) into a single streaming pipeline. No intermediate disk writes between the VLM and the embedding stages — Ray Data hands batches directly from one `map_batches` stage to the next, so the SigLIP encoders start working as soon as the first VLM-enriched batch is ready.

Each row of the output parquet carries the structured catalog JSON *and* two 1152-dimensional embeddings (image + text), ready to load into a vector store.

The deep dive: [`notebooks/03_enrich_and_embed.ipynb`](notebooks/03_enrich_and_embed.ipynb).


```python
!python scripts/run_enrich_and_embed.py
```


```python
# Inspect a sample row of the final output.
import pyarrow.parquet as pq

out_path = f"/mnt/cluster_storage/vlm-distillation-catalog-enrichment/enc_vlm_enriched_with_embeddings_{N_ROWS}.parquet"
tbl = pq.read_table(out_path)
print(f"rows: {tbl.num_rows}")
print(f"schema: {tbl.schema.names}")
row = tbl.slice(0, 1).to_pylist()[0]
print(f"\ntitle: {row['title'][:100]}")
print(f"raw_output (3B student): {row['raw_output']}")
print(f"image_embedding dim: {len(row['image_embedding'])}")
print(f"text_embedding dim:  {len(row['text_embedding'])}")
```

## Run as a scheduled Anyscale Job

The same scripts run unchanged as Anyscale Jobs — no rewrite, no second codebase. [`job_config.yaml`](job_config.yaml) submits Stage 3 on the same 4× L4 cluster used for the workspace runs, which is the daily / weekly cadence most teams use to refresh their catalog:

```bash
anyscale job submit --config-file job_config.yaml --env HF_TOKEN=$HF_TOKEN
```

Swap the `entrypoint` field to `scripts/run_teacher_batch_label.py` or `scripts/run_distill_student_lora.py` to submit Stages 1 or 2 as jobs.

## Clean up

Remove the cached parquets, adapters, and intermediate checkpoints from cluster storage.


```python
import shutil

shutil.rmtree("/mnt/cluster_storage/vlm-distillation-catalog-enrichment", ignore_errors=True)
print("cluster storage cleared")
```
