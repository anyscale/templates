# Distributed VLA Fine-Tuning with Ray

This template fine-tunes the [PI0.5](https://www.physicalintelligence.company/blog/pi05) Vision-Language-Action (VLA) model on a [LeRobot](https://github.com/huggingface/lerobot) robotics dataset stored in S3, using [Ray](https://docs.ray.io/) on [Anyscale](https://anyscale.com).

It leverages Ray's ability to independently scale two distinct compute tiers — CPU nodes for streaming data preprocessing and GPU nodes for distributed training — so that neither side sits idle and GPU stalls waiting for data are eliminated.

```
+-----------+       +------------+       +-----------+
| S3 Bucket | ----> | Ray Data   | ----> | Ray Train |
| (LeRobot  |       | (CPU pool) |       | (N GPUs)  |
| mp4+pqt)  |       |            |       |           |
+-----------+       +------------+       +-----------+
                     |                    |
                     | - read parquet     | - load PI0.5
                     | - decode mp4       | - freeze backbone
                     | - rename cameras   | - train action heads
                     | - transpose HWC    | - mixed-precision
                     |   -> CHW float32   | - gradient accum
                     | - stream batches   | - checkpoint & resume
                     +--------------------+---------------------
```

For the full walkthrough, open **[vla.ipynb](vla.ipynb)**.

## Files

| File | Description |
|------|-------------|
| `vla.ipynb` | Interactive notebook — step-by-step walkthrough |
| `vla.py` | Job script — same pipeline, submittable with `uv run python vla.py` |
| `util.py` | Training utilities — model loading, checkpointing, collation, training step helpers |
| `lerobot_datasource.py` | Custom Ray Data datasource for LeRobot v3 datasets |

## Setup

### Dependencies

This template uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync
```

### HuggingFace Token

PI0.5 depends on [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) as a vision backbone. Google requires you to **accept the model license** before the weights can be downloaded.

1. Navigate to the [google/paligemma-3b-pt-224 model page](https://huggingface.co/google/paligemma-3b-pt-224) and accept the license agreement.
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Export the token:

```bash
export HF_TOKEN=hf_...
```

## Usage

### Notebook

Open `vla.ipynb` and run cells top-to-bottom. When prompted for a kernel, select the Python environment named **vla** (`.venv/bin/python`).

### Job script

```bash
export HF_TOKEN=hf_...
uv run python vla.py
```

To scale: change `num_workers` in the `ScalingConfig`. The training code, data pipeline, and checkpointing all adapt automatically.

## GPU Requirements

This template supports both **A100** and **L4** GPUs but requires different
configurations for each:

| | A100 (80 GB) | L4 (24 GB) |
|---|---|---|
| `batch_size` | 4 | 1 |
| `grad_accum` | 2 | 8 |
| `num_workers` | 4 | 4 |

**A100s** have enough VRAM to run larger batch sizes with minimal gradient
accumulation. This keeps GPU utilization high and the data pipeline
straightforward.

**L4s** require `batch_size=1` to fit in 24 GB VRAM. To compensate, increase
`grad_accum` so the effective batch size stays reasonable. However, smaller
per-step batches mean faster consumption, which can cause the Ray Data
pipeline to fall behind and spill to disk. If you see frequent object store
spillage, reduce the data pipeline's throughput to match the training speed —
for example, lower the `map_batches` concurrency or decrease the number of
CPU data workers so the producer and consumer stay in balance.
