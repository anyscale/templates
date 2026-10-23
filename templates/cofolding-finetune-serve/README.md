# Fine-tune and serve a co-folding model on GPUs with Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/cofolding-finetune-serve"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/cofolding-finetune-serve" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 60 min without GPUs, 2 h with

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/cofolding-finetune-serve
```

Co-folding models such as Boltz-2 and OpenFold3 predict how a small molecule binds its protein target from
sequence and chemical structure alone. Their published accuracy is measured on curated benchmarks and does
not transfer evenly to an active drug discovery campaign.

This template evaluates those models against 2,147 crystal structures from 3 antiviral campaigns, fine-tunes
a general OpenFold3 checkpoint on a fragment screen with Ray Train, and serves the result with Ray Serve. On
an EV-A71 2A campaign, 66 fragment-screen structures raise pose accuracy on held-out follow-on compounds
from 23.9% to 44.8%, in 2.4 hours on 8 L4 GPUs trained on spot capacity.

![Before and after fine-tuning](https://raw.githubusercontent.com/anyscale/templates/main/templates/cofolding-finetune-serve/docs/images/before_after_accuracy.gif)

## Prerequisites

- Intermediate-level Python experience
- Familiarity with structure prediction or molecular docking is helpful, not required
- GPUs are optional: every notebook displays stored results without them

## Notebooks

| Notebook | Ray library | Description |
|---|---|---|
| `01_ray_and_offline_inference.ipynb` | Ray Core, Ray Data | Tasks, actors and objects, then distributed prediction and pose scoring across 3 campaigns |
| `02_train_and_tune.ipynb` | Ray Train, Ray Tune | Fine-tune OpenFold3 across 8 GPUs, resume from a checkpoint after spot preemption, sweep the learning rate |
| `03_serve.ipynb` | Ray Serve | A results viewer and prediction endpoint, with per-request checkpoint swaps and canary rollout |

Each notebook stands alone, so you can start at any of the 3.

## Setup


```python
!python scripts/setup_data.py
```

About 500 MB. That is enough to run every notebook with `USE_GPUS = False`, the default: all tables, figures
and the before and after comparison read scored results downloaded here, on CPU, in about a minute.

To predict or fine-tune on GPUs, add the model weights and set `USE_GPUS = True` in the Setup cell:

```bash
python scripts/setup_data.py --with-weights   # Boltz-2 and OpenFold3, about 15 GB
```

| Setting | Default | Meaning |
|---|---|---|
| `USE_GPUS` | `False` | `False` displays stored results. `True` runs prediction and fine-tuning live, starting GPU nodes |
| `AIDDD_SHARED` | `/mnt/user_storage/aiddd` | Where data, weights and results live. Every node reads it |
| `AIDDD_SERVICE_URL` | unset | A deployed prediction service for notebook 03. Unset, the notebook starts Ray Serve in the workspace |

## Results

Share of predicted poses within 2 Å of the crystal structure, on late follow-on compounds: the last design
cycles of each campaign, and the hardest stage for every model.

| Model | EV-A71 2A | SARS-CoV-2 Mpro | MERS-CoV Mpro |
|---|---|---|---|
| Boltz-2 | 1.0% | 91.7% | 90.7% |
| OpenFold3 145k (general) | 23.9% | 87.7% | 73.5% |
| OpenFold3 145k, fine-tuned here | **44.8%** | 62.1% | 68.8% |

A model fine-tuned for 1 target is a specialist: it gains on that target and gives up accuracy on the
others. Measured on curated binding events, with a PoseBusters-valid crystal reference and suspected
crystallographic artefacts excluded.

## What Ray contributes

- Moving the fine-tune from 2 GPUs to 8 is `--num-workers 8`. OpenFold3's own PyTorch Lightning training
  loop runs unchanged inside each Ray Train worker.
- Training ran on spot capacity. Across these runs 9 nodes were reclaimed and every run finished, because a
  checkpoint every 100 steps caps the cost of a reclaimed node at 100 steps.
- Batch inference scored 402 new predictions in 19 minutes on 12 L4 GPUs at 80% GPU utilization.
- Ray Serve holds several checkpoints on the same replicas and picks one per request, so a new model
  reaches users behind a canary rather than a redeploy.

## Testing


```python
!python tests/test_template.py
```

CPU-only, about 1 minute. Checks that a clean environment reproduces the published accuracy figures and the
training loss curve.

## Data and attribution

- EV-A71 2A structures and affinities: OpenBind consortium release, Zenodo 10.5281/zenodo.22142262 (CC0).
- Published co-folding benchmark: `OpenBind-Consortium/EV-A71_2A_benchmark` (Apache-2.0).
- SARS-CoV-2 and MERS-CoV Mpro: Fragalysis structures with ASAP/Polaris potency, built by
  `scripts/prepare_mpro.py`.
- Boltz-2 and OpenFold3 weights are downloaded from their own public sources.

`of3-ob-2025-06-30-174k` (OpenBind-0) comes from the consortium that produced the EV-A71 2A data, and its
date is after almost all of those structures. It scores 91% on EV-A71 2A late follow-ons but 28% on the
fragment screen, a pattern consistent with having seen related data for this target. Results involving it
are reported with that caveat.
