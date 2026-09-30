# On-policy distillation for LLMs with SkyRL

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/skyrl-opd-template"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/skyrl-opd-template" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/skyrl-opd-template
```

**⏱️ Time to complete**: ~1 hour (≈15–20 min cluster startup + image pull, ≈40 min training & eval).

This template runs **on-policy distillation (OPD)** on Anyscale with [SkyRL](https://github.com/NovaSky-AI/SkyRL). A small **student** model (Qwen3-0.6B) learns to solve GSM8K math word problems by imitating a larger **teacher** (Qwen3-4B), on a single 4×L4 node.

OPD combines the strengths of RL and distillation. The student generates its own attempts — on-policy, like RL — but instead of a single correct/incorrect reward at the end, the **teacher grades every token** the student produces — dense, like distillation. That dense signal makes OPD more sample-efficient than RL for teaching a small model a skill the teacher already has. See the [Thinking Machines writeup](https://thinkingmachines.ai/blog/on-policy-distillation/) and [NovaSky's OPD recipe](https://novasky-ai.notion.site/on-policy-distillation).

The runnable code lives next to this notebook in two files: **`opd_trainer.py`** (the method) and **`run_opd_gsm8k.sh`** (the config + launch).

## How it works

OPD reuses SkyRL's RL training loop with two changes:

**The teacher replaces the reference model.** Standard RL keeps a frozen copy of the policy as a reference to measure drift. OPD puts the **teacher** in that slot, so for every token the student samples, you also get the teacher's probability for that same token.

**The reward is per-token agreement with the teacher.** Each token is rewarded by `teacher_logprob − student_logprob` — positive when the teacher likes the token more than the student does. The update pulls the student toward the teacher, token by token, on the student's own rollouts. This is reverse-KL distillation.

During training the student never sees whether its final answer was correct — only whether the teacher agrees with each token. Accuracy still climbs, because it's imitating a teacher that *is* good at getting them right. Correctness is measured only for evaluation.

**Same tokenizer required.** The teacher scores the student's exact token IDs, so both models must share a tokenizer. This template stays within the Qwen3 family (`Qwen3-0.6B-Base` student, `Qwen3-4B` teacher).

## Setup

This template uses the `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0` image, so all system dependencies are already installed.

Clone SkyRL, pin the commit the image is built for, and copy in the two companion files:

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/
git checkout 94d53895
cp ../run_opd_gsm8k.sh ../opd_trainer.py .
```

`opd_trainer.py` is the OPD trainer — the ~15-line method (shown below). `run_opd_gsm8k.sh` supplies the GSM8K + 4×L4 config and launches it.

## Prepare the dataset

Convert GSM8K into the [Parquet schema SkyRL expects](https://docs.skyrl.ai/docs/datasets/dataset-preparation). The `--max_train_dataset_length 512` cap keeps this a quick ~8-step demo; drop it to distill on the full dataset (many more steps, several hours):

```bash
uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir /mnt/cluster_storage/data/gsm8k --max_train_dataset_length 512
```

This writes `train.parquet` and `validation.parquet`.

## Run on-policy distillation

Set the teacher and launch. Use the W&B logger (set `WANDB_API_KEY` in the [Dependencies tab](https://docs.anyscale.com/development#environment-variables)) or `LOGGER=console` to print to stdout:

```bash
SKYRL_RAY_PG_TIMEOUT_IN_S=90 \
DATA_DIR=/mnt/cluster_storage/data/gsm8k \
STUDENT_MODEL=Qwen/Qwen3-0.6B-Base \
TEACHER_MODEL=Qwen/Qwen3-4B \
LOGGER=wandb \
bash run_opd_gsm8k.sh
```

The script wires up OPD — teacher in the reference slot, per-token reverse-KL reward, `no_op` advantage — and starts with `eval_before_train=true`, which records the student's **baseline GSM8K accuracy**. Watch it climb as distillation proceeds. In a reference run on 4×L4, the Qwen3-0.6B student went from **0% → ~50%** GSM8K accuracy over the 8 steps.

## The recipe

All of OPD is essentially one line, in `opd_trainer.py`:

```python
rewards = -(action_log_probs - teacher_action_log_probs) * loss_mask
```

Each token's reward is how much more the teacher likes it than the student. Everything else is a standard SkyRL run — read the full trainer in `opd_trainer.py`.

## Next steps

- **Swap the task or models** — any same-tokenizer student/teacher pair, any SkyRL environment.
- **Sharpen the signal** — upgrade the per-token estimator (k1) to **full-distribution reverse-KL** for larger runs.
- **Serve the distilled student** with [Ray Serve LLM on Anyscale](https://console.anyscale.com/template-preview/deployment-serve-llm).
