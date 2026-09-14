"""
Show one GRPO group end to end on a real training row: 4 sampled completions from the
base model, their rewards from rewards.py, and the group-normalised advantages.

    RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
      uv run --frozen python rollout_demo.py --config configs/grpo_smoke.yaml --row 0

Runs as one Ray task on one GPU. Sampling settings (temperature, max tokens, model,
revision, num_generations) come from the same YAML the trainer uses. Writes
results/grpo_rollout_example.md and prints it.
"""

from __future__ import annotations

import argparse
import os
import sys

import ray
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


@ray.remote(num_gpus=1)
def rollout_group(cfg: dict, row_index: int) -> str:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl.data_utils import prepare_multimodal_messages

    sys.path.insert(0, HERE)
    from rewards import format_reward, label_reward
    from train_grpo_trl import to_trl_rows

    row = to_trl_rows(os.path.join(cfg["data_dir"], "train.parquet"), limit=row_index + 1)[row_index]
    g = cfg["num_generations"]

    proc = AutoProcessor.from_pretrained(cfg["model_name_or_path"], revision=cfg["model_revision"], padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_name_or_path"], revision=cfg["model_revision"], dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    prompt = prepare_multimodal_messages(row["prompt"], images=row["images"])
    inputs = proc.apply_chat_template(
        [prompt] * g, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True
    ).to("cuda")
    torch.manual_seed(cfg.get("seed", 42))
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=cfg["temperature"],
            top_p=cfg.get("top_p", 1.0),
            max_new_tokens=cfg["max_completion_length"],
        )
    comp_ids = out[:, inputs["input_ids"].shape[1]:]
    texts = proc.batch_decode(comp_ids, skip_special_tokens=True)
    lengths = [(ids != proc.tokenizer.pad_token_id).sum().item() for ids in comp_ids]

    completions = [[{"role": "assistant", "content": t}] for t in texts]
    gt = [row["ground_truth"]] * g
    r_label = label_reward(completions, gt)
    r_format = format_reward(completions)
    rewards = torch.tensor([a + b for a, b in zip(r_label, r_format)])
    # GRPO, scale_rewards="group": (r - group mean) / (group std + 1e-4), unbiased std as in TRL
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    lines = [
        f"# One GRPO group, training row {row_index + 1}",
        "",
        f"Ground truth: **{row['ground_truth']}**. Base model `{cfg['model_name_or_path']}` "
        f"(revision {cfg['model_revision'][:8]}), {g} samples, temperature {cfg['temperature']}, "
        f"max {cfg['max_completion_length']} new tokens. The patch is #1 in `grpo_train_samples.png`.",
        "",
        "| # | tokens | label_reward | format_reward | reward | advantage |",
        "|---|---|---|---|---|---|",
    ]
    for i in range(g):
        lines.append(f"| {i + 1} | {lengths[i]} | {r_label[i]:.1f} | {r_format[i]:.1f} | {rewards[i]:.1f} | {adv[i]:+.2f} |")
    lines += [
        "",
        f"Group mean reward {rewards.mean():.3f}, std {rewards.std():.3f}. "
        "Advantage = (reward − mean) / (std + 1e-4), applied to every token of that completion.",
        "",
    ]
    for i, t in enumerate(texts):
        lines += [f"## Completion {i + 1}  (reward {rewards[i]:.1f}, advantage {adv[i]:+.2f})", "", t.strip(), ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(HERE, "configs/grpo_smoke.yaml"))
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "results/grpo_rollout_example.md"))
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ray.init(runtime_env={"env_vars": {"HF_HOME": os.environ.get("HF_HOME", "/mnt/cluster_storage/hf_cache")}})
    md = ray.get(rollout_group.remote(cfg, args.row))
    print(md)
    # The task runs on the worker; the head has this directory, so write here.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
