"""
Stacked bar of one GRPO step's phases, per step, from the run's log_history.jsonl.

    python plot_step_breakdown.py /mnt/cluster_storage/trl_grpo/pathvlm_2b_trl/log_history.jsonl out.png

Reads the `timing/*_s` keys that grpo_step_timing.py injected into the HF log stream.
This is the chart that argues for a faster rollout engine: the generate slice is the
part vLLM / SkyRL take out of the step.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASES = ["generate", "sync_wait", "reward", "forward", "backward", "optimizer", "tokenize", "other"]


def main(path: str, out: str) -> None:
    steps = [json.loads(l) for l in open(path) if "timing/step_s" in l]
    x = [r["step"] for r in steps]
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(x) + 2), 4))
    bottom = [0.0] * len(x)
    for ph in PHASES:
        vals = [r.get(f"timing/{ph}_s", 0.0) for r in steps]
        ax.bar(x, vals, bottom=bottom, label=ph, width=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("seconds")
    tot = sum(r["timing/step_s"] for r in steps) / max(len(steps), 1)
    gen = sum(r.get("timing/generate_s", 0) for r in steps) / max(len(steps), 1)
    ax.set_title(f"TRL GRPO step breakdown, rank 0  (mean step {tot:.1f}s, generate {100 * gen / tot:.0f}%)")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out} ({len(steps)} steps)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "step_breakdown.png")
