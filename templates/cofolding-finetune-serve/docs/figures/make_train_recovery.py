"""The 8-GPU worker group losing a node and resuming from its last checkpoint.

    python docs/figures/make_train_recovery.py        # writes docs/images/train_recovery.png

Drawn from the run of 2026-10-23: 8 workers, 1 complex per worker per step, a checkpoint
every 100 steps, and a spot node reclaimed at step 700.
"""

from matplotlib.patches import FancyBboxPatch

from common import BAD, GOOD, INK, INK2, MUTED, OUT, RULE, plt

N_GPUS = 8
PREEMPTED = 2          # the rank whose node was reclaimed
CHECKPOINT_STEP = 700


def gpu_row(ax, y, states, label):
    """One worker group: N_GPUS boxes, coloured by state ('run', 'lost', 'resume')."""
    w, gap = 1.0, 0.18
    face = {"run": "#E8F2EF", "lost": "#FBEBE7", "resume": "#E8F2EF"}
    edge = {"run": GOOD, "lost": BAD, "resume": GOOD}
    for i, st in enumerate(states):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, 0.62, boxstyle="round,pad=0.015,rounding_size=0.08",
                                    fc=face[st], ec=edge[st], lw=1.4 if st == "lost" else 1.0, zorder=3))
        ax.text(x + w / 2, y + 0.37, f"GPU {i}", ha="center", va="center", fontsize=9, color=INK)
        ax.text(x + w / 2, y + 0.17, "lost" if st == "lost" else "training",
                ha="center", va="center", fontsize=8, color=BAD if st == "lost" else INK2)
    ax.text(-0.45, y + 0.31, label, ha="right", va="center", fontsize=10, color=INK, fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_axis_off()
    ax.set_xlim(-3.4, N_GPUS * 1.18 + 0.2)
    ax.set_ylim(-0.4, 2.62)

    before = ["run"] * N_GPUS
    before[PREEMPTED] = "lost"
    gpu_row(ax, 1.42, before, f"at step {CHECKPOINT_STEP}")
    gpu_row(ax, 0.18, ["resume"] * N_GPUS, "after the restart")

    # the reclaimed node, called out above the worker it took down
    px = PREEMPTED * 1.18 + 0.5
    ax.annotate("spot node reclaimed", xy=(px, 2.09), xytext=(px, 2.26), fontsize=9, color=BAD,
                ha="center", va="bottom", arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.1))
    ax.annotate("", xy=(px, 1.26), xytext=(px, 1.02),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(px + 0.2, 1.14, "Ray Train restarts the worker group; every worker reloads\n"
                            f"the checkpoint written at step {CHECKPOINT_STEP}",
            fontsize=9, color=INK2, va="center", linespacing=1.5)
    ax.text(N_GPUS * 1.18 - 0.18, -0.12, f"training continues from step {CHECKPOINT_STEP}, not from 0",
            fontsize=9.5, color=GOOD, ha="right", va="center", fontweight="bold")
    ax.plot([-3.3, N_GPUS * 1.18 + 0.1], [1.06, 1.06], color=RULE, lw=0.8, zorder=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "train_recovery.png", dpi=170)
    print("wrote", OUT / "train_recovery.png")


if __name__ == "__main__":
    main()
