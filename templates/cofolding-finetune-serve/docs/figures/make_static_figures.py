"""Static README figures, drawn from the benchmark outputs.

    python docs/figures/make_static_figures.py

workflow.png, architecture.png, pose_accuracy.png, affinity.png, finetune.png,
throughput.png, gpu_cluster.png
"""

import json

import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import (BG, INK, INK2, MUTED, RULE, MODEL_COLORS, MODEL_NAMES, MODEL_SHORT, OUT, SHARED, plt,
                    load_results, panel_label)

DPI = 300
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.titlesize": 9.5})
TARGET_NAMES = {"ev71-2a": "EV-A71 2A protease", "sars2-mpro": "SARS-CoV-2 main protease"}
STAGES = ["fragment", "early follow-on", "late follow-on"]
STAGE_LABELS = ["Fragment\nscreen", "Early\nfollow-ons", "Late\nfollow-ons"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", OUT / name)


def model_legend(fig, y=0.965, x=0.07, short=False):
    step = 0.9 / max(1, len(MODEL_COLORS))     # fits however many models are plotted
    for k, m in enumerate(MODEL_COLORS):
        xk = x + k * step
        fig.patches.append(Rectangle((xk, y - 0.012), 0.012, 0.024, color=MODEL_COLORS[m], lw=0,
                                     transform=fig.transFigure, figure=fig))
        fig.text(xk + 0.018, y, (MODEL_SHORT if short else MODEL_NAMES)[m], fontsize=8.5, color=INK, va="center")


# ---------------------------------------------------------------------------
def workflow():
    boxes = [("Data", "Public datasets", "2,147 crystal structures\nwith measured affinity,\n3 antiviral targets"),
             ("Predict", "Ray Data", "Boltz-2, OpenBind-0 and\nfine-tuned OpenFold3 on\nautoscaling L4 GPUs"),
             ("Score", "Ray Data", "Ligand RMSD of every\npredicted pose against\nits crystal structure"),
             ("Fine-tune", "Ray Train", "OpenFold3 on the\nEV-A71 2A fragment\nscreen, multi-GPU"),
             ("Serve", "Ray Serve", "Live pose and affinity\nprediction, 3D results\nviewer")]
    fig = plt.figure(figsize=(7.2, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    bw, gap, x0 = 0.172, 0.0245, 0.02
    for k, (title, tech, body) in enumerate(boxes):
        bx = x0 + k * (bw + gap)
        ax.add_patch(FancyBboxPatch((bx, 0.08), bw, 0.84, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc=BG, ec=INK2, lw=0.7))
        ax.text(bx + 0.012, 0.83, f"{k + 1}  {title}", fontsize=9.5, fontweight="bold", color=INK, va="top")
        ax.text(bx + 0.012, 0.64, tech, fontsize=8, fontweight="bold", color=MODEL_COLORS["boltz2"], va="top")
        ax.text(bx + 0.012, 0.47, body, fontsize=6.8, color=INK2, va="top", linespacing=1.4)
        if k < len(boxes) - 1:
            ax.annotate("", xy=(bx + bw + gap - 0.002, 0.5), xytext=(bx + bw + 0.002, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=6))
    save(fig, "workflow.png")


# ---------------------------------------------------------------------------
def architecture():
    lanes = [("boltz2", 4, 2, "pose + affinity"), ("openbind0", 2, 3, "pose"), ("of3p2", 2, 2, "pose")]
    fig = plt.figure(figsize=(7.2, 3.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100), ax.set_ylim(0, 45), ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f4f2ee"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.45)

    box(1.5, 13.5, 13, 16, "Ligands", "699 unique SMILES\n(EV-A71 2A),\nchecked once\nwith RDKit")
    ys = [33, 20, 7]
    ax.plot([14.5, 18], [21.5, 21.5], color=INK2, lw=0.8)
    ax.plot([18, 18], [ys[2] + 3.2, ys[0] + 3.2], color=INK2, lw=0.8)
    for (m, n_gpu, per, out), y in zip(lanes, ys):
        ax.annotate("", xy=(21.2, y + 3.2), xytext=(18, y + 3.2),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))
        ax.text(21.5, y + 8.3, MODEL_NAMES[m], fontsize=8, fontweight="bold", color=INK, va="bottom")
        for g in range(n_gpu):
            gx = 21.5 + g * 9.2
            ax.add_patch(Rectangle((gx, y), 8.4, 6.6, fc=BG, ec=INK2, lw=0.6))
            ax.text(gx + 0.6, y + 6.0, "L4", fontsize=5.5, color=MUTED, va="top")
            sw = (8.4 - 1.2 - 0.5 * (per - 1)) / per
            for p in range(per):
                ax.add_patch(Rectangle((gx + 0.6 + p * (sw + 0.5), y + 0.7), sw, 3.9, color=MODEL_COLORS[m], lw=0))
        ax.text(60.5, y + 4.6, f"{n_gpu} GPUs, {per} predictors per GPU", fontsize=6.9, color=INK, va="center")
        ax.text(60.5, y + 1.9, out, fontsize=6.9, color=MUTED, va="center")
        ax.plot([77, 79.2], [y + 3.2, y + 3.2], color=INK2, lw=0.8)
    ax.plot([79.2, 79.2], [ys[2] + 3.2, ys[0] + 3.2], color=INK2, lw=0.8)
    ax.annotate("", xy=(80.8, 21.5), xytext=(79.2, 21.5),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))
    box(81, 13.5, 17.5, 16, "Join and score", "on CPU nodes:\nligand RMSD against\nevery crystal,\nParquet output")
    ax.text(1.5, 1.2, "Each model runs its own Ray Data pipeline, all at the same time, on an 8-GPU budget split by "
            "cost per ligand. Each coloured block is 1 predictor process.", fontsize=6.6, color=MUTED)
    save(fig, "architecture.png")


# ---------------------------------------------------------------------------
# Models fine-tuned on 1 target are plotted on that target only: their score on another
# protein is a different claim (specialization), and side by side it reads as "fine-tuning
# made the model worse". That comparison lives in the text instead.
TARGET_SPECIFIC = {"of3ft": "ev71-2a", "of3ray": "ev71-2a"}


def models_for(tkey):
    return [m for m in MODEL_COLORS if TARGET_SPECIFIC.get(m, tkey) == tkey]


def pose_accuracy():
    data = {t: load_results(t) for t in TARGET_NAMES}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), gridspec_kw=dict(height_ratios=[1, 0.95]))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.20, hspace=0.46, wspace=0.2)
    model_legend(fig, short=True)
    grid = np.logspace(np.log10(0.1), np.log10(30), 300)
    for col, (tkey, df) in enumerate(data.items()):
        # (a, b) cumulative success rate vs RMSD threshold
        ax = axes[0, col]
        for m in models_for(tkey):
            r = df[f"{m}_lig_rmsd"].dropna().to_numpy()
            ax.plot(grid, [(r < g).mean() * 100 for g in grid], color=MODEL_COLORS[m], lw=1.6)
        ax.axvline(2, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.text(2.15, 4, "2 Å", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
        ax.set_xlim(0.1, 30), ax.set_ylim(0, 100)
        ax.set_xlabel("ligand RMSD threshold (Å)")
        ax.set_ylabel("poses within threshold (%)" if col == 0 else "")
        ax.set_title(f"{TARGET_NAMES[tkey]}   n = {len(df)}", pad=8)
        ax.grid(color=RULE, lw=0.5)
        panel_label(ax, "ab"[col], x=-0.1)
        # (c, d) success rate by campaign stage
        ax = axes[1, col]
        gx = np.arange(3)
        shown = models_for(tkey)
        n = len(shown)
        bw = 0.74 / n                      # the whole group spans 0.74 of the slot
        for k, m in enumerate(shown):
            v = [100 * (df[df.stage == s][f"{m}_lig_rmsd"] < 2).mean() for s in STAGES]
            xs = gx + (k - (n - 1) / 2) * bw
            ax.bar(xs, v, bw * 0.88, color=MODEL_COLORS[m], lw=0, zorder=3)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv + 1.5, f"{vv:.0f}", ha="center", va="bottom", fontsize=6.2, color=INK2)
        ns = [int((df.stage == s).sum()) for s in STAGES]
        ax.set_xticks(gx, [f"{l}\nn = {n}" for l, n in zip(STAGE_LABELS, ns)], fontsize=7.5)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 108), ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("poses within 2 Å (%)" if col == 0 else "")
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
        panel_label(ax, "cd"[col], x=-0.1)
    fig.text(0.085, 0.072, "Curated binding events (valid reference pose, no suspected crystallographic artefact). "
             "1 sampled pose per ligand.\nThe fine-tuned model was trained on EV-A71 2A fragments, so it is shown "
             "on that target only.", fontsize=6.8, color=MUTED, va="top", linespacing=1.6)
    save(fig, "pose_accuracy.png")


# ---------------------------------------------------------------------------
def affinity():
    from scipy.stats import spearmanr
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.19, wspace=0.3)
    labels = {"ev71-2a": "measured pKD", "sars2-mpro": "measured pIC50"}
    for k, tkey in enumerate(TARGET_NAMES):
        d = load_results(tkey).drop_duplicates("ligand_id").dropna(subset=["experimental_pKD",
                                                                           "boltz2_affinity_pred_value"])
        x = d.experimental_pKD.to_numpy()
        # Boltz-2 reports log10(IC50 / uM); 6 - value converts it to a pIC50-like scale (higher = tighter)
        y = 6 - d.boltz2_affinity_pred_value.to_numpy()
        rho = spearmanr(x, y).statistic
        ax = axes[k]
        ax.scatter(x, y, s=9, color=MODEL_COLORS["boltz2"], alpha=0.55, lw=0)
        ax.set_xlabel(labels[tkey])
        ax.set_ylabel("Boltz-2 predicted affinity\n(6 − log10 IC50 in µM)" if k == 0 else "")
        ax.set_title(f"{TARGET_NAMES[tkey]}", pad=8)
        ax.text(0.03, 0.96, f"Spearman ρ = {rho:.2f}\nn = {len(d)} compounds", transform=ax.transAxes,
                va="top", fontsize=8, color=INK, bbox=dict(fc=BG, ec="none", pad=1.5, alpha=0.9))
        ax.grid(color=RULE, lw=0.5)
        panel_label(ax, "ab"[k], x=-0.14)
    fig.text(0.105, 0.02, "Each point is a compound with a measured affinity. Higher is tighter binding on both axes.",
             fontsize=6.8, color=MUTED)
    save(fig, "affinity.png")


# ---------------------------------------------------------------------------
def finetune():
    runs = sorted((SHARED / "finetune" / "runs").glob("*/checkpoint_*/loss_curve.json"))
    curve = pd.DataFrame(json.loads(runs[-1].read_text()))
    curve = curve[curve.step > 1]  # step 1 includes kernel compilation
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.2, wspace=0.28)
    spec = [("loss", "training loss", "lower is better", MODEL_COLORS["boltz2"]),
            ("lddt_inter_protein_ligand_step", "protein-ligand lDDT", "higher is better", MODEL_COLORS["of3ft"])]
    for k, (col, name, hint, color) in enumerate(spec):
        ax = axes[k]
        s = curve[col]
        ax.plot(curve.step, s, color=color, lw=0.6, alpha=0.35)
        ax.plot(curve.step, s.rolling(20, min_periods=20).mean(), color=color, lw=1.8)
        first, last = s.iloc[:20].mean(), s.iloc[-20:].mean()
        digits = 3 if col == "loss" else 2
        ax.set_title(f"{name}: {first:.{digits}f} → {last:.{digits}f}", pad=8)
        ax.text(1.0, 1.02, hint, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.set_xlabel("optimizer step")
        ax.set_xlim(0, curve.step.max() + 1)
        ax.grid(color=RULE, lw=0.5)
        panel_label(ax, "ab"[k], x=-0.12)
    fig.text(0.085, 0.02, "66 EV-A71 2A fragment structures, 2 L4 GPUs, from the OpenBind-0 weights. Thin line: "
             "each step. Thick line: 20-step mean. Titles: mean of first vs last 20 steps.", fontsize=6.4, color=MUTED)
    save(fig, "finetune.png")
    print(f"  loss {curve.loss.iloc[:20].mean():.3f} -> {curve.loss.iloc[-20:].mean():.3f}; lDDT "
          f"{curve.lddt_inter_protein_ligand_step.iloc[:20].mean():.2f} -> "
          f"{curve.lddt_inter_protein_ligand_step.iloc[-20:].mean():.2f}")


# ---------------------------------------------------------------------------
def throughput():
    """Predictors per GPU and vCPUs per node, from the sweep of 2026-10-22.

    $AIDDD_SHARED/runs/pack/{cpu8-20261022-085733,cpu16-20261022-085429}/summary.json:
    1 L4, EV-A71 2A, batches of 8. Each point is 64 ligands for OpenFold3, 24 for Boltz-2.
    """
    sec = {
        ("OpenFold3", 8): {1: 23.8, 2: 20.2, 3: 20.7, 4: 19.5},
        ("OpenFold3", 16): {1: 23.1, 2: 14.0, 3: 14.0, 4: 14.1},
        ("Boltz-2", 8): {1: 39.2, 2: 29.5},
        ("Boltz-2", 16): {1: 40.0, 2: 28.0},
    }
    cpu_color = {8: "#9FB0BC", 16: MODEL_COLORS["boltz2"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.76, bottom=0.31, wspace=0.26)
    for ax, model in zip(axes, ("OpenFold3", "Boltz-2")):
        per = sorted(sec[(model, 16)])
        x = np.arange(len(per))
        for k, cpus in enumerate((8, 16)):
            v = [3600 / sec[(model, cpus)][n] for n in per]
            ax.bar(x + (k - 0.5) * 0.38, v, 0.34, color=cpu_color[cpus], lw=0, zorder=3,
                   label=f"{cpus} vCPU per node")
            for xx, vv in zip(x + (k - 0.5) * 0.38, v):
                ax.text(xx, vv + 6, f"{vv:.0f}", ha="center", va="bottom", fontsize=6.4, color=INK2)
        ax.set_xticks(x, [str(n) for n in per])
        ax.set_xlabel("predictors sharing 1 L4 GPU")
        ax.set_ylim(0, 300)
        ax.set_ylabel("ligands per hour" if model == "OpenFold3" else "")
        ax.set_title(model, pad=6)
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.1, borderpad=0.2)
    fig.text(0.1, 0.93, "2 predictors per GPU, on a node with enough CPU", fontsize=9.5, fontweight="bold")
    fig.text(0.1, 0.18, "1 L4, EV-A71 2A, batches of 8 ligands. OpenFold3 gains only when the node has 16 vCPU, "
             "because it\nfeaturizes on CPU; past 2 predictors nothing changes. Boltz-2 gains from the 2nd "
             "predictor either way.", fontsize=6.4, color=MUTED, va="top", linespacing=1.5)
    save(fig, "throughput.png")


# ---------------------------------------------------------------------------
GPU_LOGS = os.environ.get("AIDDD_GPU_LOGS", str(SHARED / "results_base-of3p2" / "mers-mpro_gpu"))
# What the run was, for the figure title. Set AIDDD_GPU_LABEL when pointing GPU_LOGS elsewhere.
GPU_LOGS_LABEL = os.environ.get("AIDDD_GPU_LABEL", "MERS-CoV Mpro benchmark")


def gpu_cluster():
    import matplotlib.colors as mcolors
    # Per-node samples written beside the results by aiddd_pipeline.py. GPU_LOGS picks the run.
    files = sorted(Path(GPU_LOGS).glob("*.csv"))
    if not files:
        print(f"no GPU samples under {GPU_LOGS}: keeping the existing gpu_cluster.png")
        return
    logs = [pd.read_csv(f, names=["t", "util", "mem"]) for f in files]
    t0 = min(l.t.min() for l in logs)
    logs.sort(key=lambda l: l.t.min())
    end = max(l.t.max() for l in logs)
    nmin = int(np.ceil((end - t0) / 60))
    grid = np.full((len(logs), nmin), np.nan)
    for i, l in enumerate(logs):
        b = ((l.t - t0) // 60).astype(int)
        grid[i, :] = l.groupby(b).util.mean().reindex(range(nmin)).to_numpy()
    all_up = max(l.t.min() for l in logs)
    first_down = min(l.t.max() for l in logs)
    steady = pd.concat([l[(l.t >= all_up) & (l.t <= first_down)] for l in logs]).util.mean()
    overall = pd.concat(logs).util.mean()
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    fig.subplots_adjust(left=0.09, right=0.9, top=0.8, bottom=0.25)
    cmap = mcolors.LinearSegmentedColormap.from_list("u", ["#f1efeb", "#9fb6cf", MODEL_COLORS["boltz2"], "#0f2f52"])
    cmap.set_bad(BG)
    im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=0, vmax=100,
                   extent=(0, nmin, len(logs) - 0.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(len(logs)), [f"GPU {i + 1}" for i in range(len(logs))], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("minutes since the first GPU node joined")
    for s in ax.spines.values():
        s.set_visible(False)
    for x in [(all_up - t0) / 60, (first_down - t0) / 60]:
        ax.axvline(x, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text((all_up - t0) / 60 + 0.5, -0.9, f"all {len(logs)} GPUs up", fontsize=7, color=INK2, va="bottom")
    ax.text((first_down - t0) / 60 - 0.5, -0.9, "first GPU released", fontsize=7, color=INK2, va="bottom", ha="right")
    cax = fig.add_axes([0.92, 0.25, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.ax.set_title("GPU %", fontsize=7, color=INK2, pad=4)
    fig.text(0.085, 0.95, f"GPU utilization per node, {GPU_LOGS_LABEL} ({len(logs)} L4 GPUs)", fontsize=9.5,
             fontweight="bold")
    fig.text(0.085, 0.02, f"1-minute means of nvidia-smi samples taken every 5 s. Mean while all {len(logs)} GPUs were up: "
             f"{steady:.0f}%; over the whole run: {overall:.0f}%.\nEach node is released once its model has no work "
             f"left, so the cluster shrinks as the run finishes.", fontsize=6.6, color=MUTED, linespacing=1.5)
    save(fig, "gpu_cluster.png")
    print(f"  steady {steady:.1f}%, overall {overall:.1f}%, all up at {(all_up - t0) / 60:.1f} min, "
          f"first down {(first_down - t0) / 60:.1f} min, end {(end - t0) / 60:.1f} min")


if __name__ == "__main__":
    workflow()
    architecture()
    pose_accuracy()
    affinity()
    finetune()
    throughput()
    gpu_cluster()
