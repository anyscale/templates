"""docs/images/campaign.gif: how a fragment grows into a potent compound, and how
well each model follows it. Real crystal poses from the EV-A71 2A release.

    python docs/figures/make_campaign_gif.py
"""

import textwrap

import numpy as np
from matplotlib.patches import Rectangle

from common import (BG, CRYSTAL, GOOD, BAD, INK, INK2, MUTED, RULE, MODEL_COLORS, MODEL_SHORT, OUT,
                    ca_trace, draw_backbone, draw_ligand, fig_to_image, load_crystal, load_results, plt,
                    rot_x, rot_y, save_gif, smooth_trace, view_matrix)

W, H, DPI = 8.0, 4.5, 120
NEW = "#7A4E9E"  # atoms added since the previous compound
SERIES = [
    ("x0926a", "A71EV2A-x0926a", "Fragment screen",
     "Small molecules (fragments) are soaked into protein crystals. They bind weakly, "
     "but each crystal shows exactly where. This 14-atom fragment starts the series."),
    ("x4431a", "A71EV2A-x4431a", "Early follow-on",
     "Chemists keep the fragment's anchor and add new groups (purple) that reach further "
     "into the pocket."),
    ("x7259a", "A71EV2A-x7259a", "Late follow-on",
     "Many design cycles later, the compound is twice the size and binds about 27x more "
     "tightly. Its new groups never appeared in the fragment screen."),
]

res = load_results("ev71-2a")
info = {c: res[res.complex_name == c].iloc[0] for _, c, *_ in SERIES}
xt = {lid: load_crystal("ev71-2a", lid, c) for lid, c, *_ in SERIES}

ref = xt["x7259a"]
ca, _ = ca_trace(ref["ca"])
trace = smooth_trace(ca)
pocket = ref["lig"].mean(0)
R0 = rot_x(-12) @ view_matrix(ca, pocket)
SPAN = 12.5


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def kd_text(pkd):
    if not np.isfinite(pkd):
        return "not measured"
    kd_nm = 10 ** (9 - pkd)
    return f"pKD {pkd:.1f}  (KD ≈ {kd_nm:,.0f} nM)"


def new_atoms(prev, cur):
    """Atoms of `cur` with no atom of `prev` within 1.2 A."""
    if prev is None:
        return np.zeros(len(cur["lig"]), bool)
    d = np.linalg.norm(cur["lig"][:, None] - prev["lig"][None], axis=-1).min(1)
    return d > 1.2


frames, durs = [], []


def frame(stage_idx, lig_draw, title, body, facts, models, ghost=None):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    bar = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bar.set_xlim(0, 1), bar.set_ylim(0, 1), bar.axis("off")
    bar.text(0.02, 0.955, "A chemical series from the EV-A71 2A campaign, crystal structures in the same pocket",
             fontsize=9, color=MUTED, va="top")
    # stage tracker
    names = ["Fragment", "Early follow-on", "Late follow-on", "Does each model keep up?"]
    x0, x1, y = 0.045, 0.955, 0.055
    w = (x1 - x0) / len(names)
    for i, n in enumerate(names):
        bar.add_patch(Rectangle((x0 + i * w + 0.002, y + 0.022), w - 0.004, 0.006,
                                color=INK if i <= stage_idx else RULE, lw=0))
        bar.text(x0 + i * w + 0.003, y - 0.008, n, fontsize=10, va="center",
                 color=INK if i == stage_idx else (INK2 if i < stage_idx else MUTED),
                 fontweight="bold" if i == stage_idx else "normal")

    ax = fig.add_axes([0.02, 0.12, 0.52, 0.8])
    ax.set_xlim(-SPAN * 1.15, SPAN * 1.15), ax.set_ylim(-SPAN, SPAN)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    draw_backbone(ax, trace, R0, pocket, lw=(1.2, 4.5))
    if ghost is not None:
        draw_ligand(ax, ghost["lig"], ghost["bonds"], R0, pocket, "#cfcac2", lw=2.4, atoms=False, zorder=3)
    lig_draw(ax)
    lens = plt.Circle((0, 0), SPAN * 0.98, transform=ax.transData)
    for art in ax.collections + ax.lines:
        art.set_clip_path(lens)
    ax.add_patch(plt.Circle((0, 0), SPAN * 0.98, fill=False, ec=RULE, lw=0.8))

    bar.text(0.585, 0.88, title, fontsize=18, fontweight="bold", color=INK, va="top")
    lines = textwrap.wrap(body, 38)
    for k, line in enumerate(lines):
        bar.text(0.585, 0.775 - k * 0.055, line, fontsize=12, color=INK2, va="baseline")
    yy = 0.775 - len(lines) * 0.055 - 0.04
    for k, (lab, val) in enumerate(facts):
        bar.text(0.585, yy - k * 0.055, lab, fontsize=10.5, color=MUTED, va="baseline")
        bar.text(0.70, yy - k * 0.055, val, fontsize=11.5, color=INK, va="baseline")
    if models:
        yy = yy - len(facts) * 0.055 - 0.03
        bar.text(0.585, yy, "Predicted pose error (ligand RMSD)", fontsize=10.5, color=MUTED, va="baseline")
        for k, m in enumerate(MODEL_COLORS):
            v = models[m]
            yk = yy - 0.058 * (k + 1)
            bar.add_patch(Rectangle((0.585, yk - 0.004), 0.011, 0.03, color=MODEL_COLORS[m], lw=0))
            bar.text(0.603, yk, MODEL_SHORT[m], fontsize=11.5, color=INK, va="baseline")
            bar.text(0.82, yk, f"{v:.1f} Å", fontsize=11.5, color=INK, va="baseline", ha="right")
            bar.text(0.84, yk, "correct" if v < 2 else "wrong", fontsize=11.5, va="baseline", fontweight="bold",
                     color=GOOD if v < 2 else BAD)
    return fig_to_image(fig, DPI)


def stage(idx, n=22):
    lid, cname, title, body = SERIES[idx]
    cur = xt[lid]
    prev = xt[SERIES[idx - 1][0]] if idx else None
    new = new_atoms(prev, cur)
    row = info[cname]
    # each atom grows out of the nearest atom of the previous compound
    if prev is None:
        start = np.repeat(cur["lig"].mean(0, keepdims=True), len(cur["lig"]), 0)
    else:
        nn = np.linalg.norm(cur["lig"][:, None] - prev["lig"][None], axis=-1).argmin(1)
        start = prev["lig"][nn]
    facts = [("Size", f"{int(row.heavy_atoms)} heavy atoms"), ("Binding", kd_text(row.experimental_pKD))]
    models = {m: float(row[f"{m}_lig_rmsd"]) for m in MODEL_COLORS}
    for i in range(n):
        a = ease(i / (n * 0.6))

        def draw(ax, a=a):
            xyz = start + a * (cur["lig"] - start)
            p = (xyz - pocket) @ R0.T
            for b0, b1 in cur["bonds"]:
                c = NEW if (new[b0] or new[b1]) else CRYSTAL
                ax.plot(p[[b0, b1], 0], p[[b0, b1], 1], color=c, lw=3.2, solid_capstyle="round", zorder=5)
            ax.scatter(p[:, 0], p[:, 1], s=22, c=[NEW if k else CRYSTAL for k in new], zorder=6, linewidths=0)

        frames.append(frame(idx, draw, title, body, facts, models if i >= n * 0.6 else None,
                            ghost=prev))
        durs.append(80)
    durs[-1] = 5200


def summary(n=26):
    stages = ["fragment", "early follow-on", "late follow-on"]
    labels = ["Fragment\nscreen", "Early\nfollow-ons", "Late\nfollow-ons"]
    acc = {m: [100 * (res[res.stage == s][f"{m}_lig_rmsd"] < 2).mean() for s in stages] for m in MODEL_COLORS}
    ns = [int((res.stage == s).sum()) for s in stages]
    for i in range(n + 1):
        t = ease(i / (n * 0.7))
        fig = plt.figure(figsize=(W, H), dpi=DPI)
        bar = fig.add_axes([0, 0, 1, 1], zorder=-1)
        bar.set_xlim(0, 1), bar.set_ylim(0, 1), bar.axis("off")
        bar.text(0.045, 0.94, "Does each model keep up as the chemistry changes?", fontsize=18,
                 fontweight="bold", color=INK, va="top")
        bar.text(0.045, 0.86, "Correct poses (ligand RMSD < 2 Å), 860 curated EV-A71 2A crystal structures, "
                 "by campaign stage.", fontsize=11.5, color=INK2, va="top")
        x0, x1, y = 0.045, 0.955, 0.055
        names = ["Fragment", "Early follow-on", "Late follow-on", "Does each model keep up?"]
        w = (x1 - x0) / 4
        for k, nme in enumerate(names):
            bar.add_patch(Rectangle((x0 + k * w + 0.002, y + 0.022), w - 0.004, 0.006, color=INK, lw=0))
            bar.text(x0 + k * w + 0.003, y - 0.008, nme, fontsize=10, va="center",
                     color=INK if k == 3 else INK2, fontweight="bold" if k == 3 else "normal")
        ax = fig.add_axes([0.1, 0.29, 0.6, 0.48])
        gx = np.arange(3)
        bw = 0.26
        for k, m in enumerate(MODEL_COLORS):
            v = np.array(acc[m]) * t
            ax.bar(gx + (k - 1) * (bw + 0.02), v, bw, color=MODEL_COLORS[m], lw=0, zorder=3)
            if t > 0.999:
                for g in gx:
                    ax.text(g + (k - 1) * (bw + 0.02), acc[m][g] + 2, f"{acc[m][g]:.0f}", ha="center",
                            va="bottom", fontsize=10, color=INK)
        ax.set_xticks(gx, [f"{l}\nn = {n_}" for l, n_ in zip(labels, ns)], fontsize=10.5)
        ax.set_ylim(0, 105), ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("correct poses (%)", fontsize=10.5)
        ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)
        ax.tick_params(axis="x", length=0)
        # legend, ink text with colour swatch
        for k, m in enumerate(MODEL_COLORS):
            yk = 0.74 - k * 0.055
            bar.add_patch(Rectangle((0.76, yk - 0.012), 0.014, 0.032, color=MODEL_COLORS[m], lw=0))
            bar.text(0.782, yk + 0.004, MODEL_SHORT[m], fontsize=11.5, color=INK, va="center")
        if t > 0.999:
            note = ("Boltz-2 misses almost every follow-on. The fine-tuned model trained on these fragments, "
                    "so only its follow-on scores are a fair test.")
            for k, line in enumerate(textwrap.wrap(note, 27, break_on_hyphens=False)):
                bar.text(0.76, 0.53 - k * 0.052, line, fontsize=10.5, color=INK2, va="baseline")
        frames.append(fig_to_image(fig, DPI))
        durs.append(80)
    durs[-1] = 6000


if __name__ == "__main__":
    for k in range(3):
        stage(k)
    summary()
    size = save_gif(frames, durs, OUT / "campaign.gif", extra=(NEW, "#cfcac2"))
    print(f"{len(frames)} frames, {sum(durs) / 1000:.1f} s, {size / 1e6:.1f} MB")
