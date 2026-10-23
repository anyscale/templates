"""docs/images/workflow.gif: the whole workflow in six steps, from real outputs.

    python docs/figures/make_hero_gif.py

1 the question, 2 the crystal answer key, 3 two models predict the pose,
4 the pose is scored, 5 every complex on a GPU cluster, 6 the Ray workflow.
"""

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import (BG, CRYSTAL, GOOD, BAD, INK, INK2, MONO, MUTED, RULE, MODEL_COLORS, OUT, SHARED, TARGETS,
                    ca_trace, draw_backbone, draw_ligand, fig_to_image, load_crystal, load_prediction_in_crystal_frame,
                    load_results, plt, rot_x, rot_y, save_gif, smooth_trace, view_matrix)

W, H, DPI = 8.0, 4.5, 120  # 960 x 540 px
LIGAND = ("x7259a", "A71EV2A-x7259a")
SMILES = "CSc1ncnc(N2CCC[C@@H]2C2CC2)c1CC(=O)N1CCC[C@H](C(N)=O)C1"
SEQ = TARGETS["ev71-2a"].path("fasta").read_text().splitlines()[1].strip()
STEPS = ["Question", "Answer key", "Predict", "Score", "Scale up", "Workflow"]
SHOWN = {"openbind0": "OpenBind-0", "boltz2": "Boltz-2"}


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
xtal = load_crystal("ev71-2a", *LIGAND)
pred = {m: load_prediction_in_crystal_frame(SHARED / "predictions" / m / f"{LIGAND[0]}.cif", xtal) for m in SHOWN}
res = load_results("ev71-2a")
row = res[res.complex_name == LIGAND[1]].iloc[0]
rmsd = {m: float(row[f"{m}_lig_rmsd"]) for m in SHOWN}

ca, _ = ca_trace(xtal["ca"])
trace = smooth_trace(ca)
center = ca.mean(0)
R0 = rot_x(-10) @ view_matrix(ca, xtal["lig"].mean(0))
span = np.abs((ca - center) @ R0.T)[:, :2].max() + 4

rng = np.random.default_rng(7)
noise = {m: rng.normal(size=(len(pred[m]["lig"]), 3)) for m in SHOWN}
wobble = {m: rng.normal(size=(len(pred[m]["lig"]), 3)) for m in SHOWN}


def assignment(P, pe, Q, qe):
    from scipy.optimize import linear_sum_assignment
    pairs = []
    for el in set(pe) & set(qe):
        pi = [i for i, e in enumerate(pe) if e == el]
        qi = [i for i, e in enumerate(qe) if e == el]
        r, c = linear_sum_assignment(np.linalg.norm(P[pi][:, None] - Q[qi][None], axis=-1))
        pairs += [(pi[a], qi[b]) for a, b in zip(r, c)]
    return pairs


pairs = {m: assignment(pred[m]["lig"], pred[m]["el"], xtal["lig"], xtal["el"]) for m in SHOWN}

# Scale-up grid: every curated EV-A71 2A and SARS-CoV-2 Mpro binding event, in campaign order.
grids = {}
for tkey in ["ev71-2a", "sars2-mpro"]:
    d = load_results(tkey).copy()
    d["order"] = d.stage.map({"fragment": 0, "early follow-on": 1, "late follow-on": 2})
    d = d.sort_values(["order", "xtal_index"])
    grids[tkey] = {m: (d[f"{m}_lig_rmsd"] < 2).to_numpy() for m in MODEL_COLORS}
    grids[tkey]["n"] = len(d)
    grids[tkey]["stage_breaks"] = np.searchsorted(d.order.to_numpy(), [1, 2])


# ---------------------------------------------------------------------------
# Frame scaffolding
# ---------------------------------------------------------------------------
def new_frame(step, caption=True):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    bar = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bar.set_xlim(0, 1), bar.set_ylim(0, 1), bar.axis("off")
    # step tracker along the bottom
    x0, x1, y = 0.045, 0.955, 0.055
    w = (x1 - x0) / len(STEPS)
    for i, name in enumerate(STEPS):
        done, now = i < step, i == step
        bar.add_patch(Rectangle((x0 + i * w + 0.002, y + 0.022), w - 0.004, 0.006,
                                color=INK if (done or now) else RULE, lw=0))
        bar.text(x0 + i * w + 0.003, y - 0.008, f"{i + 1}  {name}", fontsize=10,
                 color=INK if now else (INK2 if done else MUTED), fontweight="bold" if now else "normal",
                 va="center", ha="left")
    if caption:
        bar.text(0.02, 0.955, "EV-A71 2A protease with compound x7259a (late follow-on, pKD 7.9)", fontsize=9,
                 color=MUTED, ha="left", va="top")
    return fig, bar


def text_block(bar, step_title, body, x=0.585, y=0.86, width=38):
    """Title plus body text, one line per call so every line sits on the same grid."""
    import textwrap
    bar.text(x, y, step_title, fontsize=18, fontweight="bold", color=INK, va="top", ha="left")
    for k, line in enumerate(textwrap.wrap(body, width)):
        bar.text(x, y - 0.13 - k * 0.056, line, fontsize=12, color=INK2, va="baseline", ha="left")
    return y - 0.1 - len(textwrap.wrap(body, width)) * 0.056


def mol_axes(fig):
    ax = fig.add_axes([0.02, 0.13, 0.53, 0.83])
    ax.set_xlim(-span, span), ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def key(ax, items):
    """Small colour key in the bottom-left corner of the structure panel."""
    for k, (label, color) in enumerate(items):
        y = 0.04 + (len(items) - 1 - k) * 0.055
        ax.plot([0.03, 0.08], [y, y], color=color, lw=3, transform=ax.transAxes, solid_capstyle="round")
        ax.text(0.1, y, label, fontsize=10, color=INK2, va="center", transform=ax.transAxes)


def view(t):
    """Slow rocking rotation for depth."""
    return rot_y(18 * np.sin(2 * np.pi * t)) @ R0


def tag(ax, xy, text, color, dx=4.5, dy=4.0, ha="left"):
    ax.annotate(text, xy=xy, xytext=(xy[0] + dx, xy[1] + dy), fontsize=11, color=INK, fontweight="bold",
                ha=ha, va="center", arrowprops=dict(arrowstyle="-", color=color, lw=1.2, shrinkA=0, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none"))


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
frames, durs = [], []


def add(img, ms):
    frames.append(img)
    durs.append(ms)


def scene_question(n=26):
    for i in range(n):
        t = i / n
        fig, bar = new_frame(0)
        ax = mol_axes(fig)
        R = view(t * 0.5)
        draw_backbone(ax, trace, R, center)
        p = (xtal["lig"].mean(0) - center) @ R.T
        ax.add_patch(plt.Circle(p[:2], 5.5, fill=False, ls=(0, (2, 2)), lw=1.1, color=INK2, zorder=6))
        ax.text(p[0], p[1] - 7.5, "binding pocket", fontsize=10.5, color=INK2, ha="center", va="top", zorder=6)
        text_block(bar, "Where does a drug bind?",
                   "Co-folding models predict the protein and the compound together in 3D, "
                   "from 2 text inputs only.")
        k = int(len(SEQ) * ease(t * 1.6))
        bar.text(0.585, 0.50, "Protein sequence", fontsize=10.5, color=MUTED, va="top")
        seq = SEQ[:k]
        bar.text(0.585, 0.445, (seq[:34] + ("\n" + seq[34:68] if k > 34 else "")) + (" …" if k > 68 else ""),
                 fontsize=9.5, family=MONO, color=INK, va="top", linespacing=1.3)
        s = SMILES[:int(len(SMILES) * ease(t * 1.6 - 0.3))]
        bar.text(0.585, 0.31, "Compound (SMILES)", fontsize=10.5, color=MUTED, va="top")
        bar.text(0.585, 0.255, s[:34] + ("\n" + s[34:] if len(s) > 34 else ""), fontsize=9.5,
                 family=MONO, color=INK, va="top", linespacing=1.3)
        add(fig_to_image(fig, DPI), 90)
    durs[-1] = 2600


def scene_crystal(n=22):
    for i in range(n):
        t = 0.5 + i / n * 0.5
        fig, bar = new_frame(1)
        ax = mol_axes(fig)
        R = view(t * 0.5)
        draw_backbone(ax, trace, R, center)
        a = ease(i / (n * 0.4))
        p = draw_ligand(ax, xtal["lig"], xtal["bonds"], R, center, CRYSTAL, alpha=a)
        if a > 0.95:
            c = p.mean(0)
            tag(ax, (c[0] - 3.5, c[1] + 2.5), "crystal pose", CRYSTAL, dx=-7, dy=6.5, ha="right")
        text_block(bar, "The answer key: crystals",
                   "A crystal structure shows where the compound actually binds. "
                   "2,147 structures from 3 antiviral campaigns serve as ground truth. "
                   "The models never see them.")
        add(fig_to_image(fig, DPI), 90)
    durs[-1] = 3000


def scene_predict(n=36):
    for i in range(n):
        t = i / n
        fig, bar = new_frame(2)
        ax = mol_axes(fig)
        R = view(0.5 + 0.0)
        draw_backbone(ax, trace, R, center)
        draw_ligand(ax, xtal["lig"], xtal["bonds"], R, center, CRYSTAL, alpha=0.18, atoms=False, lw=2.2)
        a = ease(t / 0.8)
        for m in SHOWN:
            x0 = pred[m]["lig"]
            sigma = 14 * (1 - a) ** 1.3
            start = center + 0.0 * x0
            x = (1 - a) * (start + sigma * noise[m]) + a * x0 + 0.6 * (1 - a) * np.sin(i / 3) * wobble[m]
            draw_ligand(ax, x, pred[m]["bonds"] if a > 0.9 else [], R, center, MODEL_COLORS[m],
                        size=14 if a < 0.9 else 16)
            if a > 0.98:
                c = ((x0 - center) @ R.T).mean(0)
                tag(ax, (c[0] + 2.5, c[1] + 2), SHOWN[m], MODEL_COLORS[m], dx=6, dy=6)
        key(ax, [("crystal pose (faint)", CRYSTAL), ("OpenBind-0", MODEL_COLORS["openbind0"]),
                 ("Boltz-2", MODEL_COLORS["boltz2"])])
        text_block(bar, "2 models, 2 predictions",
                   "Each model refines randomly placed atoms into a 3D pose. "
                   "From identical inputs, OpenBind-0 and Boltz-2 reach different answers.")
        add(fig_to_image(fig, DPI), 90)
    durs[-1] = 2800


def scene_score(n=30):
    for i in range(n):
        t = i / n
        fig, bar = new_frame(3)
        ax = mol_axes(fig)
        R = view(0.5)
        draw_backbone(ax, trace, R, center)
        q = draw_ligand(ax, xtal["lig"], xtal["bonds"], R, center, CRYSTAL)
        a = ease(t / 0.5)
        for m in SHOWN:
            p = draw_ligand(ax, pred[m]["lig"], pred[m]["bonds"], R, center, MODEL_COLORS[m], alpha=0.9)
            for pi, qi in pairs[m]:
                e = p[pi] + a * (q[qi] - p[pi])
                ax.plot([p[pi, 0], e[0]], [p[pi, 1], e[1]], color=MUTED, lw=0.7, ls=(0, (1.5, 1.5)), zorder=4)
        key(ax, [("crystal pose", CRYSTAL), ("OpenBind-0", MODEL_COLORS["openbind0"]),
                 ("Boltz-2", MODEL_COLORS["boltz2"]), ("atom-to-atom error", MUTED)])
        text_block(bar, "Score against the crystal",
                   "Ligand RMSD: the typical distance between each predicted atom and its crystal "
                   "position. Below 2 Å, about 1 bond length, a pose counts as correct.")
        for k, m in enumerate(SHOWN):
            if a < 0.99:
                continue
            ok = rmsd[m] < 2
            y = 0.36 - k * 0.09
            bar.add_patch(Rectangle((0.585, y - 0.02), 0.012, 0.04, color=MODEL_COLORS[m], lw=0))
            bar.text(0.607, y, SHOWN[m], fontsize=12, color=INK, va="center", fontweight="bold")
            bar.text(0.76, y, f"{rmsd[m]:.1f} Å", fontsize=12, color=INK, va="center")
            bar.text(0.84, y, "correct pose" if ok else "wrong pocket", fontsize=12,
                     color=GOOD if ok else BAD, va="center", fontweight="bold")
        add(fig_to_image(fig, DPI), 90)
    durs[-1] = 4000


def grid_layout(g, rows):
    """Column/row of every binding event; each campaign stage starts a new column after a gap."""
    pos, col, starts = [], 0, []
    bounds = [0, *g["stage_breaks"], g["n"]]
    for a, b in zip(bounds[:-1], bounds[1:]):
        starts.append(col)
        for j in range(b - a):
            pos.append((col + j // rows, j % rows))
        col += int(np.ceil((b - a) / rows)) + 1
    return pos, starts, col - 1


def scene_scale(n=40):
    import matplotlib.colors as mcolors
    names = {"ev71-2a": "EV-A71 2A protease", "sars2-mpro": "SARS-CoV-2 main protease"}
    labels = {"boltz2": "Boltz-2", "openbind0": "OpenBind-0", "of3ft": "OpenFold3, fine-tuned\non EV-A71 2A"}
    rows, cpx, pw = 15, 5, int(W * DPI)
    ph = int(H * DPI)
    lay = {k: grid_layout(g, rows) for k, g in grids.items()}
    x_px = {"ev71-2a": 215}
    x_px["sars2-mpro"] = x_px["ev71-2a"] + lay["ev71-2a"][2] * cpx + 70
    top_px, row_px = 160, rows * cpx + 26
    pale = tuple(int(255 * v) for v in mcolors.to_rgb("#e4e0d9"))
    for i in range(n + 1):
        t = ease(i / n)
        fig, bar = new_frame(4, caption=False)
        bar.text(0.045, 0.94, "Scale up: every crystal structure, 3 models", fontsize=18,
                 fontweight="bold", color=INK, va="top")
        bar.text(0.045, 0.855, "Ray Data runs the 3 models on autoscaling L4 GPUs. 1 square = 1 crystal structure,\n"
                 "in campaign order. Filled: within 2 Å of the crystal. Pale: wrong pose.",
                 fontsize=11, color=INK2, va="top", linespacing=1.4)
        for tkey in grids:
            bar.text(x_px[tkey] / pw, 1 - (top_px - 8) / ph, names[tkey], fontsize=10.5, fontweight="bold",
                     color=INK, va="bottom")
        done = t >= 0.999
        for r_, m in enumerate(MODEL_COLORS):
            yc = 1 - (top_px + r_ * row_px + rows * cpx / 2) / ph
            bar.text(0.045, yc, labels[m], fontsize=10, color=INK, fontweight="bold", va="center", linespacing=1.2)
            for tkey, g in grids.items():
                if done:
                    bar.text((x_px[tkey] + lay[tkey][2] * cpx + 10) / pw, yc, f"{100 * g[m].mean():.0f}%",
                             fontsize=13, fontweight="bold", color=INK, va="center")
        yb = 1 - (top_px + 2 * row_px + rows * cpx + 6) / ph
        for tkey, g in grids.items():
            _, starts, _ = lay[tkey]
            for k, (c, lab) in enumerate(zip(starts, ["fragment screen", "early follow-ons", "late follow-ons"])):
                x = (x_px[tkey] + c * cpx) / pw
                drop = 0.04 if k == 0 else 0.0
                bar.plot([x, x], [yb + 0.008, yb - drop - 0.004], color=MUTED, lw=0.7)
                bar.text(x + 0.004, yb - drop, lab, fontsize=8.5, color=MUTED, va="top")
        img = fig_to_image(fig, DPI)
        px = img.load()
        for r_, m in enumerate(MODEL_COLORS):
            col = tuple(int(255 * v) for v in mcolors.to_rgb(MODEL_COLORS[m]))
            for tkey, g in grids.items():
                pos = lay[tkey][0]
                for j in range(int(round(t * g["n"]))):
                    c, r = pos[j]
                    x0, y0 = x_px[tkey] + c * cpx, top_px + r_ * row_px + r * cpx
                    fill = col if g[m][j] else pale
                    for dx in range(cpx - 1):
                        for dy in range(cpx - 1):
                            px[x0 + dx, y0 + dy] = fill
        add(img, 80)
    durs[-1] = 5000


def scene_workflow(n=30):
    boxes = [("Data", "Public datasets", "2,147 crystal\nstructures with\nmeasured affinity"),
             ("Predict", "Ray Data", "3 models on\nautoscaling\nL4 GPUs"),
             ("Score", "Ray Data", "Ligand RMSD\nagainst every\ncrystal"),
             ("Fine-tune", "Ray Train", "OpenFold3 145k\non 66 fragment\nstructures, 8 GPUs"),
             ("Serve", "Ray Serve", "Pose and affinity\nfor a new molecule,\n3D viewer")]
    for i in range(n):
        fig, bar = new_frame(5, caption=False)
        bar.text(0.045, 0.94, "What you build here", fontsize=18, fontweight="bold", color=INK, va="top")
        bar.text(0.045, 0.855, "A reproducible pipeline, from public crystal data to a model you can query live.",
                 fontsize=12, color=INK2, va="top")
        lit = min(len(boxes), int(i / 4) + 1)
        bw, gap, x = 0.165, 0.0262, 0.045
        for k, (title, tech, body) in enumerate(boxes):
            on = k < lit
            bx = x + k * (bw + gap)
            bar.add_patch(FancyBboxPatch((bx, 0.30), bw, 0.42, boxstyle="round,pad=0,rounding_size=0.012",
                                         fc=BG, ec=INK if on else RULE, lw=1.1 if on else 0.8))
            bar.text(bx + 0.015, 0.67, f"{k + 1}", fontsize=9, color=MUTED, va="top")
            bar.text(bx + 0.015, 0.605, title, fontsize=15, fontweight="bold", color=INK if on else MUTED, va="top")
            bar.text(bx + 0.015, 0.525, tech, fontsize=10.5, color=MODEL_COLORS["boltz2"] if on else MUTED, va="top",
                     fontweight="bold")
            bar.text(bx + 0.015, 0.455, body, fontsize=9.8, color=INK2 if on else MUTED, va="top", linespacing=1.35)
            if k < len(boxes) - 1:
                bar.annotate("", xy=(bx + bw + gap - 0.003, 0.51), xytext=(bx + bw + 0.003, 0.51),
                             arrowprops=dict(arrowstyle="-|>", color=INK2 if k + 1 < lit else RULE, lw=0.9,
                                             mutation_scale=8))
        if lit == len(boxes):
            bar.text(0.045, 0.235, "Benchmark: 1 job per target on 8 to 12 L4 GPUs, about 80% GPU utilization.\n"
                     "Fine-tuning: 1,000 optimizer steps on 8 GPUs, about 2 hours.",
                     fontsize=11, color=INK2, va="top", linespacing=1.5)
        add(fig_to_image(fig, DPI), 120)
    durs[-1] = 5000


if __name__ == "__main__":
    scene_question()
    scene_crystal()
    scene_predict()
    scene_score()
    scene_scale()
    scene_workflow()
    size = save_gif(frames, durs, OUT / "workflow.gif", colors=128)
    print(f"{len(frames)} frames, {sum(durs) / 1000:.1f} s, {size / 1e6:.1f} MB")
