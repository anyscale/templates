"""Before/after GIFs for the fine-tuning section.

    python docs/figures/make_before_after_gifs.py                      # all 3, current numbers
    python docs/figures/make_before_after_gifs.py --only pose \\
        --after-cif <our pose of x7056a> --after-rmsd 0.6 \\
        --after-source "Checkpoint: ours, Ray Train on 8 GPUs"
    python docs/figures/make_before_after_gifs.py --only accuracy --ours 65.0

docs/images/before_after_pose.gif      x7056a: OF3-p2 145k pose, then the fine-tuned pose
docs/images/before_after_accuracy.gif  same model, new target; what fine-tuning recovers
docs/images/fault_tolerance.gif        spot node reclaimed at step ~700, with and without checkpoints

Accuracy values are measured in this repository's pipeline (curated late follow-ons,
pose within 2 A). Pass --ours once our Ray Train checkpoint has been benchmarked.
"""

import argparse
import textwrap

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import (BG, CRYSTAL, GOOD, BAD, INK, INK2, MUTED, RULE, FT_COLORS, MODEL_COLORS, OUT,
                    ca_trace, draw_backbone, draw_ligand, fig_to_image, load_crystal,
                    load_prediction_in_crystal_frame, plt, rot_x, rot_y, save_gif, smooth_trace, view_matrix)

W, H, DPI = 8.0, 4.5, 120  # 960 x 540 px
BEFORE, AFTER, THEIRS, BOLTZ = FT_COLORS["of3p2"], FT_COLORS["of3ray"], MODEL_COLORS["of3ft"], MODEL_COLORS["boltz2"]


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def canvas(header=None):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    bar = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bar.set_xlim(0, 1), bar.set_ylim(0, 1), bar.axis("off")
    if header:
        bar.text(0.02, 0.955, header, fontsize=9, color=MUTED, va="top")
    return fig, bar


def frag_phrase(a):
    return f"{a.n_fragments} fragment structures" if a.n_fragments else "the fragment-screen structures"


def text_block(bar, title, body, x=0.585, y=0.86, width=36):
    bar.text(x, y, title, fontsize=18, fontweight="bold", color=INK, va="top")
    lines = textwrap.wrap(body, width)
    for k, line in enumerate(lines):
        bar.text(x, y - 0.13 - k * 0.056, line, fontsize=12, color=INK2, va="baseline")
    return y - 0.13 - len(lines) * 0.056


def rmsd_row(bar, y, color, label, value, alpha=1.0):
    ok = value < 2
    bar.add_patch(Rectangle((0.585, y - 0.02), 0.012, 0.04, color=color, lw=0, alpha=alpha))
    bar.text(0.607, y, label, fontsize=12, color=INK, va="center", fontweight="bold", alpha=alpha)
    bar.text(0.80, y, f"{value:.1f} Å", fontsize=12, color=INK, va="center", ha="right", alpha=alpha)
    bar.text(0.82, y, "correct pose" if ok else "wrong pose", fontsize=12, va="center", fontweight="bold",
             color=GOOD if ok else BAD, alpha=alpha)


# ---------------------------------------------------------------------------
# 1. Pose: 1 late follow-on with no close fragment (--ligand)
# ---------------------------------------------------------------------------
def pose_gif(a):
    xtal = load_crystal("ev71-2a", a.ligand, f"A71EV2A-{a.ligand}")
    before = load_prediction_in_crystal_frame(a.before_cif, xtal)
    after = load_prediction_in_crystal_frame(a.after_cif, xtal)
    morph = before["el"] == after["el"]          # same atom order: move each atom to its new place
    ca, _ = ca_trace(xtal["ca"])
    trace = smooth_trace(ca)
    pocket = xtal["lig"].mean(0)
    R0 = rot_x(-12) @ view_matrix(ca, pocket)
    span = 15.0
    header = f"EV-A71 2A protease · late follow-on {a.ligand}{a.pkd} · never seen in training"
    frames, durs = [], []

    def frame(t_rot, draw, title, body, rows):
        fig, bar = canvas(header)
        ax = fig.add_axes([0.02, 0.06, 0.54, 0.86])
        ax.set_xlim(-span, span), ax.set_ylim(-span, span)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        R = rot_y(16 * np.sin(2 * np.pi * t_rot)) @ R0
        draw_backbone(ax, trace, R, pocket, lw=(1.2, 4.5))
        draw(ax, R)
        lens = plt.Circle((0, 0), span * 0.98, transform=ax.transData)
        for art in ax.collections + ax.lines:
            art.set_clip_path(lens)
        ax.add_patch(plt.Circle((0, 0), span * 0.98, fill=False, ec=RULE, lw=0.8))
        y = text_block(bar, title, body) - 0.08
        for k, row in enumerate(rows):
            rmsd_row(bar, y - k * 0.09, *row)
        if a.after_source:
            for k, line in enumerate(textwrap.wrap(a.after_source, 46)):
                bar.text(0.585, 0.09 - k * 0.04, line, fontsize=9.5, color=MUTED, va="bottom")
        frames.append(fig_to_image(fig, DPI))

    def crystal(ax, R, alpha=1.0, faint=False):
        draw_ligand(ax, xtal["lig"], xtal["bonds"], R, pocket, CRYSTAL, alpha=0.35 if faint else alpha,
                    lw=2.4 if faint else 2.8, atoms=not faint)

    # Scene 1: the crystal pose
    n = 18
    for i in range(n):
        frame(i / 60, lambda ax, R, i=i: crystal(ax, R, alpha=ease(i / 8)),
              "The answer key", f"The crystal structure shows where {a.ligand} binds in the pocket.", [])
        durs.append(110)
    durs[-1] = 2600

    # Scene 2: the general model's pose
    n = 24
    for i in range(n):
        t = ease(i / 10)

        def draw(ax, R, t=t):
            crystal(ax, R, faint=True)
            draw_ligand(ax, before["lig"], before["bonds"], R, pocket, BEFORE, alpha=t)
        frame((18 + i) / 60, draw, "Before fine-tuning",
              "A general OpenFold3 model (OF3-p2 145k) places the compound in the wrong orientation.",
              [(BEFORE, "OF3-p2 145k", a.before_rmsd, t)])
        durs.append(110)
    durs[-1] = 3600

    # Scene 3: after fine-tuning, the pose moves onto the crystal
    n = 30
    for i in range(n):
        t = ease(i / 22)

        def draw(ax, R, t=t):
            crystal(ax, R, faint=True)
            if morph:
                xyz = before["lig"] + t * (after["lig"] - before["lig"])
                col = BEFORE if t < 0.5 else AFTER
                draw_ligand(ax, xyz, after["bonds"] if t > 0.5 else before["bonds"], R, pocket, col)
            else:
                draw_ligand(ax, before["lig"], before["bonds"], R, pocket, BEFORE, alpha=1 - t)
                draw_ligand(ax, after["lig"], after["bonds"], R, pocket, AFTER, alpha=t)
        frame((42 + i) / 60, draw, "After fine-tuning",
              f"The same model, fine-tuned on {frag_phrase(a)} of this protein, reproduces the crystal pose.",
              [(BEFORE, "OF3-p2 145k", a.before_rmsd, 1.0)] + ([(AFTER, a.after_label, a.after_rmsd, 1.0)]
                                                              if t > 0.95 else []))
        durs.append(110)
    durs[-1] = 3000

    # Scene 4: hold on the overlay with the crystal at full strength
    n = 20
    for i in range(n):
        def draw(ax, R):
            crystal(ax, R)
            draw_ligand(ax, after["lig"], after["bonds"], R, pocket, AFTER, alpha=0.9)
        frame((72 + i) / 60, draw, "After fine-tuning",
              "Dark: crystal. Teal: prediction. The compound now sits in the right place,\nin the right orientation.",
              [(BEFORE, "OF3-p2 145k", a.before_rmsd, 1.0), (AFTER, a.after_label, a.after_rmsd, 1.0)])
        durs.append(110)
    durs[-1] = 4500
    size = save_gif(frames, durs, OUT / "before_after_pose.gif", extra=(AFTER, BEFORE))
    return len(frames), sum(durs), size


# ---------------------------------------------------------------------------
# 2. Accuracy: same model, new target; what fine-tuning recovers
# ---------------------------------------------------------------------------
def accuracy_gif(a):
    sars = [("Boltz-2", 91.7, BOLTZ), ("OF3-p2 145k", 87.7, BEFORE)]
    ev = [("Boltz-2", 1.0, BOLTZ), ("OF3-p2 145k", 23.9, BEFORE),
          ("Ours\n(Ray Train)", a.ours, AFTER), ("OpenBind\nfine-tune", 78.1, THEIRS)]
    scenes = [("A well-studied target", "General models place 88% to 92% of SARS-CoV-2 Mpro follow-ons "
               "correctly.", 1, 0),
              ("Same model, new target: 88% → 24%", "On EV-A71 2A the general OpenFold3 model drops to 24%, "
               "and Boltz-2 to 1%.", 1, 2),
              ("Fine-tuning on the fragment screen", f"The same model, fine-tuned on {frag_phrase(a)} of the new "
               "protein.", 1, 4)]
    frames, durs = [], []
    for s, (title, body, n_sars, n_ev) in enumerate(scenes):
        prev_ev = scenes[s - 1][3] if s else 0
        n = 26
        for i in range(n):
            t = ease(i / 18)
            fig, bar = canvas()
            bar.text(0.045, 0.94, title, fontsize=18, fontweight="bold", color=INK, va="top")
            bar.text(0.045, 0.855, body, fontsize=12, color=INK2, va="top")
            bar.text(0.045, 0.05, "Late follow-on compounds, share of poses within 2 Å of the crystal structure. "
                     "All models scored in the same pipeline.", fontsize=9, color=MUTED, va="bottom")
            for k, (panel, data, shown, grow_from) in enumerate(
                    [("SARS-CoV-2 Mpro", sars, 2, 0 if s == 0 else 2), ("EV-A71 2A", ev, n_ev, prev_ev)]):
                ax = fig.add_axes([0.07 + k * 0.33, 0.17, 0.22 if k == 0 else 0.5, 0.56])
                ax.set_xlim(-0.6, len(data) - 0.4), ax.set_ylim(0, 105)
                ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"], fontsize=10)
                if k:
                    ax.set_yticks([])
                    ax.spines["left"].set_visible(False)
                ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)
                ax.set_title(panel, fontsize=12.5, loc="left", pad=10)
                ax.set_xticks(range(len(data)), [d[0] for d in data], fontsize=10)
                ax.tick_params(axis="x", length=0)
                for j, (label, val, col) in enumerate(data):
                    if j >= shown:
                        continue
                    g = 1.0 if j < grow_from else t
                    if val is None:                       # our run: not benchmarked yet
                        if g > 0.95:
                            ax.text(j, 8, "in\ntraining", ha="center", va="bottom", fontsize=11, color=col,
                                    fontweight="bold", linespacing=1.1, zorder=4)
                        continue
                    ax.bar(j, val * g, 0.62, color=col, lw=0, zorder=3)
                    if g > 0.95:
                        ax.text(j, val + 2.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=12,
                                fontweight="bold", color=INK)
            frames.append(fig_to_image(fig, DPI))
            durs.append(90)
        durs[-1] = 4200 if s < 2 else 6000
    size = save_gif(frames, durs, OUT / "before_after_accuracy.gif", extra=(AFTER, BEFORE, THEIRS))
    return len(frames), sum(durs), size


# ---------------------------------------------------------------------------
# 3. Fault tolerance: a spot node is reclaimed at step ~700
# ---------------------------------------------------------------------------
def fault_gif(a):
    step_s = 8.0                               # measured on 8 L4 workers
    total, fail, every = 1000, 700, 100
    h = lambda steps: steps * step_s / 3600     # noqa: E731
    lanes = [("Without checkpoints", "restarts from step 0", fail + total, BAD),
             ("With checkpoints every 100 steps", "resumes from step 700", total + 10, GOOD)]
    t_end = h(fail + total)
    frames, durs = [], []
    n = 70
    for i in range(n + 1):
        now = t_end * ease(i / n) * 1.0
        fig, bar = canvas()
        bar.text(0.045, 0.94, "A spot node is reclaimed during training", fontsize=18, fontweight="bold",
                 color=INK, va="top")
        bar.text(0.045, 0.855, "1,000 steps on 8 GPUs, about 8 s each. Ray Train restarts the workers "
                 "in both cases.", fontsize=12, color=INK2, va="top")
        ax = fig.add_axes([0.06, 0.27, 0.9, 0.47])
        ax.set_xlim(0, t_end * 1.02), ax.set_ylim(-0.5, 1.6)
        ax.set_yticks([]), ax.spines["left"].set_visible(False)
        ax.set_xlabel("hours", fontsize=10.5)
        for k, (name, what, steps_done, col) in enumerate(lanes):
            y = 1 - k
            ax.text(0, y + 0.36, name, fontsize=12.5, fontweight="bold", color=INK, va="bottom")
            ax.add_patch(Rectangle((0, y - 0.13), t_end, 0.26, color="#F3F5F7", lw=0, zorder=1))
            # progress: (start hour, end hour, start step); the reclaim happens at step 700
            if k == 0:
                segs = [(0, h(fail), 0), (h(fail), h(fail + total), 0)]
            else:
                segs = [(0, h(fail), 0), (h(fail) + 0.02, h(fail) + 0.02 + h(total - fail), fail)]
            for j, (s0, s1, start) in enumerate(segs):
                if now <= s0:
                    continue
                e = min(now, s1)
                lost = k == 0 and j == 0 and now >= h(fail)     # the 700 steps that must be redone
                ax.add_patch(Rectangle((s0, y - 0.13), e - s0, 0.26, color=BAD if lost else AFTER,
                                       lw=0, zorder=2, alpha=0.55 if lost else 0.9))
                if lost:
                    ax.text(h(fail) / 2, y, "lost", fontsize=11, color=BG, ha="center", va="center",
                            fontweight="bold", zorder=4)
            if k == 1:
                for c in range(every, total, every):
                    tc = h(c) if c <= fail else h(fail) + 0.02 + h(c - fail)
                    if now >= tc:
                        ax.plot([tc, tc], [y - 0.17, y + 0.17], color=INK, lw=1.1, zorder=3)
            if now >= h(fail):
                ax.plot(h(fail), y, marker="X", ms=13, color=BAD, mec=BG, mew=1.5, zorder=5)
            finish = h(fail + total) if k == 0 else h(fail) + 0.02 + h(total - fail)
            if now >= finish - 1e-6:
                ax.text(finish, y - 0.25, f"done at {finish:.1f} h", fontsize=11, color=INK, ha="right", va="top")
        if now >= h(fail):
            ax.text(h(fail), 1.6, "node reclaimed at step 700", fontsize=10.5, color=BAD, ha="center", va="top",
                    fontweight="bold")
            bar.text(0.045, 0.045, "Without: restarts from step 0, about 1.5 h lost.   With: resumes from the "
                     "step-700 checkpoint, at most 100 steps lost.", fontsize=11, color=INK2, va="bottom")
        frames.append(fig_to_image(fig, DPI))
        durs.append(90)
        if i == int(n * h(fail) / t_end):     # pause on the moment of the failure
            durs[-1] = 2200
    durs[-1] = 6000
    size = save_gif(frames, durs, OUT / "fault_tolerance.gif", extra=(AFTER,))
    return len(frames), sum(durs), size


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["pose", "accuracy", "fault"], default=None)
    ap.add_argument("--ligand", default="x7056a", help="EV-A71 2A ligand id for the pose GIF")
    ap.add_argument("--pkd", default=" (pKD 6.1)", help="shown after the ligand id; empty string to omit")
    ap.add_argument("--before-cif", default="/mnt/user_storage/aiddd/predictions_base-of3p2/1022-2240/of3p2/x7056a.cif")
    ap.add_argument("--before-rmsd", type=float, default=6.01)
    ap.add_argument("--after-cif", default="/mnt/user_storage/aiddd/predictions/of3ft/x7056a.cif",
                    help="the fine-tuned pose of x7056a (default: the OpenBind team's released fine-tune)")
    ap.add_argument("--after-rmsd", type=float, default=0.40)
    ap.add_argument("--after-label", default="Fine-tuned")
    ap.add_argument("--after-source", default="Fine-tuned checkpoint: the OpenBind team's release, also started from 145k")
    ap.add_argument("--n-fragments", type=int, default=None,
                    help="training-set size to name on the slides (66 for our runs; default: not named)")
    ap.add_argument("--ours", type=float, default=None,
                    help="our Ray Train fine-tune, %% of EV-A71 2A late follow-ons within 2 A (default: 'training')")
    args = ap.parse_args()
    from common import GIF_PACE
    for name, fn in [("pose", pose_gif), ("accuracy", accuracy_gif), ("fault", fault_gif)]:
        if args.only in (None, name):
            n, ms, size = fn(args)
            print(f"{name}: {n} frames, {ms * GIF_PACE / 1000:.1f} s, {size / 1e6:.2f} MB")
