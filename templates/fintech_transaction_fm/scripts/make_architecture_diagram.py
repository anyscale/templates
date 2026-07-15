"""Regenerates assets/architecture.png — the Part 1 architecture swimlane.

    python scripts/make_architecture_diagram.py

Colors are the validated reference palette from the dataviz method (slots 1 and 6);
text uses ink tokens, identity is carried by lane labels, never color alone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a897f"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, BAND = "#fcfcfb", "#f2f2ef"

fig, ax = plt.subplots(figsize=(14, 6.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2)
ax.axis("off")

for y0, y1 in ((4.0, 5.9), (1.9, 3.8), (0.35, 1.35)):
    ax.add_patch(FancyBboxPatch((1.35, y0), 12.5, y1 - y0, boxstyle="round,pad=0.04",
                                fc=BAND, ec="none", zorder=0))
ax.text(0.12, 4.95, "CPU workers\nRay Data\nautoscale 0 \u2192 N", va="center", ha="left",
        fontsize=9.5, color=BLUE, fontweight="bold", linespacing=1.5)
ax.text(0.12, 2.85, "GPU workers\nRay Train / Serve\nautoscale 0 \u2192 8\u00d7A10G", va="center",
        ha="left", fontsize=9.5, color=ORANGE, fontweight="bold", linespacing=1.5)
ax.text(0.12, 0.85, "Shared\nstorage", va="center", ha="left",
        fontsize=9.5, color=INK2, fontweight="bold", linespacing=1.5)

def box(x, y, w, h, title, sub, nb, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc="white", ec=color, lw=1.6, zorder=3))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=10,
            color=INK, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h / 2 - 0.12, sub, ha="center", va="center", fontsize=8.2,
            color=INK2, zorder=4, linespacing=1.4)
    ax.text(x + w / 2, y + 0.17, nb, ha="center", va="center", fontsize=7.3,
            color=MUTED, zorder=4)

def chip(x, label):
    w = 1.5
    ax.add_patch(FancyBboxPatch((x, 0.55), w, 0.6, boxstyle="round,pad=0.03",
                                fc="white", ec=MUTED, lw=1.2, zorder=3))
    ax.text(x + w / 2, 0.85, label, ha="center", va="center", fontsize=8.4,
            color=INK2, zorder=4)
    return x + w / 2

def arrow(x0, y0, x1, y1, color=INK2, lw=1.2, cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", color=color,
                                 lw=lw, mutation_scale=11, zorder=2,
                                 connectionstyle=cs, shrinkA=2, shrinkB=2))

box(1.7, 4.25, 2.3, 1.35, "Prepare data", "split 24M txns,\ntokenize \u2192 corpus", "notebooks 2\u20133", BLUE)
box(4.45, 2.15, 2.0, 1.35, "Pretrain", "next-token LM\n8 GPUs, 2 h", "notebook 4", ORANGE)
box(7.0, 4.25, 1.9, 1.35, "Tokenize", "per-transaction\ntokens", "notebook 5", BLUE)
box(7.0, 2.15, 1.9, 1.35, "Embed", "model forward\npass only", "notebook 5", ORANGE)
box(9.35, 2.15, 1.9, 1.35, "Train detectors", "3 XGBoost fits +\n2 fine-tunes", "notebooks 6\u20137", ORANGE)
box(11.75, 2.15, 1.9, 1.35, "Serve", "embedding +\nfraud score API", "notebook 8", ORANGE)

c_csv = chip(1.7, "TabFormer csv")
c_corp = chip(4.45, "splits \u00b7 corpus")
c_mod = chip(7.0, "model")
c_emb = chip(9.35, "embeddings")
c_det = chip(11.75, "detectors")

arrow(c_csv, 1.15, c_csv, 4.25)
arrow(2.85, 4.25, c_corp - 0.35, 1.18, cs="arc3,rad=0.14")
arrow(c_corp + 0.2, 1.15, 5.45, 2.15)
arrow(6.1, 2.15, c_mod - 0.35, 1.18, cs="arc3,rad=-0.25")
arrow(c_mod + 0.2, 1.15, 7.75, 2.15)
arrow(8.6, 2.15, c_emb - 0.35, 1.18, cs="arc3,rad=-0.25")
arrow(c_emb + 0.2, 1.15, 10.1, 2.15)
arrow(10.95, 2.15, c_det - 0.35, 1.18, cs="arc3,rad=-0.22")
arrow(c_det + 0.2, 1.15, 12.5, 2.15)

arrow(7.95, 4.25, 7.95, 3.5, color=BLUE, lw=1.8)
ax.text(8.1, 3.85, "streams", fontsize=8.2, color=BLUE, style="italic")

ax.set_title("One elastic cluster \u2014 each stage on the hardware it needs",
             fontsize=12, color=INK, pad=10, loc="left", x=0.02)
plt.tight_layout()
plt.savefig("assets/architecture.png", bbox_inches="tight", facecolor=SURFACE)
print("saved assets/architecture.png")
