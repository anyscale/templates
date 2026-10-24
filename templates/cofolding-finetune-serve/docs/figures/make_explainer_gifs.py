import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from PIL import Image
import io
from pathlib import Path

for _f in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams["font.family"] = "Source Sans 3"

# Same palette as common.py
INK, MUTED, SOFT, RULE = "#1F2933", "#3E4C59", "#7B8794", "#E4E7EB"
SURF, EDGE = "#F3F5F7", "#CDD3DA"
TEAL, CORAL, BLUE, VIOLET, WHITE = "#0F8B7A", "#C8553D", "#2563A6", "#6C63D8", "#FFFFFF"
OUT = str(Path(__file__).resolve().parents[1] / "images")
W, H = 1440, 810
XU, YU = 160.0, 90.0

STEP, HOLD, FADE, NFADE = 230, 6500, 80, 14   # frame durations in ms


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t*t*(3-2*t)


def canvas():
    fig = plt.figure(figsize=(W/100, H/100), dpi=88)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XU); ax.set_ylim(0, YU); ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    return fig, ax


def finish(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=WHITE)
    plt.close(fig); buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def chrome(ax, title, stages, active, footer):
    ax.text(10, 81, title, fontsize=30, color=INK, va="center", weight=500)
    ax.text(150, 81, f"{active+1} / {len(stages)}", fontsize=17, color=SOFT,
            va="center", ha="right", weight=400)
    x = 10.0
    for i, s in enumerate(stages):
        on = i == active
        done = i < active
        w = 3.4 + 1.42*len(s)
        ax.add_patch(FancyBboxPatch((x, 71.5), w, 5.0,
                     boxstyle="round,pad=0.5,rounding_size=1.4",
                     fc=BLUE if on else WHITE, ec="none" if on else RULE,
                     lw=1.2, zorder=3))
        ax.text(x+w/2, 74.0, s, fontsize=16.5, ha="center", va="center",
                color=WHITE if on else (MUTED if done else SOFT),
                zorder=4, weight=500 if on else 400)
        x += w + 2.6
    ax.plot([10, 150], [67.5, 67.5], color=RULE, lw=1.2)
    ax.plot([10, 150], [12.5, 12.5], color=RULE, lw=1.2)
    ax.text(10, 7.6, "IN THE WORKSHOP", fontsize=13, color=BLUE, weight=500)
    ax.text(38, 7.6, footer, fontsize=16.5, color=MUTED, weight=400)


def mol(ax, P, edges, color, alpha=1.0, r=1.85, lw=3.4, z=5, grow=None):
    for a, b in edges:
        if grow is None or max(a, b) < grow - 0.3:
            ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=color,
                    lw=lw, alpha=alpha, solid_capstyle="round", zorder=z)
    for j, p in enumerate(P):
        if grow is None:
            ax.add_patch(Circle(p, r, fc=color, ec=WHITE, lw=1.0,
                                alpha=alpha, zorder=z+1))
        elif j < grow:
            g = min(grow-j, 1.0)
            ax.add_patch(Circle(p, r*g, fc=color, ec=WHITE, lw=1.0,
                                alpha=alpha, zorder=z+1))


def assemble(scenes, path):
    imgs, durs = [], []
    for k, (n, fn) in enumerate(scenes):
        for i in range(n):
            imgs.append(fn(i/(n-1) if n > 1 else 1.0)); durs.append(STEP)
        durs[-1] = HOLD
        if k+1 < len(scenes):
            end, nxt = imgs[-1].astype(np.float32), scenes[k+1][1](0.0).astype(np.float32)
            for f in range(1, NFADE+1):
                a = ease(f/(NFADE+1))
                imgs.append((end*(1-a)+nxt*a).astype(np.uint8)); durs.append(FADE)
    # 1 shared palette for every frame: stable colours and a small file
    frames = [Image.fromarray(a) for a in imgs]
    w, h = frames[0].size
    picks = frames[::max(1, len(frames) // 16)][:16]
    sample = Image.new("RGB", (w, h * len(picks)))
    for i, f in enumerate(picks):
        sample.paste(f, (0, h * i))
    pal = sample.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(path, len(imgs), f"{sum(durs) / 1000:.1f}s")


# ------------------------------------------------------------------ GIF 1
SEQ = "GFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNF"
SMI = "Cn1cnc(Cn2c(=O)nc(Nc3cc4cn(C)nc4cc3Cl)n(Cc3cc(F)c(F)cc3F)c2=O)n1"
T1 = "How a co-folding model predicts a binding pose"
S1 = ["Inputs", "Evolution", "Prediction", "Score"]
F1 = ["2,147 complexes from 3 antiviral discovery campaigns",
      "1 precomputed alignment per target, reused by every prediction",
      "Boltz-2 and OpenBind-0 fan out across GPUs with Ray Data",
      "Every pose scored against its crystal structure"]

rng = np.random.default_rng(7)
ANG = np.radians(np.array([90, 150, 210, 270, 330, 30.0]))
POSE = np.column_stack([51 + 7.4*np.cos(ANG), 41.5 + 7.4*np.sin(ANG)])
_t = np.radians(14)
_R = np.array([[np.cos(_t), -np.sin(_t)], [np.sin(_t), np.cos(_t)]])
_c = POSE.mean(0)
XTAL = (POSE-_c) @ _R.T + _c + np.array([2.4, -1.6])
START = np.column_stack([rng.uniform(32, 70, 6), rng.uniform(57, 64, 6)])
RING = [(i, (i+1) % 6) for i in range(6)]


def bowl1():
    pts = [(14, 18), (14, 52), (28, 52)]
    for x in np.linspace(28, 74, 56):
        pts.append((x, 34 + ((x-51)/23.0)**2 * 18))
    return pts + [(74, 52), (88, 52), (88, 18)]


def g1_inputs(u):
    fig, ax = canvas(); chrome(ax, T1, S1, 0, F1[0])
    ax.text(10, 60, "PROTEIN SEQUENCE", fontsize=13.5, color=SOFT, weight=500)
    ax.text(10, 54.5, SEQ[:int(ease(min(u/0.38, 1))*len(SEQ))], fontsize=18,
            color=INK, family="Liberation Mono")
    if u > 0.36:
        ax.text(10, 43, "LIGAND SMILES", fontsize=13.5, color=SOFT, weight=500)
        ax.text(10, 37.5, SMI[:int(ease((u-0.36)/0.40)*len(SMI))], fontsize=17,
                color=INK, family="Liberation Mono")
    if u > 0.78:
        ax.text(10, 24, "That is the entire input.", fontsize=26, color=INK,
                alpha=ease((u-0.78)/0.12), weight=500)
        ax.text(10, 18, "No 3D structure is given for either one.",
                fontsize=20, color=MUTED, alpha=ease((u-0.86)/0.14))
    return finish(fig)


def g1_msa(u):
    fig, ax = canvas(); chrome(ax, T1, S1, 1, F1[1])
    ax.text(10, 62, "Hundreds of versions of the same protein, from other species",
            fontsize=19, color=MUTED)
    cols, rows, cw, ch, hot = 46, 9, 3.0, 3.1, [12, 31]
    x0, y0 = 10, 56
    shown = int(ease(min(u/0.44, 1))*rows)
    pal = ["#E3E6EA", "#D6E4F7", "#D8EEE7", "#F8E2D9", "#E4E2F7", "#F7EEDC"]
    for r in range(shown):
        for k in range(cols):
            col = pal[(k*7+r*3) % len(pal)]
            if k in hot:
                col = CORAL if (r % 3) else TEAL
            ax.add_patch(Rectangle((x0+k*cw, y0-r*ch), cw*0.82, ch*0.68,
                                   fc=col, ec="none", zorder=3))
    if u > 0.50:
        a = ease((u-0.50)/0.20)
        for k in hot:
            ax.add_patch(FancyBboxPatch((x0+k*cw-0.7, y0-(rows-1)*ch-0.8),
                         cw*0.82+1.4, (rows-1)*ch+3.0,
                         boxstyle="round,pad=0,rounding_size=0.8",
                         fc="none", ec=INK, lw=2.0, alpha=a, zorder=6))
    if u > 0.66:
        ax.text(10, 24, "Columns that change together are in contact in 3D.",
                fontsize=24, color=INK, alpha=ease((u-0.66)/0.16), weight=500)
        ax.text(10, 18, "Most of the folding accuracy comes from that signal.",
                fontsize=19, color=MUTED, alpha=ease((u-0.74)/0.16))
    return finish(fig)


def g1_predict(u):
    fig, ax = canvas(); chrome(ax, T1, S1, 2, F1[2])
    ax.add_patch(Polygon(bowl1(), closed=True, fc=SURF, ec=EDGE, lw=1.6, zorder=2))
    ax.text(18, 22, "protein", fontsize=17, color=SOFT)
    ax.text(10, 62, "The model places the ligand and shapes the pocket at once",
            fontsize=24, color=INK, weight=500)
    t = ease(max((u-0.08)/0.74, 0))
    P = START + (POSE-START)*t + (1-t)*rng.normal(0, 1.3, START.shape)
    mol(ax, P, RING, TEAL, alpha=0.28+0.72*t, r=1.45+0.4*t)
    if u > 0.76:
        ax.text(102, 44, "The pocket is not rigid.", fontsize=19, color=MUTED,
                alpha=ease((u-0.76)/0.12))
        ax.text(102, 39, "A loop here swings open", fontsize=19, color=MUTED,
                alpha=ease((u-0.82)/0.12))
        ax.text(102, 34, "when a ligand binds.", fontsize=19, color=MUTED,
                alpha=ease((u-0.86)/0.12))
    return finish(fig)


def g1_score(u):
    fig, ax = canvas(); chrome(ax, T1, S1, 3, F1[3])
    ax.add_patch(Polygon(bowl1(), closed=True, fc=SURF, ec=EDGE, lw=1.6, zorder=2))
    mol(ax, POSE, RING, TEAL)
    a = ease(min(u/0.22, 1))
    mol(ax, XTAL, RING, CORAL, alpha=a, z=3)
    if u > 0.28:
        b = ease((u-0.28)/0.20)
        for p, q in zip(POSE, XTAL):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=1.6,
                    ls=(0, (2.5, 2.5)), alpha=b, zorder=8)
    ax.add_patch(Circle((11, 62), 1.5, fc=TEAL, ec="none"))
    ax.text(14.5, 62, "predicted pose", fontsize=18, color=INK, va="center")
    ax.add_patch(Circle((44, 62), 1.5, fc=CORAL, ec="none", alpha=a))
    ax.text(47.5, 62, "crystal structure", fontsize=18, color=INK,
            va="center", alpha=a)
    if u > 0.42:
        v = 1.19*ease((u-0.42)/0.26)
        ax.text(98, 50, f"{v:.2f} Å", fontsize=40, color=INK, weight=500)
        ax.text(98, 44, "ligand RMSD", fontsize=19, color=INK)
        ax.text(98, 39.5, "average gap between matched atoms",
                fontsize=16, color=MUTED)
    if u > 0.72:
        g = ease((u-0.72)/0.20)
        ax.add_patch(FancyBboxPatch((98, 27), 46, 7.0,
                     boxstyle="round,pad=0.5,rounding_size=1.6",
                     fc="#E9F5F0", ec=TEAL, lw=1.6, alpha=g, zorder=9))
        ax.text(121, 30.6, "under 2 Å, counted correct", fontsize=19,
                color="#0A6B51", ha="center", va="center", alpha=g,
                zorder=10, weight=500)
    return finish(fig)


assemble([(26, g1_inputs), (24, g1_msa), (30, g1_predict), (30, g1_score)],
         OUT + "/cofolding_explained.gif")


# ------------------------------------------------------------------ GIF 2
T2 = "From fragment screen to optimized compound"
S2 = ["Fragment screen", "Early follow-on", "Late follow-on"]
_A = np.array([
    [30, 26.3], [34.3, 28.0], [25.8, 28.4], [36.8, 31.8], [22.6, 31.4],
    [31.2, 31.6], [39.4, 35.4], [18.8, 35.0], [27.3, 35.6], [35.4, 39.0],
    [21.6, 39.2], [29.0, 41.0]], float)
ATOMS = np.column_stack([(_A[:, 0]-30)*1.62 + 45, (_A[:, 1]-22)*1.52 + 34])
BONDS = [(0,1),(0,2),(0,5),(1,3),(2,4),(3,6),(4,7),(5,8),(8,9),(8,10),(9,11)]
PH = [dict(n=3, word="weak", pot=0.18, sim=1.00,
           cap="Tiny molecules soaked into crystals",
           sub="A few stick, showing where to build",
           foot="66 fragment structures fine-tune OpenFold3 with Ray Train"),
      dict(n=7, word="moderate", pot=0.55, sim=0.52,
           cap="Chemists grow the fragment outward,",
           sub="into the next sub-pocket along",
           foot="Accuracy is split by campaign stage, never averaged"),
      dict(n=12, word="potent", pot=0.93, sim=0.21,
           cap="Potent compounds fill the pocket",
           sub="with chemistry the screen never saw",
           foot="Follow-on compounds are the held-out test")]


def bowl2():
    pts = [(8, 18), (8, 52), (18, 52)]
    for x in np.linspace(18, 72, 56):
        pts.append((x, 34 + ((x-45)/27.0)**2 * 18))
    return pts + [(72, 52), (86, 52), (86, 18)]


def meter(ax, x, y, w, frac, color, label, value):
    ax.text(x, y+5.2, label, fontsize=16.5, color=SOFT, weight=500)
    ax.add_patch(FancyBboxPatch((x, y), w, 3.4,
                 boxstyle="round,pad=0,rounding_size=1.7",
                 fc="#EEF0F3", ec="none", zorder=3))
    if frac > 0.01:
        ax.add_patch(FancyBboxPatch((x, y), max(w*frac, 3.4), 3.4,
                     boxstyle="round,pad=0,rounding_size=1.7",
                     fc=color, ec="none", zorder=4))
    ax.text(x+w+2.5, y+1.7, value, fontsize=18, color=INK, va="center")


def g2(k, final=False):
    s = PH[k]; prev = PH[k-1] if k else None

    def fn(u):
        fig, ax = canvas()
        chrome(ax, T2, S2, k,
               "Live prediction on a new molecule with Ray Serve" if final else s["foot"])
        ax.add_patch(Polygon(bowl2(), closed=True, fc=SURF, ec=EDGE, lw=1.6, zorder=2))
        ax.text(12, 22, "protein", fontsize=17, color=SOFT)
        t = 1.0 if final else ease(u)
        n0 = prev["n"] if prev else 0
        p0 = prev["pot"] if prev else 0.0
        s0 = prev["sim"] if prev else 1.0
        mol(ax, ATOMS, BONDS, TEAL, grow=n0 + (s["n"]-n0)*t)
        meter(ax, 96, 54, 34, p0+(s["pot"]-p0)*t, CORAL, "BINDING STRENGTH", s["word"])
        meter(ax, 96, 41, 34, s0+(s["sim"]-s0)*t, VIOLET,
              "SIMILARITY TO THE FRAGMENTS", f"{s0+(s['sim']-s0)*t:.2f}")
        if final:
            ax.text(96, 29, "Every compound binds this pocket", fontsize=20,
                    color=INK, alpha=ease(u/0.42), weight=500)
            ax.text(96, 23.5, "The model has to hold up throughout",
                    fontsize=18, color=MUTED, alpha=ease((u-0.3)/0.42))
        else:
            ax.text(96, 29, s["cap"], fontsize=19, color=INK, weight=500)
            ax.text(96, 23.5, s["sub"], fontsize=18, color=MUTED)
        return finish(fig)
    return fn


assemble([(24, g2(0)), (24, g2(1)), (24, g2(2)), (18, g2(2, final=True))],
         OUT + "/campaign_explained.gif")
