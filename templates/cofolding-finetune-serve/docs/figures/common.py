"""Shared style, data loading and 3D drawing helpers for the README figures.

Every figure is drawn from this template's own outputs under $AIDDD_SHARED
(default /mnt/user_storage/aiddd): crystal structures, predicted CIFs, the
scored Parquet results, the per-node GPU logs and the Ray Train loss curve.
"""

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from aiddd_pipeline import (SHARED, TARGETS, _binding_chain_ca, _kabsch,  # noqa: E402
                            _load_cif, _ca_by_resnum, _pred_ligand)

OUT = ROOT / "docs" / "images"

# ---------------------------------------------------------------------------
# Style: journal figure conventions. White ground, Source Sans 3 (OFL, in
# fonts/), thin axes, ticks out, labels in ink (never in the series color).
# ---------------------------------------------------------------------------
for f in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

INK = "#1F2933"
INK2 = "#3E4C59"
MUTED = "#7B8794"
RULE = "#E4E7EB"
BG = "#ffffff"
FONT = "Source Sans 3"
MONO = "Liberation Mono"

# Model identity, fixed order: the 3 benchmark models. Distinct in hue and
# lightness, so every pair also separates under colour-vision deficiency.
MODEL_COLORS = {"boltz2": "#2563A6", "openbind0": "#D9922E", "of3p2": "#7A8796", "of3ft": "#C8553D"}
MODEL_NAMES = {"boltz2": "Boltz-2", "openbind0": "OpenBind-0", "of3p2": "OpenFold3 145k (general)",
               "of3ft": "OpenFold3 145k, fine-tuned on EV-A71 2A fragments"}
MODEL_SHORT = {"boltz2": "Boltz-2", "openbind0": "OpenBind-0", "of3p2": "OF3 145k",
               "of3ft": "OF3 145k, fine-tuned"}
# The fine-tuning story: the general OF3-p2 145k model and our Ray Train fine-tune of it.
# Kept out of MODEL_COLORS, which the benchmark figures iterate over.
FT_COLORS = {"of3p2": "#7A8796", "of3ray": "#0F8B7A"}
FT_NAMES = {"of3p2": "OpenFold3 145k (general)", "of3ray": "Fine-tuned with Ray Train"}
CRYSTAL = "#22272E"
GOOD = "#1E8A5A"
BAD = "#C8553D"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.dpi": 200,
})


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right", color=INK)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _latest_ours():
    """The newest benchmarked Ray Train run (results_ft-p2-*), ignoring smoke tests."""
    runs = sorted((p for p in SHARED.glob("results_ft-p2-*") if not any(w in p.name for w in ("smoke", "test"))),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def load_results(target_key):
    """Scored binding events on the curated set (same filter as scripts/compare_targets.py).

    The benchmark folder holds boltz2, openbind0 and of3ft. The general 145k model and our own
    fine-tunes are scored into their own folders, so they are merged on here by complex."""
    path = SHARED / "results" / target_key
    df = pd.concat([pd.read_parquet(f) for f in sorted(path.glob("*.parquet"))], ignore_index=True)
    ours = _latest_ours()
    for folder, key in ((SHARED / "results_base-of3p2" / target_key, "of3p2"),
                        ((ours / target_key) if ours else None, "of3ray")):
        files = sorted(folder.glob("*.parquet")) if folder and folder.exists() else []
        if not files:
            continue
        extra = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        cols = ["complex_name"] + [c for c in extra.columns if c.startswith(f"{key}_")]
        if len(cols) > 1:
            df = df.drop(columns=[c for c in cols[1:] if c in df]).merge(extra[cols], on="complex_name", how="left")
    meta = pd.read_csv(TARGETS[target_key].path("metadata"))
    keep = meta.pb_valid_ref.astype(bool) & ~meta.suspected_artefact.astype(bool)
    return df[df.complex_name.isin(set(meta.loc[keep, "complex_name"]))]


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
def crystal_dir(target_key, ligand_id, complex_name):
    return TARGETS[target_key].structures / ligand_id / complex_name


def read_sdf(path):
    """Heavy atoms and bonds of an SDF ligand."""
    lines = Path(path).read_text().splitlines()
    n, nb = int(lines[3][:3]), int(lines[3][3:6])
    xyz, el, keep = [], [], {}
    for i, l in enumerate(lines[4:4 + n]):
        e = l[31:34].strip().upper()
        if e != "H":
            keep[i] = len(xyz)
            xyz.append([float(l[0:10]), float(l[10:20]), float(l[20:30])])
            el.append(e)
    bonds = []
    for l in lines[4 + n:4 + n + nb]:
        a, b = int(l[0:3]) - 1, int(l[3:6]) - 1
        if a in keep and b in keep:
            bonds.append((keep[a], keep[b]))
    return np.array(xyz), el, bonds


def bonds_by_distance(xyz):
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    i, j = np.where((d < 1.9) & (d > 0.5))
    return [(a, b) for a, b in zip(i, j) if a < b]


def load_crystal(target_key, ligand_id, complex_name):
    from Bio.PDB import PDBParser
    d = crystal_dir(target_key, ligand_id, complex_name)
    lig, el, bonds = read_sdf(d / f"{complex_name}_ligand_ref.sdf")
    model = PDBParser(QUIET=True).get_structure("x", str(d / f"{complex_name}_complex_ref.pdb"))[0]
    ca = _binding_chain_ca(model, lig)
    return {"ca": ca, "lig": lig, "el": el, "bonds": bonds}


def load_prediction_in_crystal_frame(cif, crystal):
    """Predicted protein CA and ligand, superposed on the crystal protein (as scored)."""
    pred = _load_cif(str(cif))
    pca = _ca_by_resnum(pred, "A")
    common = [i for i in sorted(set(pca) & set(crystal["ca"])) if pca[i][0] == crystal["ca"][i][0]]
    P = np.array([pca[i][1] for i in common])
    Q = np.array([crystal["ca"][i][1] for i in common])
    R, t = _kabsch(P, Q)
    lp, le = _pred_ligand(pred)
    lp = lp @ R.T + t
    return {"lig": lp, "el": le, "bonds": bonds_by_distance(lp)}


def ca_trace(ca_dict):
    keys = sorted(ca_dict)
    return np.array([ca_dict[k][1] for k in keys]), np.array(keys)


# ---------------------------------------------------------------------------
# 3D -> 2D drawing
# ---------------------------------------------------------------------------
def rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def view_matrix(points, toward):
    """Rotation that puts the principal axes of `points` in the screen plane and
    turns the view so `toward` (e.g. the pocket) faces the viewer."""
    c = points - points.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    R = Vt.copy()
    if np.linalg.det(R) < 0:
        R[2] *= -1
    v = R @ (toward - points.mean(0))
    if v[2] < 0:
        R = rot_y(180) @ R
    return R


def smooth_trace(xyz, k=4):
    """Catmull-Rom spline through CA atoms, for a ribbon-like backbone line."""
    p = np.vstack([xyz[0], xyz, xyz[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, k, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(xyz[-1])
    return np.array(out)


def draw_backbone(ax, trace_xyz, R, center, color="#b8b2a7", lw=(1.0, 3.2), alpha=(0.25, 0.95), zorder=1):
    """Depth-cued backbone: near segments darker and thicker."""
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    p = (trace_xyz - center) @ R.T
    seg = np.stack([p[:-1, :2], p[1:, :2]], axis=1)
    z = (p[:-1, 2] + p[1:, 2]) / 2
    zn = (z - z.min()) / (np.ptp(z) + 1e-9)
    order = np.argsort(z)
    base = np.array(mcolors.to_rgb(color))
    far = np.array(mcolors.to_rgb(BG))
    cols = [(*(far + (base - far) * (alpha[0] + (alpha[1] - alpha[0]) * s)), 1) for s in zn[order]]
    widths = lw[0] + (lw[1] - lw[0]) * zn[order]
    lc = LineCollection(seg[order], colors=cols, linewidths=widths, capstyle="round", zorder=zorder)
    ax.add_collection(lc)
    return p


def draw_ligand(ax, xyz, bonds, R, center, color, lw=2.6, zorder=5, alpha=1.0, atoms=True, size=16):
    p = (xyz - center) @ R.T
    for a, b in bonds:
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color=color, lw=lw, solid_capstyle="round",
                zorder=zorder, alpha=alpha)
    if atoms:
        ax.scatter(p[:, 0], p[:, 1], s=size, color=color, zorder=zorder + 0.1, alpha=alpha, linewidths=0)
    return p


def fig_to_image(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


GIF_PACE = 2.4   # every frame duration is multiplied by this; larger plays slower


def save_gif(frames, durations, path, colors=128, extra=()):
    """Quantize every frame to one shared palette (stable colours, small file).
    Named colours (models, crystal, good/bad) are pinned so they never drift."""
    import matplotlib.colors as mcolors
    pinned = list(dict.fromkeys([BG, INK, INK2, MUTED, RULE, CRYSTAL, GOOD, BAD, *MODEL_COLORS.values(),
                                 *FT_COLORS.values(), *extra]))
    w, h = frames[0].size
    k = min(len(frames), 16)
    sample = Image.new("RGB", (w, h * k))
    step = max(1, len(frames) // k)
    for i, f in enumerate(frames[::step][:k]):
        sample.paste(f, (0, h * i))
    free = sample.quantize(colors=colors - len(pinned), method=Image.Quantize.MEDIANCUT)
    rgb = [tuple(int(255 * v) for v in mcolors.to_rgb(c)) for c in pinned]
    rgb += [tuple(free.getpalette()[3 * i:3 * i + 3]) for i in range(colors - len(pinned))]
    pal = Image.new("P", (1, 1))
    pal.putpalette([v for c in rgb for v in c] + [0] * 3 * (256 - len(rgb)))
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=[round(d * GIF_PACE) for d in durations], loop=0, optimize=True, disposal=1)
    return Path(path).stat().st_size
