"""
Display helpers for the template notebooks: an inline 3D viewer and result tables.

The Ray code lives in the notebook itself. This module only draws things, so the
notebook cells stay about Ray and the science.
"""

import html
import json
import os
import uuid
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

from aiddd_pipeline import SHARED, STAGES, TARGETS, pose_in_crystal_frame, relocate_paths

USER = os.environ.get("ANYSCALE_USERNAME") or os.environ.get("USER") or "user"
# Everything you write goes here. AIDDD_MINE keeps it off SHARED when SHARED is
# read-only, as it is when the image ships the data.
MINE = Path(os.environ.get("AIDDD_MINE") or SHARED / "runs" / USER)

MODEL_NAMES = {"boltz2": "Boltz-2", "openbind0": "OpenBind-0", "of3p2": "OpenFold3 145k (general)",
               "of3ray": "Fine-tuned with Ray Train", "of3ft": "OpenBind fine-tune (released)"}
COLORS = {"crystal": "#22272E", "boltz2": "#2563A6", "openbind0": "#D9922E", "of3p2": "#7A8796",
          "of3ray": "#0F8B7A", "of3ft": "#C8553D"}   # same palette as docs/figures/common.py
MODEL_ORDER = ["boltz2", "openbind0", "of3p2", "of3ray", "of3ft"]
# Results of the 145k base and of our fine-tunes are scored into their own folders
# (results_base-of3p2/, results_<run>/) and joined onto the benchmark by complex.
BASE_RESULTS = SHARED / "results_base-of3p2"
OURS_RUN = os.environ.get("AIDDD_OURS_RUN")   # e.g. ft-p2-evfrag-1000s-8g-1023-0140; default: newest ft-p2-* run
TARGET_KEYS = ["ev71-2a", "sars2-mpro", "mers-mpro"]
warnings.filterwarnings("ignore", category=FutureWarning)
_3DMOL = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"



# ---------------------------------------------------------------------------
# Static picture shown in every viewer box until 3Dmol.js draws over it. Saved outputs
# of an untrusted notebook, and browsers that block scripts in notebook outputs, never
# run the viewer script, so they show this picture instead of an empty box.
# ---------------------------------------------------------------------------
def _atoms(model):
    """Heavy atoms of one viewer model as (xyz, chain, resname, atom name), plus SDF bonds."""
    import numpy as np
    lines = model["text"].splitlines()
    if model["fmt"] == "sdf":
        n_atoms, n_bonds = int(lines[3][:3]), int(lines[3][3:6])
        rows = [l.split() for l in lines[4:4 + n_atoms]]
        keep = [i for i, r in enumerate(rows) if r[3] != "H"]
        xyz = np.array([[float(v) for v in rows[i][:3]] for i in keep])
        index = {old: new for new, old in enumerate(keep)}
        bonds = [(index[a - 1], index[b - 1]) for a, b in
                 ((int(l[:3]), int(l[3:6])) for l in lines[4 + n_atoms:4 + n_atoms + n_bonds])
                 if a - 1 in index and b - 1 in index]
        return xyz, ["L"] * len(xyz), ["LIG"] * len(xyz), [""] * len(xyz), bonds
    recs = [l for l in lines if l.startswith(("ATOM", "HETATM")) and l[76:78].strip() != "H"]
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in recs]).reshape(-1, 3)
    return xyz, [l[21] for l in recs], [l[17:20].strip() for l in recs], [l[12:16].strip() for l in recs], None


def _snapshot(models, height):
    """PNG data URI: protein backbone trace (grey) and each ligand as sticks in its model's colour."""
    import base64
    import io

    import numpy as np
    from matplotlib.figure import Figure
    trace, ligands = None, []
    for m in models:
        xyz, chain, resn, name, bonds = _atoms(m)
        for sel, style in m["styles"]:
            if "cartoon" in style and trace is None:
                trace = xyz[[n == "CA" for n in name]]
            if "stick" in style:
                pick = np.array([sel.get("chain") in (None, c) for c in chain], bool)
                if pick.any():
                    sub = xyz[pick]
                    if bonds is None:   # PDB ligand: bond atoms closer than 1.9 A
                        d = np.linalg.norm(sub[:, None] - sub[None], axis=-1)
                        sub_bonds = list(zip(*np.nonzero(np.triu((d > 0.4) & (d < 1.9)))))
                    else:
                        sub_bonds = bonds
                    ligands.append((sub, sub_bonds, style["stick"].get("color", "#2b2b2b")))
    if not ligands:
        return ""
    allig = np.vstack([l[0] for l in ligands])
    center = allig.mean(0)
    # orient so the ligands spread across the image plane (principal axes of their atoms)
    _, _, vt = np.linalg.svd(allig - center)
    rot = vt.T
    proj = lambda pts: (pts - center) @ rot
    w_in, h_in = 8.2, height / 100
    fig = Figure(figsize=(w_in, h_in), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_aspect("equal")
    lig2d = proj(allig)
    half = max(np.abs(lig2d[:, 0]).max() * h_in / w_in, np.abs(lig2d[:, 1]).max(), 4.0) + 5.0
    ax.set_xlim(-half * w_in / h_in, half * w_in / h_in); ax.set_ylim(-half, half)
    if trace is not None and len(trace) > 1:
        t = proj(trace)
        near = np.linalg.norm(trace - center, axis=1) < 2.2 * half
        for i in range(len(t) - 1):
            if near[i] and near[i + 1] and np.linalg.norm(trace[i + 1] - trace[i]) < 4.2:
                depth = 1 / (1 + np.exp(t[i, 2] / 4))   # nearer residues are darker
                ax.plot(t[i:i + 2, 0], t[i:i + 2, 1], color="#bdb8b0", lw=5, alpha=0.25 + 0.5 * depth,
                        solid_capstyle="round", zorder=1)
    for k, (pts, bonds, color) in enumerate(ligands):
        q = proj(pts)
        for a, b in bonds:
            ax.plot(q[[a, b], 0], q[[a, b], 1], color=color, lw=3.2, solid_capstyle="round", zorder=3 + k)
        ax.scatter(q[:, 0], q[:, 1], s=9, color=color, zorder=3 + k)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _placeholder(models, height):
    try:
        img = _snapshot(models, height)
    except Exception:   # the picture is a convenience; never let it break the cell
        img = ""
    # only styles that survive JupyterLab's sanitizer for untrusted output (no inset, no object-fit)
    pic = (f'<img src="{img}" alt="structure" style="position:absolute;top:0;left:0;width:100%;height:auto">'
           if img else "")
    return pic + ('<div style="position:absolute;right:10px;bottom:6px;font:11px system-ui,sans-serif;'
                  'color:#8a857c">Static view. Run the cell for the interactive 3D view.</div>')


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------
def _viewer_js(div, payload, zoom, zoom_out):
    """Script that draws one viewer. JupyterLab runs an output's script before the output is
    attached to the page, so it waits until both the viewer box and 3Dmol.js are available."""
    return f"""
(function() {{
  function draw(el) {{
    el.innerHTML = "";
    var v = $3Dmol.createViewer(el, {{backgroundColor: "white"}});
    var models = {payload};
    models.forEach(function(m, i) {{
      v.addModel(m.text, m.fmt);
      v.setStyle({{model: i}}, {{}});
      m.styles.forEach(function(s) {{ s[0].model = i; v.addStyle(s[0], s[1]); }});
    }});
    var zoom = {zoom};
    if (zoom === "ligands")   // the reference ligand (model 1) and every predicted ligand (chain B)
      zoom = {{predicate: function(a) {{ return a.model === 1 || (a.model > 1 && a.chain === "B"); }}}};
    v.zoomTo(zoom); v.zoom({zoom_out}); v.render();
  }}
  if (!window.$3Dmol && !document.getElementById("aiddd-3dmol")) {{
    var s = document.createElement("script"); s.id = "aiddd-3dmol"; s.src = "{_3DMOL}";
    s.onerror = function() {{ window.aiddd3dmolFailed = true; }};
    document.head.appendChild(s);
  }}
  var tries = 0;
  (function attempt() {{
    var el = document.getElementById("{div}");
    if (el && window.$3Dmol) {{ draw(el); return; }}
    if (el && window.aiddd3dmolFailed) {{
      el.lastChild.textContent = "Could not download 3Dmol.js from cdnjs.cloudflare.com. Check the internet access, then run this cell again.";
      return;
    }}
    if (++tries < 600) setTimeout(attempt, 50);
  }})();
}})();"""


def _viewer(models, legend="", height=420, zoom=None, zoom_out=0.85):
    """models: [{"text", "fmt", "styles": [(selection, style), ...]}]. Loads 3Dmol.js once."""
    div = "v" + uuid.uuid4().hex[:10]
    js = _viewer_js(div, json.dumps(models), json.dumps(zoom or {}), zoom_out)
    display(HTML(f'<div style="font:13px system-ui,sans-serif;margin:4px 0">{legend}</div>'
                 f'<div id="{div}" style="width:100%;max-width:820px;height:{height}px;position:relative;'
                 f'border:1px solid #e5e5e5;border-radius:6px">{_placeholder(models, height)}</div><script>{js}</script>'))


def _chip(color, text):
    return (f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;width:10px;'
            f'height:10px;border-radius:2px;background:{color};margin-right:5px"></span>{html.escape(text)}</span>')


def _stick(color):
    return {"stick": {"color": color, "radius": 0.22}}


def show_complex(row, models=("boltz2", "openbind0"), height=420):
    """One crystal complex: the crystal ligand (dark) and each model's predicted ligand,
    moved onto the crystal the same way the scorer does it. Grey ribbon: the crystal protein."""
    ref_pdb = Path(row["ref_pdb"])
    ref_sdf = ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf"))
    shown = [{"text": ref_pdb.read_text(), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.55}}),
                         ({"byres": True, "within": {"distance": 5, "sel": {"resn": "LIG"}}, "not": {"resn": "LIG"}},
                          {"line": {"color": "#9a958c"}})]},
             {"text": ref_sdf.read_text(), "fmt": "sdf", "styles": [({}, _stick(COLORS["crystal"]))]}]
    chips = [_chip(COLORS["crystal"], "crystal structure")]
    for k in models:
        cif = row.get(f"{k}_cif")
        if not isinstance(cif, str) or not Path(cif).exists():
            continue
        try:
            pdb = pose_in_crystal_frame(cif, ref_pdb)
        except Exception as e:
            chips.append(f"<i>{MODEL_NAMES.get(k, k)}: could not superpose ({e})</i>")
            continue
        shown.append({"text": pdb, "fmt": "pdb", "styles": [({"chain": "B"}, _stick(COLORS.get(k, "#2e9e4f")))]})
        rmsd = row.get(f"{k}_lig_rmsd")
        verdict = "" if pd.isna(rmsd) else f"  {rmsd:.1f} Å {'✓' if rmsd < 2 else '✗'}"
        chips.append(_chip(COLORS.get(k, "#2e9e4f"), MODEL_NAMES.get(k, k) + verdict))
    title = f"<b>{row['complex_name']}</b> · {row.get('stage', '')}"
    if pd.notna(row.get("experimental_pKD")):
        title += f" · measured affinity {row['experimental_pKD']:.1f}"
    # frame the crystal ligand and every predicted ligand, so a pose far from the pocket stays in view
    _viewer(shown, legend=title + "<br>" + "".join(chips), height=height, zoom="ligands")


def show_live_overlay(row, poses, height=420):
    """Several live poses of one molecule over its crystal, one color each, like show_complex.
    poses: [(label, pose_pdb, color, rmsd or None)], each pose already moved onto the crystal
    (the Service's /predict answer: "cif" with "format": "pdb")."""
    ref_pdb = Path(row["ref_pdb"])
    ref_sdf = ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf"))
    shown = [{"text": ref_pdb.read_text(), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.55}}),
                         ({"byres": True, "within": {"distance": 5, "sel": {"resn": "LIG"}}, "not": {"resn": "LIG"}},
                          {"line": {"color": "#9a958c"}})]},
             {"text": ref_sdf.read_text(), "fmt": "sdf", "styles": [({}, _stick(COLORS["crystal"]))]}]
    chips = [_chip(COLORS["crystal"], "crystal structure")]
    for label, pdb, color, rmsd in poses:
        shown.append({"text": pdb, "fmt": "pdb", "styles": [({"chain": "B"}, _stick(color))]})
        chips.append(_chip(color, label + ("" if rmsd is None else f"  {rmsd:.1f} Å {'✓' if rmsd < 2 else '✗'}")))
    _viewer(shown, legend=f"<b>{row['complex_name']}</b><br>" + "".join(chips), height=height, zoom="ligands")


def show_aligned(pose_pdb, parent, label="new molecule (live)", height=420):
    """Like show_analog, for a pose already moved onto the crystal (PDB text), such as the
    `cif` field the Service's /predict returns with "format": "pdb"."""
    ref_pdb = Path(parent["ref_pdb"])
    ref_sdf = ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf"))
    _viewer([{"text": ref_pdb.read_text(), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.55}})]},
             {"text": ref_sdf.read_text(), "fmt": "sdf", "styles": [({}, _stick(COLORS["crystal"]))]},
             {"text": pose_pdb, "fmt": "pdb",
              "styles": [({"chain": "B"}, _stick("#2e9e4f"))]}],
            legend=_chip(COLORS["crystal"], f"parent {parent['complex_name']} (crystal)") + _chip("#2e9e4f", label),
            height=height, zoom="ligands")


def show_analog(cif_path, parent, label="new molecule (Boltz-2, live)", height=420):
    """A live prediction for a new molecule, moved onto the crystal of the compound it was
    made from: does the model keep the parent's binding mode?"""
    ref_pdb = Path(parent["ref_pdb"])
    ref_sdf = ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf"))
    _viewer([{"text": ref_pdb.read_text(), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.55}})]},
             {"text": ref_sdf.read_text(), "fmt": "sdf", "styles": [({}, _stick(COLORS["crystal"]))]},
             {"text": pose_in_crystal_frame(str(cif_path), ref_pdb), "fmt": "pdb",
              "styles": [({"chain": "B"}, _stick("#2e9e4f"))]}],
            legend=_chip(COLORS["crystal"], f"parent {parent['complex_name']} (crystal)") + _chip("#2e9e4f", label),
            height=height, zoom="ligands")


def _cif_to_pdb(cif_text):
    """mmCIF text to PDB text (3Dmol draws ribbons more reliably from PDB).

    The Service answers with PDB text in its `cif` field (`"format": "pdb"`), so accept
    either: an mmCIF file must open with a `data_` directive, and anything else is already
    PDB and goes through untouched.
    """
    if not cif_text.lstrip().startswith("data_"):
        return cif_text
    import io
    import tempfile
    from Bio.PDB import PDBIO
    from aiddd_pipeline import _load_cif
    with tempfile.NamedTemporaryFile("w", suffix=".cif") as f:
        f.write(cif_text)
        f.flush()
        model = _load_cif(f.name)
    for res in model.get_residues():
        if len(res.resname) > 3:
            res.resname = "LIG"
    out, buf = PDBIO(), io.StringIO()
    out.set_structure(model)
    out.save(buf)
    return buf.getvalue()


def show_prediction(cif_text, label="", height=380):
    """A live prediction on its own (no crystal to compare with): protein ribbon, ligand sticks."""
    _viewer([{"text": _cif_to_pdb(cif_text), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.7}}),
                         ({"chain": "B"}, _stick(COLORS["boltz2"]))]}],
            legend=_chip(COLORS["boltz2"], label or "Boltz-2 prediction"), height=height,
            zoom={"chain": "B"}, zoom_out=0.35)   # centred on the ligand, pocket in view


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _parquet(path):
    files = sorted(Path(path).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def ours_run():
    """The Ray Train fine-tune shown as "ours": AIDDD_OURS_RUN, or the newest benchmarked ft-p2-* run."""
    if OURS_RUN:
        return OURS_RUN
    runs = sorted((p for p in SHARED.glob("results_ft-p2-*") if not any(w in p.name for w in ("smoke", "test"))),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1].name[len("results_"):] if runs else None


def load_results(target):
    """Scored binding events from the full benchmark, on the curated set (valid crystal
    reference, no suspected crystallographic artefact), one row per crystal complex.
    Adds the 145k base (of3p2_*) and our fine-tune (of3ray_*) where they have been scored."""
    df = relocate_paths(_parquet(SHARED / "results" / target))
    for path, key in ((BASE_RESULTS / target, "of3p2"), (SHARED / f"results_{ours_run()}" / target, "of3ray")):
        extra = _parquet(path)
        extra = relocate_paths(extra) if extra is not None else None
        if extra is not None and f"{key}_lig_rmsd" in extra:
            cols = ["complex_name"] + [c for c in extra.columns if c.startswith(f"{key}_")]
            df = df.drop(columns=[c for c in cols[1:] if c in df]).merge(extra[cols], on="complex_name", how="left")
    meta = pd.read_csv(TARGETS[target].path("metadata"))
    keep = meta.pb_valid_ref.astype(bool) & ~meta.suspected_artefact.astype(bool)
    return df[df.complex_name.isin(set(meta.loc[keep, "complex_name"]))].reset_index(drop=True)


def success(rmsd):
    """Share of poses within 2 Å of the crystal, as a percentage."""
    rmsd = rmsd.astype(float).dropna()
    return round(100 * (rmsd < 2).mean(), 1) if len(rmsd) else float("nan")


def _models(dfs, models):
    """The requested models, in display order, that have scores in at least one table."""
    models = models or MODEL_ORDER
    return [m for m in MODEL_ORDER if m in models and any(f"{m}_lig_rmsd" in d for d in dfs)]


def glance(results, models=None):
    """% of poses under 2 Å per model, per target and campaign phase."""
    cols = {}
    for t, df in results.items():
        for phase, mask in (("fragments", df.fragment_screen), ("follow-ons", ~df.fragment_screen)):
            if mask.any():
                cols[f"{TARGETS[t].name}, {phase}"] = {
                    MODEL_NAMES[m]: success(df.loc[mask, f"{m}_lig_rmsd"]) if f"{m}_lig_rmsd" in df else float("nan")
                    for m in _models(results.values(), models)}
    return pd.DataFrame(cols)


def by_stage(df, models=None):
    return pd.DataFrame({st: {MODEL_NAMES[m]: success(df.loc[df.stage == st, f"{m}_lig_rmsd"])
                              for m in _models([df], models)}
                         for st in STAGES if (df.stage == st).any()})


def before_after(results, models=("of3p2", "of3ray", "of3ft")):
    """Late follow-ons, % within 2 Å: the fine-tuning target (EV-A71 2A) next to the
    proteases it was not trained on. A model not yet scored shows as "pending"."""
    rows = {}
    for m in models:
        rows[MODEL_NAMES[m]] = {
            f"{TARGETS[t].name} (late follow-ons)": (success(df.loc[df.stage == "late follow-on", f"{m}_lig_rmsd"])
                                                     if f"{m}_lig_rmsd" in df else "pending")
            for t, df in results.items()}
    return pd.DataFrame(rows).T


def loss_curve(run):
    """Per-step metrics of a finished finetune_ray_train.py run, or None."""
    found = sorted((SHARED / "finetune" / "runs" / run).glob("checkpoint_*/loss_curve.json")) if run else []
    return pd.DataFrame(json.loads(found[-1].read_text())) if found else None


def serve_link(path="/"):
    """Where this Workspace exposes Ray Serve (port 8000) to your browser."""
    domain = os.environ.get("ANYSCALE_SESSION_DOMAIN")
    # The Workspace proxy serves the Ray Dashboard at the domain root and other ports via ?port=
    url = f"https://{domain}{path}?port=8000" if domain else f"http://localhost:8000{path}"
    display(HTML(f'<p style="font-size:15px">Open the viewer: <a href="{url}" target="_blank" rel="noopener noreferrer"><b>{url}</b></a>'
                 f'<br><span style="color:#777">If the link reports an invalid referer, copy it into a new '
                 f'browser tab instead of clicking it.</span></p>'))
    return url
