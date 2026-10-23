"""
Stage 3: Ray Serve app with a leaderboard and a 3D overlay viewer.

    serve run serve_viewer:app

Then open the Serve URL from the Anyscale Workspace "Ports" tab (port 8000).
Reads aiddd_pipeline.py output (one row per crystal complex). Pick a model at
the top, click a complex to see the predicted pose (green) over the X-ray
ligand (orange). Uses 3Dmol.js from a CDN in the browser.

Live prediction: paste a SMILES at the top. Two model deployments run in parallel,
each on half an L4, so one of each shares a GPU:
  FineTunedLive  pose, from OpenFold3 fine-tuned on the EV-A71 2A fragment screen
  Boltz2Live     binding affinity (OpenFold3 does not predict affinity)
Both reuse the pipeline's predictor classes. Repeat SMILES come back from the caches
under predictions/live/ and predictions/live_of3ft/.

pip install "ray[serve]" fastapi
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from ray import serve
from ray.serve.handle import DeploymentHandle

from aiddd_pipeline import pose_in_crystal_frame

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
RESULTS = Path(os.environ.get("AIDDD_RESULTS", SHARED / "results" / "ev71-2a"))
STRUCTURES = SHARED / "raw" / "OpenBind_EV-A71_2A" / "structures"
MODELS = {"boltz2": "Boltz-2", "openbind0": "OpenBind-0", "of3ft": "OF3 fine-tuned",
          "of3ray": "OF3 Ray Train fine-tune"}

api = FastAPI()

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Co-folding benchmark on Ray</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:0;display:grid;grid-template-columns:380px 1fr;height:100vh}
 #list{overflow:auto;border-right:1px solid #ddd;padding:12px}
 #view{position:relative} #v{width:100%;height:100%}
 table{border-collapse:collapse;width:100%;font-size:13px}
 td,th{padding:4px 6px;border-bottom:1px solid #eee;text-align:left}
 tr:hover{background:#f4f6ff;cursor:pointer} .good{color:#1a7f37;font-weight:600}
 h2{font-size:16px;margin:4px 0 10px} #stats{font-size:13px;color:#444;margin-bottom:10px}
 .tab{display:inline-block;margin:0 6px 8px 0;padding:3px 8px;border:1px solid #ccd;border-radius:12px;font-size:12px;color:#334;text-decoration:none} .tab.on{background:#334;color:#fff}
 #live{padding-bottom:10px;margin-bottom:10px;border-bottom:1px solid #ddd}
 #legend{position:absolute;top:10px;left:10px;background:#fff9;padding:6px 10px;border-radius:6px;font-size:13px}
 details{font-size:13px;line-height:1.45;margin:0 0 12px;padding:8px 10px;background:#f6f7f9;border-radius:6px}
 summary{cursor:pointer;font-weight:600} details dt{font-weight:600;margin-top:8px} details dd{margin:2px 0 0 0;color:#333}
 th[title]{cursor:help;text-decoration:underline dotted} #lread{font-size:12px;color:#444;margin-top:4px;line-height:1.4}
</style></head><body>
<div id="list">
<form id="live" onsubmit="predict(event)"><h2>Live prediction</h2>
 <input id="smi" placeholder="SMILES, e.g. CNC(=O)c1nn(C)c2ccccc12" style="width:100%;box-sizing:border-box;padding:5px">
 <button style="margin:6px 0">Predict</button> <span id="lstat"></span><div id="lread"></div></form>
<details><summary>How to read this page</summary><dl>
 <dt>The question</dt><dd>Each row is an X-ray crystal structure of a compound bound to the EV-A71 2A protease. The model
  received only the protein sequence and the compound's SMILES string; the crystal structure is the answer key.</dd>
 <dt>Model tabs</dt><dd><b>Boltz-2</b> and <b>OpenBind-0</b> are general-purpose co-folding models. <b>OF3 fine-tuned</b> is
  OpenFold3 adapted to this protein using its fragment-screen structures only.</dd>
 <dt>Lig RMSD (Å)</dt><dd>Average distance between each predicted ligand atom and the same atom in the crystal, after the
  two proteins are superposed. <b>Below 2 Å (green) counts as a correct pose</b>; a carbon-carbon bond is about 1.5 Å.
  Values of 2 to 4 Å usually mean the compound is in the pocket but rotated; above 5 Å it is in the wrong place.</dd>
 <dt>Stage</dt><dd><b>Fragment</b>: a small molecule from the initial crystallographic screen. <b>Follow-on</b>: a larger
  compound designed later by medicinal chemists. Follow-ons are the harder and more useful test, because they contain
  chemistry the screen never showed.</dd>
 <dt>pKD</dt><dd>Measured binding affinity, −log<sub>10</sub> of the dissociation constant K<sub>D</sub> in molar.
  <b>Higher means tighter binding</b>, and each unit is a tenfold change: pKD 5 = 10 µM, 6 = 1 µM, 7 = 100 nM, 9 = 1 nM.
  Fragments typically bind at 3 to 5; optimized compounds reach 7 or more. Blank means not measured.</dd>
 <dt>Colours in the 3D view</dt><dd><b style="color:#2e9e4f">Green</b>: the model's predicted pose.
  <b style="color:#f28c28">Orange</b>: the crystal pose. Grey ribbon: the predicted protein. Good predictions overlap.</dd>
 <dt>Live prediction</dt><dd>The <b>pose</b> comes from OF3 fine-tuned, the most accurate model on this protein's
  follow-on compounds. The <b>affinity</b> comes from Boltz-2, because OpenFold3 does not predict affinity.
  <b>Ligand ipTM</b> (0 to 1) is the pose model's confidence in the protein-ligand interface; as a
  rough guide, above 0.8 is confident and below 0.5 is unreliable. <b>Affinity</b> is Boltz-2's predicted
  log<sub>10</sub>(IC<sub>50</sub> in µM), so <b>lower means tighter binding</b>: 2 ≈ 100 µM (weak), 0 = 1 µM,
  −2 ≈ 10 nM (potent). <b>P(binder)</b> is Boltz-2's predicted probability that the molecule binds at all. Confidence
  is not accuracy: check each model's record on the leaderboard before trusting a pose.</dd>
</dl></details>
<h2>__MODEL__: __N__ crystal complexes</h2><div>__TABS__</div><div id="stats">__STATS__</div>
<table><tr><th title="Crystal structure identifier">Complex</th>
<th title="Distance between predicted and crystal ligand (Å). Below 2 Å (green) is a correct pose; lower is better.">Lig RMSD (Å)</th>
<th title="Fragment: initial screen. Follow-on: larger compound designed later; the harder test.">Stage</th>
<th title="Measured affinity, −log10 KD. Higher binds tighter; 6 = 1 µM, 9 = 1 nM.">pKD</th></tr>__ROWS__</table></div>
<div id="view"><div id="v"></div>
<div id="legend"><b>green</b> __MODEL__ &nbsp; <b>orange</b> X-ray &nbsp; <span id="cur"></span></div></div>
<script>
// Behind the Anyscale Workspace proxy the page is opened as ...?port=8000; keep that on every request.
const PORT = new URLSearchParams(location.search).get("port");
function u(path){ return PORT ? path + (path.includes("?") ? "&" : "?") + "port=" + PORT : path; }
document.querySelectorAll("a.tab").forEach(a => a.href = u(a.getAttribute("href")));
let viewer = $3Dmol.createViewer("v",{backgroundColor:"white"});
async function show(cid){
  document.getElementById("cur").textContent = cid;
  viewer.clear();
  const r = await fetch(u("/structure/pred/__KEY__/"+cid));
  viewer.addModel(await r.text(), r.headers.get("x-format") || "cif");
  viewer.setStyle({model:0},{cartoon:{color:"#cfd8dc",opacity:0.6}});
  viewer.setStyle({model:0,chain:"B"},{stick:{color:"#2e9e4f",radius:0.25}});
  const g = await fetch(u("/structure/gt/"+cid));
  if (g.ok){ viewer.addModel(await g.text(), "pdb");
    viewer.setStyle({model:1},{});
    viewer.setStyle({model:1,resn:"LIG"},{stick:{color:"#f28c28",radius:0.25}}); }
  viewer.zoomTo({chain:"B"}); viewer.render();
}
async function predict(ev){
  ev.preventDefault();
  const smi = document.getElementById("smi").value.trim(), st = document.getElementById("lstat");
  if(!smi) return;
  st.textContent = "predicting on GPU (about a minute; cached repeats are instant)...";
  document.getElementById("lread").textContent = "";
  const r = await fetch(u("/predict"), {method:"POST", headers:{"Content-Type":"application/json"},
                                     body: JSON.stringify({smiles: smi})});
  const j = await r.json();
  if(!r.ok || j.status === "failed" || j.status === "skipped"){ st.textContent = "failed: " + (j.error || r.status); return; }
  const m = j.metrics, f = v => (typeof v === "number" ? v.toFixed(2) : "n/a");
  const aff = m.affinity_pred_value, ic50 = typeof aff === "number" ? Math.pow(10, aff) : null;
  const iptm = j.pose_iptm;
  const conf = iptm >= 0.8 ? "confident" : (iptm >= 0.5 ? "moderately confident" : "not confident");
  document.getElementById("lread").innerHTML = (ic50 === null ? "" :
    `Predicted IC<sub>50</sub> ≈ <b>${ic50 >= 1 ? ic50.toFixed(ic50 >= 10 ? 0 : 1) + " µM" : (ic50 * 1000).toFixed(0) + " nM"}</b>
     (pIC<sub>50</sub> ${(6 - aff).toFixed(1)}; ${aff > 1 ? "weak" : (aff > -1 ? "moderate" : "potent")} binding). `) +
    `${j.pose_model} is ${conf} about this pose. Boltz-2 estimates a ${Math.round(100 * m.affinity_probability_binary)}% chance that the molecule binds. ` +
    `No crystal structure exists for a new molecule, so there is no RMSD to check.`;
  st.innerHTML = `pose: ${j.pose_model}, ipTM <b>${f(iptm)}</b> &middot; affinity <b>${f(m.affinity_pred_value)}</b> (log10 IC50, &micro;M) &middot; P(binder) <b>${f(m.affinity_probability_binary)}</b> &middot; ${j.seconds.toFixed(0)} s`;
  document.getElementById("cur").textContent = "live: " + smi;
  viewer.clear(); viewer.addModel(j.cif, "cif");
  viewer.setStyle({model:0},{cartoon:{color:"#cfd8dc",opacity:0.6}});
  viewer.setStyle({model:0,chain:"B"},{stick:{color:"#2e9e4f",radius:0.25}});
  viewer.zoomTo({chain:"B"}); viewer.render();
}
const first = document.querySelector("tr[data-cid]"); if(first) show(first.dataset.cid);
</script></body></html>"""


def predict_one(pred, smiles: str) -> dict:
    """One SMILES through a pipeline predictor, as a JSON-ready dict."""
    from aiddd_pipeline import validate
    lig_id = "smi_" + hashlib.sha1(smiles.encode()).hexdigest()[:12]
    batch = validate(pd.DataFrame({"ligand_id": [lig_id], "smiles": [smiles]}))
    r = pred(batch).iloc[0]
    k = pred.key
    out = {"ligand_id": lig_id, "status": r[f"{k}_status"], "error": r[f"{k}_error"],
           "metrics": json.loads(r[f"{k}_metrics"]), "seconds": float(r[f"{k}_batch_seconds"])}
    if r[f"{k}_cif"]:
        out["cif"] = Path(r[f"{k}_cif"]).read_text()
    return out


# Half an L4 per replica: one Boltz2Live and one FineTunedLive share each GPU (about 5 and
# 3 GB of its 24 GB). Two replicas of each fit the two L4s the Service is given.
@serve.deployment(ray_actor_options={"num_gpus": 0.5, "accelerator_type": "L4"},
                  autoscaling_config={"min_replicas": 1, "max_replicas": 2,
                                      "target_ongoing_requests": 1})
class Boltz2Live:
    """Boltz-2 affinity (and its own pose). Same predictor class as the batch pipeline."""

    def __init__(self):
        from aiddd_pipeline import Boltz2Predictor

        class LivePredictor(Boltz2Predictor):
            key = "live"   # cache under predictions/live/, separate from the benchmark

        self.pred = LivePredictor()

    def __call__(self, smiles: str) -> dict:
        return predict_one(self.pred, smiles)


@serve.deployment(ray_actor_options={"num_gpus": 0.5, "accelerator_type": "L4"},
                  autoscaling_config={"min_replicas": 1, "max_replicas": 2,
                                      "target_ongoing_requests": 1})
class FineTunedLive:
    """Pose from OpenFold3 fine-tuned on the EV-A71 2A fragment screen (the of3ft checkpoint)."""

    def __init__(self):
        from aiddd_pipeline import OF3_FT_CKPT, TARGETS, OF3Predictor
        self.pred = OF3Predictor(key="live_of3ft", ckpt_path=str(OF3_FT_CKPT),
                                 binary="/opt/of3-legacy/bin/run_openfold", target="ev71-2a",
                                 out_dir=str(TARGETS["ev71-2a"].pred_root / "live_of3ft"))

    def __call__(self, smiles: str) -> dict:
        return predict_one(self.pred, smiles)


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class Viewer:
    def __init__(self, live: DeploymentHandle, pose: DeploymentHandle = None):
        self.live = live   # Boltz-2: affinity
        self.pose = pose   # fine-tuned OpenFold3: pose (None: use Boltz-2's pose)
        # File by file: a column that is all null in one file and numeric in another
        # will not merge as one Parquet dataset.
        self.df = pd.concat([pd.read_parquet(f) for f in sorted(RESULTS.glob("*.parquet"))], ignore_index=True)
        self.models = [k for k in MODELS if f"{k}_lig_rmsd" in self.df]

    @api.get("/", response_class=HTMLResponse)
    def index(self, model: str = ""):
        key = model if model in self.models else self.models[0]
        df = self.df.dropna(subset=[f"{key}_cif"]).sort_values(f"{key}_lig_rmsd", na_position="last")
        rows = []
        for _, r in df.iterrows():
            rmsd = r[f"{key}_lig_rmsd"]
            cls = "good" if pd.notna(rmsd) and rmsd < 2 else ""
            pkd = "" if pd.isna(r.experimental_pKD) else f"{r.experimental_pKD:.1f}"
            rows.append(
                f'<tr data-cid="{r.complex_name}" onclick="show(\'{r.complex_name}\')">'
                f"<td>{r.complex_name.split('-', 1)[1]}</td><td class='{cls}'>{'' if pd.isna(rmsd) else f'{rmsd:.2f}'}</td>"
                f"<td>{'fragment' if r.fragment_screen else 'follow-on'}</td><td>{pkd}</td></tr>")
        ok = df[f"{key}_lig_rmsd"].dropna()
        stats = (f"median ligand RMSD {ok.median():.2f} &Aring; &middot; <b>{(ok < 2).mean():.0%} of poses correct</b> (under 2 &Aring;)"
                 if len(ok) else "not scored yet")
        tabs = "".join(f'<a class="tab{" on" if k == key else ""}" href="/?model={k}">{MODELS[k]}</a>'
                       for k in self.models)
        return (PAGE.replace("__MODEL__", MODELS[key]).replace("__KEY__", key).replace("__N__", str(len(df)))
                .replace("__TABS__", tabs).replace("__STATS__", stats).replace("__ROWS__", "".join(rows)))

    @api.post("/predict")
    async def predict(self, body: dict):
        smiles = str(body.get("smiles", "")).strip()
        if not smiles or len(smiles) > 300:
            return JSONResponse({"status": "failed", "error": "need a SMILES under 300 chars"}, 400)
        # Both models run at the same time, each on its own replica
        if self.pose is None:
            aff = await self.live.remote(smiles)
            pose, pose_model = aff, MODELS["boltz2"]
        else:
            aff, pose = await asyncio.gather(self.live.remote(smiles), self.pose.remote(smiles))
            pose_model = MODELS["of3ft"]
        iptm = pose["metrics"].get("ligand_iptm", pose["metrics"].get("iptm"))
        out = {**pose, "pose_model": pose_model, "pose_iptm": iptm,
               "metrics": aff["metrics"], "affinity_status": aff["status"],
               "seconds": max(pose["seconds"], aff["seconds"])}
        return JSONResponse(out, 200 if out["status"] in ("ok", "cached") else 422)

    @api.get("/structure/pred/{key}/{cid}", response_class=PlainTextResponse)
    def pred(self, key: str, cid: str):
        row = self.df[self.df.complex_name == cid]
        if key not in self.models or row.empty or not row.iloc[0][f"{key}_cif"]:
            return PlainTextResponse("not found", status_code=404)
        cif = row.iloc[0][f"{key}_cif"]
        try:   # move the prediction onto its crystal, so the two ligands can be compared
            return PlainTextResponse(pose_in_crystal_frame(cif, row.iloc[0].ref_pdb), headers={"x-format": "pdb"})
        except Exception:
            return PlainTextResponse(Path(cif).read_text(), headers={"x-format": "cif"})

    @api.get("/structure/gt/{cid}", response_class=PlainTextResponse)
    def gt(self, cid: str):
        row = self.df[self.df.complex_name == cid]
        if "ref_pdb" in self.df and not row.empty and Path(row.iloc[0].ref_pdb).exists():
            return Path(row.iloc[0].ref_pdb).read_text()
        hits = list(STRUCTURES.glob(f"*/{cid}/{cid}_complex_ref.pdb"))
        if hits:
            return hits[0].read_text()
        return PlainTextResponse("no ground truth", status_code=404)


def build_app(accelerator: str = "L4"):
    """The Viewer composed with both model deployments, on the given GPU type."""
    gpu = {"num_gpus": 0.5, "accelerator_type": accelerator}
    return Viewer.bind(Boltz2Live.options(ray_actor_options=gpu).bind(),
                       FineTunedLive.options(ray_actor_options=gpu).bind())


app = build_app()
