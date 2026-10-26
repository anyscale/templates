"""
Stage 3: Ray Serve app with a leaderboard and a 3D overlay viewer.

    serve run serve_viewer:app

Then open the Serve URL from the Anyscale Workspace "Ports" tab (port 8000).
Reads aiddd_pipeline.py output (one row per crystal complex). Pick a model at
the top, click a complex to see the predicted pose (green) over the X-ray
ligand (orange). Uses 3Dmol.js from a CDN in the browser.

Live prediction: paste a SMILES at the top. Two model deployments, each on half an L4:
  OF3Pool     pose, from OpenFold3. Ray Serve model multiplexing: one deployment serves the
              base model and every program specialist, chosen per request by model id, and
              keeps the most recently used checkpoints loaded in memory
  Boltz2Live  binding affinity (OpenFold3 does not predict affinity)
The model registry is OF3Pool's user_config; the Viewer's user_config "route" names the model
id that answers this program (EV-A71 2A), and a /predict request can name another one. Both
change with `anyscale service deploy --in-place` (jobs/demo_*.yaml): no restart.
If the molecule has a crystal structure, the page also draws the crystal ligand and reports
the live pose's RMSD to it. Every answer says where it came from: the pose cache, a model
already in memory, or a model loaded for this request. Poses are cached under
predictions/live/ and predictions/live_of3ft_<version>/.

pip install "ray[serve]" fastapi
"""

import asyncio
import os
import resource
from pathlib import Path

import pandas as pd
import ray
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from ray import serve
from ray.serve.handle import DeploymentHandle

from aiddd_pipeline import (
    LIVE_MODELS,
    OF3_LEGACY_PY,
    pose_in_crystal_frame,
    relocate_paths,
    score_pose,
)
from of3_live import LiveModel, cached_answer, node_zone, predict_one

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
RESULTS = Path(os.environ.get("AIDDD_RESULTS", SHARED / "results" / "ev71-2a"))
STRUCTURES = SHARED / "raw" / "OpenBind_EV-A71_2A" / "structures"
MODELS = {
    "boltz2": "Boltz-2",
    "openbind0": "OpenBind-0",
    "of3ft": "OF3 fine-tuned",
    "of3ray": "OF3 Ray Train fine-tune",
}

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
 #hist table{margin-top:8px;font-size:14px} #hist tr:hover{background:none;cursor:default}
 #hist .new td{background:#eef7f0} #hist .ver{font-family:ui-monospace,monospace;font-size:12px;color:#556}
 #lstat{font-size:14px;line-height:1.5}
 #step{font-size:16px;font-weight:600;color:#1f3a68;background:#eef3fb;border-radius:6px;padding:6px 10px;margin:0 0 8px}
 #step:empty{display:none} #step.off{color:#8a4b00;background:#fdf3e7}
 #pool{font-size:14px;color:#334;margin-top:8px;line-height:1.5} #pool code{background:#eef1f6;padding:0 4px;border-radius:3px;font-size:14px}
</style></head><body>
<div id="list">
<form id="live" onsubmit="predict(event)"><h2>Live prediction</h2>
 <div id="step"></div>
 <input id="smi" placeholder="SMILES, e.g. CNC(=O)c1nn(C)c2ccccc12" style="width:100%;box-sizing:border-box;padding:5px">
 <button style="margin:6px 0">Predict</button>
 <label id="freshbox" style="display:none;font-size:12px;color:#444" title="Run the model even if this pose is cached: about 25 s with the model in memory, up to 100 s if it must load first">
  <input type="checkbox" id="fresh"> run live, skip the cached pose</label>
 <span id="lstat"></span><div id="lread"></div><div id="hist"></div><div id="pool"></div></form>
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
 <dt>Live prediction</dt><dd>The <b>pose</b> comes from the OpenFold3 model named in the result line: the general
  base model, or a specialist fine-tuned on this protein's fragment screen. The result line also says where the
  answer came from: the <b>pose cache</b> (computed earlier, returned at once), a model <b>already in memory</b>,
  or a model <b>loaded for this request</b>. The <b>affinity</b> comes from Boltz-2, because OpenFold3 does not
  predict affinity. If the molecule has a crystal structure, the crystal ligand is drawn in orange and the result
  line gives the ligand RMSD.
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
  const fresh = document.getElementById("fresh").checked;
  st.textContent = fresh ? "predicting live on GPU (about 25 s; longer if the model must load)..."
                          : "predicting (cached poses return at once; a new molecule takes 1 to 2 minutes)...";
  document.getElementById("lread").textContent = "";
  const r = await fetch(u("/predict"), {method:"POST", headers:{"Content-Type":"application/json"},
                                     body: JSON.stringify({smiles: smi, fresh})});
  const j = await r.json();
  if(!r.ok || j.status === "failed" || j.status === "skipped"){ st.textContent = "failed: " + (j.error || r.status); return; }
  const m = j.metrics, f = v => (typeof v === "number" ? v.toFixed(2) : "n/a");
  const aff = m.affinity_pred_value, ic50 = typeof aff === "number" ? Math.pow(10, aff) : null;
  const iptm = j.pose_iptm, key = j.answer_key;
  const conf = iptm >= 0.8 ? "confident" : (iptm >= 0.5 ? "moderately confident" : "not confident");
  document.getElementById("lread").innerHTML = (ic50 === null ? "" :
    `Predicted IC<sub>50</sub> ≈ <b>${ic50 >= 1 ? ic50.toFixed(ic50 >= 10 ? 0 : 1) + " µM" : (ic50 * 1000).toFixed(0) + " nM"}</b>
     (pIC<sub>50</sub> ${(6 - aff).toFixed(1)}; ${aff > 1 ? "weak" : (aff > -1 ? "moderate" : "potent")} binding). `) +
    `${j.pose_model} is ${conf} about this pose. Boltz-2 estimates a ${Math.round(100 * m.affinity_probability_binary)}% chance that the molecule binds. ` +
    (!key ? `No crystal structure exists for this molecule, so there is no RMSD to check.` :
     key.lig_rmsd === null ? `Crystal ${key.complex} exists, but this pose could not be scored against it.` :
     `This molecule has a crystal structure (${key.complex}, orange): the predicted ligand is <b>${key.lig_rmsd.toFixed(1)} &Aring;</b>
      from it, ${key.lig_rmsd < 2 ? "a correct pose" : "a wrong pose"} (correct is under 2 &Aring;).`);
  st.innerHTML = `pose: ${j.pose_model} (${j.models.pose.version}), ipTM <b>${f(iptm)}</b>` +
    (key && key.lig_rmsd !== null ? ` &middot; RMSD <b>${f(key.lig_rmsd)}</b> &Aring;` : "") +
    ` &middot; affinity <b>${f(m.affinity_pred_value)}</b> (log10 IC50, &micro;M) &middot; P(binder) <b>${f(m.affinity_probability_binary)}</b>` +
    `<br>${source(j)}` +
    (j.zone ? `<br>GPU replicas: OpenFold3 in <b>${j.zone}</b>, Boltz-2 in <b>${j.affinity_zone || "?"}</b>` : "");
  record(smi, j);
  pool();
  document.getElementById("cur").textContent = `live (${j.pose_model}): ${smi}`;
  viewer.clear(); viewer.addModel(j.cif, j.format || "cif");
  viewer.setStyle({model:0},{cartoon:{color:"#cfd8dc",opacity:0.6}});
  viewer.setStyle({model:0,chain:"B"},{stick:{color:"#2e9e4f",radius:0.25}});
  if (key){ const g = await fetch(u("/structure/gt/"+key.complex));
    if (g.ok){ viewer.addModel(await g.text(), "pdb");
      viewer.setStyle({model:1},{});
      viewer.setStyle({model:1,resn:"LIG"},{stick:{color:"#f28c28",radius:0.25}}); } }
  viewer.zoomTo({chain:"B"}); viewer.render();
}
// Every live answer for the molecule on screen, oldest first. After an in-place update the
// same SMILES is answered by the new model version: its row is highlighted under the old one.
// Kept in the page only: reload it to start the demo clean.
const answers = [];
// Where a pose came from: the pose cache, a model the OpenFold3 replica already held, or one
// Ray Serve loaded for this request (model multiplexing: load() runs once per model id).
const secs = v => v < 10 ? v.toFixed(1) : v.toFixed(0);
function source(j){
  const s = secs(j.seconds);
  if (j.source === "cache") return `<b>pose from cache</b> (computed earlier) &middot; ${s} s` +
    (j.load_seconds ? `, ${secs(j.load_seconds)} s of it loading the model` : "");
  if (j.source === "memory") return `<b>computed live</b>, model already in memory &middot; ${s} s`;
  return `<b>computed live</b>, model loaded for this request (${j.load_seconds.toFixed(0)} s of ${s} s)`;
}
function record(smi, j){
  const key = j.answer_key, last = answers.filter(h => h.smi === smi).pop();
  const version = j.models.pose.version;
  const row = {smi, version, name: j.pose_model, iptm: j.pose_iptm, aff: j.metrics.affinity_pred_value,
               rmsd: key && key.lig_rmsd !== null ? key.lig_rmsd : null,
               how: j.source === "cache" ? "cache" : j.source === "memory" ? "live" : "live + load", s: j.seconds};
  if (last && last.version === version) Object.assign(last, row); else answers.push(row);
  const rows = answers.filter(h => h.smi === smi), f = v => (typeof v === "number" ? v.toFixed(2) : "n/a");
  document.getElementById("hist").innerHTML = rows.length < 2 ? "" :
    `<table><tr><th>Answered by</th><th>RMSD (&Aring;)</th><th>ipTM</th><th>Affinity</th><th>How</th></tr>` +
    rows.map((h, i) => `<tr class="${i === rows.length - 1 ? "new" : ""}"><td>${h.name}<br><span class="ver">version ${h.version}</span></td>` +
      `<td class="${h.rmsd !== null && h.rmsd < 2 ? "good" : ""}">${f(h.rmsd)}</td><td>${f(h.iptm)}</td><td>${f(h.aff)}</td>` +
      `<td>${h.how}<br><span class="ver">${secs(h.s)} s</span></td></tr>`).join("") +
    `</table>`;
}
// The OpenFold3 pool: which model ids its replica holds in memory now (GET /models).
async function pool(){
  try {
    const p = await (await fetch(u("/models"))).json();
    document.getElementById("pool").innerHTML = `This program's requests go to <code>${p.route}</code>. ` +
      `In memory on the OpenFold3 replica (up to ${p.max_resident}): ` +
      (p.resident.length ? p.resident.map(i => `<code>${i}</code>`).join(" ") : "none yet") + `.` +
      (p.zone ? ` OpenFold3 replica in <code>${p.zone}</code>.` : "");
    document.getElementById("freshbox").style.display = p.allow_fresh ? "inline" : "none";
    const step = document.getElementById("step");
    step.textContent = p.step || "Not on a demo step: apply jobs/demo_1_base.yaml";
    step.className = p.step ? "" : "off";
  } catch(e) {}
}
pool();
const first = document.querySelector("tr[data-cid]"); if(first) show(first.dataset.cid);
</script></body></html>"""


def canonical(smiles: str):
    """RDKit canonical SMILES, so one molecule written two ways still finds its crystal."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def live_cache(pred, version: str) -> Path:
    """One cache folder per model version, so a new model never returns an old model's
    cached poses. "base" keeps the original folder."""
    out = pred.t.pred_root / (
        pred.key if version == "base" else f"{pred.key}_{version}"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


# A Serve config (or `anyscale service deploy --in-place`) that changes only user_config
# calls reconfigure() on the running replicas, with no restart.
#
# GPU memory is not the limit (about 5 GB for Boltz-2 and 3 GB for OpenFold3 of an L4's 24 GB); host
# memory is. A live Boltz-2 prediction takes a 32 GB L4 node to about 18 GB, and OF3Pool holds about 15 GB
# with 3 models loaded: together on one node, Ray's memory monitor stops one of them. So each GPU
# deployment also reserves its host memory ("memory" in ray_actor_options). Ray offers actors about 70%
# of a node's free memory (the object store gets the rest), about 22 GB here, so it never places two of
# them on one node. The reservation is for placement only: it does not cap what a replica uses.
GPU_REPLICA_MEMORY = 15e9

# Ray 2.59 runs a sync method on the replica's event loop, so a minute-long prediction there would
# hold up every other request. These deployments answer cached poses straight from disk, and run a
# live prediction in a thread, one at a time per replica.
@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.5,
        "memory": GPU_REPLICA_MEMORY,
        "accelerator_type": "L4",
    },
    # 1 replica: cached affinities need no second one
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_ongoing_requests": 1,
    },
    max_ongoing_requests=32,
    user_config={"version": "base", "checkpoint": None},
)
class Boltz2Live:
    """Boltz-2 affinity (and its own pose). Same predictor class as the batch pipeline."""

    def __init__(self):
        from aiddd_pipeline import Boltz2Predictor

        class LivePredictor(Boltz2Predictor):
            key = "live"  # cache under predictions/live/, separate from the benchmark

        self.pred = LivePredictor()
        self.gpu = asyncio.Lock()
        self.zone = node_zone()

    def reconfigure(self, cfg: dict):
        """checkpoint: a Boltz-2 structure checkpoint (`boltz predict --checkpoint`); None is
        the released boltz2_conf.ckpt. Affinity always uses the released affinity checkpoint.
        """
        ckpt = cfg.get("checkpoint")
        self.pred.extra_args = ["--checkpoint", ckpt] if ckpt else []
        self.pred.out = live_cache(self.pred, cfg["version"])
        self.model = {
            "name": cfg.get("name", MODELS["boltz2"]),
            "version": cfg["version"],
            "checkpoint": ckpt,
        }

    async def __call__(self, smiles: str) -> dict:
        out = await asyncio.to_thread(cached_answer, self.pred.out, smiles)
        if out is None:
            async with self.gpu:
                out = await asyncio.to_thread(predict_one, self.pred, smiles)
        return {**out, "model": self.model, "zone": self.zone}


# How many model ids one OF3Pool replica keeps loaded. Each loaded model adds about 3 GB to the
# replica's host memory (measured on the Service: 8.5, 11.5 and 14.6 GB with 1, 2 and 3 loaded; the
# weights move to the GPU per request). 3 keeps every demo model loaded, so a room that mixes
# ids never waits on a reload. A further id evicts the least recently used one: Ray Serve calls
# its __del__, which frees it. Notebook 03's 6.1 deployment has the same settings.
MAX_RESIDENT = 3


@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.5,
        "memory": GPU_REPLICA_MEMORY,
        "accelerator_type": "L4",
        # OF3-p2 checkpoints only load in OpenFold3 0.4.x
        "runtime_env": {"py_executable": OF3_LEGACY_PY},
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_ongoing_requests": 1,
    },
    max_ongoing_requests=32,
    user_config={"models": LIVE_MODELS},
)
class OF3Pool:
    """OpenFold3 poses from many checkpoints in one deployment: Ray Serve model multiplexing.

    user_config {"models": {model_id: {"version", "name", "checkpoint"}}} is the registry. An
    in-place update that adds an id makes it servable at once; its weights load on the first
    request for it. Requests name the id with handle.options(multiplexed_model_id=...) or the
    serve_multiplexed_model_id HTTP header, and Ray Serve routes each to a replica that already
    holds it when one exists."""

    def __init__(self):
        self.models, self.loaded = {}, {}
        self.gpu = asyncio.Lock()  # OpenFold3 runs one prediction at a time per replica
        self.zone = node_zone()

    async def reconfigure(self, cfg: dict):
        """A new registry, in effect at once. This does not wait for the GPU lock: while a replica
        reconfigures, Ray Serve sends it no requests, so a wait behind a live prediction or a load
        would stop every request to the deployment. Weights whose id now names other weights are
        replaced by that id's next request, in __call__; an id that left the registry keeps its
        weights until Ray Serve evicts it.
        """
        self.models = cfg["models"]

    @serve.multiplexed(max_num_models_per_replica=MAX_RESIDENT)
    async def load(self, model_id: str) -> LiveModel:
        """Runs on the first request for model_id on this replica. Builds OpenFold3 and reads the
        checkpoint once; the returned model stays loaded until evicted."""
        async with self.gpu:
            model = await asyncio.to_thread(LiveModel, self.models[model_id])
        self.loaded[model_id] = model
        return model

    async def __call__(self, smiles: str, fresh: bool = False) -> dict:
        """fresh: ignore the cached pose, so the model really runs."""
        model_id = serve.get_multiplexed_model_id()
        if model_id not in self.models:
            raise KeyError(
                f"unknown model id {model_id!r}; registered: {sorted(self.models)}"
            )
        spec = self.models[model_id]
        cached = lambda: asyncio.to_thread(cached_answer, self.cache_dir(spec), smiles)
        if (
            not fresh and (out := await cached()) is not None
        ):  # no model, no GPU, no waiting
            return {
                **out,
                "model": dict(spec),
                "model_id": model_id,
                "resident": self.resident(),
                "zone": self.zone,
            }
        model = await self.load(model_id)
        async with self.gpu:
            # Requests that waited here while another computed the same pose take its answer
            if not fresh and (out := await cached()) is not None:
                return {
                    **out,
                    "model": dict(spec),
                    "model_id": model_id,
                    "resident": self.resident(),
                    "zone": self.zone,
                }
            # The registry names other weights for this id now, or the weights were evicted:
            # load this id's current weights
            current = self.models[model_id]
            if not model.resident or model.spec != current:
                model.free()
                await asyncio.to_thread(model.load, current)
            out = await asyncio.to_thread(model, smiles, fresh)
        return {
            **out,
            "model_id": model_id,
            "resident": self.resident(),
            "zone": self.zone,
        }

    @staticmethod
    def cache_dir(spec: dict) -> Path:
        """A model version's pose cache folder (the one LiveModel writes): the spec's pose_cache, or
        predictions/live_of3ft_<version>."""
        from aiddd_pipeline import TARGETS

        return Path(
            spec.get("pose_cache")
            or TARGETS["ev71-2a"].pred_root / f"live_of3ft_{spec['version']}"
        )

    async def clear_cache(self, model_id: str, smiles: str) -> dict:
        """Delete one molecule's cached pose for one model id: its next request runs the model live."""
        import hashlib

        if model_id not in self.models:
            raise KeyError(
                f"unknown model id {model_id!r}; registered: {sorted(self.models)}"
            )
        lig_id = "smi_" + hashlib.sha1(smiles.encode()).hexdigest()[:12]
        cache = self.cache_dir(self.models[model_id])
        removed = []
        for f in (
            cache / f"{lig_id}.json",
            cache / f"{lig_id}.cif",
        ):  # .json first: _cached needs both
            if f.exists():
                f.unlink()
                removed.append(f.name)
        return {
            "model_id": model_id,
            "version": self.models[model_id]["version"],
            "removed": removed,
        }

    def resident(self) -> list:
        """Model ids this replica holds in memory now."""
        return [i for i, m in self.loaded.items() if m.resident]

    async def status(self) -> dict:
        rss = (
            int(open("/proc/self/statm").read().split()[1])
            * resource.getpagesize()
            / 1e9
        )
        # what Ray reserved for this replica: its GPU share and host memory
        assigned = ray.get_runtime_context().get_assigned_resources()
        return {
            "registry": {
                i: {k: v for k, v in m.items() if k != "checkpoint"}
                for i, m in self.models.items()
            },
            "resident": self.resident(),
            "max_resident": MAX_RESIDENT,
            "replica_memory_gb": round(rss, 1),
            "replica_gpus": assigned.get("GPU", 0),
            "replica_memory_reserved_gb": round(assigned.get("memory", 0) / 1e9, 1),
            "zone": self.zone,
        }


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    # 2 from the start: a room arrives faster than autoscaling reacts
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 4,
        "target_ongoing_requests": 16,
    },
    max_ongoing_requests=100,  # requests wait here on GPU calls; Ray Serve's default is 5
    user_config={"route": "ev71-openbind", "allow_fresh": False},
)
@serve.ingress(api)
class Viewer:
    def __init__(self, live: DeploymentHandle, pool: DeploymentHandle):
        self.live = live  # Boltz-2: affinity
        self.pool = pool  # OpenFold3: poses, multiplexed by model id
        self.scored = {}  # (pose file, mtime, crystal) -> RMSD and aligned pose
        # File by file: a column that is all null in one file and numeric in another
        # will not merge as one Parquet dataset.
        self.df = relocate_paths(
            pd.concat(
                [pd.read_parquet(f) for f in sorted(RESULTS.glob("*.parquet"))],
                ignore_index=True,
            )
        )
        self.models = [k for k in MODELS if f"{k}_lig_rmsd" in self.df]
        # Canonical SMILES -> first crystal of that compound: the answer key for live predictions
        self.known = {}
        for r in (
            self.df.dropna(subset=["smiles"]).sort_values("complex_name").itertuples()
        ):
            if (c := canonical(r.smiles)) is not None:
                self.known.setdefault(c, r)

    def reconfigure(self, cfg: dict):
        """route: the model id that answers this program's requests. Repointing it is how a
        new version goes live. A /predict request can name another id.
        allow_fresh: honor "fresh" (run the model even when the pose is cached). Off for a
        shared room: 40 fresh requests on one OpenFold3 replica would queue for half an hour.
        """
        self.route = cfg["route"]
        self.allow_fresh = cfg.get("allow_fresh", False)
        self.step = cfg.get(
            "step", ""
        )  # the demo step that set this config (jobs/demo_*.yaml)

    def answer_key(self, row, cif_path: str) -> dict:
        """For a molecule with a crystal structure: the live pose's ligand RMSD to that crystal,
        and the pose moved onto it, so the page can draw the two ligands together. Scoring takes
        about 0.25 s of CPU, so each pose file is scored once per replica (keyed by its path and
        modification time: a fresh prediction that overwrites it is scored again)."""
        key = (cif_path, Path(cif_path).stat().st_mtime_ns, row.complex_name)
        if key not in self.scored:
            if len(self.scored) > 2000:
                self.scored.clear()
            self.scored[key] = self._score(row, cif_path)
        return self.scored[key]

    def _score(self, row, cif_path: str) -> dict:
        ref_pdb = Path(row.ref_pdb)
        ref_sdf = ref_pdb.with_name(
            ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf")
        )
        try:
            rmsd, _ = score_pose(cif_path, ref_pdb, ref_sdf, {})
            return {
                "answer_key": {"complex": row.complex_name, "lig_rmsd": float(rmsd)},
                "cif": pose_in_crystal_frame(cif_path, ref_pdb),
                "format": "pdb",
            }
        except Exception as e:
            return {
                "answer_key": {
                    "complex": row.complex_name,
                    "lig_rmsd": None,
                    "error": str(e),
                }
            }

    @api.get("/", response_class=HTMLResponse)
    def index(self, model: str = ""):
        key = model if model in self.models else self.models[0]
        df = self.df.dropna(subset=[f"{key}_cif"]).sort_values(
            f"{key}_lig_rmsd", na_position="last"
        )
        rows = []
        for _, r in df.iterrows():
            rmsd = r[f"{key}_lig_rmsd"]
            cls = "good" if pd.notna(rmsd) and rmsd < 2 else ""
            pkd = "" if pd.isna(r.experimental_pKD) else f"{r.experimental_pKD:.1f}"
            rows.append(
                f'<tr data-cid="{r.complex_name}" onclick="show(\'{r.complex_name}\')">'
                f"<td>{r.complex_name.split('-', 1)[1]}</td><td class='{cls}'>{'' if pd.isna(rmsd) else f'{rmsd:.2f}'}</td>"
                f"<td>{'fragment' if r.fragment_screen else 'follow-on'}</td><td>{pkd}</td></tr>"
            )
        ok = df[f"{key}_lig_rmsd"].dropna()
        stats = (
            f"median ligand RMSD {ok.median():.2f} &Aring; &middot; <b>{(ok < 2).mean():.0%} of poses correct</b> (under 2 &Aring;)"
            if len(ok)
            else "not scored yet"
        )
        tabs = "".join(
            f'<a class="tab{" on" if k == key else ""}" href="/?model={k}">{MODELS[k]}</a>'
            for k in self.models
        )
        return (
            PAGE.replace("__MODEL__", MODELS[key])
            .replace("__KEY__", key)
            .replace("__N__", str(len(df)))
            .replace("__TABS__", tabs)
            .replace("__STATS__", stats)
            .replace("__ROWS__", "".join(rows))
        )

    @api.post("/predict")
    async def predict(self, body: dict):
        smiles = str(body.get("smiles", "")).strip()
        if not smiles or len(smiles) > 300:
            return JSONResponse(
                {"status": "failed", "error": "need a SMILES under 300 chars"}, 400
            )
        model_id = body.get("model") or self.route
        fresh = (
            bool(body.get("fresh")) and self.allow_fresh
        )  # run the model even if the pose is cached
        # The pose and affinity models run at the same time, each on its own replica.
        # multiplexed_model_id sends the pose request to a replica that holds that model.
        try:
            aff, pose = await asyncio.gather(
                self.live.remote(smiles),
                self.pool.options(multiplexed_model_id=model_id).remote(smiles, fresh),
            )
        except Exception as e:
            cause = (
                getattr(e, "cause", None) or e
            )  # the replica's own exception, not Ray's wrapper
            code = 404 if isinstance(cause, KeyError) else 500
            return JSONResponse(
                {"status": "failed", "error": f"{type(cause).__name__}: {cause}"[:600]},
                code,
            )
        iptm = pose["metrics"].get("ligand_iptm", pose["metrics"].get("iptm"))
        out = {
            **pose,
            "pose_model": pose["model"]["name"],
            "pose_iptm": iptm,
            "metrics": aff["metrics"],
            "affinity_status": aff["status"],
            "seconds": max(pose["seconds"], aff["seconds"]),
            "models": {"pose": pose["model"], "affinity": aff["model"]},
        }
        out["affinity_zone"] = aff.get("zone")
        c = await asyncio.to_thread(canonical, smiles)
        if (row := self.known.get(c) if c else None) is not None and pose.get(
            "cif_path"
        ):
            # RDKit scoring and alignment: in a thread, so the replica keeps serving others
            out.update(await asyncio.to_thread(self.answer_key, row, pose["cif_path"]))
        return JSONResponse(out, 200 if out["status"] in ("ok", "cached") else 422)

    @api.post("/cache/clear")
    async def cache_clear(self, body: dict):
        """{"compound": "x7259a" or "smiles": "...", "model": "base" (default: the route)}. Deletes that pose
        from the cache so the next request computes it live. Affinity (Boltz-2) stays cached.
        """
        smiles = str(body.get("smiles") or "").strip()
        if not smiles and body.get("compound"):
            rows = self.df[
                self.df.complex_name.str.endswith("-" + str(body["compound"]))
            ].dropna(subset=["smiles"])
            if rows.empty:
                return JSONResponse(
                    {"error": f"no compound {body['compound']!r} in the results"}, 404
                )
            smiles = rows.iloc[0].smiles
        if not smiles:
            return JSONResponse({"error": "need a compound or a smiles"}, 400)
        try:
            out = await self.pool.clear_cache.remote(
                body.get("model") or self.route, smiles
            )
        except Exception as e:
            cause = getattr(e, "cause", None) or e
            return JSONResponse(
                {"error": f"{type(cause).__name__}: {cause}"[:400]},
                404 if isinstance(cause, KeyError) else 500,
            )
        return {**out, "smiles": smiles}

    @api.get("/models")
    async def models_status(self):
        """The registry, the route, and which model ids the OF3Pool replica holds in memory."""
        return {
            "route": self.route,
            "allow_fresh": self.allow_fresh,
            "step": self.step,
            **(await self.pool.status.remote()),
        }

    @api.get("/structure/pred/{key}/{cid}", response_class=PlainTextResponse)
    def pred(self, key: str, cid: str):
        row = self.df[self.df.complex_name == cid]
        if key not in self.models or row.empty or not row.iloc[0][f"{key}_cif"]:
            return PlainTextResponse("not found", status_code=404)
        cif = row.iloc[0][f"{key}_cif"]
        try:  # move the prediction onto its crystal, so the two ligands can be compared
            return PlainTextResponse(
                pose_in_crystal_frame(cif, row.iloc[0].ref_pdb),
                headers={"x-format": "pdb"},
            )
        except Exception:
            return PlainTextResponse(Path(cif).read_text(), headers={"x-format": "cif"})

    @api.get("/structure/gt/{cid}", response_class=PlainTextResponse)
    def gt(self, cid: str):
        row = self.df[self.df.complex_name == cid]
        if (
            "ref_pdb" in self.df
            and not row.empty
            and Path(row.iloc[0].ref_pdb).exists()
        ):
            return Path(row.iloc[0].ref_pdb).read_text()
        hits = list(STRUCTURES.glob(f"*/{cid}/{cid}_complex_ref.pdb"))
        if hits:
            return hits[0].read_text()
        return PlainTextResponse("no ground truth", status_code=404)


def build_app(accelerator: str = "L4"):
    """The Viewer composed with the model deployments, on the given GPU type. Each GPU deployment
    asks for half a GPU and reserves its host memory, so on the Service's 2 single-GPU nodes
    OF3Pool and Boltz-2 each have a node to themselves. Default registry:
    aiddd_pipeline.LIVE_MODELS; default route: ev71-openbind."""
    return Viewer.bind(
        Boltz2Live.options(
            ray_actor_options={
                "num_gpus": 0.5,
                "memory": GPU_REPLICA_MEMORY,
                "accelerator_type": accelerator,
            }
        ).bind(),
        OF3Pool.options(
            ray_actor_options={
                "num_gpus": 0.5,
                "memory": GPU_REPLICA_MEMORY,
                "accelerator_type": accelerator,
                "runtime_env": {"py_executable": OF3_LEGACY_PY},
            }
        ).bind(),
    )


app = build_app()
