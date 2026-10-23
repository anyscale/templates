"""
End-to-end OpenBind EV-A71 2A evaluation on Ray Data.

    python aiddd_pipeline.py --limit 8 --max-gpus 2                  # smoke test
    python aiddd_pipeline.py --models boltz2,openbind0,of3ft         # full benchmark

One streaming Ray Data pipeline, five stages, each on the hardware it needs:

    metadata.csv (925 complexes -> 699 unique ligands)
      -> validate   CPU   RDKit-parse SMILES, size, skip already-predicted rows
      -> boltz2     GPU   Boltz-2 pose + affinity, batch per `boltz predict`
      -> openbind0  GPU   OpenFold3 with the OpenBind-0 checkpoint
      -> of3ft      GPU   (optional) OF3 fine-tuned on EV-A71 2A fragments
      -> score      CPU   explode back to complexes, Calpha-align, ligand RMSD
      -> parquet + summary vs. the published OpenBind numbers, split by
         fragment screen vs follow-on, campaign stage, and similarity to fragments

Predictions are made once per unique ligand and scored against every crystal
binding event of that ligand (925 events, 699 ligands: ~25% fewer GPU calls).
Outputs are cached per ligand on shared storage, so a rerun after a failure
only predicts what is missing.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ray

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
BOLTZ_CACHE = SHARED / "boltz_cache"
OF3_CACHE = SHARED / "of3_cache"
OF3_OB_CKPT = OF3_CACHE / "of3-ob-2025-06-30-174k.pt"   # OpenBind-0
OF3_OB_URL = "https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-ob-2025-06-30-174k.pt"
OF3_P2_CKPT = OF3_CACHE / "of3-p2-145k.pt"           # OF3-p2: start of the released fine-tune
OF3_P2_URL = "https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-p2-145k.pt"
OF3_LEGACY = "/opt/of3-legacy/bin/run_openfold"       # OF3-p2 weights only load in OpenFold3 0.4.x
OF3_FT_CKPT = OF3_CACHE / "of3p2-ft-ev2a.ckpt"
OF3_FT_URL = ("https://openfold3-data.s3.amazonaws.com/openfold3-parameters/"
              "openbind/of3p2-ft-ev2a.ckpt")
PRED_ROOT = SHARED / "predictions"


@dataclass(frozen=True)
class Target:
    """Everything target specific. Each release folder has the same layout:
    metadata CSV, one FASTA, one MSA, structures/<ligand>/<complex>/<complex>_{complex,ligand}_ref.*"""
    key: str
    name: str
    release: Path
    metadata: str
    fasta: str
    msa: str
    pred_root: Path
    affinity_label: str = "pKD"
    published: Optional[Path] = None

    @property
    def structures(self) -> Path:
        return self.release / "structures"

    def path(self, name: str) -> Path:
        return self.release / getattr(self, name)


TARGETS = {
    "ev71-2a": Target(
        "ev71-2a", "EV-A71 2A protease", SHARED / "raw" / "OpenBind_EV-A71_2A",
        "EV-A71_2A_metadata.csv", "EV-A71_2A.fasta", "EV-A71_2A_cfdb_msa.a3m",
        pred_root=PRED_ROOT, affinity_label="pKD",
        published=(SHARED / "EV-A71_2A_benchmark" / "structure" / "processed_outputs"
                   / "final_cofolding_pose_data.parquet")),
    # Built by scripts/prepare_mpro.py from Fragalysis structures and ASAP/Polaris potency.
    "sars2-mpro": Target(
        "sars2-mpro", "SARS-CoV-2 Mpro", SHARED / "raw" / "SARS-CoV-2_Mpro",
        "metadata.csv", "protein.fasta", "protein.a3m",
        pred_root=PRED_ROOT / "sars2-mpro", affinity_label="pIC50"),
    "mers-mpro": Target(
        "mers-mpro", "MERS-CoV Mpro", SHARED / "raw" / "MERS-CoV_Mpro",
        "metadata.csv", "protein.fasta", "protein.a3m",
        pred_root=PRED_ROOT / "mers-mpro", affinity_label="pIC50"),
}

# Fixed output columns per model so every Ray Data block has the same schema.
# Anything else the model reports is kept in <key>_metrics as JSON.
BOLTZ_METRICS = ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                 "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
                 "affinity_pred_value", "affinity_probability_binary"]
OF3_METRICS = ["sample_ranking_score", "avg_plddt", "gpde", "ptm", "iptm", "has_clash"]

# model key -> (display name, published method name in the OpenBind benchmark)
MODELS = {
    "boltz2": ("Boltz-2", "boltz-2"),
    "openbind0": ("OpenBind-0 (OF3)", None),
    "of3p2": ("OF3-p2 145k (before fine-tuning)", "of3p2"),
    "of3ft": ("OF3 fine-tuned on EV-A71 2A", "of3p2_ft"),
    "of3ray": ("OF3 fine-tuned with Ray Train", None),   # your own weights from finetune_ray_train.py
}
FINETUNE_RUNS = SHARED / "finetune" / "runs"


def latest_finetune_ckpt() -> Optional[Path]:
    """Most recent weights exported by finetune_ray_train.py."""
    found = sorted(FINETUNE_RUNS.glob("*/checkpoint_*/of3_ev71_ft.pt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def start_gpu_monitor(out_dir: Optional[str], every_s: float = 5.0):
    """Sample this node's GPU utilization in the background, once per node.

    The first predictor on a node takes a lock file; the others return. Samples go to
    <out>/_gpu/<node>.csv as `unix_time,util_percent,mem_mib`, summarized at the end.
    """
    if not out_dir:
        return
    import socket
    import threading
    d = Path(out_dir + "_gpu")
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / f"{socket.gethostname()}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return

    def loop():
        if not shutil.which("nvidia-smi"):
            return
        with open(d / f"{socket.gethostname()}.csv", "a") as f:
            while True:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
                for line in r.stdout.strip().splitlines():
                    f.write(f"{time.time():.0f},{line.replace(' ', '')}\n")
                f.flush()
                time.sleep(every_s)

    threading.Thread(target=loop, daemon=True).start()


def gpu_report(out_dir: str) -> str:
    d = Path(out_dir + "_gpu")
    files = sorted(d.glob("*.csv")) if d.exists() else []
    if not files:
        return "GPU utilization: no samples"
    per_node = []
    for f in files:
        v = pd.read_csv(f, names=["t", "util", "mem"])
        per_node.append(v.util.mean())
    allv = pd.concat(pd.read_csv(f, names=["t", "util", "mem"]) for f in files)
    # The samples stay next to the results (the next run into --out clears them), so
    # utilization can still be read after the job log is gone.
    return (f"GPU utilization: {allv.util.mean():.0f}% mean over {len(files)} GPU nodes "
            f"(per node {', '.join(f'{u:.0f}%' for u in per_node)}; "
            f"{(allv.util >= 50).mean():.0%} of samples at 50% or more)")


def read_first_sequence(path: Path) -> str:
    """First FASTA record only. (A multi-record FASTA must not be concatenated.)"""
    seq, seen_header = [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if seen_header:
                break
            seen_header = True
        elif line:
            seq.append(line)
    return "".join(seq)


# ---------------------------------------------------------------------------
# Stage 1 (CPU): validate ligands before they can cost GPU time.
# ---------------------------------------------------------------------------
def validate(batch: pd.DataFrame) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    ok, heavy = [], []
    for smi in batch["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        ok.append(mol is not None)
        heavy.append(mol.GetNumHeavyAtoms() if mol is not None else 0)
    batch["smiles_ok"] = ok
    batch["heavy_atoms"] = heavy
    return batch


# ---------------------------------------------------------------------------
# Stage 2 (GPU): Boltz-2
# ---------------------------------------------------------------------------
class Boltz2Predictor:
    """One actor per GPU. Each batch = one `boltz predict` call over N ligands."""

    key = "boltz2"

    def __init__(self, target: str = "ev71-2a", diffusion_samples: int = 1, recycling_steps: int = 3,
                 affinity_samples: int = 5, monitor_dir: Optional[str] = None,
                 affinity: bool = True, extra_args: tuple = (), out_dir: Optional[str] = None):
        start_gpu_monitor(monitor_dir)
        self.t = TARGETS[target]
        self.protein = read_first_sequence(self.t.path("fasta"))
        self.samples = diffusion_samples
        self.recycling = recycling_steps
        # The affinity head runs its own diffusion ensemble (5 samples by default), which is
        # over half of Boltz-2's GPU time. Fewer samples are faster; cache them separately.
        self.affinity = affinity
        self.affinity_samples = affinity_samples
        self.extra_args = list(extra_args)   # more `boltz predict` flags (tune_boltz.py)
        suffix = "" if affinity_samples == 5 else f"_aff{affinity_samples}"
        self.out = Path(out_dir) if out_dir else self.t.pred_root / f"{self.key}{suffix}"
        self.out.mkdir(parents=True, exist_ok=True)

    def _yaml(self, smiles: str) -> str:
        return (
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {self.protein}\n      msa: {self.t.path('msa')}\n"
            f"  - ligand:\n      id: B\n      smiles: '{smiles}'\n"
            + ("properties:\n  - affinity:\n      binder: B\n" if self.affinity else "")
        )

    def _cached(self, lig_id):
        cif, meta = self.out / f"{lig_id}.cif", self.out / f"{lig_id}.json"
        if cif.exists() and meta.exists():
            return {"metrics": json.loads(meta.read_text()), "cif": str(cif), "status": "cached"}
        return None

    def _save(self, lig_id, src_cif, metrics):
        cif = self.out / f"{lig_id}.cif"
        shutil.copy(src_cif, cif)
        (self.out / f"{lig_id}.json").write_text(json.dumps(metrics))
        return {"metrics": metrics, "cif": str(cif), "status": "ok"}

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        t0 = time.time()
        rows, todo = {}, []
        for lig_id, smi, ok in zip(batch.ligand_id, batch.smiles, batch.smiles_ok):
            if not ok:
                rows[lig_id] = {"status": "skipped", "error": "invalid SMILES"}
            elif (c := self._cached(lig_id)) is not None:
                rows[lig_id] = c
            else:
                todo.append((lig_id, smi))

        if todo:
            work = Path(tempfile.mkdtemp(prefix="boltz_"))
            (work / "in").mkdir()
            for lig_id, smi in todo:
                (work / "in" / f"{lig_id}.yaml").write_text(self._yaml(smi))
            proc = subprocess.run(
                ["boltz", "predict", str(work / "in"), "--out_dir", str(work / "out"),
                 "--cache", str(BOLTZ_CACHE), "--output_format", "mmcif",
                 "--recycling_steps", str(self.recycling),
                 "--diffusion_samples", str(self.samples), "--num_workers", "2",
                 "--diffusion_samples_affinity", str(self.affinity_samples), *self.extra_args],
                capture_output=True, text=True)
            pred = work / "out" / "boltz_results_in" / "predictions"
            for lig_id, _ in todo:
                d = pred / lig_id
                try:
                    metrics = json.loads((d / f"confidence_{lig_id}_model_0.json").read_text())
                    aff = d / f"affinity_{lig_id}.json"
                    if aff.exists():
                        metrics.update(json.loads(aff.read_text()))
                    metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                    rows[lig_id] = self._save(lig_id, d / f"{lig_id}_model_0.cif", metrics)
                except Exception as e:
                    rows[lig_id] = {"status": "failed", "error": _err(e, proc)}
            shutil.rmtree(work, ignore_errors=True)

        return _attach(batch, rows, self.key, BOLTZ_METRICS, time.time() - t0, len(todo))


# ---------------------------------------------------------------------------
# Stage 3/4 (GPU): OpenFold3 with an OpenBind checkpoint
# ---------------------------------------------------------------------------
class OF3Predictor:
    """OpenFold3 `run_openfold predict` per batch. key/ckpt pick the checkpoint."""

    def __init__(self, key: str, ckpt_path: str, binary: str = "run_openfold",
                 target: str = "ev71-2a", diffusion_samples: int = 1, data_workers: int = 2,
                 monitor_dir: Optional[str] = None, out_dir: Optional[str] = None):
        start_gpu_monitor(monitor_dir)
        self.key = key
        self.t = TARGETS[target]
        self.ckpt_path, self.binary = ckpt_path, binary
        self.samples = diffusion_samples
        self.data_workers = data_workers
        self.protein = read_first_sequence(self.t.path("fasta"))
        self.out = Path(out_dir) if out_dir else self.t.pred_root / key
        self.out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("OPENFOLD_CACHE", str(OF3_CACHE))
        # OpenFold3 only reads MSA files whose name it recognizes (a key of its
        # max_seq_counts config), and silently runs without one otherwise.
        self.msa = self.t.pred_root / "msa_of3" / "cfdb_uniref30.a3m"
        if not self.msa.exists():
            self.msa.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.msa.with_suffix(f".{os.getpid()}.tmp")
            shutil.copy(self.t.path("msa"), tmp)
            tmp.rename(self.msa)

    _cached = Boltz2Predictor._cached
    _save = Boltz2Predictor._save

    def _query(self, smiles):
        return {"chains": [
            {"molecule_type": "protein", "chain_ids": ["A"], "sequence": self.protein,
             "main_msa_file_paths": [str(self.msa)]},
            {"molecule_type": "ligand", "chain_ids": ["B"], "smiles": smiles}],
            "use_paired_msas": False}

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        t0 = time.time()
        rows, todo = {}, []
        for lig_id, smi, ok in zip(batch.ligand_id, batch.smiles, batch.smiles_ok):
            if not ok:
                rows[lig_id] = {"status": "skipped", "error": "invalid SMILES"}
            elif (c := self._cached(lig_id)) is not None:
                rows[lig_id] = c
            else:
                todo.append((lig_id, smi))

        if todo:
            work = Path(tempfile.mkdtemp(prefix=f"{self.key}_"))
            qjson = work / "query.json"
            qjson.write_text(json.dumps(
                {"seeds": [42], "queries": {lig: self._query(smi) for lig, smi in todo}}))
            # OpenFold3 starts 10 data loader processes by default, each importing torch.
            # That oversubscribes an 8 vCPU node and stalls when predictors share a GPU.
            runner = work / "runner.yml"
            runner.write_text(f"data_module_args:\n  num_workers: {self.data_workers}\n")
            cmd = [self.binary, "predict", "--query_json", str(qjson), "--runner_yaml", str(runner),
                   "--output_dir", str(work / "out"), "--use_msa_server", "False",
                   "--use_templates", "False", "--num_diffusion_samples", str(self.samples),
                   "--num_model_seeds", "1", "--inference_ckpt_path", self.ckpt_path]
            # The fine-tuned checkpoint stores pathlib objects, which recent PyTorch
            # refuses to unpickle by default. Both checkpoints are trusted downloads.
            env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            for lig_id, _ in todo:
                try:
                    samples = sorted((work / "out" / lig_id).glob(f"seed_*/{lig_id}_seed_*_sample_*_model.cif"))
                    if not samples:
                        raise FileNotFoundError("no OF3 output structure")
                    best, best_score, best_agg = samples[0], -np.inf, {}
                    for s in samples:  # rank diffusion samples by OF3's own ranking score
                        agg = json.loads(Path(str(s).replace("_model.cif", "_confidences_aggregated.json")).read_text())
                        score = agg.get("sample_ranking_score", agg.get("avg_plddt", 0.0))
                        if score > best_score:
                            best, best_score, best_agg = s, score, agg
                    metrics = {k: float(v) for k, v in best_agg.items() if isinstance(v, (int, float, bool))}
                    rows[lig_id] = self._save(lig_id, best, metrics)
                except Exception as e:
                    rows[lig_id] = {"status": "failed", "error": _err(e, proc)}
            shutil.rmtree(work, ignore_errors=True)

        return _attach(batch, rows, self.key, OF3_METRICS, time.time() - t0, len(todo))


def _err(e, proc):
    tail = proc.stderr[-600:] if proc.returncode else ""
    return f"{type(e).__name__}: {e} | {tail}"[:900]


def _attach(batch, rows, key, metric_cols, seconds, n_predicted):
    """Add one fixed set of <key>_* columns, whatever happened to each ligand."""
    out = batch.copy()
    recs = [rows[l] for l in batch.ligand_id]
    out[f"{key}_status"] = [r.get("status") for r in recs]
    out[f"{key}_error"] = [r.get("error") for r in recs]
    out[f"{key}_cif"] = [r.get("cif") for r in recs]
    for m in metric_cols:
        out[f"{key}_{m}"] = [float(r.get("metrics", {}).get(m, np.nan)) for r in recs]
    out[f"{key}_metrics"] = [json.dumps(r.get("metrics", {})) for r in recs]
    out[f"{key}_batch_seconds"] = float(seconds)
    out[f"{key}_batch_predicted"] = int(n_predicted)
    return out


# ---------------------------------------------------------------------------
# Stage 5 (CPU): explode ligand -> crystal complexes, score every model's pose
# ---------------------------------------------------------------------------
def _ref_paths(complex_name, structures: Path):
    short = complex_name.split("-", 1)[1]
    hits = list(structures.glob(f"*/{complex_name}"))
    d = hits[0] if hits else structures / short / complex_name
    return d / f"{complex_name}_complex_ref.pdb", d / f"{complex_name}_ligand_ref.sdf"


def _read_sdf_heavy(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[3][:3])
    xyz, el = [], []
    for l in lines[4:4 + n]:
        e = l[31:34].strip().upper()
        if e != "H":
            xyz.append([float(l[0:10]), float(l[10:20]), float(l[20:30])])
            el.append(e)
    return np.array(xyz), el


def _ca_by_resnum(model, chain_id):
    return {res.id[1]: (res.get_resname(), res["CA"].coord)
            for res in model[chain_id] if res.id[0] == " " and "CA" in res}


def _binding_chain_ca(model, lig_xyz):
    """Crystal protein chain the ligand sits against. Many EV-A71 2A crystals have
    two copies in the asymmetric unit; aligning to the wrong one adds ~30-40 A."""
    best, best_n = None, -1
    for chain in model:
        ca = np.array([r["CA"].coord for r in chain if r.id[0] == " " and "CA" in r])
        if len(ca) < 50:
            continue
        n = int((np.linalg.norm(ca[:, None] - lig_xyz[None], axis=-1).min(1) < 10).sum())
        if n > best_n:
            best, best_n = chain.id, n
    return _ca_by_resnum(model, best)


def _pred_ligand(model, chain_id="B"):
    for chain in model:
        if chain.id == chain_id:
            atoms = [a for res in chain for a in res if a.element.upper() != "H"]
            return np.array([a.coord for a in atoms]), [a.element.upper() for a in atoms]
    return None, None


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - pc @ R.T


def _element_matched_rmsd(P, pe, Q, qe):
    from scipy.optimize import linear_sum_assignment
    sq, n = 0.0, 0
    for el in set(pe) & set(qe):
        pi = [i for i, e in enumerate(pe) if e == el]
        qi = [i for i, e in enumerate(qe) if e == el]
        D = np.linalg.norm(P[pi][:, None] - Q[qi][None], axis=-1)
        r, c = linear_sum_assignment(D)
        sq, n = sq + (D[r, c] ** 2).sum(), n + len(r)
    return float(np.sqrt(sq / n)) if n else np.nan


def _load_cif(path: str):
    """Parse an mmCIF with Biopython, filling in columns OpenFold3 leaves out."""
    import io
    from Bio.PDB import MMCIFIO, MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    d = MMCIF2Dict(path)
    n = len(d["_atom_site.id"])
    missing = [k for k in ("_atom_site.occupancy", "_atom_site.B_iso_or_equiv") if k not in d]
    if not missing:
        return MMCIFParser(QUIET=True).get_structure("p", path)[0]
    for k in missing:
        d[k] = ["1.0" if k.endswith("occupancy") else "0.0"] * n
    io_ = MMCIFIO()
    io_.set_dict(d)
    buf = io.StringIO()
    io_.save(buf)
    buf.seek(0)
    return MMCIFParser(QUIET=True).get_structure("p", buf)[0]


def _superpose(pred, exp_ca):
    """Rotation and translation that put the predicted protein on the crystal one (CA atoms)."""
    pred_ca = _ca_by_resnum(pred, "A")
    common = [i for i in sorted(set(pred_ca) & set(exp_ca)) if pred_ca[i][0] == exp_ca[i][0]]
    if len(common) < 50:
        raise ValueError(f"only {len(common)} matching CA residues")
    P = np.array([pred_ca[i][1] for i in common])
    Q = np.array([exp_ca[i][1] for i in common])
    R, t = _kabsch(P, Q)
    return R, t, float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))


def score_pose(pred_cif: str, ref_pdb: Path, ref_sdf: Path, exp_ca_cache: dict):
    from Bio.PDB import PDBParser
    lq, lqe = _read_sdf_heavy(ref_sdf)
    if str(ref_pdb) not in exp_ca_cache:
        exp = PDBParser(QUIET=True).get_structure("e", str(ref_pdb))[0]
        exp_ca_cache[str(ref_pdb)] = _binding_chain_ca(exp, lq)
    pred = _load_cif(pred_cif)
    R, t, ca_rmsd = _superpose(pred, exp_ca_cache[str(ref_pdb)])
    lp, lpe = _pred_ligand(pred)
    if lp is None or not len(lp):
        raise ValueError("no predicted ligand on chain B")
    return _element_matched_rmsd(lp @ R.T + t, lpe, lq, lqe), ca_rmsd


def pose_in_crystal_frame(pred_cif: str, ref_pdb: Path) -> str:
    """The predicted complex moved onto its crystal (the superposition score_pose uses), as
    PDB text. Models predict in their own coordinate frame, so without this a predicted and
    a crystal ligand drawn together would sit far apart even when the pose is right."""
    import io
    from Bio.PDB import PDBIO, PDBParser
    ref_pdb = Path(ref_pdb)
    lq, _ = _read_sdf_heavy(ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf")))
    exp_ca = _binding_chain_ca(PDBParser(QUIET=True).get_structure("e", str(ref_pdb))[0], lq)
    pred = _load_cif(pred_cif)
    R, t, _ = _superpose(pred, exp_ca)
    for atom in pred.get_atoms():
        atom.transform(R.T, t)   # Biopython applies coord @ rot + tran
    for res in pred.get_residues():
        if len(res.resname) > 3:   # Boltz-2 names the ligand LIG1: too wide for a PDB column
            res.resname = "LIG"
    out, buf = PDBIO(), io.StringIO()
    out.set_structure(pred)
    out.save(buf)
    return buf.getvalue()


def make_scorer(model_keys, target: str = "ev71-2a"):
    structures = TARGETS[target].structures

    def score(row: dict) -> list[dict]:
        cache, out = {}, []
        for complex_name in row["complexes"]:
            ref_pdb, ref_sdf = _ref_paths(complex_name, structures)
            r = {k: v for k, v in row.items() if k != "complexes"}
            r["complex_name"] = complex_name
            r["target"] = target
            r["ref_pdb"] = str(ref_pdb)
            for k in model_keys:
                r[f"{k}_lig_rmsd"], r[f"{k}_ca_rmsd"], r[f"{k}_score_error"] = np.nan, np.nan, None
                if row.get(f"{k}_cif"):
                    try:
                        r[f"{k}_lig_rmsd"], r[f"{k}_ca_rmsd"] = score_pose(row[f"{k}_cif"], ref_pdb, ref_sdf, cache)
                    except Exception as e:
                        r[f"{k}_score_error"] = f"{type(e).__name__}: {e}"[:300]
            out.append(r)
        return out
    return score


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_ligands(target: Target, limit=None):
    m = pd.read_csv(target.path("metadata"))
    if "stage" not in m:
        m["stage"] = None
    m["curated"] = m.pb_valid_ref.astype(bool) & ~m.suspected_artefact.astype(bool)
    lig = (m.groupby("compound_group", sort=False)
             .agg(smiles=("smiles", "first"), fragment_screen=("fragment_screen", "first"),
                  covalent=("covalent", "first"), experimental_pKD=("experimental_pKD", "first"),
                  complexes=("complex_name", list), curated_any=("curated", "any"),
                  stage=("stage", "first"))
             .reset_index().rename(columns={"compound_group": "ligand_id"}))
    lig = add_progression(lig)
    if limit:
        # a reproducible mix of fragments and follow-ons for smoke tests
        lig = pd.concat([lig[lig.fragment_screen].head(limit // 4),
                         lig[~lig.fragment_screen].head(limit - limit // 4)])
    return m, lig


STAGES = ["fragment", "early follow-on", "late follow-on"]
SIM_BINS = [(0.0, 0.3, "far (<0.3)"), (0.3, 0.5, "mid (0.3-0.5)"), (0.5, 1.01, "near (>=0.5)")]


def add_progression(lig):
    """Two views of compound progression:
    stage: fragment screen, then follow-ons split into early and late. A target's
           metadata can set this explicitly (the Mpro targets use the time-based
           Polaris train/test split); otherwise follow-ons are split at the median
           crystal id, since crystal ids follow campaign time.
    frag_sim: max Morgan Tanimoto to any *other* fragment-screen ligand, i.e. how far
           a compound has moved from what the fragment screen (and the fine-tune) saw."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    lig = lig.copy()
    # Crystal ids look like x0152a (EV-A71) or a0080a / b0110a (Mpro): order by letter, then number.
    first = lig.complexes.str[0].str.extract(r"-([a-z]?)(\d{3,})")
    letter = first[0].fillna("").map(lambda c: ord(c) - 96 if c else 0)
    lig["xtal_index"] = letter * 100000 + first[1].astype(float)
    cut = lig.loc[~lig.fragment_screen, "xtal_index"].median()
    by_order = np.where(lig.fragment_screen, STAGES[0],
                        np.where(lig.xtal_index <= cut, STAGES[1], STAGES[2]))
    lig["stage"] = lig["stage"].where(lig["stage"].isin(STAGES), by_order)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) if (m := Chem.MolFromSmiles(s)) is not None else None for s in lig.smiles]
    frag_idx = [i for i, f in enumerate(lig.fragment_screen) if f and fps[i] is not None]
    frag_fps = [fps[i] for i in frag_idx]
    sims = []
    for i, fp in enumerate(fps):
        if fp is None or not frag_fps:
            sims.append(np.nan)
            continue
        ts = DataStructs.BulkTanimotoSimilarity(fp, frag_fps)
        ts = [t for j, t in zip(frag_idx, ts) if j != i]
        sims.append(max(ts) if ts else np.nan)
    lig["frag_sim"] = sims
    return lig


def summarize(df, model_keys, curated, target: Target):
    d = df[df.complex_name.isin(curated)]
    lines = []
    pub = pd.read_parquet(target.published) if target.published and target.published.exists() else None
    if pub is not None:
        pub = (pub[(pub["rank"] == 0) & ~pub.filtered]
               .sort_values("rank_score", ascending=False).groupby(["method", "complex_id"]).head(1))
    for k in model_keys:
        name, pub_method = MODELS[k]
        groups = [(label, d.fragment_screen == frag) for frag, label in ((True, "fragments"), (False, "follow-ons"))]
        if "stage" in d:
            groups += [(f"stage: {st}", d.stage == st) for st in STAGES]
        if "frag_sim" in d:
            groups += [(f"sim: {lab}", (d.frag_sim >= lo) & (d.frag_sim < hi)) for lo, hi, lab in SIM_BINS]
        for label, mask in groups:
            sub = d[mask]
            s = sub[f"{k}_lig_rmsd"].dropna()
            if not len(s):
                continue
            line = (f"{name:30s} {label:24s} n={len(s):4d}  median RMSD {s.median():5.2f} A"
                    f"  <2A {(s < 2).mean():6.1%}")
            if pub is not None and pub_method:
                p = pub[(pub.method == pub_method) & pub.complex_id.isin(sub.complex_name)].lig_rmsd
                if len(p):
                    line += f"   | published (same complexes) {p.median():5.2f} A, {(p < 2).mean():6.1%}"
            lines.append(line)
    if "boltz2_affinity_pred_value" in d:
        # Cast to numpy floats first: Ray Data returns Arrow-backed columns, where NaN is not
        # treated as missing and dropna() would keep ligands with no measured affinity.
        a = (d.drop_duplicates("ligand_id")[["experimental_pKD", "boltz2_affinity_pred_value"]]
             .astype(float).dropna())
        if len(a) > 5:
            # Boltz-2 affinity_pred_value is log10(IC50 in uM): lower = tighter, so negate.
            rho = (-a.boltz2_affinity_pred_value).corr(a.experimental_pKD, method="spearman")
            lines.append(f"Boltz-2 affinity vs experimental {target.affinity_label}: "
                         f"Spearman {rho:.2f} (n={len(a)} ligands)")
    return "\n".join(lines)


# Relative GPU time per ligand, measured on an L4 with the default packing. The
# EV-A71 2A run of 2026-10-20 (308 ligands, 12 L4) put the true ratio at 1.6, not 2:
# Boltz-2 took 8 GPUs and 47.1 min, OpenBind-0 took 4 and 58.9, so Boltz-2's eight
# GPUs idled for the last twelve minutes. Pass --gpu-split to override per run,
# which is necessary whenever the stages run on different GPU types.
GPU_COST = {"boltz2": 1.6, "openbind0": 1.0, "of3p2": 1.0, "of3ft": 1.0, "of3ray": 1.0}


def split_gpus(model_keys, total: int, override: Optional[str] = None) -> dict:
    """Whole GPUs per model stage, proportional to cost, at least one each.

    `override` is a comma list like "boltz2=7,openbind0=13", which wins outright.
    """
    if override:
        share = {}
        for part in override.split(","):
            k, _, v = part.partition("=")
            k = k.strip()
            if k not in model_keys:
                raise SystemExit(f"--gpu-split names {k!r}, which is not in --models {model_keys}")
            share[k] = int(v)
        missing = [k for k in model_keys if k not in share]
        if missing:
            raise SystemExit(f"--gpu-split must cover every model; missing {missing}")
        return share
    w = {k: GPU_COST[k] for k in model_keys}
    share = {k: max(1, int(total * c / sum(w.values()))) for k, c in w.items()}
    # Hand any GPUs lost to rounding to the most expensive stages first.
    for k in sorted(w, key=w.get, reverse=True):
        if sum(share.values()) >= total:
            break
        share[k] += 1
    return share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="ev71-2a", choices=list(TARGETS),
                    help="which protein target to benchmark")
    ap.add_argument("--models", default="boltz2,openbind0",
                    help=f"comma list from {list(MODELS)}")
    ap.add_argument("--limit", type=int, default=None, help="first N ligands (smoke test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--boltz-per-gpu", type=int, default=2,
                    help="Boltz-2 predictors sharing one GPU (one featurizes while another computes)")
    ap.add_argument("--of3-per-gpu", type=int, default=3,
                    help="OpenFold3 predictors sharing one GPU")
    ap.add_argument("--cpus-per-gpu-node", type=int, default=8,
                    help="vCPUs on one GPU worker (8 on g6.2xlarge, 16 on g6.4xlarge). Each predictor "
                         "reserves this divided by the predictors sharing its GPU, less one for Ray")
    ap.add_argument("--gpu-split", default=None,
                    help="GPUs per model, e.g. 'boltz2=7,openbind0=13', instead of splitting "
                         "--max-gpus by cost. Use it when the stages run on different GPU types")
    ap.add_argument("--max-gpus", type=int, default=8,
                    help="GPUs for the whole pipeline, split across model stages by their cost")
    ap.add_argument("--accelerator", default="L4", help="GPU type for every GPU stage")
    ap.add_argument("--of3-accelerator", default=None, help="override GPU type for OF3 stages, e.g. L40S")
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--affinity-samples", type=int, default=5,
                    help="Boltz-2 affinity ensemble size (Boltz default 5). Lower is faster")
    ap.add_argument("--affinity-only", action="store_true",
                    help="only ligands with an experimental affinity value")
    ap.add_argument("--of3ray-ckpt", default=None,
                    help="weights for the of3ray model (default: latest finetune_ray_train.py run)")
    ap.add_argument("--of3ray-legacy", action="store_true",
                    help="of3ray weights were fine-tuned from of3-p2-145k (finetune_ray_train.py --base p2), "
                         "so predict with the legacy OpenFold3 0.4.3")
    ap.add_argument("--stage", default=None, choices=STAGES,
                    help="only ligands from one campaign stage, e.g. 'late follow-on'")
    ap.add_argument("--pred-dir", default=None,
                    help="write predictions to <pred-dir>/<model> instead of the shared cache, "
                         "so a live run recomputes every ligand")
    ap.add_argument("--out", default=None, help="output Parquet directory (default: results/<target>)")
    args = ap.parse_args()
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    assert all(k in MODELS for k in model_keys), model_keys

    target = TARGETS[args.target]
    args.out = args.out or str(SHARED / "results" / target.key)
    for p in (target.path("metadata"), target.path("fasta"), target.path("msa"), target.structures, BOLTZ_CACHE):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run `python scripts/setup_data.py` first (README step 2).")
    for key, ckpt, url in (("openbind0", OF3_OB_CKPT, OF3_OB_URL), ("of3p2", OF3_P2_CKPT, OF3_P2_URL),
                           ("of3ft", OF3_FT_CKPT, OF3_FT_URL)):
        if key in model_keys and not ckpt.exists():
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {key} checkpoint -> {ckpt}")
            subprocess.run(["curl", "-sSL", "-o", str(ckpt), url], check=True)

    of3ray_ckpt = None
    if "of3ray" in model_keys:
        of3ray_ckpt = Path(args.of3ray_ckpt) if args.of3ray_ckpt else latest_finetune_ckpt()
        if of3ray_ckpt is None or not of3ray_ckpt.exists():
            raise SystemExit("No Ray Train weights found. Run finetune_ray_train.py first (README step 7).")
        print(f"of3ray weights: {of3ray_ckpt}")
        print("  note: predictions are cached per ligand; delete predictions/of3ray after retraining")

    ray.init()
    meta, lig = load_ligands(target, None if args.affinity_only else args.limit)
    if args.affinity_only:
        lig = lig[lig.experimental_pKD.notna()]
        lig = lig.sample(min(args.limit, len(lig)), random_state=0) if args.limit else lig
    if args.stage:
        lig = lig[lig.stage == args.stage]
    print(f"Target: {target.name}")
    # Sort by size so each batch holds similar ligands: no batch waits on one giant.
    lig = lig.iloc[lig.smiles.str.len().argsort(kind="stable")].reset_index(drop=True)
    print(f"{len(lig)} unique ligands -> {int(lig.complexes.str.len().sum())} crystal complexes; "
          f"models={model_keys}, batch={args.batch_size}, GPUs={args.max_gpus}")

    n_blocks = max(1, len(lig) // args.batch_size)
    lig = validate(lig)   # RDKit parse, a second for the whole set: done once, on the driver

    # A single 150 to 300 residue complex uses under 5 GB and cannot keep an L4 busy on its
    # own, so several predictors share each GPU: while one loads weights or featurizes on
    # CPU, another runs on the GPU. CPUs are split so the predictors on one g6.2xlarge
    # (8 vCPU) fit together.
    #
    # Pools have a fixed size. Ray Data only asks for more nodes when CPU or memory is
    # busy, so fractional GPU actors that fit on the first nodes would never trigger a
    # scale up. A fixed size is always requested, and splitting the GPUs by each model's
    # cost per ligand lets every model finish at about the same time.
    gpus = split_gpus(model_keys, args.max_gpus, args.gpu_split)
    per_gpu = {k: args.boltz_per_gpu if k == "boltz2" else args.of3_per_gpu for k in model_keys}
    # Models predicted with the legacy OpenFold3 0.4.3 take more host RAM per process (of3ft
    # also loads a 6.7 GB checkpoint): three per node OOM-killed the raylet on 32 GB nodes.
    # Say so, because a silent clamp makes a packing sweep look like it plateaus when the
    # extra actors never ran.
    legacy = {"of3ft", "of3p2"} | ({"of3ray"} if args.of3ray_legacy else set())
    for k in legacy & set(per_gpu):
        if per_gpu[k] > 2:
            print(f"{k}: --of3-per-gpu {per_gpu[k]} clamped to 2 (legacy OpenFold3, host RAM)")
            per_gpu[k] = 2
    print("GPUs per model: " + ", ".join(f"{k} {g} ({per_gpu[k]} per GPU)" for k, g in gpus.items()))

    def predictor(k):
        out_dir = {"out_dir": str(Path(args.pred_dir) / k)} if args.pred_dir else {}
        if k == "boltz2":
            return Boltz2Predictor, {"target": target.key, "diffusion_samples": args.diffusion_samples,
                                     "affinity_samples": args.affinity_samples, "monitor_dir": args.out,
                                     **out_dir}
        # of3p2 and of3ft are OF3-p2 checkpoints: they need the legacy openfold3 in /opt/of3-legacy.
        if k == "of3ft":
            ck = {"ckpt_path": str(OF3_FT_CKPT), "binary": OF3_LEGACY}
        elif k == "of3p2":
            ck = {"ckpt_path": str(OF3_P2_CKPT), "binary": OF3_LEGACY}
        elif k == "of3ray":
            ck = {"ckpt_path": str(of3ray_ckpt), **({"binary": OF3_LEGACY} if args.of3ray_legacy else {})}
        else:
            ck = {"ckpt_path": str(OF3_OB_CKPT)}
        return OF3Predictor, {"key": k, "target": target.key, "monitor_dir": args.out,
                              "diffusion_samples": args.diffusion_samples, **ck, **out_dir}

    def run_model(k):
        """One independent Ray Data pipeline per model. Models never wait on each other:
        chained stages would all run at the speed of the slowest one."""
        cls, kwargs = predictor(k)
        n = min(gpus[k] * per_gpu[k], n_blocks)   # no more actors than batches
        acc = args.accelerator if k == "boltz2" else (args.of3_accelerator or args.accelerator)
        # One batch in flight per actor: batches take minutes, so a queued batch would
        # wait behind a busy actor while another actor sits idle at the end of the run.
        pool = ray.data.ActorPoolStrategy(size=n, max_tasks_in_flight_per_actor=1)
        ds = ray.data.from_pandas(lig[["ligand_id", "smiles", "smiles_ok"]]).repartition(n_blocks)
        out = ds.map_batches(cls, fn_constructor_kwargs=kwargs, batch_size=args.batch_size,
                             batch_format="pandas", num_gpus=1 / per_gpu[k],
                             num_cpus=max(1, args.cpus_per_gpu_node // per_gpu[k] - 1),
                             accelerator_type=acc,
                             compute=pool).to_pandas()
        print(f"  {MODELS[k][0]}: finished after {(time.time() - t0) / 60:.1f} min", flush=True)
        return out[["ligand_id"] + [c for c in out.columns if c.startswith(f"{k}_")]]

    for d in (args.out, args.out + "_gpu"):
        if Path(d).exists():
            shutil.rmtree(d)   # write_parquet adds files: never mix with an earlier run
    t0 = time.time()
    with ThreadPoolExecutor(len(model_keys)) as pool:
        per_model = list(pool.map(run_model, model_keys))
    for res in per_model:
        lig = lig.merge(res, on="ligand_id", how="left")

    # Score every model's pose against every crystal binding event (CPU).
    scored = (ray.data.from_pandas(lig).repartition(max(1, len(lig) // 16))
              .flat_map(make_scorer(model_keys, target.key), num_cpus=1))
    scored.write_parquet(args.out)
    wall = time.time() - t0

    df = ray.data.read_parquet(args.out).to_pandas()
    print(f"\nDone in {wall / 60:.1f} min. {len(df)} complexes scored -> {args.out}")
    print(gpu_report(args.out))
    for k in model_keys:
        st = df.drop_duplicates("ligand_id")[f"{k}_status"].value_counts().to_dict()
        print(f"  {MODELS[k][0]}: {st}")
        per = df.drop_duplicates("ligand_id")
        per = per[per[f"{k}_batch_predicted"] > 0]
        if len(per):
            b = per.drop_duplicates(f"{k}_batch_seconds")
            print(f"    {b[f'{k}_batch_seconds'].sum() / b[f'{k}_batch_predicted'].sum():.1f} GPU-s per ligand")
        err = df[f"{k}_error"].dropna()
        if len(err):
            print(f"    first error: {err.iloc[0][:300]}")
    curated = set(meta.loc[meta.curated, "complex_name"])
    print("\nCurated benchmark (PoseBusters-valid reference, no suspected artefact):")
    print(summarize(df, model_keys, curated, target))


if __name__ == "__main__":
    main()
