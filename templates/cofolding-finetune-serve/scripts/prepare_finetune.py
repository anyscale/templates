"""
Build an OpenFold3 training set from one or more targets' crystal structures.

    python scripts/prepare_finetune.py                           # EV-A71 2A fragments (66)
    python scripts/prepare_finetune.py --targets ev71-2a,sars2-mpro      # both screens (105)
    python scripts/prepare_finetune.py --targets ev71-2a,sars2-mpro \\
        --stages "fragment,early follow-on"                      # plus early compounds (540)
    python scripts/prepare_finetune.py --limit 4 --check         # quick test, then load with OF3
    python scripts/prepare_finetune.py --paper-set --out $AIDDD_SHARED/finetune/ev71-2a-frag80
                                                                 # the OpenBind paper's set (80)

Selection is two separate decisions. The quality filters never move: a PoseBusters-valid
reference pose, no suspected crystallographic artefact, no covalent ligand, one binding
event per compound. Which campaign stages to train on is a design choice, set by --stages,
and every stage spent on training is one the benchmark can no longer use to measure
generalization. Held-out targets are simply the ones left out of --targets.

Output, in the layout OF3's WeightedPDBDataset reads (finetune_ray_train.py points at it):

    $AIDDD_SHARED/finetune/ev71-2a/
        structures/<complex>/<complex>.npz    cleaned AtomArray: one protein chain, Zn, ligand
        reference_mols/<id>.sdf               one per residue type, Zn and ligand (annotated SDF)
        alignments/EV-A71_2A/cfdb_uniref30.a3m  the release MSA, shared by every complex
        training_cache.json                   chains, interfaces, clusters (sampling weights)

Why not OF3's own mmCIF preprocessing: the release ships REFMAC PDB files whose
ligand residue (LIG) is not in the Chemical Component Dictionary, so the standard
cleanup would delete it. Instead each complex is rebuilt from the sequence: every
residue of the 150 aa construct (coordinates from the crystal, NaN where unresolved,
so residue numbers line up with the MSA columns), the structural zinc, and the
ligand from its reference SDF with bond orders checked against the SMILES.
Waters, DMSO and sulfate are dropped.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import pandas as pd
from biotite.interface.rdkit import from_mol
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.pipelines.preprocessing.structure import _process_component
from openfold3.core.data.primitives.structure.component import mol_from_atomarray
from openfold3.core.data.primitives.structure.interface import get_interface_chain_id_pairs
from openfold3.core.data.resources.residues import PROTEIN_RESTYPE_1TO3, MoleculeType

RDLogger.DisableLog("rdApp.*")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiddd_pipeline import SHARED, STAGES, TARGETS  # noqa: E402

MSA_NAME = "cfdb_uniref30"      # must be a key of OF3's MSASettings.max_seq_counts
PROTEIN, ZINC, LIGAND = "1", "2", "3"   # chain ids in the training AtomArrays

# Per target: the alignment representative every complex of that target shares, and
# the protein cluster id used for sampling weights. The structures, metadata, FASTA
# and MSA all come from TARGETS, so a target built by scripts/prepare_mpro.py needs
# no new paths here.
TARGET_SPECS = {
    "ev71-2a":    dict(msa_id="EV-A71_2A", cluster="ev71_2a"),
    "sars2-mpro": dict(msa_id="SARS-CoV-2_Mpro", cluster="sars2_mpro"),
    "mers-mpro":  dict(msa_id="MERS-CoV_Mpro", cluster="mers_mpro"),
}
ANNOTATIONS = ["chain_id", "res_id", "ins_code", "res_name", "hetero", "atom_name",
               "element", "charge", "occupancy"]


def select_complexes(meta: pd.DataFrame, stages=("fragment",), stage_of=None,
                     paper_set: bool = False) -> pd.DataFrame:
    """Clean reference pose, wanted campaign stages, one binding event per compound.

    The quality filters are never relaxed: a pose that PoseBusters rejects, a ligand
    modelled at a crystal contact rather than the pocket, or a covalent ligand would
    all teach the model the wrong thing.

    Which stages to train on is a design choice instead. `stage_of` maps complex_name
    to campaign stage and comes from the scored results, so training and evaluation
    can never disagree about what counts as a follow-on compound. Without it only the
    fragment screen can be selected, from the metadata's own flag.

    `paper_set` reproduces the OpenBind paper's fine-tuning set instead (79 to 80 EV-A71
    fragment complexes): artefacts are still dropped, but PoseBusters-invalid references
    and every binding event of a compound are kept, to compare with its released checkpoint.
    """
    meta = meta.copy()
    for c in ("fragment_screen", "pb_valid_ref", "suspected_artefact", "covalent"):
        meta[c] = meta[c].astype(bool)
    ok = ~meta.suspected_artefact & ~meta.covalent
    if not paper_set:
        ok &= meta.pb_valid_ref

    stages = tuple(stages)
    if stage_of is None:
        if stages != ("fragment",):
            raise ValueError(f"stages {stages} need --results for the stage labels")
        ok &= meta.fragment_screen
    else:
        ok &= meta.complex_name.map(stage_of).isin(stages)

    meta = meta[ok].sort_values("complex_name")
    if not paper_set:
        meta = meta.drop_duplicates("compound_group")
    return meta.reset_index(drop=True)


def stage_labels(results_root: Path, target_key: str) -> dict:
    """complex_name -> campaign stage, as the benchmark staged it."""
    files = sorted((results_root / target_key).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No scored results under {results_root / target_key}; "
                         "the campaign stages are derived there. Run aiddd_pipeline.py "
                         "for this target first, or pass --stages fragment.")
    df = pd.concat([pd.read_parquet(f, columns=["complex_name", "stage"]) for f in files],
                   ignore_index=True).dropna(subset=["stage"])
    return dict(zip(df.complex_name, df.stage))


def blank(n: int) -> struc.AtomArray:
    """Empty AtomArray with the annotation set every segment shares."""
    arr = struc.AtomArray(n)
    arr.set_annotation("charge", np.zeros(n, int))
    arr.set_annotation("occupancy", np.ones(n))
    return arr


def from_template(template: struc.AtomArray, coord: np.ndarray, chain: str) -> struc.AtomArray:
    arr = blank(len(template))
    for name in ANNOTATIONS[:-1]:
        arr.set_annotation(name, template.get_annotation(name))
    arr.coord = coord
    arr.chain_id[:] = chain
    arr.bonds = template.bonds
    return arr


def build_protein(seq: str, crystal: struc.AtomArray) -> struc.AtomArray:
    """Full-length chain from CCD residues; crystal coordinates where resolved."""
    resolved = {(r, a): xyz for r, a, xyz in zip(crystal.res_id, crystal.atom_name, crystal.coord)}
    residues = []
    for res_id, aa in enumerate(seq, start=1):
        res = struc.info.residue(PROTEIN_RESTYPE_1TO3[aa])
        res = res[(res.element != "H") & (res.atom_name != "OXT")]   # OF3 drops terminal OXT
        res.res_id[:] = res_id
        coord = np.array([resolved.get((res_id, a), [np.nan] * 3) for a in res.atom_name])
        residues.append(from_template(res, coord, PROTEIN))
    chain = struc.concatenate(residues)
    chain.bonds = struc.connect_via_residue_names(chain)
    chain.occupancy[np.isnan(chain.coord).any(axis=1)] = 0.0
    return chain


def load_ligand(sdf: Path, smiles: str) -> Chem.Mol:
    """Crystal pose with bond orders taken from the SMILES (falls back to the SDF's own)."""
    mol = Chem.MolFromMolFile(str(sdf), removeHs=True)
    ref = Chem.MolFromSmiles(smiles)
    if mol is None or ref is None:
        raise ValueError("ligand does not parse")
    if mol.GetNumHeavyAtoms() != ref.GetNumHeavyAtoms():
        raise ValueError(f"partial ligand: {mol.GetNumHeavyAtoms()} of {ref.GetNumHeavyAtoms()} atoms")
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
    except ValueError:
        pass
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def build_ligand(mol: Chem.Mol, res_name: str) -> struc.AtomArray:
    lig = from_mol(mol, conformer_id=0, add_hydrogen=False)
    lig.res_name[:] = res_name
    lig.res_id[:] = 1
    lig.hetero[:] = True
    lig.atom_name = struc.create_atom_names(lig)
    lig.element = np.char.upper(lig.element)
    return from_template(lig, lig.coord, LIGAND)


def resolution(pdb_path: Path) -> float:
    for line in open(pdb_path):
        if "RESOLUTION RANGE HIGH" in line:
            return float(line.split(":")[1])
    return 2.0


def release_date(pdb_path: Path) -> str:
    try:
        return datetime.strptime(open(pdb_path).readline()[50:59].strip(), "%d-%b-%y").strftime("%Y-%m-%d")
    except ValueError:
        return "2026-01-01"


def build_complex(row, seq: str, lig_id: str, structures: Path) -> tuple[struc.AtomArray, Chem.Mol]:
    """One training AtomArray: the protein copy the ligand binds, its zinc, the ligand.

    `structures` is the target's crystal-structure root. The zinc step is skipped
    for a target that has none, so the Mpro targets need no special case.
    """
    folder = structures / row.compound_group / row.complex_name
    crystal = pdb.PDBFile.read(folder / f"{row.complex_name}_complex_ref.pdb").get_structure(
        model=1, altloc="occupancy", extra_fields=["occupancy"])
    crystal = crystal[crystal.element != "H"]
    mol = load_ligand(folder / f"{row.complex_name}_ligand_ref.sdf", row.smiles)
    lig_xyz = mol.GetConformer().GetPositions()

    # Crystals hold one or two protein copies: keep the one the ligand contacts
    protein = crystal[struc.filter_amino_acids(crystal) & ~crystal.hetero]
    dist = {c: np.min(np.linalg.norm(protein.coord[protein.chain_id == c][:, None] - lig_xyz, axis=-1))
            for c in np.unique(protein.chain_id)}
    chain = protein[protein.chain_id == min(dist, key=dist.get)]
    parts = [build_protein(seq, chain)]

    # Structural zinc of that copy (coordinated by Cys/His, within 3 A of the chain)
    zinc = crystal[crystal.res_name == "ZN"]
    if len(zinc):
        d = np.min(np.linalg.norm(zinc.coord[:, None] - chain.coord, axis=-1), axis=1)
        if d.min() < 3.0:
            zn = struc.info.residue("ZN")
            parts.append(from_template(zn, zinc.coord[[d.argmin()]], ZINC))

    parts.append(build_ligand(mol, lig_id))
    atoms = struc.concatenate(parts)

    # Annotations OF3's mmCIF preprocessing writes
    chains = {PROTEIN: (1, MoleculeType.PROTEIN, "A"), ZINC: (2, MoleculeType.LIGAND, "B"),
              LIGAND: (3, MoleculeType.LIGAND, "C")}
    atoms.set_annotation("entity_id", np.array([chains[c][0] for c in atoms.chain_id]))
    atoms.set_annotation("label_entity_id", atoms.entity_id.copy())
    atoms.set_annotation("molecule_type_id", np.array([int(chains[c][1]) for c in atoms.chain_id]))
    atoms.set_annotation("label_asym_id", np.array([chains[c][2] for c in atoms.chain_id]))
    atoms.set_annotation("auth_asym_id", atoms.label_asym_id.copy())
    atoms.set_annotation("label_seq_id", np.where(atoms.chain_id == PROTEIN, atoms.res_id.astype(str), "."))
    atoms.set_annotation("auth_seq_id", atoms.res_id.copy())
    for prefix in ("label", "auth"):
        atoms.set_annotation(f"{prefix}_comp_id", atoms.res_name.copy())
        atoms.set_annotation(f"{prefix}_atom_id", atoms.atom_name.copy())
    atoms.set_annotation("sym_id", np.zeros(len(atoms), int))
    return atoms, mol


def cache_entry(atoms: struc.AtomArray, row, lig_id: str, n_total: int, pdb_path: Path,
                msa_id: str, protein_cluster: str) -> dict:
    """Training-cache record. Clusters: the protein and zinc are shared by every complex
    (one big cluster each), every compound is its own cluster, so OF3's sampler favours
    the protein-ligand interface (AF3 SI 2.5.1)."""
    chains = {}
    for cid in np.unique(atoms.chain_id):
        is_prot = cid == PROTEIN
        ref = None if is_prot else ("ZN" if cid == ZINC else lig_id)
        cluster = {PROTEIN: (protein_cluster, n_total), ZINC: ("zn", n_total),
                   LIGAND: (row.compound_group, 1)}[cid]
        chains[cid] = {
            "label_asym_id": atoms.label_asym_id[atoms.chain_id == cid][0],
            "auth_asym_id": atoms.label_asym_id[atoms.chain_id == cid][0],
            "entity_id": int(atoms.entity_id[atoms.chain_id == cid][0]),
            "molecule_type": "PROTEIN" if is_prot else "LIGAND",
            "reference_mol_id": ref,
            "alignment_representative_id": msa_id if is_prot else None,
            "template_ids": [] if is_prot else None,
            "cluster_id": cluster[0], "cluster_size": cluster[1]}
    interfaces = {}
    for a, b in get_interface_chain_id_pairs(atoms, distance_threshold=5.0):
        a, b = str(a), str(b)
        size = 1 if LIGAND in (a, b) else n_total
        interfaces[f"{a}_{b}"] = {"cluster_id": f"{chains[a]['cluster_id']}_{chains[b]['cluster_id']}",
                                  "cluster_size": size}
    return {"release_date": release_date(pdb_path), "resolution": resolution(pdb_path),
            "chains": chains, "interfaces": interfaces}


def ref_mol_record(meta: dict) -> dict:
    keys = ("conformer_gen_strategy", "fallback_conformer_pdb_id", "canonical_smiles",
            "fallback_conformer_source")
    return {**{k: meta.get(k) for k in keys}, "set_fallback_to_nan": False}


def check(out: Path, n: int = 4) -> None:
    """Load the set through OF3's own WeightedPDBDataset and featurize a few samples."""
    from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
    from openfold3.projects.of3_all_atom.config.dataset_configs import TrainingDatasetSpec
    from finetune_ray_train import dataset_paths

    spec = TrainingDatasetSpec.model_validate({
        "name": "openbind-ev71", "dataset_class": "WeightedPDBDataset", "mode": "train",
        "weight": 1.0, "config": {"crop": {"token_crop": {"token_budget": 192}},
                                  "dataset_paths": dataset_paths(out)}})
    ds = WeightedPDBDataset(spec.config)
    print(f"WeightedPDBDataset: {len(ds)} datapoints (chains + interfaces)")
    for i in np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False):
        f = ds[int(i)]
        resolved = f["ground_truth"]["atom_resolved_mask"] if "ground_truth" in f else f.get("atom_resolved_mask")
        print(f"  {f['pdb_id']:<18} {f['preferred_chain_or_interface']:<12} "
              f"tokens={int(f['token_mask'].sum())} msa={tuple(f['msa'].shape)}"
              + (f" resolved_atoms={int(resolved.sum())}" if resolved is not None else ""))


def build_target(tkey: str, stages, results_root, out: Path, letter: str, limit=None, paper_set=False):
    """Write one target's complexes into a shared training set. Returns its cache rows."""
    target, spec = TARGETS[tkey], TARGET_SPECS[tkey]
    msa_id, seq_path = spec["msa_id"], target.path("fasta")
    seq = "".join(open(seq_path).read().split("\n")[1:]).strip()

    stage_of = None if tuple(stages) == ("fragment",) else stage_labels(results_root, tkey)
    rows = select_complexes(pd.read_csv(target.path("metadata")), stages, stage_of, paper_set)
    if limit:
        rows = rows.head(limit)
    (out / f"alignments/{msa_id}").mkdir(parents=True, exist_ok=True)

    # One MSA per target, query line first, shared by every complex of that target.
    msa = target.path("msa")
    assert open(msa).read().split("\n")[1].strip() == seq, f"{tkey}: MSA query != FASTA"
    shutil.copy(msa, out / "alignments" / msa_id / f"{MSA_NAME}.a3m")

    structures, ref_mols = {}, {}
    for i, row in enumerate(rows.itertuples()):
        # 5 characters max, and unique across targets: one letter per target.
        lig_id = f"{letter}{i:04d}"
        pdb_path = (target.structures / row.compound_group / row.complex_name
                    / f"{row.complex_name}_complex_ref.pdb")
        try:
            atoms, _ = build_complex(row, seq, lig_id, target.structures)
        except Exception as e:   # report and skip a broken entry
            print(f"  skip {row.complex_name}: {e}")
            continue
        (out / "structures" / row.complex_name).mkdir(parents=True, exist_ok=True)
        write_atomarray_to_npz(atoms, out / "structures" / row.complex_name / f"{row.complex_name}.npz")
        structures[row.complex_name] = cache_entry(atoms, row, lig_id, len(rows), pdb_path,
                                                   msa_id, spec["cluster"])
        lig = atoms[atoms.chain_id == LIGAND]
        ref_mols[lig_id] = ref_mol_record(_process_component(lig_id, mol_from_atomarray(lig), 1,
                                                             out / "reference_mols"))
    print(f"  {tkey}: {len(structures)} complexes from {len(rows)} selected")
    return structures, ref_mols, seq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="ev71-2a",
                    help=f"comma list from {list(TARGET_SPECS)}; several are pooled into one set")
    ap.add_argument("--stages", default="fragment",
                    help=f"comma list from {STAGES}. Anything past 'fragment' needs --results, "
                         "and spends that stage as training data instead of held-out evaluation")
    ap.add_argument("--results", type=Path, default=SHARED / "results",
                    help="scored results root, where the campaign stage of each complex is read from")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: $AIDDD_SHARED/finetune/<targets>-<stages>")
    ap.add_argument("--limit", type=int, default=None, help="first N complexes per target (testing)")
    ap.add_argument("--check", action="store_true", help="load a few samples with OF3 afterwards")
    ap.add_argument("--paper-set", action="store_true",
                    help="the OpenBind paper's selection: keep PoseBusters-invalid references and "
                         "every binding event (80 EV-A71 fragments instead of 66)")
    args = ap.parse_args()

    tkeys = [t.strip() for t in args.targets.split(",")]
    stages = [s.strip() for s in args.stages.split(",")]
    assert all(t in TARGET_SPECS for t in tkeys), tkeys
    assert all(s in STAGES for s in stages), stages
    out = args.out or (SHARED / "finetune" /
                       f"{'+'.join(tkeys)}-{'+'.join(s.split()[0] for s in stages)}")
    for sub in ("structures", "reference_mols"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print(f"targets {tkeys}, stages {stages} -> {out}")
    structures, ref_mols, seqs = {}, {}, []
    for letter, tkey in zip("ABCDEFG", tkeys):
        s, r, seq = build_target(tkey, stages, args.results, out, letter, args.limit, args.paper_set)
        structures.update(s)
        ref_mols.update(r)
        seqs.append(seq)

    # Reference molecules for every residue type across all targets, and zinc, from
    # the CCD copy in biotite.
    for code in sorted({PROTEIN_RESTYPE_1TO3[a] for seq in seqs for a in seq}) + ["ZN"]:
        res = struc.info.residue(code)
        mol = mol_from_atomarray(res[res.element != "H"])
        ref_mols[code] = ref_mol_record(_process_component(code, mol, 1, out / "reference_mols"))

    cache = {"_type": "ClusteredDatasetCache",
             "name": f"openbind-{'+'.join(tkeys)}-{'+'.join(s.split()[0] for s in stages)}",
             "structure_data": structures, "reference_molecule_data": ref_mols}
    (out / "training_cache.json").write_text(json.dumps(cache, indent=1))
    print(f"{len(structures)} complexes, {len(ref_mols)} reference molecules -> {out}")

    if args.check:
        check(out)


if __name__ == "__main__":
    main()
