"""
Build two more benchmark targets: SARS-CoV-2 Mpro and MERS-CoV Mpro.

    python scripts/prepare_mpro.py                  # both targets (about 10 min, 300 MB)
    python scripts/prepare_mpro.py --target mers-mpro

Sources (all public):
    Crystal structures   Fragalysis target CoV-Mpro (ASAP Discovery and COVID Moonshot), CC-BY 4.0
    Potency (pIC50)      ASAP x Polaris x OpenADMET antiviral challenge, MIT
                         huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge
    MSA                  ColabFold MSA server, one query per protein

The output has the same layout as the OpenBind EV-A71 2A release, so the
pipeline treats all three targets the same way:

    $AIDDD_SHARED/raw/SARS-CoV-2_Mpro/
        metadata.csv           one row per crystal binding event
        protein.fasta          protein.a3m
        structures/<compound>/<complex>/<complex>_complex_ref.pdb
                                        <complex>_ligand_ref.sdf

How progression is defined (column `stage`):
    fragment          SARS-CoV-2 only: COVID Moonshot XChem fragment screen (16 heavy atoms or fewer)
    early follow-on   ASAP compounds in the Polaris training set
    late follow-on    ASAP compounds in the Polaris test set (made later in the campaign)
    (unset)           ASAP compounds without potency: staged by crystal order in the pipeline

Binding events far from the catalytic cysteine (crystal contact or surface sites)
are flagged `suspected_artefact` and left out of the curated benchmark.
"""

import argparse
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
FRAGALYSIS = "https://fragalysis.diamond.ac.uk/api/site_observations/?format=json&target=41&limit=5000"
POTENCY = ("https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge/"
           "resolve/main/Potency.csv")

TARGETS = {
    "sars2-mpro": dict(folder="SARS-CoV-2_Mpro", prefix="SARS-CoV-2-Mpro",
                       pic50="pIC50 (SARS-CoV-2 Mpro)", cys=145, moonshot_fragments=True),
    "mers-mpro": dict(folder="MERS-CoV_Mpro", prefix="MERS-CoV-Mpro",
                      pic50="pIC50 (MERS-CoV Mpro)", cys=148, moonshot_fragments=False),
}
FRAGMENT_MAX_HEAVY_ATOMS = 16
ACTIVE_SITE_CUTOFF = 8.0   # Angstrom from the catalytic cysteine SG to the nearest ligand atom
THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "aiddd-benchmark"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
    raise RuntimeError("unreachable")


def heavy_atoms(smiles: str) -> int:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    return m.GetNumHeavyAtoms() if m is not None else 0


def select_observations(obs: pd.DataFrame, potency: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per binding event, with the columns the pipeline expects."""
    asap = obs[obs.prefix_tooltip == cfg["prefix"]].copy()
    asap["compound_group"] = asap.compound_code.str.rsplit("-", n=1).str[0]
    asap["complex_name"] = asap.code
    asap = asap.merge(potency[["Molecule Name", "Set", cfg["pic50"]]], how="left",
                      left_on="compound_group", right_on="Molecule Name")
    asap["experimental_pKD"] = asap[cfg["pic50"]]
    asap["fragment_screen"] = False
    asap["stage"] = asap.Set.map({"Train": "early follow-on", "Test": "late follow-on"})
    parts = [asap]

    if cfg["moonshot_fragments"]:
        # Legacy entries have PDB-style codes; 5r** is the XChem fragment screen deposition.
        legacy = obs[(obs.prefix_tooltip == "") & obs.code.str.startswith("5r")].copy()
        legacy = legacy[legacy.smiles.map(heavy_atoms).between(1, FRAGMENT_MAX_HEAVY_ATOMS)]
        legacy["complex_name"] = "SARS-" + legacy.code
        # The same fragment can appear in several crystals: group by SMILES.
        first_code = legacy.groupby("smiles").code.transform("first")
        legacy["compound_group"] = "frag-" + first_code.str.replace("-", "_")
        legacy["experimental_pKD"] = np.nan
        legacy["fragment_screen"] = True
        legacy["stage"] = "fragment"
        parts.append(legacy)

    df = pd.concat(parts, ignore_index=True)
    df = df[df.smiles.notna() & (df.smiles != "missing")]
    df["covalent"] = False
    df["pb_valid_ref"] = True
    return df


def download_structures(df: pd.DataFrame, root: Path, workers: int = 16) -> pd.DataFrame:
    def one(row):
        d = root / "structures" / row.compound_group / row.complex_name
        pdb, sdf = d / f"{row.complex_name}_complex_ref.pdb", d / f"{row.complex_name}_ligand_ref.sdf"
        try:
            if not (pdb.exists() and sdf.exists()):
                d.mkdir(parents=True, exist_ok=True)
                pdb.write_bytes(fetch(row.bound_file))
                sdf.write_bytes(fetch(row.ligand_mol))
            return True
        except Exception as e:
            print(f"  skip {row.complex_name}: {type(e).__name__}: {e}")
            return False

    with ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(one, df.itertuples()))
    return df[ok]


def ligand_coords(sdf: Path) -> np.ndarray:
    lines = sdf.read_text().splitlines()
    n = int(lines[3][:3])
    return np.array([[float(l[0:10]), float(l[10:20]), float(l[20:30])] for l in lines[4:4 + n]])


def residues(pdb: Path):
    """{chain: {resnum: (resname, {atom: xyz})}} for standard protein residues."""
    out = {}
    for l in pdb.read_text().splitlines():
        if l.startswith("ATOM") and l[17:20] in THREE_TO_ONE:
            res = out.setdefault(l[21], {}).setdefault(int(l[22:26]), (l[17:20], {}))
            res[1][l[12:16].strip()] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return out


def flag_active_site(df: pd.DataFrame, root: Path, cys: int) -> pd.DataFrame:
    """Mark binding events that are not in the catalytic site of any protein chain."""
    flags = []
    for row in df.itertuples():
        d = root / "structures" / row.compound_group / row.complex_name
        lig = ligand_coords(d / f"{row.complex_name}_ligand_ref.sdf")
        dist = np.inf
        for chain in residues(d / f"{row.complex_name}_complex_ref.pdb").values():
            sg = chain.get(cys, (None, {}))[1].get("SG")
            if sg is not None:
                dist = min(dist, float(np.linalg.norm(lig - sg, axis=1).min()))
        flags.append(dist > ACTIVE_SITE_CUTOFF)
    df = df.copy()
    df["suspected_artefact"] = flags
    return df


def protein_sequence(df: pd.DataFrame, root: Path, n_files: int = 40) -> str:
    """Full construct sequence, numbered from residue 1, merged across several crystals
    so a residue missing from one structure is filled in from another."""
    seen = {}
    for row in df.head(n_files).itertuples():
        d = root / "structures" / row.compound_group / row.complex_name
        for chain in residues(d / f"{row.complex_name}_complex_ref.pdb").values():
            for num, (name, _) in chain.items():
                seen.setdefault(num, THREE_TO_ONE[name])
    nums = sorted(n for n in seen if n >= 1)
    missing = sorted(set(range(1, nums[-1] + 1)) - set(nums))
    if missing:
        raise SystemExit(f"Residues {missing[:10]} never resolved; cannot build a gap-free sequence.")
    return "".join(seen[n] for n in range(1, nums[-1] + 1))


def build_msa(sequence: str, out: Path):
    """One ColabFold MSA query per protein, using the client that ships with Boltz."""
    from boltz.data.msa.mmseqs2 import run_mmseqs2
    a3m = run_mmseqs2([sequence], prefix=str(out.parent / "msa_tmp"), use_env=True, use_filter=True)
    out.write_text(a3m[0])


def prepare(key: str, obs: pd.DataFrame, potency: pd.DataFrame):
    cfg = TARGETS[key]
    root = SHARED / "raw" / cfg["folder"]
    root.mkdir(parents=True, exist_ok=True)
    print(f"\n== {key} -> {root}")

    df = select_observations(obs, potency, cfg)
    print(f"  {len(df)} binding events, {df.compound_group.nunique()} compounds; downloading structures")
    df = download_structures(df, root)
    df = flag_active_site(df, root, cfg["cys"])

    seq = protein_sequence(df, root)
    (root / "protein.fasta").write_text(f">{key}\n{seq}\n")
    print(f"  protein: {len(seq)} residues")
    if not (root / "protein.a3m").exists():
        print("  building MSA (ColabFold server, a few minutes)")
        try:
            build_msa(seq, root / "protein.a3m")
        except ImportError:
            print("  boltz is not installed here: rerun inside the project image to build the MSA")

    cols = ["complex_name", "compound_group", "smiles", "experimental_pKD", "covalent", "fragment_screen",
            "pb_valid_ref", "suspected_artefact", "stage"]
    df[cols].to_csv(root / "metadata.csv", index=False)

    s = df.drop_duplicates("compound_group")
    print(f"  wrote metadata.csv: {len(df)} binding events, {len(s)} compounds, "
          f"{int(s.fragment_screen.sum())} fragments, {int(s.experimental_pKD.notna().sum())} with pIC50, "
          f"{int(df.suspected_artefact.sum())} events outside the active site")
    print("  stages:", s.stage.fillna("by crystal order").value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=list(TARGETS), action="append",
                    help="target to build (repeatable; default: both)")
    args = ap.parse_args()

    print("Fetching Fragalysis observations and Polaris potency")
    raw = json.loads(fetch(FRAGALYSIS))
    obs = pd.DataFrame(raw["results"] if isinstance(raw, dict) else raw)
    obs = obs[obs.superseded.astype(str) != "True"]
    potency = pd.read_csv(io.BytesIO(fetch(POTENCY)))
    for key in args.target or list(TARGETS):
        prepare(key, obs, potency)
    print("\nDone. Run the benchmark with: python aiddd_pipeline.py --target sars2-mpro")


if __name__ == "__main__":
    main()
