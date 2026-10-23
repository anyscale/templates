"""
Compare models across targets and compound progression in one table.

    python scripts/compare_targets.py                      # every target with results
    python scripts/compare_targets.py --csv summary.csv    # also save the table

Reads the Parquet output of aiddd_pipeline.py for each target
($AIDDD_SHARED/results/<target>) and reports, per model, target and stage:
the number of curated binding events, median ligand RMSD, and the share of
poses under 2 A. A second table shows how well Boltz-2's predicted affinity
ranks compounds (Spearman correlation with the experimental values).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root
from aiddd_pipeline import MODELS, SHARED, STAGES, TARGETS


def load(target_key: str):
    path = SHARED / "results" / target_key
    if not path.exists():
        return None
    # One file per Ray Data block; a block whose column is all empty gets a null type,
    # so read the files one by one instead of as a single dataset.
    files = sorted(path.glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    meta = pd.read_csv(TARGETS[target_key].path("metadata"))
    curated = meta.loc[meta.pb_valid_ref.astype(bool) & ~meta.suspected_artefact.astype(bool), "complex_name"]
    return df[df.complex_name.isin(set(curated))]


def pose_table(results: dict) -> pd.DataFrame:
    rows = []
    for tkey, df in results.items():
        for mkey, (mname, _) in MODELS.items():
            col = f"{mkey}_lig_rmsd"
            if col not in df or df[col].notna().sum() == 0:
                continue
            for stage in ["all"] + STAGES:
                s = (df if stage == "all" else df[df.stage == stage])[col].dropna()
                if len(s):
                    rows.append({"model": mname, "target": TARGETS[tkey].name, "stage": stage, "n": len(s),
                                 "median_rmsd": round(s.median(), 2), "pct_under_2A": round(100 * (s < 2).mean(), 1)})
    return pd.DataFrame(rows)


def affinity_table(results: dict) -> pd.DataFrame:
    rows = []
    for tkey, df in results.items():
        if "boltz2_affinity_pred_value" not in df:
            continue
        a = df.drop_duplicates("ligand_id").dropna(subset=["experimental_pKD", "boltz2_affinity_pred_value"])
        if len(a) > 5:
            # Boltz-2 reports log10(IC50 in uM): lower is tighter, so negate before correlating.
            rho = (-a.boltz2_affinity_pred_value).corr(a.experimental_pKD, method="spearman")
            rows.append({"target": TARGETS[tkey].name, "experimental": TARGETS[tkey].affinity_label,
                         "n_ligands": len(a), "spearman": round(rho, 2)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="also write the pose table to this CSV file")
    args = ap.parse_args()

    results = {k: df for k in TARGETS if (df := load(k)) is not None}
    if not results:
        raise SystemExit(f"No results under {SHARED / 'results'}. Run aiddd_pipeline.py first.")

    poses = pose_table(results)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print("Pose accuracy (curated binding events)\n")
        print(poses.to_string(index=False))
        print("\nBoltz-2 affinity ranking\n")
        print(affinity_table(results).to_string(index=False))
    if args.csv:
        poses.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
