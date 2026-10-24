"""
Does more compute fix Boltz-2's EV-A71 2A follow-ons? A Ray Tune sweep over Boltz-2's
inference settings.

    python scripts/tune_boltz.py                                     # 8 settings, 1 L4 each, about 1 h
    python scripts/tune_boltz.py --target sars2-mpro --ligands 48

Boltz-2 places only 3% of EV-A71 2A follow-on poses within 2 A with its default settings
(1 diffusion sample, 3 recycling steps). This sweep asks whether that is a sampling problem
(more samples, more recycling) or a physics problem (inference-time steering with
--use_potentials), or neither. Each setting is one Tune trial on its own L4:

    Tuner             one trial per setting, all running at once on an autoscaling cluster
    grid_search       recycling steps x diffusion samples x steering
    tune.report       pose accuracy so far, after every batch of ligands

Every trial predicts the same random sample of curated follow-on ligands (affinity off:
this is about poses) and scores them against the crystals, exactly like the benchmark.
Predictions are cached per setting, so a rerun only predicts what is missing.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ray
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repository root
from ray import tune

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))


def ligand_set(target: str, n: int, seed: int = 0) -> pd.DataFrame:
    """A fixed random sample of curated follow-on ligands, keeping only curated complexes."""
    from aiddd_pipeline import TARGETS, load_ligands, validate
    meta, lig = load_ligands(TARGETS[target])
    curated = set(meta.loc[meta.curated, "complex_name"])
    lig = lig[~lig.fragment_screen & lig.curated_any]
    lig = validate(lig.sample(min(n, len(lig)), random_state=seed).reset_index(drop=True))
    lig["complexes"] = lig.complexes.map(lambda cs: [c for c in cs if c in curated])
    return lig


def setting_name(c: dict) -> str:
    return f"rec{c['recycling_steps']}_samples{c['diffusion_samples']}" + ("_steered" if c["use_potentials"] else "")


def trial(config: dict, ligands: pd.DataFrame):
    """One Boltz-2 setting: predict every ligand in batches and score the poses as they come."""
    from aiddd_pipeline import Boltz2Predictor, make_scorer
    predictor = Boltz2Predictor(
        target=config["target"], diffusion_samples=config["diffusion_samples"],
        recycling_steps=config["recycling_steps"], affinity=False,
        extra_args=["--use_potentials"] if config["use_potentials"] else [],
        out_dir=str(SHARED / "tune" / "boltz2" / config["target"] / setting_name(config)))
    score = make_scorer(["boltz2"], config["target"])
    rmsd, t0 = [], time.time()
    for start in range(0, len(ligands), config["batch_size"]):
        batch = ligands.iloc[start:start + config["batch_size"]]
        pred = predictor(batch[["ligand_id", "smiles", "smiles_ok"]].copy())
        rows = pred.merge(batch[["ligand_id", "complexes"]], on="ligand_id").to_dict("records")
        rmsd += [r["boltz2_lig_rmsd"] for row in rows for r in score(row)]
        s = pd.Series(rmsd, dtype=float)
        tune.report({"ligands_done": start + len(batch), "complexes": len(s),
                     "success": float((s < 2).mean()), "median_rmsd": float(s.median()),
                     "failed": int(s.isna().sum()), "minutes": (time.time() - t0) / 60})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="ev71-2a", choices=["ev71-2a", "sars2-mpro", "mers-mpro"])
    ap.add_argument("--ligands", type=int, default=32, help="random curated follow-on ligands")
    ap.add_argument("--recycling", default="3,10", help="recycling steps to try")
    ap.add_argument("--samples", default="1,5", help="diffusion samples to try (best by confidence is kept)")
    ap.add_argument("--steering", default="off,on", help="--use_potentials off and/or on")
    ap.add_argument("--batch-size", type=int, default=8, help="ligands per boltz call")
    ap.add_argument("--max-concurrent", type=int, default=8, help="trials (GPUs) at the same time")
    ap.add_argument("--accelerator", default="L4")
    ap.add_argument("--name", default=f"tune-boltz2-{time.strftime('%Y%m%d-%H%M%S')}")
    args = ap.parse_args()

    ray.init()
    ligands = ligand_set(args.target, args.ligands)
    print(f"{len(ligands)} follow-on ligands, {int(ligands.complexes.str.len().sum())} curated complexes "
          f"({args.target})")

    param_space = {
        "target": args.target,
        "batch_size": args.batch_size,
        "recycling_steps": tune.grid_search([int(x) for x in args.recycling.split(",")]),
        "diffusion_samples": tune.grid_search([int(x) for x in args.samples.split(",")]),
        "use_potentials": tune.grid_search([s.strip() == "on" for s in args.steering.split(",")]),
    }
    # One L4 per trial. The tiny accelerator_type resource pins trials to that GPU type.
    resources = {"CPU": 6, "GPU": 1, f"accelerator_type:{args.accelerator}": 0.001}
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trial, ligands=ligands), resources),
        param_space=param_space,
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent),
        run_config=tune.RunConfig(name=args.name, storage_path=str(SHARED / "tune" / "runs")),
    )
    start = time.time()
    grid = tuner.fit()

    rows = []
    for r in grid:
        m = r.metrics or {}
        rows.append({"setting": setting_name(r.config), "recycling_steps": r.config["recycling_steps"],
                     "diffusion_samples": r.config["diffusion_samples"], "steered": r.config["use_potentials"],
                     "success": m.get("success", np.nan), "median_rmsd": m.get("median_rmsd", np.nan),
                     "complexes": m.get("complexes"), "minutes": m.get("minutes", np.nan),
                     "error": str(r.error)[:200] if r.error else None})
    df = pd.DataFrame(rows).sort_values(["success", "median_rmsd"], ascending=[False, True])
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Boltz-2 on {args.target} follow-ons, "
          f"share of poses under 2 A:")
    for row in df.itertuples():
        if row.error:
            print(f"  {row.setting:28s} failed: {row.error}")
        else:
            print(f"  {row.setting:28s} {row.success:6.1%}  median {row.median_rmsd:5.2f} A  "
                  f"n={row.complexes}  {row.minutes:5.1f} min")
    out = SHARED / "tune" / "runs" / args.name / "summary.csv"
    df.to_csv(out, index=False)
    print(f"  table: {out}")


if __name__ == "__main__":
    main()
