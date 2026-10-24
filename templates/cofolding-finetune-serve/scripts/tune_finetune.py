"""
Tune the OpenFold3 fine-tune with Ray Tune: which learning rate gives the best poses?

    python scripts/tune_finetune.py                                   # 4 learning rates, 1 GPU each, about 35 min
    python scripts/tune_finetune.py --lrs 1e-5,3e-5,1e-4,3e-4,1e-3 --max-steps 300 --workers-per-trial 2

Each Tune trial is one complete Ray Train run (finetune_ray_train.py) with its own
learning rate. Trials run side by side, each on its own GPU nodes:

    Tuner             runs the trials concurrently, up to --max-concurrent at a time
    ASHAScheduler     stops a trial early when its protein-ligand lDDT falls behind the others
    TorchTrainer      the same Ray Train run as step 7, with TuneReportCallback streaming
                      every training step to Tune
    validation        each finished trial predicts a fixed set of held-out EV-A71 2A early
                      follow-on compounds with its new weights, on a GPU, and scores the poses

Trials are ranked by validation pose accuracy (share of poses under 2 A), not by training
loss: the training loss of a fine-tune from OpenBind-0 is nearly flat and says little about
poses. The late follow-on compounds are never used here, so a benchmark on them afterwards
(aiddd_pipeline.py --models of3ray --of3ray-ckpt <best weights>) is still a held-out test.
"""

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ray
from ray import tune
from ray.tune.integration.ray_train import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repository root
import finetune_ray_train as ft

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
TARGET = "ev71-2a"


def validation_set(n: int, seed: int = 0) -> pd.DataFrame:
    """A fixed random sample of curated early follow-on ligands of EV-A71 2A."""
    from aiddd_pipeline import TARGETS, load_ligands, validate
    meta, lig = load_ligands(TARGETS[TARGET])
    curated = set(meta.loc[meta.curated, "complex_name"])
    lig = lig[(lig.stage == "early follow-on") & lig.curated_any]
    lig = validate(lig.sample(min(n, len(lig)), random_state=seed).reset_index(drop=True))
    lig["complexes"] = lig.complexes.map(lambda cs: [c for c in cs if c in curated])
    return lig


def pose_stats(rmsd: pd.Series) -> dict:
    rmsd = rmsd.astype(float)
    return {"val_success": float((rmsd < 2).mean()), "val_median_rmsd": float(rmsd.median()),
            "val_n": int(len(rmsd))}


@ray.remote(num_gpus=1, num_cpus=4)
def predict_and_score(weights: str, val: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Predict the validation ligands with one set of weights (one OF3 call) and score them."""
    from aiddd_pipeline import OF3Predictor, make_scorer
    predictor = OF3Predictor(key="of3tune", ckpt_path=weights, target=TARGET, data_workers=3)
    predictor.out = Path(out_dir)          # one prediction folder per trial, not the shared cache
    predictor.out.mkdir(parents=True, exist_ok=True)
    pred = predictor(val[["ligand_id", "smiles", "smiles_ok"]].copy())
    rows = pred.merge(val[["ligand_id", "complexes"]], on="ligand_id").to_dict("records")
    score = make_scorer(["of3tune"], TARGET)
    return pd.DataFrame([r for row in rows for r in score(row)])


def trial(config: dict, val: pd.DataFrame):
    """One Tune trial: a full Ray Train fine-tune, then validation of its weights."""
    args = ft.parser().parse_args([])
    for k, v in config["base_args"].items():
        setattr(args, k, v)
    args.lr = config["lr"]
    args.storage = args.storage / config["sweep"]       # every trial's run next to the Tune results
    args.name = tune.get_context().get_trial_name()
    # TuneReportCallback forwards every ray.train.report from the training workers to Tune,
    # which is what ASHA watches to stop weak trials early.
    result = ft.make_trainer(args, callbacks=[TuneReportCallback()]).fit()

    weights = Path(result.checkpoint.path) / "of3_ev71_ft.pt"
    # placement_group=None: run outside the trial's 1-CPU reservation, on a GPU of its own.
    scored = ray.get(predict_and_score.options(
        accelerator_type=args.accelerator,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None)).remote(
        str(weights), val, str(args.storage / args.name / "val_predictions")))
    stats = pose_stats(scored.of3tune_lig_rmsd)
    stats["val_failed"] = int(scored.of3tune_lig_rmsd.isna().sum())
    tune.report({**result.metrics, **stats, "weights": str(weights)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs", default="3e-5,1e-4,3e-4,1e-3", help="learning rates to try, one trial each")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--workers-per-trial", type=int, default=1, help="GPUs per trial")
    ap.add_argument("--max-concurrent", type=int, default=4, help="trials running at the same time")
    ap.add_argument("--grace-steps", type=int, default=60,
                    help="ASHA: every trial trains at least this many steps before it can be stopped")
    ap.add_argument("--no-early-stop", action="store_true", help="run every trial to --max-steps")
    ap.add_argument("--val-ligands", type=int, default=24, help="held-out early follow-on ligands")
    ap.add_argument("--accelerator", default="L4")
    ap.add_argument("--name", default=f"tune-of3-ev71-{time.strftime('%Y%m%d-%H%M%S')}")
    args = ap.parse_args()

    base = ft.parser().parse_args([])
    ft.check_inputs(base)
    ray.init()

    val = validation_set(args.val_ligands)
    print(f"Validation set: {len(val)} early follow-on ligands, {int(val.complexes.str.len().sum())} "
          f"crystal complexes (EV-A71 2A). Late follow-ons stay held out.")

    # The two starting points, scored on the same complexes from the benchmark results
    # (CPU only, no prediction needed).
    files = glob.glob(str(SHARED / "results" / TARGET / "*.parquet"))
    if files:
        bench = pd.concat([pd.read_parquet(f) for f in files])
        bench = bench[bench.complex_name.isin({c for cs in val.complexes for c in cs})]
        for key, label in (("openbind0", "OpenBind-0 (starting weights)"), ("of3ft", "released fine-tune (of3ft)")):
            if f"{key}_lig_rmsd" in bench:
                s = pose_stats(bench[f"{key}_lig_rmsd"])
                print(f"  {label:32s} {s['val_success']:6.1%} under 2 A, median {s['val_median_rmsd']:.2f} A")

    lrs = [float(x) for x in args.lrs.split(",")]
    scheduler = None if args.no_early_stop else ASHAScheduler(
        time_attr="step", metric="lddt_avg20", mode="max",
        # max_t one past the last step: ASHA stops any trial that reaches max_t, which would
        # cut off the validation that runs after training.
        max_t=args.max_steps + 1, grace_period=min(args.grace_steps, args.max_steps), reduction_factor=2)
    param_space = {
        "lr": tune.grid_search(lrs),
        "sweep": args.name,
        "base_args": {"max_steps": args.max_steps, "num_workers": args.workers_per_trial,
                      "accelerator": args.accelerator},
    }
    tuner = tune.Tuner(
        # The trial itself only drives the run: training and validation request their own GPUs.
        # with_parameters puts the validation set in the object store once for all trials.
        tune.with_resources(tune.with_parameters(trial, val=val), {"CPU": 1}),
        param_space=param_space,
        tune_config=tune.TuneConfig(scheduler=scheduler, max_concurrent_trials=args.max_concurrent),
        run_config=tune.RunConfig(name=args.name, storage_path=str(base.storage)),
    )
    start = time.time()
    grid = tuner.fit()

    rows = []
    for r in grid:
        m = r.metrics or {}
        rows.append({"lr": r.config["lr"], "steps": m.get("step"), "lddt_avg20": m.get("lddt_avg20"),
                     "val_success": m.get("val_success"), "val_median_rmsd": m.get("val_median_rmsd"),
                     "weights": m.get("weights"), "error": str(r.error)[:200] if r.error else None})
    df = pd.DataFrame(rows).sort_values("lr")
    print(f"\nDone in {(time.time() - start) / 60:.1f} min: {len(df)} trials")
    for row in df.itertuples():
        if row.error:
            print(f"  lr {row.lr:.0e}  failed: {row.error}")
        elif row.val_success is None or np.isnan(row.val_success):
            print(f"  lr {row.lr:.0e}  stopped early by ASHA at step {row.steps} (lDDT {row.lddt_avg20:.3f})")
        else:
            print(f"  lr {row.lr:.0e}  {row.steps} steps  lDDT {row.lddt_avg20:.3f}  "
                  f"validation {row.val_success:6.1%} under 2 A, median {row.val_median_rmsd:.2f} A")
    done = df.dropna(subset=["val_success"])
    if len(done):
        best = done.sort_values(["val_success", "val_median_rmsd"], ascending=[False, True]).iloc[0]
        print(f"\nBest: lr {best.lr:.0e}\n  weights: {best.weights}")
        print(f"  benchmark it: python aiddd_pipeline.py --models of3ray,openbind0 --of3ray-ckpt {best.weights}")
    out = base.storage / args.name / "summary.csv"
    df.to_csv(out, index=False)
    print(f"  table: {out}")


if __name__ == "__main__":
    main()
