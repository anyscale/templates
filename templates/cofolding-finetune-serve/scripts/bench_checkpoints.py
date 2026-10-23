"""Benchmark every fine-tuned checkpoint against every target, as Anyscale jobs.

    python scripts/bench_checkpoints.py --dry-run          # print the matrix, submit nothing
    python scripts/bench_checkpoints.py                    # submit one job per (run, target)
    python scripts/bench_checkpoints.py --runs ft-b2-ev71sars2-frags-105

Each run of finetune_ray_train.py leaves its weights under
$AIDDD_SHARED/finetune/runs/<name>/checkpoint_*/of3_ev71_ft.pt. This submits one
benchmark job per checkpoint and target, so the ablation (which data was trained on)
can be read across a single table.

Every job gets its own results directory ($AIDDD_SHARED/results_<run>/<target>) and a fresh,
time-stamped prediction directory ($AIDDD_SHARED/predictions_<run>/<stamp>). Both matter:

    results      aiddd_pipeline.py deletes its output directory before writing, so two
                 runs sharing one destroys the first run's scores.
    predictions  poses are cached per ligand under the *model key*, which is `of3ray`
                 for every fine-tuned checkpoint. Sharing a prediction directory means
                 the second checkpoint silently reuses the first one's poses and scores
                 identically to it -- which looks like "fine-tuning changed nothing".
                 Re-benchmarking the same run into the same directory is just as bad: it
                 reads every pose from the cache, so its timing measures nothing.

Runs named ft-p2-* (finetune_ray_train.py --base p2) start from OF3-p2 weights and are
predicted with the legacy OpenFold3 (--of3ray-legacy).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiddd_pipeline import SHARED, TARGETS  # noqa: E402

RUNS = SHARED / "finetune" / "runs"


def checkpoint_of(run: str) -> Path | None:
    """Weights exported by a finished run, newest checkpoint directory first."""
    found = sorted((RUNS / run).glob("checkpoint_*/of3_ev71_ft.pt"), reverse=True)
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None,
                    help="comma list of run names (default: every run with a checkpoint)")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--max-gpus", type=int, default=8,
                    help="per job. Several jobs run at once, so keep this well under the quota")
    ap.add_argument("--job-file", default="jobs/job.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = ([r.strip() for r in args.runs.split(",")] if args.runs
            else sorted(p.name for p in RUNS.iterdir() if p.is_dir()))
    targets = [t.strip() for t in args.targets.split(",")]
    assert all(t in TARGETS for t in targets), targets

    planned, skipped = [], []
    for run in runs:
        ckpt = checkpoint_of(run)
        if ckpt is None:
            skipped.append(run)
            continue
        for target in targets:
            planned.append((run, target, ckpt))

    if skipped:
        print(f"no checkpoint yet, skipping: {', '.join(skipped)}\n")
    stamp = time.strftime("%m%d-%H%M")
    for run, target, ckpt in planned:
        out = SHARED / f"results_{run}" / target
        preds = SHARED / f"predictions_{run}" / stamp
        name = f"bench-{run}-{target}"[:60]
        cmd = ["anyscale", "job", "submit", "-f", args.job_file, "--name", name, "--",
               "python", "aiddd_pipeline.py", "--target", target, "--models", "of3ray",
               "--max-gpus", str(args.max_gpus), "--of3ray-ckpt", str(ckpt),
               "--of3-per-gpu", "2", "--cpus-per-gpu-node", "16",   # g6.4xlarge, jobs/job.yaml
               "--pred-dir", str(preds), "--out", str(out)]
        if run.startswith("ft-p2-"):
            cmd.append("--of3ray-legacy")
        print(f"{name}\n    ckpt  {ckpt}\n    out   {out}\n    preds {preds}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    print(f"\n{len(planned)} job(s) {'planned' if args.dry_run else 'submitted'}")


if __name__ == "__main__":
    main()
