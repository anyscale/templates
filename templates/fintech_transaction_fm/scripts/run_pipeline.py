"""Run the full transaction-FM pipeline end to end in one command.

Usage:
    python scripts/run_pipeline.py                      # smoke (CPU)
    python scripts/run_pipeline.py --scale small        # full TabFormer, GPU
    python scripts/run_pipeline.py --source synthetic   # offline data source

Also the Anyscale Job entrypoint (see job_config.yaml). Each stage runs as a
subprocess of its canonical script, so this file stays a thin orchestrator and
the per-stage scripts remain runnable on their own.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_stage(label: str, script: str, *args: str) -> None:
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script), *args]
    print(f"=== {label}: {' '.join(cmd[1:])} ===", flush=True)
    subprocess.run(cmd, check=True)


def fresh_artifact_dirs(base: str, scale: str) -> None:
    """Remove every stage output for this scale before running.

    Ray's write_parquet APPENDS into an existing directory, and a job retry
    reuses the same cluster storage — leftovers from a previous attempt
    silently double every downstream dataset. Only the scale-independent
    source/ download cache survives; stage 01 rebuilds the raw data from it.
    """
    import shutil

    for key, path in artifact_paths(base, scale).items():
        if key == "source":
            continue
        path = path.rstrip("/")
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            continue
        print(f"[clean] removed stale {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p, default="smoke")
    p.add_argument("--source", choices=["tabformer", "synthetic"], default="tabformer")
    p.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="reuse existing stage outputs instead of starting fresh — unsafe "
        "for full reruns (parquet stages append), only for debugging",
    )
    args = p.parse_args()

    load_scale(args.scale, args.scale_config)  # fail fast on a bad scale/config
    base = get_demo_base_dir()
    if not args.keep_artifacts:
        fresh_artifact_dirs(base, args.scale)

    # Every per-scale knob lives in configs/<scale>.yaml (or the file given
    # via --scale-config); each stage loads its own block, so we only forward
    # the flags.
    scale_args = ["--scale", args.scale]
    if args.scale_config:
        scale_args += ["--scale-config", args.scale_config]

    run_stage("[1/6] data", "01_generate_data.py", *scale_args, "--source", args.source)
    run_stage("[2/6] tokenize", "02_tokenize.py", *scale_args)
    run_stage("[3/6] pretrain", "03_pretrain.py", *scale_args)
    run_stage("[4/6] extract embeddings", "04_extract_embeddings.py", *scale_args)
    run_stage("[5/6] downstream fraud eval", "05_train_downstream.py", *scale_args)

    print("=== [6/6] validate ===", flush=True)
    from scripts.validate_results import print_report, validate_pipeline

    print_report(validate_pipeline(artifact_paths(base, args.scale)))


if __name__ == "__main__":
    main()
