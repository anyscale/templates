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

from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402

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
    p.add_argument("--scale", choices=list(SCALE_MAP), default="smoke")
    p.add_argument("--source", choices=["tabformer", "synthetic"], default="tabformer")
    p.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="reuse existing stage outputs instead of starting fresh — unsafe "
        "for full reruns (parquet stages append), only for debugging",
    )
    args = p.parse_args()

    base = get_demo_base_dir()
    if not args.keep_artifacts:
        fresh_artifact_dirs(base, args.scale)

    # smoke runs on CPU; small/full use the GPUs (03 picks GPU via its own
    # per-scale presets; 04 needs the flags explicitly). All GPU replicas live
    # on the single 4xGPU worker — fat batches, no extra node spin-up. `full`
    # embeds every holdout transaction (~5M windows), so it uses all 4 GPUs.
    if args.scale == "smoke":
        embed_args = ["--num-workers", "8"]
    elif args.scale == "small":
        embed_args = ["--use-gpu", "--num-workers", "2", "--batch-size", "2048"]
    else:
        # seq 512: attention buffers cap the T4 inference batch well below
        # the seq-128 setting.
        embed_args = ["--use-gpu", "--num-workers", "4", "--batch-size", "256"]

    run_stage("[1/6] data", "01_generate_data.py", "--scale", args.scale, "--source", args.source)
    run_stage("[2/6] tokenize", "02_tokenize.py", "--scale", args.scale)
    run_stage("[3/6] pretrain", "03_pretrain.py", "--scale", args.scale)
    run_stage("[4/6] extract embeddings", "04_extract_embeddings.py", "--scale", args.scale, *embed_args)
    run_stage("[5/6] downstream fraud eval", "05_train_downstream.py", "--scale", args.scale)

    print("=== [6/6] validate ===", flush=True)
    from scripts.validate_results import print_report, validate_pipeline

    print_report(validate_pipeline(artifact_paths(base, args.scale)))


if __name__ == "__main__":
    main()
