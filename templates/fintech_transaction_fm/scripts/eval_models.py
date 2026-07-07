"""Evaluate several trained models against a shared eval set in ONE process.

The raw (~13-feature NVIDIA-style) baseline is identical across models, so it's
trained once; only each model's fm/fusion re-runs. The raw parquet is joined and
encoded once. Requires the models to share the eval set (same eval_fingerprint,
asserted) so rows align position-for-position.

Usage:
    python scripts/eval_models.py --scales full,xl,xxl \
        --base-dir /mnt/user_storage/transaction-fm
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.downstream import print_summary, run_downstream_multi  # noqa: E402
from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", required=True, help="comma-separated, e.g. full,xl,xxl")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--out", default=None, help="output dir (default: <base>/downstream/compare)")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    models = {s: artifact_paths(base, s)["embeddings"] for s in scales}
    raw_path = artifact_paths(base, scales[0])["raw"]  # identical data across scales
    out = args.out or os.path.join(base, "downstream", "compare")

    summary = run_downstream_multi(models, raw_path, out)
    print_summary(summary)
    print(f"\n[eval_models] comparison -> {out}/compare_metrics.json")


if __name__ == "__main__":
    main()
