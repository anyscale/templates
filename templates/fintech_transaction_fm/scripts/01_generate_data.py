"""Step 1 — prepare raw transaction data (real IBM TabFormer or synthetic)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    p.add_argument(
        "--source",
        choices=["tabformer", "synthetic"],
        default="tabformer",
        help="tabformer = real IBM TabFormer data (downloads ~266MB once); "
        "synthetic = offline generator (no network needed)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_scale(args.scale, args.scale_config)
    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    have_raw = os.path.exists(paths["raw"]) and os.path.exists(paths["splits"])
    if have_raw:
        print(f"[01] raw data exists -> {paths['raw']} (skipping normalize)")

    if args.source == "tabformer":
        from src.tabformer import build_benchmark, prepare_tabformer

        if not have_raw:
            prepare_tabformer(
                paths["raw"],
                paths["splits"],
                num_cards=cfg["data"]["num_cards"],
                seed=args.seed,
                source_dir=paths["source"],
            )
        # NVIDIA-protocol benchmark rows (exact repro split/sampling) — the
        # eval set every later stage keys off. Sizes overridable per scale.
        if not os.path.exists(paths["benchmark"]):
            from src.nvidia_baseline import HOLDOUT_N, TRAIN_N

            bench_cfg = cfg.get("benchmark") or {}
            build_benchmark(
                paths["raw"],
                paths["benchmark"],
                train_n=bench_cfg.get("train_n", TRAIN_N),
                holdout_n=bench_cfg.get("holdout_n", HOLDOUT_N),
            )
        else:
            print(f"[01] benchmark exists -> {paths['benchmark']} (skipping)")
    elif not have_raw:
        from src.generate_data import save_dataset

        save_dataset(paths["raw"], num_cards=cfg["data"]["num_cards"], seed=args.seed)


if __name__ == "__main__":
    main()
