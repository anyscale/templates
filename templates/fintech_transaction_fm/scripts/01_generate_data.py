"""Step 1 — prepare raw transaction data (real IBM TabFormer or synthetic)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALE_MAP), default="small")
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

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    if os.path.exists(paths["raw"]) and os.path.exists(paths["splits"]):
        print(f"[01] raw data exists -> {paths['raw']} (skipping)")
        return

    if args.source == "tabformer":
        from src.tabformer import prepare_tabformer

        prepare_tabformer(
            paths["raw"],
            paths["splits"],
            num_cards=SCALE_MAP[args.scale],
            seed=args.seed,
            source_dir=paths["source"],
        )
    else:
        from src.generate_data import save_dataset

        save_dataset(paths["raw"], num_cards=SCALE_MAP[args.scale], seed=args.seed)


if __name__ == "__main__":
    main()
