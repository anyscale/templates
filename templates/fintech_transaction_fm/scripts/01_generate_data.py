"""Step 1 — generate synthetic transaction data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generate_data import save_dataset  # noqa: E402
from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALE_MAP), default="small")
    p.add_argument("--base-dir", default=None)
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    if os.path.exists(paths["raw"]):
        print(f"[01] raw data exists -> {paths['raw']} (skipping)")
        return
    save_dataset(paths["raw"], num_cards=SCALE_MAP[args.scale])


if __name__ == "__main__":
    main()
