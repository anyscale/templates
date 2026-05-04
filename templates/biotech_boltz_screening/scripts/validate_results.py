"""
Validate Boltz-1 screening output: confidence distribution and top-10 candidates.

Usage:
  python scripts/validate_results.py --input /mnt/cluster_storage/boltz-screening/results/medium/
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def load_results(input_path: str) -> pd.DataFrame:
    """Load scored Parquet output (directory of part files or single file)."""
    if os.path.isdir(input_path):
        parts = [f for f in os.listdir(input_path) if f.endswith(".parquet")]
        dfs = [pd.read_parquet(os.path.join(input_path, p)) for p in sorted(parts)]
        return pd.concat(dfs, ignore_index=True)
    return pd.read_parquet(input_path)


def print_confidence_distribution(df: pd.DataFrame):
    """Print confidence tier distribution."""
    dist = df["confidence_tier"].value_counts()
    total = len(df)
    print(f"\n{'=' * 50}")
    print("  CONFIDENCE TIER DISTRIBUTION")
    print(f"{'=' * 50}")
    for tier in ["high", "medium", "low"]:
        count = dist.get(tier, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {tier:<10} {count:>8,}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<10} {total:>8,}")
    print(f"{'=' * 50}")

    passed = df["passed_filter"].sum() if "passed_filter" in df.columns else 0
    print(f"\n  Candidates passing filter: {passed:,} / {total:,}")


def print_top_candidates(df: pd.DataFrame, k: int = 10):
    """Print top-K candidates by confidence score."""
    top = df.sort_values("confidence", ascending=False).head(k)
    print(f"\n{'=' * 70}")
    print(f"  TOP-{k} CANDIDATES BY CONFIDENCE")
    print(f"{'=' * 70}")
    print(f"  {'Complex ID':<14} {'Confidence':>10} {'ipTM':>8} {'pLDDT':>8} {'Tier':<8} {'Residues':>8}")
    print(f"  {'-' * 64}")
    for _, row in top.iterrows():
        print(
            f"  {row['complex_id']:<14} "
            f"{row['confidence']:>10.4f} "
            f"{row['iptm']:>8.4f} "
            f"{row['plddt_mean']:>8.1f} "
            f"{row['confidence_tier']:<8} "
            f"{row['num_residues']:>8}"
        )
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to scored output directory")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidates to show")
    args = parser.parse_args()

    df = load_results(args.input)
    print(f"Loaded {len(df):,} scored complexes")
    print_confidence_distribution(df)
    print_top_candidates(df, k=args.top_k)


if __name__ == "__main__":
    main()
