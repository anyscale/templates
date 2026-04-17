"""
Validates protein embedding pipeline output.
- Checks embedding shape and stats
- Computes cosine similarity for homolog vs. random pairs
- Prints clear separation between homologs (high sim) and random pairs (low sim)

Usage:
  python scripts/validate_outputs.py --output /mnt/cluster_storage/protein-embeddings/embeddings/medium/
  python scripts/validate_outputs.py --output /path/to/embeddings --pairs /path/to/homolog_test_pairs.csv
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def load_embeddings(output_path: str) -> pd.DataFrame:
    """Load all Parquet shards from the output directory into a single DataFrame."""
    import glob
    files = glob.glob(os.path.join(output_path, "*.parquet"))
    if not files:
        # Try as a single file
        if os.path.isfile(output_path):
            files = [output_path]
        else:
            raise FileNotFoundError(f"No parquet files found at {output_path}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} sequences with embeddings")
    return df


def print_embedding_stats(df: pd.DataFrame):
    """Print summary statistics about the embeddings."""
    embeddings = np.array(df["embedding"].tolist())
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Norm stats: mean={norms.mean():.4f}, std={norms.std():.4f}, "
          f"min={norms.min():.4f}, max={norms.max():.4f}")

    if "length_bucket" in df.columns:
        print(f"\nLength bucket distribution:")
        print(df["length_bucket"].value_counts().sort_index().to_string())


def compute_pair_similarities(df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity for labeled sequence pairs.

    Returns the pairs DataFrame with an added 'cosine_sim' column.
    Homolog pairs (same protein family) should have higher similarity than random pairs.
    """
    # Build embedding lookup
    emb_lookup = {}
    for _, row in df.iterrows():
        emb_lookup[row["sequence_id"]] = np.array(row["embedding"])

    sims = []
    for _, pair in pairs_df.iterrows():
        id_a = pair["seq_id_a"]
        id_b = pair["seq_id_b"]

        if id_a not in emb_lookup or id_b not in emb_lookup:
            sims.append(np.nan)
            continue

        vec_a = emb_lookup[id_a]
        vec_b = emb_lookup[id_b]

        # Cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(vec_a, vec_b) / (norm_a * norm_b)))

    pairs_df = pairs_df.copy()
    pairs_df["cosine_sim"] = sims
    return pairs_df


def print_pair_stats(pairs_df: pd.DataFrame):
    """Print summary statistics for homolog vs. random pair similarities."""
    homologs = pairs_df[pairs_df["relationship"] == "homolog"]["cosine_sim"].dropna()
    randoms = pairs_df[pairs_df["relationship"] == "random"]["cosine_sim"].dropna()

    width = 52
    print("\n" + "=" * width)
    print("  EMBEDDING VALIDATION: HOMOLOG vs RANDOM PAIRS")
    print("=" * width)
    print(f"  {'Metric':<30} {'Homolog':<10} {'Random':<10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Count':<30} {len(homologs):<10} {len(randoms):<10}")
    print(f"  {'Mean cosine similarity':<30} {homologs.mean():<10.4f} {randoms.mean():<10.4f}")
    print(f"  {'Std cosine similarity':<30} {homologs.std():<10.4f} {randoms.std():<10.4f}")
    print(f"  {'Min cosine similarity':<30} {homologs.min():<10.4f} {randoms.min():<10.4f}")
    print(f"  {'Max cosine similarity':<30} {homologs.max():<10.4f} {randoms.max():<10.4f}")
    print("=" * width)

    if homologs.mean() > randoms.mean():
        diff = homologs.mean() - randoms.mean()
        print(f"\n  Homolog pairs are {diff:.4f} more similar on average than random pairs.")
        print(f"  This confirms the embeddings capture protein family relationships.")
    else:
        print(f"\n  WARNING: Homolog pairs are NOT more similar than random pairs.")
        print(f"  This may indicate a problem with the embeddings or corpus generation.")


def run_validation(output_path: str, pairs_path: str):
    """Full validation pipeline."""
    df = load_embeddings(output_path)
    print_embedding_stats(df)

    if os.path.exists(pairs_path):
        pairs_df = pd.read_csv(pairs_path)
        pairs_df = compute_pair_similarities(df, pairs_df)
        print_pair_stats(pairs_df)
    else:
        print(f"\nHomolog test pairs not found at {pairs_path} — skipping pair validation.")
        print("Run scripts/01_generate_corpus.py to generate test pairs.")

    # Print sample embeddings
    print(f"\nSample embeddings (first 3 sequences):")
    for _, row in df.head(3).iterrows():
        emb = np.array(row["embedding"])
        print(f"  {row['sequence_id']}: length={row.get('length', 'N/A')}, "
              f"embedding[0:5]={emb[:5].round(4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate protein embedding outputs")
    parser.add_argument(
        "--output",
        default="/mnt/cluster_storage/protein-embeddings/embeddings/medium/",
        help="Path to output Parquet directory",
    )
    parser.add_argument(
        "--pairs",
        default="/mnt/cluster_storage/protein-embeddings/raw/homolog_test_pairs.csv",
        help="Path to homolog test pairs CSV",
    )
    args = parser.parse_args()
    run_validation(args.output, args.pairs)
