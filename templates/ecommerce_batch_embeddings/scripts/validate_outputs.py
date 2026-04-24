"""
Validates embedding pipeline output.
- Checks embedding shape and stats
- Runs cosine similarity search for 3 demo queries
- Prints top-5 most similar products per query

Usage:
  python scripts/validate_outputs.py --output /mnt/cluster_storage/ecommerce-demo/embeddings/medium/
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


DEMO_QUERIES = [
    "wireless bluetooth headphones noise cancelling",
    "women's running shoes lightweight breathable",
    "daily face moisturizer SPF sensitive skin",
]


def load_embeddings(output_path: str) -> pd.DataFrame:
    """Load all Parquet shards from the output directory into a single DataFrame."""
    import glob
    files = glob.glob(os.path.join(output_path, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found at {output_path}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} products with embeddings")
    return df


def print_embedding_stats(df: pd.DataFrame):
    embeddings = np.array(df["embedding"].tolist())
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Sample norms (should be ~1.0 since normalized): {np.linalg.norm(embeddings[:5], axis=1).round(4)}")


def encode_query(query: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec = model.encode([query], normalize_embeddings=True)
    return vec[0]


def similarity_search(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Return top-k products most similar to query via dot product (cosine sim on normalized vecs)."""
    query_vec = encode_query(query)
    embeddings = np.array(df["embedding"].tolist())
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices][["product_id", "title", "category", "price"]].copy()
    results["score"] = scores[top_indices].round(4)
    return results.reset_index(drop=True)


def run_validation(output_path: str):
    df = load_embeddings(output_path)
    print_embedding_stats(df)

    print("\nLoading query encoder for similarity search...")
    for query in DEMO_QUERIES:
        print(f"\n{'─' * 60}")
        print(f"Query: \"{query}\"")
        print(f"{'─' * 60}")
        results = similarity_search(query, df)
        for _, row in results.iterrows():
            print(f"  [{row['score']:.4f}] {row['title']} — ${row['price']:.2f} ({row['category']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/mnt/cluster_storage/ecommerce-demo/embeddings/medium/",
    )
    args = parser.parse_args()
    run_validation(args.output)
