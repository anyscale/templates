"""
CPU post-processing stage for the protein embedding pipeline.

Handles:
  - Taxonomy join: broadcast join with organism metadata lookup table
  - Near-duplicate flagging: placeholder for cosine similarity dedup

Biology context:
  - Taxonomy is the hierarchical classification of organisms (domain -> class -> order -> species).
  - Joining taxonomy metadata onto embeddings lets downstream analyses filter/group
    by organism (e.g., "show me all human kinases" or "cluster bacterial vs. eukaryotic variants").
  - Near-duplicate detection flags sequences that are >95% similar in embedding space,
    which often indicates redundant entries from the same gene in different database releases.
"""
import numpy as np
import pandas as pd


def join_taxonomy(batch: dict, taxonomy_df: pd.DataFrame) -> dict:
    """Broadcast join: add taxonomy fields to each sequence based on organism_id.

    The taxonomy_df is small (~15 rows) and broadcast to all CPU workers via ray.put().
    This avoids shuffling the large embedding dataset.

    Adds columns: species, domain, class_name, order
    """
    organism_ids = batch["organism_id"]

    # Build a fast lookup dict from the taxonomy DataFrame
    tax_lookup = taxonomy_df.set_index("organism_id").to_dict("index")

    species_list = []
    domain_list = []
    class_list = []
    order_list = []

    for org_id in organism_ids:
        if isinstance(org_id, bytes):
            org_id = org_id.decode("utf-8")
        info = tax_lookup.get(org_id, {})
        species_list.append(info.get("species", "Unknown"))
        domain_list.append(info.get("domain", "Unknown"))
        class_list.append(info.get("class_name", "Unknown"))
        order_list.append(info.get("order", "Unknown"))

    batch["species"] = species_list
    batch["tax_domain"] = domain_list
    batch["tax_class"] = class_list
    batch["tax_order"] = order_list

    return batch


def flag_near_duplicates(batch: dict, threshold: float = 0.95) -> dict:
    """Flag sequences whose embeddings are near-duplicates by cosine similarity.

    This is a placeholder implementation that flags nothing — full pairwise cosine
    similarity within a batch is O(N^2) and best done as a separate offline step
    (e.g., using FAISS or ScaNN for approximate nearest neighbor search).

    In production, you would:
      1. Build a FAISS index from all embeddings
      2. Query each embedding for neighbors above the threshold
      3. Keep one representative per cluster

    For the demo, we add the column but leave it as False.
    """
    n = len(batch["sequence_id"])
    batch["is_near_duplicate"] = [False] * n
    return batch
