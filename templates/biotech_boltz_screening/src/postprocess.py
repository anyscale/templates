"""
Post-processing: classify confidence tiers, add filter flags, extract top-K.

Confidence tiers map Boltz-1's aggregate confidence score to actionable categories:
  - high   (>0.8): Strong predicted interaction. Priority for wet-lab validation.
  - medium (0.5-0.8): Moderate confidence. May warrant further computational analysis.
  - low    (<=0.5): Weak or no predicted interaction. Deprioritize.
"""
import numpy as np


def classify_and_filter(batch: dict) -> dict:
    """CPU map_batches function: classify confidence tiers and add filter flags.

    Input columns: complex_id, plddt_mean, iptm, confidence, num_residues,
                   cif_bytes, runtime_sec
    Output columns: all input columns + confidence_tier, passed_filter
    """
    n = len(batch["complex_id"])

    confidence_tiers = []
    passed_filters = []

    for i in range(n):
        conf = float(batch["confidence"][i])
        plddt = float(batch["plddt_mean"][i])

        # Classify confidence tier
        if conf > 0.8:
            tier = "high"
        elif conf > 0.5:
            tier = "medium"
        else:
            tier = "low"

        confidence_tiers.append(tier)

        # A complex passes the filter if:
        # 1. Confidence is at least medium (>0.5)
        # 2. Mean pLDDT is above 50 (basic structural quality gate)
        # 3. The prediction actually produced results (confidence > 0)
        passed = conf > 0.5 and plddt > 50.0 and conf > 0.0
        passed_filters.append(passed)

    # Pass through all input columns and add new ones
    out = {}
    for key in batch:
        out[key] = batch[key]
    out["confidence_tier"] = confidence_tiers
    out["passed_filter"] = passed_filters

    return out


def extract_top_k(df, k: int = 10, sort_by: str = "confidence"):
    """Extract the top-K candidates sorted by the given metric.

    Args:
        df: pandas DataFrame with screening results.
        k: Number of top candidates to return.
        sort_by: Column to sort by (descending).

    Returns:
        DataFrame with the top-K rows.
    """
    return (
        df.sort_values(sort_by, ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
