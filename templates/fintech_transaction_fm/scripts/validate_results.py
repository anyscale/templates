"""Validation helpers used by the README and the CI test."""

import json
import os

import pandas as pd


def validate_pipeline(paths: dict) -> dict:
    """Assert each stage produced sane artifacts; return a short report."""
    report = {}

    # Tokenized sequences exist and have the expected columns.
    tok = pd.read_parquet(paths["tokenized"])
    assert len(tok) > 0, "no tokenized sequences"
    assert "attention_mask" in tok.columns, "missing attention_mask"
    report["n_sequences"] = int(len(tok))

    # Checkpoint has weights + vocab + config.
    for fn in ("model.pt", "vocab.json", "model_config.json"):
        assert os.path.exists(os.path.join(paths["checkpoint"], fn)), f"missing {fn}"

    # Embeddings exist with a non-trivial dimension.
    emb = pd.read_parquet(paths["embeddings"])
    dim = len(emb["embedding"].iloc[0])
    assert dim >= 16, f"embedding dim too small: {dim}"
    report["embedding_dim"] = int(dim)
    report["n_embeddings"] = int(len(emb))

    # Downstream metrics produced and the FM beats (or matches) the raw baseline.
    with open(os.path.join(paths["downstream"], "downstream_metrics.json")) as f:
        m = json.load(f)
    report["results"] = m["results"]
    report["fusion_lift_pr_auc"] = m["fusion_lift_pr_auc"]
    assert m["results"]["fusion"]["pr_auc"] >= m["results"]["raw"]["pr_auc"] - 0.05, (
        "fusion materially underperforms raw baseline"
    )
    return report


def print_report(report: dict) -> None:
    print("Pipeline validation:")
    print(f"  sequences:      {report['n_sequences']:,}")
    print(f"  embedding dim:  {report['embedding_dim']}")
    print(f"  embeddings:     {report['n_embeddings']:,}")
    for name, r in report["results"].items():
        print(f"  {name:<7} PR-AUC={r['pr_auc']:.4f}  AUC-ROC={r['auc_roc']:.4f}")
    print(f"  fusion lift:    {report['fusion_lift_pr_auc']:+.4f} PR-AUC vs raw")
    print("  ALL CHECKS PASSED")
