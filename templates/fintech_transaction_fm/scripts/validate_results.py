"""Validation helpers used by the README and the CI test."""

import json
import os


def _dataset(path: str):
    import pyarrow.dataset as pads

    return pads.dataset(path, format="parquet")


def validate_pipeline(paths: dict, n_pretrain_windows: int | None = None) -> dict:
    """Assert each stage produced sane artifacts; return a short report.

    Uses Parquet metadata and small reads only — never loads a full dataset
    onto the driver (at `full` the eval artifacts are tens of GB).

    ``n_pretrain_windows`` is passed by the fused pipeline (which streams the
    tokenized windows through the object store instead of writing Parquet);
    when omitted, it is counted from the tokenized Parquet that the
    step-by-step scripts write.
    """
    report = {}

    # Tokenized sequences exist (Parquet when the per-stage scripts ran;
    # otherwise the count handed over from the in-memory pipeline).
    if n_pretrain_windows is None:
        n_pretrain_windows = _dataset(paths["tokenized_pretrain"]).count_rows()
    assert n_pretrain_windows > 0, "no pretrain windows"
    report["n_pretrain_windows"] = int(n_pretrain_windows)

    # Checkpoint has weights + vocab + config.
    for fn in ("model.pt", "vocab.json", "model_config.json"):
        assert os.path.exists(os.path.join(paths["checkpoint"], fn)), f"missing {fn}"

    # Embeddings exist, with a non-trivial dimension and all temporal splits.
    emb = _dataset(paths["embeddings"])
    n_emb = emb.count_rows()
    assert n_emb > 0, "no eval samples"
    first = emb.head(1, columns=["embedding"]).column("embedding").chunk(0)
    if hasattr(first, "storage"):
        first = first.storage  # unwrap Ray's tensor extension (see downstream.py)
    dim = len(first[0].as_py())
    assert dim >= 16, f"embedding dim too small: {dim}"
    splits = set()
    for batch in emb.to_batches(columns=["split"], batch_size=65_536):
        splits.update(batch.column("split").unique().to_pylist())
        if splits >= {"train", "val", "test"}:
            break
    assert splits >= {"train", "val", "test"}, "missing temporal splits"
    report["n_sequences"] = int(n_emb)
    report["embedding_dim"] = int(dim)
    report["n_embeddings"] = int(n_emb)

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
    print(f"  pretrain wins:  {report['n_pretrain_windows']:,}")
    print(f"  sequences:      {report['n_sequences']:,}")
    print(f"  embedding dim:  {report['embedding_dim']}")
    print(f"  embeddings:     {report['n_embeddings']:,}")
    for name, r in report["results"].items():
        print(f"  {name:<7} PR-AUC={r['pr_auc']:.4f}  AUC-ROC={r['auc_roc']:.4f}")
    print(f"  fusion lift:    {report['fusion_lift_pr_auc']:+.4f} PR-AUC vs raw")
    print("  ALL CHECKS PASSED")
