"""Validation helpers used by the README and the CI test."""

import json
import os


def _dataset(path: str):
    import pyarrow.dataset as pads

    return pads.dataset(path, format="parquet")


def validate_pipeline(
    paths: dict, n_pretrain_windows: int | None = None, strict_lift: bool = True
) -> dict:
    """Assert each stage produced sane artifacts; return a short report.

    Uses Parquet metadata and small reads only — never loads a full dataset
    onto the driver (at `full` the eval artifacts are tens of GB).

    ``n_pretrain_windows`` is passed by the fused pipeline (which streams the
    tokenized windows through the object store instead of writing Parquet);
    when omitted, it is counted from the tokenized Parquet that the
    step-by-step scripts write.

    ``strict_lift`` enforces "fusion ≥ raw" — a *quality* property that only
    holds once the FM is actually trained (``small``/``full``). At the tiny
    smoke/test scales the FM is deliberately undertrained (a couple of CPU
    epochs), so fusion can sit on either side of raw on a handful of test
    positives; there we check only that the metrics are sane.
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

    # Downstream metrics produced and the FM beats (or matches) the baseline.
    bench_path = os.path.join(paths["downstream"], "benchmark_metrics.json")
    if os.path.exists(bench_path):
        # NVIDIA-protocol path (tabformer): baseline / embedding readouts.
        with open(bench_path) as f:
            m = json.load(f)
        report["results"] = m["results"]
        report["embed_xgb_lift_ap_pct"] = m.get("embed_xgb_lift_ap_pct")
        report["strict_lift"] = strict_lift
        aps = {k: r["ap"] for k, r in m["results"].items()}
        assert all(0.0 <= v <= 1.0 for v in aps.values()), f"ap out of range: {aps}"
        if strict_lift and "embed_xgb" in m["results"]:
            assert m["results"]["embed_xgb"]["ap"] >= m["results"]["baseline"]["ap"] - 0.05, (
                "embed_xgb materially underperforms the NVIDIA baseline"
            )
        return report
    with open(os.path.join(paths["downstream"], "downstream_metrics.json")) as f:
        m = json.load(f)
    report["results"] = m["results"]
    report["fusion_lift_pr_auc"] = m["fusion_lift_pr_auc"]
    report["strict_lift"] = strict_lift
    pr = {k: m["results"][k]["pr_auc"] for k in ("raw", "fm", "fusion")}
    assert all(0.0 <= v <= 1.0 for v in pr.values()), f"pr_auc out of range: {pr}"
    if strict_lift:
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
        ap = r["ap"] if "ap" in r else r["pr_auc"]
        print(f"  {name:<10} AP={ap:.4f}  AUC-ROC={r['auc_roc']:.4f}")
    if "embed_xgb_lift_ap_pct" in report:
        if report["embed_xgb_lift_ap_pct"] is not None:
            print(f"  embed_xgb lift: {report['embed_xgb_lift_ap_pct']:+.2f}% AP vs baseline")
    else:
        print(f"  fusion lift:    {report['fusion_lift_pr_auc']:+.4f} PR-AUC vs raw")
    if not report.get("strict_lift", True):
        print("  (fusion≥raw not enforced at this scale — FM is undertrained)")
    print("  ALL CHECKS PASSED")
