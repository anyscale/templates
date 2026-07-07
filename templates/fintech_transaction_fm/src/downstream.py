"""Downstream fraud classification — the headline result.

Evaluation protocol matches NVIDIA's transaction-FM blueprint on TabFormer so
the numbers are directly comparable when run on the real data:

* **Temporal 80/10/10 split** by transaction time (cutoffs from splits.json —
  the same ones the tokenizer used). Train on the past, early-stop on val,
  report on the most recent 10%. No temporal leakage.
* **Per-transaction, last-event labels**: each sample is one target transaction
  scored from the window of history ending at it.
* **Metrics: AUC-ROC and PR-AUC (Average Precision)** — at ~0.1% fraud, AUC-ROC
  saturates and PR-AUC is the operationally meaningful number (NVIDIA frames it
  the same way).

We compare three feature sets with the SAME XGBoost recipe so the only variable
is the representation:

1. ``raw``    — ~13 hand-engineered fields of the target transaction (NVIDIA's
   XGBoost baseline: amount, time, MCC, plus merchant/channel/geo/issuer
   categoricals). Recovered by joining the raw transactions on
   (card_id, timestamp); categoricals are target-encoded (smoothed train fraud
   rate) fit on the TRAIN split only. This is the "what you have today" bar.
2. ``fm``     — the FM embedding of the history window only (no raw fields).
3. ``fusion`` — embedding concatenated with the raw fields (Nubank joint fusion).

The lift of (2)/(3) over (1) is the story. With the full baseline, AUC-ROC
saturates near the ceiling (little headroom), so PR-AUC / AP lift is the metric
that matters — same as NVIDIA's +41.76% AP framing.

``run_downstream_multi`` evaluates several models in one process: the ``raw``
baseline is identical across them (same target set), so it is trained once and
only ``fm``/``fusion`` re-run per model — and the raw join/encode happens once.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_SPLITS = ("train", "val", "test")

# NVIDIA-style hand-engineered baseline, joined from the raw transactions.
# Numerics enter directly (amount log-scaled); categoricals are target-encoded
# (smoothed train fraud rate) so each merchant/state/channel carries its
# fraud propensity. NVIDIA's exact 13-feature list isn't published; this is the
# obvious TabFormer set (12 features).
_RAW_NUM = ("amount", "hour", "day_of_week", "mcc")
_RAW_CAT = ("merchant_id", "merchant_category", "channel", "error",
            "issuer", "bin_region", "card_type", "home_state")


def _fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, w_te) -> tuple:
    import xgboost as xgb

    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        scale_pos_weight=(neg / max(pos, 1.0)),
        random_state=0,  # pinned: subsampled fits at 0.1% prevalence vary a LOT
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    # Weighted metrics undo the normal-downsampling: they estimate performance
    # at the natural fraud prevalence (what NVIDIA's blueprint reports), not on
    # the fraud-enriched sample we kept for compute reasons.
    metrics = {
        "auc_roc": float(roc_auc_score(y_te, proba, sample_weight=w_te)),
        "pr_auc": float(average_precision_score(y_te, proba, sample_weight=w_te)),
        "pr_auc_sampled": float(average_precision_score(y_te, proba)),
    }
    return metrics, proba


def _load_sorted(embeddings_path: str, want_fm: bool = True) -> dict:
    """Stream the embeddings Parquet into preallocated arrays, canonically sorted.

    pandas would materialize the embedding column as millions of small object
    arrays and vstack would copy them again (~3x peak). Streaming record
    batches into one preallocated float32 matrix keeps the peak at ~1x.

    Rows are put in canonical (card_id, raw_ts, amount, mcc) order so results
    are a function of the embedding *set*, not of parallel-writer interleaving —
    and so multiple models' rows align position-for-position (same target set).
    Returns cid/ts as well so the raw features can be joined back.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    cols = ["raw_amount", "raw_hour", "raw_dow", "raw_mcc",
            "label", "weight", "split", "card_id", "raw_ts"]
    if want_fm:
        cols = ["embedding"] + cols
    dset = pads.dataset(embeddings_path, format="parquet")
    n = dset.count_rows()
    split_values = pa.array(_SPLITS)

    X_fm = None
    amt = np.empty(n, np.float64)
    hour = np.empty(n, np.float32)
    dow = np.empty(n, np.float32)
    mcc = np.empty(n, np.float32)
    y = np.empty(n, np.int64)
    w = np.empty(n, np.float64)
    split_code = np.empty(n, np.int8)
    cid = np.empty(n, np.int64)
    ts = np.empty(n, np.int64)

    i = 0
    for batch in dset.to_batches(columns=cols, batch_size=32_768):
        m = batch.num_rows
        if m == 0:
            continue
        if want_fm:
            emb = batch.column("embedding")
            if hasattr(emb, "storage"):
                emb = emb.storage  # unwrap Ray tensor extension array
            flat = emb.flatten().to_numpy(zero_copy_only=False)
            if X_fm is None:
                X_fm = np.empty((n, len(flat) // m), np.float32)
            X_fm[i : i + m] = flat.reshape(m, -1)
        amt[i : i + m] = batch.column("raw_amount").to_numpy(zero_copy_only=False)
        hour[i : i + m] = batch.column("raw_hour").to_numpy(zero_copy_only=False)
        dow[i : i + m] = batch.column("raw_dow").to_numpy(zero_copy_only=False)
        mcc[i : i + m] = batch.column("raw_mcc").to_numpy(zero_copy_only=False)
        y[i : i + m] = batch.column("label").to_numpy(zero_copy_only=False)
        w[i : i + m] = batch.column("weight").to_numpy(zero_copy_only=False)
        codes = pc.fill_null(pc.index_in(batch.column("split"), value_set=split_values), -1)
        split_code[i : i + m] = codes.to_numpy(zero_copy_only=False)
        cid[i : i + m] = batch.column("card_id").to_numpy(zero_copy_only=False)
        ts[i : i + m] = batch.column("raw_ts").to_numpy(zero_copy_only=False)
        i += m
    assert i == n, f"read {i} rows, expected {n}"

    order = np.lexsort((mcc, amt, ts, cid))
    out = {
        "cid": cid[order], "ts": ts[order],
        "amt": amt[order], "hour": hour[order], "dow": dow[order], "mcc": mcc[order],
        "y": y[order], "w": w[order], "split_code": split_code[order],
    }
    if want_fm:
        out["X_fm"] = X_fm[order]
    return out


def _join_raw_features(cid, ts, y, raw_path: str, train_mask) -> tuple:
    """Recover NVIDIA-style hand-engineered features for each eval target.

    Joins the raw transactions on (card_id, timestamp) to pull the fields the
    embeddings don't carry (merchant, channel, geo, issuer, ...). Categoricals
    are frequency-encoded with counts fit on the TRAIN split ONLY — leakage-free
    (a fraud-rate/target encoding on all splits would inflate the baseline).
    """
    import pandas as pd
    import pyarrow.dataset as pads

    cols = ["card_id", "timestamp", *(_RAW_NUM[0:1]), "hour", "day_of_week", "mcc", *_RAW_CAT]
    cols = list(dict.fromkeys(cols))  # dedupe while preserving order
    raw = pads.dataset(raw_path, format="parquet").to_table(columns=cols).to_pandas()
    # Match the tokenizer's raw_ts: it casts timestamp to datetime64[s] (seconds
    # since epoch) — see tokenizer.py. The raw parquet is timestamp[us], so
    # normalize to seconds here too (a raw int64 cast is microseconds -> 0 match).
    raw["_ts"] = raw["timestamp"].to_numpy().astype("datetime64[s]").astype("int64")
    # (card_id, ts) is the target key; drop rare same-second dupes deterministically.
    raw = raw.drop_duplicates(["card_id", "_ts"], keep="first").set_index(["card_id", "_ts"])
    j = raw.reindex(pd.MultiIndex.from_arrays([np.asarray(cid), np.asarray(ts)]))

    matched = float(j["amount"].notna().mean())
    if matched < 0.99:
        raise RuntimeError(
            f"raw join matched only {matched:.1%} of eval targets on (card_id, "
            "raw_ts) — the raw_ts units likely differ from raw timestamp; check "
            "the tokenizer's raw_ts vs raw parquet 'timestamp' (int64 cast)."
        )

    feats, names = [], []
    # User + Card identity (ordinal), as in NVIDIA's baseline FEATURE_COLS. This
    # is the dominant fraud signal on TabFormer (fraud clusters by user, same
    # users span the temporal split) and the reason their raw baseline hits
    # ~0.99. card_id = User*100 + Card (see tabformer.py), so recover both here.
    cidv = np.asarray(cid).astype(np.int64)
    feats.append((cidv // 100).astype(np.float32)); names.append("user")
    feats.append((cidv % 100).astype(np.float32)); names.append("card")
    amt = j["amount"].to_numpy(np.float64)
    feats.append(np.sign(amt) * np.log1p(np.abs(amt))); names.append("log_amount")
    for c in ("hour", "day_of_week", "mcc"):
        feats.append(j[c].to_numpy(np.float32)); names.append(c)
    # Smoothed target (fraud-rate) encoding, fit on TRAIN only: each category ->
    # its train fraud rate, shrunk toward the global rate for rare categories.
    # This injects the categorical fraud signal a strong baseline needs;
    # frequency encoding threw it away. Train-only so test/val metrics don't
    # leak their own labels (val early-stopping guards the train overfit).
    tm = np.asarray(train_mask)
    yv = np.asarray(y).astype(np.float64)
    global_rate = float(yv[tm].mean())
    SMOOTH = 20.0
    for c in _RAW_CAT:
        vals = j[c].astype("object").to_numpy()
        stats = pd.DataFrame({"v": vals[tm], "y": yv[tm]}).groupby("v")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + SMOOTH * global_rate) / (stats["count"] + SMOOTH)
        feats.append(pd.Series(vals).map(enc).fillna(global_rate).to_numpy(np.float32))
        names.append(f"{c}_te")
    X = np.column_stack(feats).astype(np.float32)
    return X, names


def _eval_fingerprint(embeddings_path: str) -> str:
    """Order-independent hash of eval-set membership (card_id, ts, split, label).

    Metrics from two runs are comparable iff their fingerprints match — and the
    multi-model path REQUIRES equal fingerprints so the shared raw baseline and
    per-model embeddings row-align.
    """
    import hashlib

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    t = pads.dataset(embeddings_path, format="parquet").to_table(
        columns=["card_id", "raw_ts", "split", "label"]
    )
    cols = [
        t.column("card_id").to_numpy(zero_copy_only=False).astype(np.int64),
        t.column("raw_ts").to_numpy(zero_copy_only=False).astype(np.int64),
        pc.index_in(t.column("split"), value_set=pa.array(_SPLITS))
        .to_numpy(zero_copy_only=False)
        .astype(np.int64),
        t.column("label").to_numpy(zero_copy_only=False).astype(np.int64),
    ]
    order = np.lexsort(cols[::-1])
    return hashlib.sha256(np.column_stack(cols)[order].tobytes()).hexdigest()[:16]


def _masks(split_code) -> dict:
    masks = {s: split_code == c for c, s in enumerate(_SPLITS)}
    for s, m in masks.items():
        if m.sum() == 0:
            raise RuntimeError(
                f"split '{s}' is empty — re-run 01/02 so splits.json temporal "
                "cutoffs are written and applied during tokenization"
            )
    return masks


def _run_sets(feature_sets: dict, y, w, masks: dict) -> tuple:
    tr, va, te = masks["train"], masks["val"], masks["test"]
    results, proba = {}, {}
    for name, fx in feature_sets.items():
        results[name], proba[name] = _fit_eval(
            fx(tr), y[tr], fx(va), y[va], fx(te), y[te], w[te]
        )
    return results, proba


def _lifts(results: dict) -> dict:
    out = {}
    if "raw" in results:
        for k in ("fm", "fusion"):
            if k in results:
                out[f"{k}_lift_pr_auc"] = results[k]["pr_auc"] - results["raw"]["pr_auc"]
                out[f"{k}_lift_auc_roc"] = results[k]["auc_roc"] - results["raw"]["auc_roc"]
    return out


def run_downstream(embeddings_path: str, output_dir: str, raw_path: str | None = None) -> dict:
    """Train + evaluate raw/fm/fusion for one model; persist a metrics summary.

    ``raw_path`` (the raw transactions parquet) enables the full ~13-feature
    NVIDIA-style baseline via join. If omitted, falls back to the 4 fields the
    embeddings carry (amount, hour, dow, mcc) — weaker, kept for compatibility.
    """
    d = _load_sorted(embeddings_path, want_fm=True)
    masks = _masks(d["split_code"])
    y, w, X_fm = d["y"], d["w"], d["X_fm"]

    if raw_path:
        X_raw, raw_feats = _join_raw_features(d["cid"], d["ts"], y, raw_path, masks["train"])
    else:
        X_raw = np.column_stack(
            [np.sign(d["amt"]) * np.log1p(np.abs(d["amt"])), d["hour"], d["dow"], d["mcc"]]
        ).astype(np.float32)
        raw_feats = ["log_amount", "hour", "day_of_week", "mcc"]

    feature_sets = {
        "raw": lambda m: X_raw[m],
        "fm": lambda m: X_fm[m],
        "fusion": lambda m: np.hstack([X_fm[m], X_raw[m]]),
    }
    results, test_proba = _run_sets(feature_sets, y, w, masks)

    summary = {
        "protocol": (
            "temporal 80/10/10 split by transaction time; per-transaction "
            "last-event fraud labels; prevalence-weighted metrics on the "
            "held-out most-recent 10% (NVIDIA transaction-FM blueprint protocol)"
        ),
        "eval_fingerprint": _eval_fingerprint(embeddings_path),
        "raw_features": raw_feats,
        "n_samples": {s: int(m.sum()) for s, m in masks.items()},
        "fraud_rate": {s: float(y[m].mean()) for s, m in masks.items()},
        "natural_fraud_rate": {
            s: float((w[m] * y[m]).sum() / w[m].sum()) for s, m in masks.items()
        },
        "embedding_dim": int(X_fm.shape[1]),
        "results": results,
        **_lifts(results),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "downstream_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_test_predictions(output_dir, test_proba, y, w, masks["test"])
    return summary


def run_downstream_multi(models: dict, raw_path: str, output_dir: str) -> dict:
    """Evaluate several models in one process against a shared eval set.

    ``models`` maps a name -> that model's embeddings parquet path. All models
    must share the eval set (same targets) — asserted via ``eval_fingerprint``.
    The ``raw`` baseline and the raw join/encode run ONCE; only ``fm``/``fusion``
    re-run per model. Writes one comparison JSON.
    """
    names = list(models)
    fps = {n: _eval_fingerprint(p) for n, p in models.items()}
    if len(set(fps.values())) != 1:
        raise RuntimeError(
            f"eval sets differ across models — can't share the baseline / "
            f"row-align. fingerprints: {fps}"
        )

    # Shared: keys + labels (from any model, they match) and the raw baseline.
    base = _load_sorted(models[names[0]], want_fm=False)
    masks = _masks(base["split_code"])
    y, w = base["y"], base["w"]
    X_raw, raw_feats = _join_raw_features(base["cid"], base["ts"], y, raw_path, masks["train"])
    raw_only, _ = _run_sets({"raw": lambda m: X_raw[m]}, y, w, masks)

    per_model = {}
    for n in names:
        d = _load_sorted(models[n], want_fm=True)
        if not (np.array_equal(d["cid"], base["cid"]) and np.array_equal(d["ts"], base["ts"])):
            raise RuntimeError(f"model '{n}' rows don't align to the baseline after sort")
        X_fm = d["X_fm"]
        fsets = {
            "fm": lambda m, X=X_fm: X[m],
            "fusion": lambda m, X=X_fm: np.hstack([X[m], X_raw[m]]),
        }
        res, _ = _run_sets(fsets, y, w, masks)
        res = {"raw": raw_only["raw"], **res}
        per_model[n] = {
            "embedding_dim": int(X_fm.shape[1]),
            "results": res,
            **_lifts(res),
        }
        del d, X_fm

    summary = {
        "protocol": (
            "temporal 80/10/10 split; NVIDIA transaction-FM blueprint protocol; "
            "shared eval set across models (equal eval_fingerprint), raw baseline "
            "trained once"
        ),
        "eval_fingerprint": fps[names[0]],
        "raw_features": raw_feats,
        "n_samples": {s: int(m.sum()) for s, m in masks.items()},
        "natural_fraud_rate": {
            s: float((w[m] * y[m]).sum() / w[m].sum()) for s, m in masks.items()
        },
        "raw_baseline": raw_only["raw"],
        "models": per_model,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "compare_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _write_test_predictions(output_dir, test_proba, y, w, te) -> None:
    """Per-sample test scores so ROC/PR curves can be rebuilt offline (apply
    `weight` for natural-prevalence curves)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = list(test_proba)
    n_te = int(te.sum())
    pq.write_table(
        pa.table(
            {
                "feature_set": np.repeat(np.array(names), n_te),
                "label": np.tile(y[te], len(names)),
                "proba": np.concatenate([test_proba[n] for n in names]),
                "weight": np.tile(w[te], len(names)),
            }
        ),
        os.path.join(output_dir, "test_predictions.parquet"),
    )
    print(f"[05] per-sample test scores -> {output_dir}/test_predictions.parquet")


def print_summary(summary: dict) -> None:
    n = summary["n_samples"]
    nfr = summary["natural_fraud_rate"]
    print(f"protocol: {summary['protocol']}")
    print(
        f"samples  train={n['train']:,}  val={n['val']:,}  test={n['test']:,} "
        f"(natural test fraud rate {nfr['test']:.4%})"
    )
    print(f"eval fingerprint: {summary['eval_fingerprint']}")
    print(f"raw baseline features ({len(summary['raw_features'])}): "
          f"{', '.join(summary['raw_features'])}")

    def _block(results, lifts):
        print(f"{'feature set':<10} {'AUC-ROC':>10} {'PR-AUC':>10}")
        print("-" * 32)
        for name, r in results.items():
            print(f"{name:<10} {r['auc_roc']:>10.4f} {r['pr_auc']:>10.4f}")
        for k in ("fm", "fusion"):
            if f"{k}_lift_pr_auc" in lifts:
                print(f"  {k} lift: PR-AUC {lifts[f'{k}_lift_pr_auc']:+.4f}  "
                      f"AUC {lifts[f'{k}_lift_auc_roc']:+.4f}")

    if "models" in summary:  # multi
        for name, m in summary["models"].items():
            print(f"\n=== {name} ===")
            _block(m["results"], m)
    else:
        print("-" * 32)
        _block(summary["results"], summary)
