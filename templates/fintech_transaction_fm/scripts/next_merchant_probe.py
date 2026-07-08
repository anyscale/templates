"""Reco salvage probe: is next-merchant signal READABLE from the frozen embedding?

The campaign's BERT4Rec-style reco eval (06) reads the masked-position state and
regressed to HR@10 ~0.08 on the winning fraud recipe. The fraud lesson was that
the readout, not the representation, is usually the problem — so this probe
points a TRAINED head at the pooled-last embedding instead: embedding at txn t
(history up to and including t) -> predict the merchant TOKEN of txn t+1.
No leakage: the target transaction is strictly after the embedded window.

Baseline: the card's top-10 merchants over the TRAIN period (causal w.r.t. the
test period; the de-risk naive floor). Both are scored in the learned-vocab
token space (top-K ids + aggregate buckets), the same convention as 06.

    python scripts/next_merchant_probe.py --base-dir $BASE \
        --embeddings $BASE/embeddings/full_fulltest --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    import pandas as pd
    import pyarrow.dataset as pads
    import torch

    from src.merchant_vocab import _top_lookup, merchant_to_id

    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--scale", default="full")
    p.add_argument("--embeddings", default=None,
                   help="embeddings parquet dir (default: embeddings/<scale>_fulltest)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    base = args.base_dir
    emb_path = args.embeddings or f"{base}/embeddings/{args.scale}_fulltest"
    with open(f"{base}/model/{args.scale}/vocab.json") as f:
        vocab = json.load(f)
    mv = vocab["merchant_vocab"]
    assert mv, "learned merchant vocab required (this probe targets its token space)"
    lookup = _top_lookup(mv)
    n_classes = vocab["field_vocab_sizes"]["merchant_bucket"]

    # ---- raw transactions in canonical order -> next-merchant label per txn
    cols = ["card_id", "timestamp", "amount", "merchant_id"]
    raw = pads.dataset(f"{base}/raw/{args.scale}/transactions.parquet",
                       format="parquet").to_table(columns=cols).to_pandas()
    raw = raw.sort_values(["card_id", "timestamp", "amount", "merchant_id"],
                          kind="mergesort").reset_index(drop=True)
    card = raw["card_id"].to_numpy()
    ts = raw["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    amt = raw["amount"].to_numpy()
    next_merch = np.roll(raw["merchant_id"].to_numpy(), -1)
    has_next = np.r_[card[1:] == card[:-1], False]  # last txn of a card: no target
    print(f"[nm] {len(raw):,} txns, {has_next.sum():,} with a next-merchant target")

    keys = pd.DataFrame({
        "card_id": card, "_ts": ts,
        "_amt_cents": np.round(amt * 100).astype(np.int64),
        "next_merchant": next_merch, "has_next": has_next,
        "merchant_id": raw["merchant_id"].to_numpy(),
    }).drop_duplicates(["card_id", "_ts", "_amt_cents"], keep="first")

    # ---- embeddings + join
    dset = pads.dataset(emb_path, format="parquet")
    edf = dset.to_table(
        columns=["embedding", "card_id", "raw_ts", "raw_amount", "split"]
    ).to_pandas()
    edf["_ts"] = edf.pop("raw_ts").astype(np.int64)
    edf["_amt_cents"] = np.round(edf.pop("raw_amount").to_numpy() * 100).astype(np.int64)
    edf = edf.drop_duplicates(["card_id", "_ts", "_amt_cents"], keep="first")
    j = edf.merge(keys, on=["card_id", "_ts", "_amt_cents"], how="inner")
    j = j[j.has_next].reset_index(drop=True)
    y = merchant_to_id(j["next_merchant"].to_numpy(), mv, lookup).astype(np.int64)
    X = np.vstack(j["embedding"].to_numpy()).astype(np.float32)
    tr = (j["split"] == "train").to_numpy()
    te = (j["split"] == "test").to_numpy()
    print(f"[nm] joined pairs: {len(j):,} (train {tr.sum():,} / test {te.sum():,}); "
          f"{n_classes:,}-way token space")

    # ---- naive causal baseline: card's top-10 TRAIN-period merchants
    train_hist = keys[keys._ts < j.loc[tr, "_ts"].max()]  # train period upper bound
    top10 = (train_hist.groupby(["card_id", "merchant_id"]).size()
             .rename("n").reset_index()
             .sort_values(["card_id", "n"], ascending=[True, False])
             .groupby("card_id").head(10))
    top_sets = top10.groupby("card_id")["merchant_id"].agg(set).to_dict()
    te_rows = j.loc[te, ["card_id", "next_merchant"]]
    naive_hits = np.fromiter(
        (r.next_merchant in top_sets.get(r.card_id, ()) for r in te_rows.itertuples()),
        dtype=bool, count=len(te_rows))
    print(f"[nm] naive top-10-train-history HR@10 on test: {naive_hits.mean():.4f}")

    # ---- linear head on the frozen embedding
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd)
    yt = torch.as_tensor(y)
    head = torch.nn.Linear(X.shape[1], n_classes).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    tr_idx = np.flatnonzero(tr)
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(tr_idx)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                head(Xt[b].to(dev)), yt[b].to(dev))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"[nm] epoch {ep}: train ce {tot / len(perm):.4f}")

    # ---- test HR@K / NDCG@10 in token space
    ks = (1, 5, 10)
    hits = {k: 0 for k in ks}
    ndcg = 0.0
    te_idx = np.flatnonzero(te)
    with torch.no_grad():
        for i in range(0, len(te_idx), args.batch_size):
            b = te_idx[i : i + args.batch_size]
            topk = head(Xt[b].to(dev)).topk(10, dim=1).indices.cpu().numpy()
            tgt = y[b][:, None]
            match = topk == tgt
            for k in ks:
                hits[k] += int(match[:, :k].any(1).sum())
            rr = np.argwhere(match)
            ndcg += float((1.0 / np.log2(rr[:, 1] + 2)).sum())
    n_te = len(te_idx)
    results = {
        "n_train_pairs": int(tr.sum()), "n_test_pairs": int(n_te),
        "token_space": int(n_classes),
        "head": {f"hr@{k}": hits[k] / n_te for k in ks} | {"ndcg@10": ndcg / n_te},
        "naive_top10_train_history": {"hr@10": float(naive_hits.mean())},
        "reference": {"06_masked_readout_hr@10": 0.077,
                      "derisk_naive_full_history_hr@10": 0.652},
    }
    print(json.dumps(results, indent=2))
    out = f"{base}/downstream/{args.scale}_nextmerchant/probe_metrics.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[nm] -> {out}")


if __name__ == "__main__":
    main()
