"""Reco salvage probe v2: three readouts of the frozen embedding vs memorization.

Round 1 (linear head) showed the pooled-last embedding carries 5x the
next-merchant signal of the masked-position readout (HR@10 0.397 vs 0.077)
but below the naive memorization floor (0.598). This round adds the two
readouts that round buried in "next rungs":

* ``infonce_zeroshot`` — the model's OWN trained merchant scorer: pooled-last
  state -> infonce_proj -> dot product against the tied merchant embedding
  table. Zero training. Off-distribution caveat: the proj was trained on
  MASKED-position states, we query it with the unmasked last position.
* ``linear`` — round 1's head (kept for continuity).
* ``mlp`` — 2-layer MLP (512 -> hidden -> ReLU -> 21k), the trained
  non-linear untangler.

All scored on the same 2.4M test pairs against the causal naive
top-10-train-history baseline.

    python scripts/next_merchant_probe.py --base-dir $BASE --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

KS = (1, 5, 10)


def _rank_metrics(topk_fn, y, idx, batch_size):
    """HR@K/NDCG@10 for a callable idx-batch -> top-10 class ids."""
    hits = {k: 0 for k in KS}
    ndcg = 0.0
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        topk = topk_fn(b)
        match = topk == y[b][:, None]
        for k in KS:
            hits[k] += int(match[:, :k].any(1).sum())
        rr = np.argwhere(match)
        ndcg += float((1.0 / np.log2(rr[:, 1] + 2)).sum())
    n = len(idx)
    return {f"hr@{k}": hits[k] / n for k in KS} | {"ndcg@10": ndcg / n}


def main():
    import pandas as pd
    import pyarrow.dataset as pads
    import torch

    from src.merchant_vocab import _top_lookup, merchant_to_id
    from src.tokenizer import _RESERVED

    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--scale", default="full")
    p.add_argument("--embeddings", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    base = args.base_dir
    emb_path = args.embeddings or f"{base}/embeddings/{args.scale}_fulltest"
    with open(f"{base}/model/{args.scale}/vocab.json") as f:
        vocab = json.load(f)
    mv = vocab["merchant_vocab"]
    assert mv, "learned merchant vocab required"
    lookup = _top_lookup(mv)
    n_classes = vocab["field_vocab_sizes"]["merchant_bucket"]

    # ---- raw txns in canonical order -> next-merchant label per txn
    cols = ["card_id", "timestamp", "amount", "merchant_id"]
    raw = pads.dataset(f"{base}/raw/{args.scale}/transactions.parquet",
                       format="parquet").to_table(columns=cols).to_pandas()
    raw = raw.sort_values(["card_id", "timestamp", "amount", "merchant_id"],
                          kind="mergesort").reset_index(drop=True)
    card = raw["card_id"].to_numpy()
    ts = raw["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    amt = raw["amount"].to_numpy()
    next_merch = np.roll(raw["merchant_id"].to_numpy(), -1)
    has_next = np.r_[card[1:] == card[:-1], False]
    keys = pd.DataFrame({
        "card_id": card, "_ts": ts,
        "_amt_cents": np.round(amt * 100).astype(np.int64),
        "next_merchant": next_merch, "has_next": has_next,
        "merchant_id": raw["merchant_id"].to_numpy(),
    }).drop_duplicates(["card_id", "_ts", "_amt_cents"], keep="first")

    # ---- embeddings + join
    edf = pads.dataset(emb_path, format="parquet").to_table(
        columns=["embedding", "card_id", "raw_ts", "raw_amount", "split"]).to_pandas()
    edf["_ts"] = edf.pop("raw_ts").astype(np.int64)
    edf["_amt_cents"] = np.round(edf.pop("raw_amount").to_numpy() * 100).astype(np.int64)
    edf = edf.drop_duplicates(["card_id", "_ts", "_amt_cents"], keep="first")
    j = edf.merge(keys, on=["card_id", "_ts", "_amt_cents"], how="inner")
    j = j[j.has_next].reset_index(drop=True)
    y = merchant_to_id(j["next_merchant"].to_numpy(), mv, lookup).astype(np.int64)
    X = np.vstack(j["embedding"].to_numpy()).astype(np.float32)
    tr = (j["split"] == "train").to_numpy()
    te = (j["split"] == "test").to_numpy()
    te_idx = np.flatnonzero(te)
    print(f"[nm] pairs: {len(j):,} (train {tr.sum():,} / test {te.sum():,}); "
          f"{n_classes:,}-way tokens")

    dev = args.device if torch.cuda.is_available() else "cpu"
    results = {"n_train_pairs": int(tr.sum()), "n_test_pairs": int(len(te_idx)),
               "token_space": int(n_classes), "readouts": {}}

    # ---- naive causal baseline
    train_hist = keys[keys._ts < j.loc[tr, "_ts"].max()]
    top10 = (train_hist.groupby(["card_id", "merchant_id"]).size().rename("n")
             .reset_index().sort_values(["card_id", "n"], ascending=[True, False])
             .groupby("card_id").head(10))
    top_sets = top10.groupby("card_id")["merchant_id"].agg(set).to_dict()
    te_rows = j.loc[te, ["card_id", "next_merchant"]]
    naive = float(np.fromiter(
        (r.next_merchant in top_sets.get(r.card_id, ()) for r in te_rows.itertuples()),
        dtype=bool, count=len(te_rows)).mean())
    results["readouts"]["naive_top10_train_history"] = {"hr@10": naive}
    print(f"[nm] naive HR@10: {naive:.4f}")

    # ---- 1. InfoNCE zero-shot: the model's own trained merchant scorer.
    # Raw (un-normalized) embedding — the proj was trained on the model's own
    # hidden scale. Reserved token ids are masked out of the ranking.
    sd = torch.load(f"{base}/model/{args.scale}/model.pt", map_location="cpu")
    W = sd["infonce_proj.merchant_bucket.weight"].to(dev)
    b = sd["infonce_proj.merchant_bucket.bias"].to(dev)
    E = sd["dyn_emb.merchant_bucket.weight"].to(dev)           # (n_classes, d)
    Xraw = torch.as_tensor(X)

    def infonce_topk(bidx):
        with torch.no_grad():
            h = Xraw[bidx].to(dev) @ W.T + b
            logits = h @ E.T
            logits[:, :_RESERVED] = float("-inf")
            return logits.topk(10, dim=1).indices.cpu().numpy()

    results["readouts"]["infonce_zeroshot"] = _rank_metrics(
        infonce_topk, y, te_idx, args.batch_size)
    print(f"[nm] infonce_zeroshot: {results['readouts']['infonce_zeroshot']}")

    # ---- trained heads (z-scored input)
    mu, sd_ = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd_)
    yt = torch.as_tensor(y)
    tr_idx = np.flatnonzero(tr)

    def train_and_eval(name, net):
        net = net.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            perm = np.random.default_rng(ep).permutation(tr_idx)
            tot = 0.0
            for i in range(0, len(perm), args.batch_size):
                bb = perm[i : i + args.batch_size]
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(net(Xt[bb].to(dev)), yt[bb].to(dev))
                loss.backward()
                opt.step()
                tot += float(loss) * len(bb)
            print(f"[nm] {name} epoch {ep}: train ce {tot / len(perm):.4f}")

        def topk(bidx):
            with torch.no_grad():
                return net(Xt[bidx].to(dev)).topk(10, dim=1).indices.cpu().numpy()

        results["readouts"][name] = _rank_metrics(topk, y, te_idx, args.batch_size)
        print(f"[nm] {name}: {results['readouts'][name]}")

    torch.manual_seed(0)
    train_and_eval("linear", torch.nn.Linear(X.shape[1], n_classes))
    torch.manual_seed(0)
    train_and_eval("mlp", torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, n_classes),
    ))

    results["reference"] = {"06_masked_readout_hr@10": 0.077,
                            "round1_linear_hr@10": 0.397,
                            "derisk_naive_full_history_hr@10": 0.652}
    print(json.dumps(results, indent=2))
    out = f"{base}/downstream/{args.scale}_nextmerchant/probe_metrics_v2.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[nm] -> {out}")


if __name__ == "__main__":
    main()
