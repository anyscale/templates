"""Reco v3: beat the memorization floor — candidate re-ranking with causal features.

Everything scored in RAW merchant space on identical sampled events:

* ``naive_count``      — top-10 by historical count up to t (the 0.598-style floor)
* ``naive_recency``    — top-10 by count * exp(-days/tau), tau swept on val (stronger floor)
* ``mlp_solo``         — v2's MLP + time-context features (hour, dow, log-gap), token space
* ``alpha_blend``      — alpha * softmax_MLP + (1-alpha) * frequency prior, alpha swept on val
* ``ranker``           — logistic re-ranker over candidates with causal features:
                         log-count, recency decays, is-last, popularity, MLP log-softmax,
                         InfoNCE log-softmax (the "surprise" proxy, within candidates)

Every method also reported on two hard slices: next merchant NOT in the card's
top-10 (naive floor = 0 there) and NEVER seen before by the card.

    python scripts/next_merchant_v3.py --base-dir $BASE --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DECAYS_D = (7.0, 30.0, 90.0)


def hr10(df_scores, n_events):
    """df with [event, label, score] -> HR@10 with an EXPLICIT event denominator
    (events with no candidate rows count as misses, matching the naive floor)."""
    top = (df_scores.sort_values(["event", "score"], ascending=[True, False])
           .groupby("event").head(10))
    hit_events = set(top.loc[top.label == 1, "event"].to_numpy())
    return len(hit_events) / n_events


def main():
    import pandas as pd
    import pyarrow.dataset as pads
    import torch

    from src.merchant_vocab import _top_lookup, merchant_to_id

    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--scale", default="full")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-test-events", type=int, default=400_000)
    p.add_argument("--n-val-events", type=int, default=150_000)
    p.add_argument("--cand-count", type=int, default=64)
    p.add_argument("--cand-recent", type=int, default=16)
    args = p.parse_args()

    base = args.base_dir
    with open(f"{base}/model/{args.scale}/vocab.json") as f:
        vocab = json.load(f)
    mv = vocab["merchant_vocab"]
    lookup = _top_lookup(mv)
    n_classes = vocab["field_vocab_sizes"]["merchant_bucket"]

    # ---------- raw txns, canonical order ----------
    cols = ["card_id", "timestamp", "amount", "merchant_id"]
    raw = pads.dataset(f"{base}/raw/{args.scale}/transactions.parquet",
                       format="parquet").to_table(columns=cols).to_pandas()
    raw = raw.sort_values(["card_id", "timestamp", "amount", "merchant_id"],
                          kind="mergesort").reset_index(drop=True)
    card = raw["card_id"].to_numpy()
    ts = raw["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    amt_c = np.round(raw["amount"].to_numpy() * 100).astype(np.int64)
    merch = raw["merchant_id"].to_numpy()
    next_merch = np.roll(merch, -1)
    has_next = np.r_[card[1:] == card[:-1], False]
    dt_prev = np.where(np.r_[False, card[1:] == card[:-1]], ts - np.roll(ts, 1), 0)

    # global train-period popularity
    with open(f"{base}/raw/{args.scale}/splits.json") as f:
        splits = json.load(f)
    train_end = np.datetime64(splits["train_end"]).astype("datetime64[s]").astype(np.int64)
    pop = pd.Series(merch[ts < train_end]).value_counts().to_dict()

    # ---------- embeddings + join (raw row position preserved) ----------
    keys = pd.DataFrame({"card_id": card, "_ts": ts, "_amt_cents": amt_c,
                         "pos": np.arange(len(raw))})
    keys = keys.drop_duplicates(["card_id", "_ts", "_amt_cents"], keep="first")
    edf = pads.dataset(f"{base}/embeddings/{args.scale}_fulltest", format="parquet").to_table(
        columns=["embedding", "card_id", "raw_ts", "raw_amount", "raw_hour",
                 "raw_dow", "split"]).to_pandas()
    edf["_ts"] = edf.pop("raw_ts").astype(np.int64)
    edf["_amt_cents"] = np.round(edf.pop("raw_amount").to_numpy() * 100).astype(np.int64)
    edf = edf.drop_duplicates(["card_id", "_ts", "_amt_cents"], keep="first")
    j = edf.merge(keys, on=["card_id", "_ts", "_amt_cents"], how="inner")
    j = j[has_next[j["pos"].to_numpy()]].reset_index(drop=True)
    j["next_merchant"] = next_merch[j["pos"].to_numpy()]
    j["y_tok"] = merchant_to_id(j["next_merchant"].to_numpy(), mv, lookup).astype(np.int64)
    print(f"[v3] joined events with next-target: {len(j):,}")

    # sample events (identical sets for every method)
    rng = np.random.default_rng(0)
    tr_all = np.flatnonzero((j["split"] == "train").to_numpy())
    va_all = np.flatnonzero((j["split"] == "val").to_numpy())
    te_all = np.flatnonzero((j["split"] == "test").to_numpy())
    va = rng.choice(va_all, min(args.n_val_events, len(va_all)), replace=False)
    te = rng.choice(te_all, min(args.n_test_events, len(te_all)), replace=False)
    print(f"[v3] events: train {len(tr_all):,} (mlp fit), val {len(va):,} (ranker fit), "
          f"test {len(te):,} (eval)")

    # ---------- causal snapshots at sampled events ----------
    want = np.zeros(len(raw), bool)
    ev_of_pos = np.full(len(raw), -1, np.int64)
    for arr in (va, te):
        pos = j.loc[arr, "pos"].to_numpy()
        want[pos] = True
        ev_of_pos[pos] = arr
    starts = np.flatnonzero(np.r_[True, card[1:] != card[:-1]])
    ends = np.r_[starts[1:], len(raw)]
    C_ev, C_m, C_cnt, C_dt, C_last = [], [], [], [], []
    for s, e in zip(starts, ends):
        hist = {}
        last_m = -1
        for i in range(s, e):
            if want[i]:
                items = list(hist.items())
                if items:
                    by_cnt = sorted(items, key=lambda kv: -kv[1][0])[: args.cand_count]
                    by_rec = sorted(items, key=lambda kv: -kv[1][1])[: args.cand_recent]
                    seen, cands = set(), []
                    for m, (c, lt) in by_cnt + by_rec:
                        if m not in seen:
                            seen.add(m)
                            cands.append((m, c, lt))
                    ev = ev_of_pos[i]
                    C_ev.append(np.full(len(cands), ev, np.int64))
                    C_m.append(np.fromiter((m for m, _, _ in cands), np.int64, len(cands)))
                    C_cnt.append(np.fromiter((c for _, c, _ in cands), np.float32, len(cands)))
                    C_dt.append(np.fromiter(((ts[i] - lt) / 86400.0 for _, _, lt in cands),
                                            np.float32, len(cands)))
                    C_last.append(np.fromiter((1.0 if m == last_m else 0.0 for m, _, _ in cands),
                                              np.float32, len(cands)))
            m = merch[i]
            c, _ = hist.get(m, (0, 0))
            hist[m] = (c + 1, ts[i])
            last_m = m
    cev = np.concatenate(C_ev); cm = np.concatenate(C_m)
    ccnt = np.concatenate(C_cnt); cdt = np.concatenate(C_dt); clast = np.concatenate(C_last)
    print(f"[v3] candidate rows: {len(cev):,} (avg {len(cev)/ (len(va)+len(te)):.1f}/event)")

    cd = pd.DataFrame({"event": cev, "cand": cm, "cnt": ccnt, "dt": cdt, "is_last": clast})
    cd["label"] = (cd["cand"].to_numpy() ==
                   j.loc[cd["event"].to_numpy(), "next_merchant"].to_numpy()).astype(np.int8)
    cd["pop"] = np.log1p(pd.Series(cd["cand"]).map(pop).fillna(0).to_numpy())
    cd["cand_tok"] = merchant_to_id(cd["cand"].to_numpy(), mv, lookup).astype(np.int64)
    is_te_ev = np.zeros(len(j), bool); is_te_ev[te] = True
    cd["is_test"] = is_te_ev[cd["event"].to_numpy()]

    # hard slices (defined per event, from candidate frame)
    top10_hit = hr10  # alias for readability

    # ---------- MLP with time-context features (Gemini #2) ----------
    dev = args.device if torch.cuda.is_available() else "cpu"
    X = np.vstack(j["embedding"].to_numpy()).astype(np.float32)
    extra = np.stack([
        j["raw_hour"].to_numpy().astype(np.float32) / 23.0,
        j["raw_dow"].to_numpy().astype(np.float32) / 6.0,
        np.log1p(dt_prev[j["pos"].to_numpy()] / 3600.0).astype(np.float32),
    ], axis=1)
    X = np.hstack([X, extra])
    mu, sd = X[tr_all].mean(0), X[tr_all].std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd)
    yt = torch.as_tensor(j["y_tok"].to_numpy())
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, n_classes)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(tr_all)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(Xt[b].to(dev)), yt[b].to(dev))
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        print(f"[v3] mlp epoch {ep}: ce {tot / len(perm):.4f}")
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd},
               f"{base}/model/{args.scale}/next_merchant_mlp.pt")

    # per-event log-softmax gathered at candidate tokens + mlp-solo token-space HR
    sd_ = torch.load(f"{base}/model/{args.scale}/model.pt", map_location="cpu")
    W = sd_["infonce_proj.merchant_bucket.weight"].to(dev)
    bI = sd_["infonce_proj.merchant_bucket.bias"].to(dev)
    E = sd_["dyn_emb.merchant_bucket.weight"].to(dev)
    Xemb = X[:, :512]  # un-normalized embedding columns (view; batches copied below)
    cd = cd.sort_values("event", kind="mergesort").reset_index(drop=True)
    ev_order = cd["event"].to_numpy(); tok_order = cd["cand_tok"].to_numpy()
    mlp_ls = np.empty(len(cd), np.float32); inf_ls = np.empty(len(cd), np.float32)
    mlp_solo_hits = {}
    uniq_ev = np.unique(np.concatenate([va, te]))
    ptr = 0
    with torch.no_grad():
        for i in range(0, len(uniq_ev), 4096):
            evb = uniq_ev[i : i + 4096]
            ls = torch.log_softmax(net(Xt[evb].to(dev)), dim=1)
            xb = torch.as_tensor(np.ascontiguousarray(Xemb[evb])).to(dev)
            li = torch.log_softmax((xb @ W.T + bI) @ E.T, dim=1)
            top10 = ls.topk(10, dim=1).indices.cpu().numpy()
            ytok = j["y_tok"].to_numpy()[evb]
            for r, ev in enumerate(evb):
                mlp_solo_hits[ev] = bool((top10[r] == ytok[r]).any())
            pos = {int(ev): r for r, ev in enumerate(evb)}
            while ptr < len(cd) and ev_order[ptr] in pos:
                r = pos[ev_order[ptr]]
                mlp_ls[ptr] = float(ls[r, tok_order[ptr]])
                inf_ls[ptr] = float(li[r, tok_order[ptr]])
                ptr += 1
    cd["mlp_ls"] = mlp_ls; cd["inf_ls"] = inf_ls
    print(f"[v3] gathered candidate logits ({ptr:,} rows)")

    results = {"n_test_events": int(len(te)), "readouts": {}, "slices": {}}
    tecd = cd[cd.is_test]
    vacd = cd[~cd.is_test]
    n_te, n_va = len(te), len(va)

    # naive floors
    results["readouts"]["naive_count"] = {"hr@10": hr10(tecd.assign(score=tecd["cnt"]), n_te)}
    best_tau, best_v = None, -1
    for tau in DECAYS_D:
        v = hr10(vacd.assign(score=vacd["cnt"] * np.exp(-vacd["dt"] / tau)), n_va)
        if v > best_v:
            best_v, best_tau = v, tau
    results["readouts"]["naive_recency"] = {
        "tau_days": best_tau,
        "hr@10": hr10(tecd.assign(score=tecd["cnt"] * np.exp(-tecd["dt"] / best_tau)), n_te)}

    # mlp solo (token space, time-context features)
    results["readouts"]["mlp_solo_token_space"] = {
        "hr@10": float(np.mean([mlp_solo_hits[e] for e in te]))}

    # alpha blend (Gemini #1): alpha*softmax_mlp + (1-alpha)*freq prior, within candidates
    def blend_frame(frame, alpha):
        fs = frame["cnt"] / frame.groupby("event")["cnt"].transform("sum")
        return frame.assign(score=alpha * np.exp(frame["mlp_ls"]) + (1 - alpha) * fs)
    best_a, best_v = None, -1
    for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        v = hr10(blend_frame(vacd, a), n_va)
        if v > best_v:
            best_v, best_a = v, a
    results["readouts"]["alpha_blend"] = {"alpha": best_a,
                                          "hr@10": hr10(blend_frame(tecd, best_a), n_te)}

    # logistic re-ranker on causal features (inf_ls = the "surprise" proxy)
    from sklearn.linear_model import LogisticRegression

    def feats(frame):
        return np.stack([
            np.log1p(frame["cnt"]),
            np.exp(-frame["dt"] / 7.0), np.exp(-frame["dt"] / 30.0), np.exp(-frame["dt"] / 90.0),
            frame["is_last"], frame["pop"], frame["mlp_ls"], frame["inf_ls"],
        ], axis=1)
    lr_m = LogisticRegression(max_iter=300, C=1.0)
    lr_m.fit(feats(vacd), vacd["label"])
    results["readouts"]["ranker"] = {
        "hr@10": hr10(tecd.assign(score=lr_m.decision_function(feats(tecd))), n_te),
        "coef": dict(zip(["log_cnt", "rec7", "rec30", "rec90", "is_last", "pop",
                          "mlp_ls", "inf_ls"], lr_m.coef_[0].round(3).tolist()))}

    # hard slices (Gemini #3). Note: candidate-based methods are structurally 0
    # on truly-new merchants (candidates come from history); only the full-vocab
    # MLP can score there. "never_seen" also absorbs seen-but-beyond-candidate-cap.
    t10 = (tecd.sort_values(["event", "cnt"], ascending=[True, False]).groupby("event").head(10))
    in_top10 = set(t10.loc[t10.label == 1, "event"].to_numpy())
    in_cands = set(tecd.loc[tecd.label == 1, "event"].to_numpy())
    all_te = set(te.tolist())
    novel10 = all_te - in_top10
    never = all_te - in_cands
    results["slices"]["share_not_in_top10"] = len(novel10) / len(all_te)
    results["slices"]["share_never_seen_or_beyond_cap"] = len(never) / len(all_te)
    sub = tecd[tecd["event"].isin(novel10)]
    results["slices"]["not_in_top10"] = {
        "naive_count": 0.0,
        "ranker": hr10(sub.assign(score=lr_m.decision_function(feats(sub))), len(novel10)),
        "mlp_solo_token_space": float(np.mean([mlp_solo_hits[e] for e in novel10])),
    }
    results["slices"]["never_seen_or_beyond_cap"] = {
        "naive_count": 0.0,
        "ranker": None,  # structurally 0: candidates are drawn from history
        "mlp_solo_token_space": float(np.mean([mlp_solo_hits[e] for e in never]))
        if never else None,
    }

    print(json.dumps(results, indent=2))
    out = f"{base}/downstream/{args.scale}_nextmerchant/probe_metrics_v3.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[v3] -> {out}")


if __name__ == "__main__":
    main()
