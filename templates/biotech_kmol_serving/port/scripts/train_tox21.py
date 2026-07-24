"""Train the 5-member tox21 ensemble so the served demo returns real predictions.

kMoL ships **no pretrained weights** (only SchNet test fixtures), and no HuggingFace
checkpoint fits this architecture — the loader needs kMoL's exact `state_dict` keys for
5 x GraphConvolutionalNetwork (LEConv x7, hidden 96, 264,300 params). What kMoL *does*
ship is the recipe and the data, both of which this script uses verbatim:

  * `data/configs/model/tox21.json` -> the model block is **byte-identical** to a member
    of `configs/ensemble_serve.example.json`, so the checkpoints written here load into
    kmolport unchanged.
  * `data/datasets.zip::datasets/tox21.csv` -> 7,831 molecules, 12 sparse binary assay
    targets (coverage 5,810-7,265 per target, ~4-10% positives). Vendored gzipped at
    `data/tox21.csv.gz`.

Hyperparameters are taken from that config, not chosen here: AdamW(lr=0.01,
weight_decay=0.00056), OneCycleLR(max_lr=0.01, pct_start=0.3, div_factor=25,
final_div_factor=1000), masked BCEWithLogitsLoss, batch 128, 200 epochs.

**Held-out test split reproduces kMoL's `DescriptorSplitter` exactly** — MolWt via
RDKit's MolecularDescriptorCalculator, `pd.qcut(...,10, duplicates="drop")`, then
`sklearn.train_test_split(train_size=0.8, random_state=42, stratify=bins)` (see
kmol/src/kmol/data/splitters.py:124 and :366). The 5 members then come from 5-fold CV
*within* that 80% train portion, which matches the described workflow ("when we train it
produces 5 weight files ... go through all 5 and mean per molecule") and kMoL's
`cross_validation_folds: 5`. The fold construction is ours; kMoL's CV loop was not
reverse-engineered, so per-member AUC here is a real number but not a claim to reproduce
any published kMoL figure.

Folds train concurrently as 5 fractional-GPU Ray tasks (num_gpus=0.2) — the model is tiny
and one L4 fits all five comfortably.

Usage (from the bundle root):
    python scripts/train_tox21.py --epochs 200 --out-ckpt-dir checkpoints_tox21
    python scripts/train_tox21.py --epochs 5        # quick wiring check
"""
import argparse
import gzip
import io
import json
import math
import os
import sys
import time

import ray

_BUNDLE = os.environ.get("KMOL_BUNDLE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BUNDLE not in sys.path:
    sys.path.insert(0, _BUNDLE)
SHIP_DIR = os.environ.get("KMOL_SHIP_DIR", _BUNDLE)
CONFIG = os.environ.get("KMOL_CONFIG", "configs/ensemble_serve.example.json")
DATA = os.environ.get("KMOL_TOX21", "data/tox21.csv.gz")

# pandas + scikit-learn only so the split is bit-identical to kMoL's DescriptorSplitter.
# Task-scoped: nothing is installed on the workspace head.
TASK_PIP = {"pip": ["torch==2.5.1", "torch_geometric==2.6.1", "rdkit==2024.3.5", "numpy<2",
                    "pandas", "scikit-learn"]}

TARGETS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
           "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]


def roc_auc(y_true, score):
    """Rank-based ROC-AUC with tie handling. Returns None if only one class present."""
    pairs = sorted(zip(score, y_true))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:  # average ranks within ties
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


@ray.remote(num_cpus=4, runtime_env=TASK_PIP)
def prep(data_path, config_path):
    """Featurize tox21 once and reproduce kMoL's MolWt-stratified 80/20 split."""
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem, RDLogger
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    from sklearn.model_selection import train_test_split

    import kmolport
    RDLogger.DisableLog("rdApp.*")

    with gzip.open(data_path, "rt") as f:
        df = pd.read_csv(f)

    calc = MolecularDescriptorCalculator(["MolWt"])
    feat = kmolport.GraphFeaturizer()

    rows, mws = [], []
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue  # kMoL's loader would drop these too
        try:
            d = feat.featurize(r["smiles"])
        except Exception:
            continue
        y = [float(r[t]) if not pd.isna(r[t]) else float("nan") for t in TARGETS]
        d.y = torch.tensor(y, dtype=torch.float32).view(1, -1)
        rows.append(d)
        mws.append(calc.CalcDescriptors(mol)[0])

    # kMoL DescriptorSplitter: qcut into deciles, stratified 80/20, seed 42.
    bins = pd.qcut(mws, 10, labels=False, duplicates="drop").tolist()
    idx = list(range(len(rows)))
    train_idx, test_idx = train_test_split(idx, train_size=0.8, random_state=42, stratify=bins)

    # 5 members = 5-fold CV within train, stratified on the same MolWt bins.
    train_bins = [bins[i] for i in train_idx]
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = [([train_idx[i] for i in tr], [train_idx[i] for i in va])
             for tr, va in skf.split(train_idx, train_bins)]

    return rows, folds, test_idx, {"n_molecules": len(rows), "n_dropped": len(df) - len(rows),
                                   "n_train": len(train_idx), "n_test": len(test_idx)}


@ray.remote(num_gpus=0.2, num_cpus=1, runtime_env=TASK_PIP)
def train_fold(bundle, fold_id, folds, test_idx, config_path, epochs, batch_size):
    import torch
    import torch.nn.functional as F

    import kmolport
    from kmolport.abstract_network import AbstractNetwork
    from kmolport.featurizer import collate
    from kmolport.helpers import SuperFactory

    rows, _, _, _ = bundle
    tr_idx, va_idx = folds[fold_id]

    cfg = kmolport.load_config(config_path)
    member_cfg = cfg["model"]["model_configs"][fold_id]
    torch.manual_seed(1000 + fold_id)
    model = SuperFactory.create(AbstractNetwork, member_cfg).to("cuda").train()

    # Straight from data/configs/model/tox21.json.
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.00056)
    steps = max(1, math.ceil(len(tr_idx) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.01, epochs=epochs, steps_per_epoch=steps,
        pct_start=0.3, div_factor=25, final_div_factor=1000)

    def masked_bce(logits, y):
        mask = ~torch.isnan(y)
        per = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
        return (per * mask).sum() / mask.sum().clamp(min=1)

    g = torch.Generator().manual_seed(1000 + fold_id)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_idx), generator=g).tolist()
        tot = 0.0
        for s in range(steps):
            sel = [tr_idx[perm[k]] for k in range(s * batch_size, min((s + 1) * batch_size, len(perm)))]
            if not sel:
                continue
            batch = collate([rows[i] for i in sel]).to("cuda")
            logits = model({"graph": batch})
            loss = masked_bce(logits, batch.y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            history.append({"epoch": ep, "train_loss": tot / steps})

    @torch.no_grad()
    def infer(indices):
        model.eval()
        out = []
        for s in range(0, len(indices), 256):
            batch = collate([rows[i] for i in indices[s:s + 256]]).to("cuda")
            out.append(model({"graph": batch}).cpu())
        return torch.cat(out).tolist()

    val_logits, test_logits = infer(va_idx), infer(test_idx)
    buf = io.BytesIO()
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, buf)
    return {"fold": fold_id, "state": buf.getvalue(), "history": history,
            "val_logits": val_logits, "test_logits": test_logits,
            "n_train": len(tr_idx), "n_val": len(va_idx)}


def per_target_auc(logits, labels):
    """logits/labels: list of rows, 12 wide; labels may contain NaN."""
    out = {}
    for t, name in enumerate(TARGETS):
        ys, ss = [], []
        for row_l, row_y in zip(logits, labels):
            y = row_y[t]
            if y == y:  # not NaN
                ys.append(int(y)); ss.append(row_l[t])
        out[name] = roc_auc(ys, ss)
    vals = [v for v in out.values() if v is not None]
    out["MEAN"] = sum(vals) / len(vals) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out-ckpt-dir", default="checkpoints_tox21")
    ap.add_argument("--out", default="tox21_training_results.json")
    args = ap.parse_args()

    ray.init(address="auto", runtime_env={"working_dir": SHIP_DIR})
    t0 = time.perf_counter()
    print("featurizing tox21 + reproducing kMoL split ...", flush=True)
    bundle = prep.remote(DATA, CONFIG)
    rows, folds, test_idx, stats = ray.get(bundle)
    print(f"  {stats}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"training 5 folds concurrently, {args.epochs} epochs each ...", flush=True)
    res = ray.get([train_fold.remote(bundle, i, folds, test_idx, CONFIG,
                                    args.epochs, args.batch_size) for i in range(5)])

    os.makedirs(args.out_ckpt_dir, exist_ok=True)
    test_labels = [rows[i].y.view(-1).tolist() for i in test_idx]

    members, ens = [], None
    for r in sorted(res, key=lambda x: x["fold"]):
        path = os.path.join(args.out_ckpt_dir, f"model_{r['fold']}.pt")
        with open(path, "wb") as f:
            f.write(r["state"])
        auc = per_target_auc(r["test_logits"], test_labels)
        members.append({"fold": r["fold"], "n_train": r["n_train"], "n_val": r["n_val"],
                        "checkpoint": path, "test_auc": auc,
                        "final_train_loss": r["history"][-1]["train_loss"]})
        print(f"  fold {r['fold']}: mean test ROC-AUC = {auc['MEAN']:.4f}", flush=True)
        ens = r["test_logits"] if ens is None else [[a + b for a, b in zip(x, y)]
                                                   for x, y in zip(ens, r["test_logits"])]
    ens = [[v / len(res) for v in row] for row in ens]   # the ensemble is a mean of members
    ens_auc = per_target_auc(ens, test_labels)
    print(f"\nENSEMBLE (mean of 5) mean test ROC-AUC = {ens_auc['MEAN']:.4f}", flush=True)

    out = {
        "what": "5-member tox21 ensemble trained with kMoL's own recipe and dataset",
        "recipe_source": "kmol/data/configs/model/tox21.json (hyperparameters verbatim)",
        "data_source": "kmol/data/datasets.zip::datasets/tox21.csv -> data/tox21.csv.gz",
        "split": "kMoL DescriptorSplitter reproduced: MolWt deciles, stratified 80/20, seed 42;"
                 " 5 members = StratifiedKFold(5) within train (our construction)",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "dataset_stats": stats,
        "members": members,
        "ensemble_test_auc": ens_auc,
        "wall_seconds": time.perf_counter() - t0,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} and {len(members)} checkpoints to {args.out_ckpt_dir}/")


if __name__ == "__main__":
    main()
