"""Supervised fine-tuning of the pretrained decoder for fraud classification (Part 7).

This is a labeled beyond-the-blueprint extension: NVIDIA's blueprint freezes the
foundation model after pretraining and only XGBoost ever sees fraud labels. Here we put
a classification head on the decoder and update its weights on the labeled data, in two
variants:

* **single-transaction** — the head sees exactly what the frozen embedding saw (one
  tokenized transaction). Isolates "does fine-tuning beat frozen features" with no
  other change.
* **history-window** — the head sees the transaction *plus its preceding transactions*
  on the same card. This is the first configuration in the series whose inference input
  includes sequence context.

Both train on the same balanced 1M sample and score on the same 100K stratified
val/test sets as Part 6, so the numbers are directly comparable to raw/embedding/fusion.

The training function is plain PyTorch; Ray Train shards the data and wraps DDP —
the same pattern as pretraining (src/pretrain.py).
"""

import json
import math
import os

import numpy as np
import ray

from .nvsplit import _wait_for_files, ordered_parquet_files, train_parquet_files

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Data: v1 single-transaction tokens (reuses the Part 5 preparation path)
# ---------------------------------------------------------------------------

def build_single_txn_tokens(split_dir: str, out_dir: str, balanced_train: int = 1_000_000,
                            max_length: int = 128, seed: int = 42) -> dict:
    """Tokenize each labeled transaction alone → ``ft_ids_<split>.npy`` +
    ``lbl_<split>.npy`` + ``raw_<split>.parquet`` in the same row order Part 5 uses.

    Composes the same pieces as the embed stage (seeded balanced sample + order-fixing
    sort on a CPU task, then per-batch tokenization on CPU workers); the only
    difference is that the token ids are the product instead of an intermediate.
    """
    import shutil

    import ray.data

    from .nvembed import encode_txn_batch, prepare_embed_split

    ray.init(ignore_reinit_error=True)
    os.makedirs(out_dir, exist_ok=True)
    files = {"train": None, "val": "val_eval.parquet", "test": "test_eval.parquet"}
    stats = {}
    for split, fname in files.items():
        prep = ray.get(prepare_embed_split.remote(split_dir, fname, out_dir, split,
                                                  balanced_train, seed))
        shards = os.path.join(out_dir, f"_ids_{split}")
        if os.path.isdir(shards):
            shutil.rmtree(shards)
        ray.data.read_parquet(prep["prep"]) \
            .map_batches(lambda b: encode_txn_batch(b, max_length=max_length),
                         batch_format="pandas") \
            .write_parquet(shards)

        import pandas as pd
        df = pd.concat([pd.read_parquet(f) for f in ordered_parquet_files(shards)],
                       ignore_index=True).sort_values("__pos__", kind="mergesort")
        ids = np.stack([np.asarray(r, dtype=np.int32) for r in df["ids"]])
        np.save(os.path.join(out_dir, f"ft_ids_{split}.npy"), ids)
        shutil.rmtree(shards)
        os.remove(prep["prep"])
        stats[split] = {"rows": int(len(ids)), "fraud": prep["fraud"]}
    _wait_for_files([os.path.join(out_dir, f"ft_ids_{s}.npy") for s in SPLITS])
    return stats


# ---------------------------------------------------------------------------
# Data: v2 history-window tokens
# ---------------------------------------------------------------------------

def _recover_targets(split_dir: str, shards_dir: str, balanced_train: int, seed: int):
    """Reconstruct, for every labeled row of each split, its source-row id (``__seq__``)
    plus label — by re-running the exact seeded selections of Part 2/Part 5 on data that
    still carries ``__seq__``. Returns {split: DataFrame[User, Card, __seq__, label]} in
    the same row order as ``lbl_<split>.npy``."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from .nvsplit import normalize_batch

    meta = json.load(open(os.path.join(split_dir, "split_meta.json")))
    tr_cut, te_cut = pd.Timestamp(meta["train_cutoff"]), pd.Timestamp(meta["test_cutoff"])
    max_users = meta.get("max_users")

    cols = ["User", "Card", "Year", "Month", "Day", "Time", "Is Fraud?", "__seq__"]
    src = pd.concat([pd.read_parquet(f, columns=cols)
                     for f in ordered_parquet_files(shards_dir)], ignore_index=True)
    if max_users is not None:
        src = src[src["User"] < max_users]
    src = normalize_batch(src).sort_values("__seq__", kind="mergesort").reset_index(drop=True)
    src["label"] = (src["Is Fraud?"].astype(str).str.lower() == "yes").astype("int32")

    out = {}
    rng_parts = {
        "train": src[src["date"] < tr_cut],
        "val": src[(src["date"] >= tr_cut) & (src["date"] < te_cut)],
        "test": src[src["date"] >= te_cut],
    }
    for split, part in rng_parts.items():
        part = part.reset_index(drop=True)
        y = part["label"].to_numpy()
        if split == "train":  # the balanced sample (prepare_embed_split, verbatim RNG)
            f, nrm = np.where(y == 1)[0], np.where(y == 0)[0]
            rng = np.random.RandomState(seed)
            nf = min(len(f), int(balanced_train * 0.1))
            nn = min(len(nrm), balanced_train - nf)
            sel = np.concatenate([rng.choice(f, nf, replace=False),
                                  rng.choice(nrm, nn, replace=False)])
            rng.shuffle(sel)
            part = part.iloc[sel].reset_index(drop=True)
        else:  # the 100K stratified eval sample (nvsplit.stratified_eval, verbatim RNG)
            eval_samples = int(meta["eval_samples"])
            if eval_samples < len(part):
                _, keep_idx = train_test_split(np.arange(len(part)), test_size=eval_samples,
                                               stratify=y, random_state=seed)
                part = part.iloc[keep_idx].reset_index(drop=True)
        # Final row order everywhere downstream is the preprocess sort (user, card, time);
        # within a card, CSV order == time order, so (User, Card, __seq__) reproduces it.
        part = part.sort_values(["User", "Card", "__seq__"], kind="mergesort")
        out[split] = part[["User", "Card", "__seq__", "label"]].reset_index(drop=True)
    return out


def tokenize_card_windows(group, targets_by_card: dict, window_txns: int,
                          window_tokens: int, merchant_hash: int = 2000):
    """One card's rows in → one token window per labeled target on that card. The
    window is the target transaction plus up to ``window_txns - 1`` preceding
    transactions, in the pretraining corpus format (``<bos> t1 <sep> ... tN <eos>``),
    so the fine-tuned model reads exactly the format it was pretrained on."""
    import pandas as pd

    from .nvidia_tokenizer import FinancialTabularTokenizer
    from .nvtokenize_cpu import preprocess_cpu, transform_cpu

    user, card = int(group["User"].iloc[0]), int(group["Card"].iloc[0])
    targets = targets_by_card.get((user, card))
    if not targets:
        return pd.DataFrame({"split": [], "pos": [], "label": [], "ids": []})

    global _TOK  # per-worker tokenizer cache (module-level in nvcorpus does the same)
    try:
        _TOK
    except NameError:
        _TOK = None
    if _TOK is None:
        _TOK = FinancialTabularTokenizer(merchant_hash_size=merchant_hash,
                                         category_hierarchy=True, temporal_encoding=True)

    group = group.sort_values("__seq__", kind="mergesort").reset_index(drop=True)
    seqs = group["__seq__"].to_numpy(np.int64)
    td = transform_cpu(preprocess_cpu(group), merchant_hash_size=merchant_hash)
    txn_text = td.iloc[:, 0].str.cat([td[c] for c in td.columns[1:]], sep=" ")

    rows = []
    for split, pos, tseq, label in targets:
        i = int(np.searchsorted(seqs, tseq))
        lo = max(0, i - window_txns + 1)
        line = "<bos> " + " <sep> ".join(txn_text.iloc[lo:i + 1]) + " <eos>"
        ids = np.asarray(_TOK.encode(line, max_length=window_tokens), dtype=np.int32)
        rows.append({"split": split, "pos": int(pos), "label": int(label),
                     "ids": ids.tolist()})
    return pd.DataFrame(rows)


def build_history_windows(split_dir: str, shards_dir: str, out_dir: str,
                          balanced_train: int = 1_000_000, window_txns: int = 19,
                          window_tokens: int = 256, seed: int = 42) -> dict:
    """History-window tokens for every labeled row → ``ft_hist_ids_<split>.npy``, row
    order identical to ``lbl_<split>.npy`` (asserted via the carried labels)."""
    import shutil

    import pandas as pd
    import ray.data

    ray.init(ignore_reinit_error=True)
    os.makedirs(out_dir, exist_ok=True)

    targets = _recover_targets(split_dir, shards_dir, balanced_train, seed)
    by_card = {}
    for split, df in targets.items():
        for pos, (u, c, s, y) in enumerate(df.itertuples(index=False, name=None)):
            by_card.setdefault((int(u), int(c)), []).append((split, pos, int(s), int(y)))

    shards = os.path.join(out_dir, "_hist_tmp")
    if os.path.isdir(shards):
        shutil.rmtree(shards)
    ray.data.read_parquet(ordered_parquet_files(shards_dir)) \
        .groupby(["User", "Card"]) \
        .map_groups(lambda g: tokenize_card_windows(g, by_card, window_txns, window_tokens),
                    batch_format="pandas") \
        .write_parquet(shards)

    df = pd.concat([pd.read_parquet(f) for f in ordered_parquet_files(shards)],
                   ignore_index=True)
    stats = {}
    for split in SPLITS:
        part = df[df["split"] == split].sort_values("pos", kind="mergesort")
        ids = np.stack([np.asarray(r, dtype=np.int32) for r in part["ids"]])
        lbl_ref = np.load(os.path.join(out_dir, f"lbl_{split}.npy"))
        assert np.array_equal(part["label"].to_numpy(np.int64), lbl_ref.astype(np.int64)), \
            f"{split}: window row order does not match lbl_{split}.npy"
        np.save(os.path.join(out_dir, f"ft_hist_ids_{split}.npy"), ids)
        stats[split] = {"rows": int(len(ids)), "window_tokens": int(window_tokens)}
    shutil.rmtree(shards)
    _wait_for_files([os.path.join(out_dir, f"ft_hist_ids_{s}.npy") for s in SPLITS])
    return stats


# ---------------------------------------------------------------------------
# Model + Ray Train fine-tune
# ---------------------------------------------------------------------------

def build_classifier(hf_dir: str):
    """The pretrained decoder with a fraud head: last real token's hidden state →
    one logit. The backbone starts from the Part 4 weights and is fully trainable."""
    import torch
    from transformers import AutoModelForCausalLM

    class FraudClassifier(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.backbone = lm.model  # the LlamaModel inside LlamaForCausalLM
            self.head = torch.nn.Linear(lm.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            h = self.backbone(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
            last = attention_mask.sum(dim=1) - 1  # last real token per row
            pooled = h[torch.arange(h.size(0), device=h.device), last]
            return self.head(pooled).squeeze(-1)

    lm = AutoModelForCausalLM.from_pretrained(hf_dir)
    return FraudClassifier(lm)


def _predict(model, ids: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    import torch
    model.eval()
    out = np.empty(len(ids), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            b = torch.from_numpy(ids[i:i + batch_size].astype(np.int64)).to(device)
            attn = (b != 0).long()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(b, attn)
            out[i:i + len(b)] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def train_func_ft(config: dict):
    import torch
    from sklearn.metrics import average_precision_score

    torch.manual_seed(config.get("seed", 0))
    model = build_classifier(config["hf_dir"])
    model = ray.train.torch.prepare_model(model)
    base = model.module if hasattr(model, "module") else model
    device = next(base.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config.get("weight_decay", 0.01))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    val_ids = np.load(config["val_ids"])
    val_lbl = np.load(config["val_lbl"]).astype(int)
    rank = ray.train.get_context().get_world_rank()

    shard = ray.train.get_dataset_shard("train")
    best_ap = -1.0
    for epoch in range(config["epochs"]):
        model.train()
        running, n = 0.0, 0
        for batch in shard.iter_torch_batches(batch_size=config["batch_size"],
                                              dtypes={"input_ids": torch.long,
                                                      "label": torch.float32}):
            ids, y = batch["input_ids"], batch["label"]
            attn = (ids != 0).long()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                logits = model(ids, attn)
                loss = loss_fn(logits.float(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n += 1

        metrics = {"epoch": epoch, "loss": running / max(n, 1)}
        if rank == 0:
            val_ap = float(average_precision_score(val_lbl, _predict(base, val_ids, device)))
            metrics["val_ap"] = val_ap
            print(f"[finetune] epoch {epoch + 1}/{config['epochs']}  "
                  f"loss={metrics['loss']:.4f}  val_ap={val_ap:.4f}", flush=True)
            if val_ap > best_ap:  # keep the best-on-val weights, like NB05's early stop
                best_ap = val_ap
                os.makedirs(config["out_dir"], exist_ok=True)
                torch.save(base.state_dict(), os.path.join(config["out_dir"], "model.pt"))
                with open(os.path.join(config["out_dir"], "meta.json"), "w") as f:
                    json.dump({"best_epoch": epoch, "val_ap": val_ap,
                               "variant": config.get("variant", "?")}, f)
        ray.train.report(metrics)


def finetune(hf_dir: str, tokens_dir: str, out_dir: str, variant: str = "single",
             epochs: int = 3, batch_size: int = 64, lr: float = 2e-5,
             num_workers: int = 1, use_gpu: bool = False, storage_base: str = None) -> dict:
    """Fine-tune on ``ft_ids_*`` (variant='single') or ``ft_hist_ids_*``
    (variant='history'); saves the best-on-val model to ``out_dir``."""
    import ray.data
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    prefix = "ft_ids" if variant == "single" else "ft_hist_ids"
    ids = np.load(os.path.join(tokens_dir, f"{prefix}_train.npy"))
    lbl = np.load(os.path.join(tokens_dir, "lbl_train.npy")).astype(np.float32)
    train_ds = (ray.data.from_numpy(ids).rename_columns({"data": "input_ids"})
                .zip(ray.data.from_numpy(lbl).rename_columns({"data": "label"})))

    trainer = TorchTrainer(
        train_func_ft,
        train_loop_config={
            "hf_dir": hf_dir, "out_dir": out_dir, "variant": variant,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "val_ids": os.path.join(tokens_dir, f"{prefix}_val.npy"),
            "val_lbl": os.path.join(tokens_dir, "lbl_val.npy"),
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={"train": train_ds},
        run_config=RunConfig(name=f"transaction_fm_finetune_{variant}",
                             storage_path=os.path.join(storage_base or os.path.dirname(out_dir),
                                                       "ray_results")),
    )
    trainer.fit()
    _wait_for_files([os.path.join(out_dir, "model.pt")])
    return json.load(open(os.path.join(out_dir, "meta.json")))


# ---------------------------------------------------------------------------
# Scoring: the fine-tuned model alone, and fused with the raw features
# ---------------------------------------------------------------------------

@ray.remote
def _score_task(hf_dir, model_dir, tokens_dir, variant, use_gpu, emb_dir):
    import sys
    sys.path.insert(0, ".")
    import pandas as pd
    import torch
    import xgboost as xgb
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import OrdinalEncoder

    from src.finetune import _predict, build_classifier
    from src.nvscore import _PARAMS

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = build_classifier(hf_dir)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"),
                                     map_location=device))
    model.to(device)

    prefix = "ft_ids" if variant == "single" else "ft_hist_ids"
    scores, y = {}, {}
    for sp in SPLITS:
        ids = np.load(os.path.join(tokens_dir, f"{prefix}_{sp}.npy"))
        scores[sp] = _predict(model, ids, device)
        y[sp] = np.load(os.path.join(tokens_dir, f"lbl_{sp}.npy")).astype(int)

    out = {"finetuned": {"auc_roc": float(roc_auc_score(y["test"], scores["test"])),
                         "pr_auc": float(average_precision_score(y["test"], scores["test"]))}}

    # SFT + raw fusion: the fine-tuned score as one extra column next to the 13 raw
    # features, fit with NVIDIA's fusion param set (their NB05 recipe, unchanged).
    raw = {sp: pd.read_parquet(os.path.join(emb_dir, f"raw_{sp}.parquet")) for sp in SPLITS}
    pre = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         make_column_selector(dtype_include=["object", "category"])),
        remainder="passthrough")
    X = {"train": pre.fit_transform(raw["train"]),
         "val": pre.transform(raw["val"]), "test": pre.transform(raw["test"])}
    Xf = {sp: np.column_stack([np.asarray(X[sp], dtype=np.float32), scores[sp]])
          for sp in SPLITS}
    clf = xgb.XGBClassifier(**_PARAMS["fusion"], scale_pos_weight=1.0, tree_method="hist",
                            device=("cuda" if use_gpu else "cpu"),
                            early_stopping_rounds=20, eval_metric="auc")
    clf.fit(Xf["train"], y["train"], eval_set=[(Xf["val"], y["val"])], verbose=False)
    p = clf.predict_proba(Xf["test"])[:, 1]
    out["finetuned_plus_raw"] = {"auc_roc": float(roc_auc_score(y["test"], p)),
                                 "pr_auc": float(average_precision_score(y["test"], p)),
                                 "best_iteration": int(clf.best_iteration)}
    np.save(os.path.join(model_dir, "test_scores.npy"), scores["test"])
    return out


def score_finetuned(hf_dir: str, model_dir: str, tokens_dir: str, emb_dir: str,
                    variant: str = "single", use_gpu: bool = False) -> dict:
    """Test-set AP/AUC for the fine-tuned model, alone and fused with the raw
    features — same eval as Part 6, so the numbers slot into the same table."""
    ray.init(ignore_reinit_error=True)
    opts = {"num_gpus": 1, "num_cpus": 8} if use_gpu else {"num_cpus": 4}
    res = ray.get(_score_task.options(**opts).remote(hf_dir, model_dir, tokens_dir,
                                                     variant, use_gpu, emb_dir))
    with open(os.path.join(model_dir, "scores.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res
