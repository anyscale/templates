"""Embed single transactions with our pretrained decoder, using the vendored NVIDIA
tokenizer + ``HuggingFaceDecoderInference`` (last-token pooling). Mirrors NVIDIA's NB04
(and ``scripts/nvidia_repro/run_ours_full.py``): each transaction is tokenized alone
(``<bos>`` + its field tokens + ``<eos>``, padded to ``max_length``) and embedded from the
last real token.

For each split (train balanced-sampled ~10% fraud, val_eval, test_eval) writes, in a
deterministic row order:
* ``embed_<split>.npy`` — the embedding matrix,
* ``lbl_<split>.npy``   — the int labels,
* ``raw_<split>.parquet`` — NVIDIA's 13 raw feature columns (so Part 6 needs no tokenizer).
"""
import os

import ray

from src.nvsplit import _wait_for_files

# NVIDIA's 13 raw feature columns (Hour derived from Time; Amount parsed from "$..,..").
FC = ["User", "Card", "Year", "Month", "Day", "Hour", "Amount", "Use Chip",
      "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC"]


@ray.remote(num_gpus=1, num_cpus=8)
def _embed(hf_dir: str, split_dir: str, out_dir: str, balanced_train: int,
           max_length: int, batch_size: int, merchant_hash: int, seed: int) -> dict:
    import sys
    import time

    import numpy as np

    sys.path.insert(0, ".")  # worker cwd = Ray working_dir (template) → src.* resolves
    import cudf
    from src.nvidia_tokenizer import FinancialTabularTokenizer, FinancialTokenizerPipeline
    from src.decoder_inference import HuggingFaceDecoderInference
    from src.nvsplit import train_parquet_files

    tok = FinancialTabularTokenizer(merchant_hash_size=merchant_hash, category_hierarchy=True,
                                    temporal_encoding=True)
    inf = HuggingFaceDecoderInference(model_path=hf_dir, tokenizer=tok, pooling="last_token")
    print(f"[nvembed] model loaded, embed_dim {inf.embedding_dim}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    files = {"train": None, "val": "val_eval.parquet", "test": "test_eval.parquet"}
    stats = {}
    for split, fname in files.items():
        # train may be the reference single file or the distributed sharded dir
        src_files = (train_parquet_files(split_dir) if fname is None
                     else [os.path.join(split_dir, fname)])
        gdf = cudf.read_parquet(src_files)
        if "__seq__" in gdf.columns:  # sharded split: restore CSV row order, then drop
            gdf = gdf.sort_values("__seq__").drop(columns=["__seq__"]).reset_index(drop=True)
        fr = gdf["Is Fraud?"]
        lbl = ((fr == "Yes") | (fr == "1")).astype("int32").to_pandas().to_numpy()
        if split == "train":  # balanced ~10%-fraud training sample (NVIDIA NB01)
            f = np.where(lbl == 1)[0]
            nrm = np.where(lbl == 0)[0]
            rng = np.random.RandomState(seed)
            nf = min(len(f), int(balanced_train * 0.1))
            nn = min(len(nrm), balanced_train - nf)
            sel = np.concatenate([rng.choice(f, nf, replace=False), rng.choice(nrm, nn, replace=False)])
            rng.shuffle(sel)
            gdf = gdf.iloc[sel].reset_index(drop=True)
            lbl = lbl[sel]
        base = gdf.to_pandas()
        base["Hour"] = base["Time"].str.split(":", n=1, expand=True)[0].astype(int)
        base["Amount"] = (base["Amount"].str.replace("$", "", regex=False)
                          .str.replace(",", "", regex=False).astype(float))
        base = base.reset_index(drop=True)

        gdf["__row_id__"] = np.arange(len(gdf), dtype=np.int64)
        pip = FinancialTokenizerPipeline(merchant_hash_size=merchant_hash)
        gp = pip.preprocess(gdf)
        rid = gp["__row_id__"].to_pandas().to_numpy(np.int64)
        lbl = lbl[rid]
        pip.fit(gp)
        tdf = pip.transform(gp)
        ids = pip.encode(tdf, max_length=max_length)      # one token seq per transaction
        t0 = time.time()
        emb = inf.extract_embeddings_batched(ids, batch_size=batch_size, show_progress=False)

        np.save(os.path.join(out_dir, f"embed_{split}.npy"), emb)
        np.save(os.path.join(out_dir, f"lbl_{split}.npy"), lbl)
        base.loc[rid, FC].reset_index(drop=True).to_parquet(os.path.join(out_dir, f"raw_{split}.parquet"))
        stats[split] = {"rows": int(len(emb)), "dim": int(emb.shape[1]),
                        "fraud": int(lbl.sum()), "fraud_rate": float(lbl.mean())}
        print(f"[nvembed] {split}: {emb.shape} fraud {int(lbl.sum())}/{len(lbl)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return {"embed_dim": int(inf.embedding_dim), "splits": stats}


# ---------------------------------------------------------------------------
# Ray Data implementation — streaming CPU→GPU embed.
#
# The seeded, order-sensitive preamble (balanced train sample, the preprocess sort that
# fixes output row order) runs once per split on a CPU task, which also writes lbl_/raw_
# immediately — they never needed a GPU. Tokenization then streams per-batch on CPU
# workers, GPU actors do ONLY the forward pass, and rows carry ``__pos__`` so assembly
# restores the reference order no matter where blocks land.
# ---------------------------------------------------------------------------

PREP_COLS = ["amt_val", "merch_hash", "mcc_int", "mcc_str", "hour", "dow", "month",
             "card", "chip_upper", "zip3", "state_clean", "cust"]


@ray.remote(num_cpus=12)  # 12 CPUs only fits the 128GB node class — the full train
def prepare_embed_split(split_dir: str, fname, out_dir: str, split: str,  # frame needs it
                        balanced_train: int, seed: int) -> dict:
    """The reference preamble, verbatim on CPU: load the split in CSV order, draw the
    seeded balanced train sample, derive the 13 raw feature columns, run the preprocess
    sort that defines output row order. Writes ``lbl_<split>.npy`` +
    ``raw_<split>.parquet`` directly and a prepared parquet (tokenizer inputs +
    ``__pos__``) for the streaming stage."""
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd

    from src.nvsplit import train_parquet_files
    from src.nvtokenize_cpu import preprocess_cpu

    if fname is None:  # train — may be the sharded distributed split
        pdf = pd.concat([pd.read_parquet(f) for f in train_parquet_files(split_dir)],
                        ignore_index=True)
        if "__seq__" in pdf.columns:
            pdf = (pdf.sort_values("__seq__", kind="mergesort")
                      .drop(columns=["__seq__"]).reset_index(drop=True))
    else:
        pdf = pd.read_parquet(os.path.join(split_dir, fname))

    fr = pdf["Is Fraud?"]
    lbl = ((fr == "Yes") | (fr == "1")).astype("int32").to_numpy()
    if split == "train":  # balanced ~10%-fraud training sample (NVIDIA NB01)
        f = np.where(lbl == 1)[0]
        nrm = np.where(lbl == 0)[0]
        rng = np.random.RandomState(seed)
        nf = min(len(f), int(balanced_train * 0.1))
        nn = min(len(nrm), balanced_train - nf)
        sel = np.concatenate([rng.choice(f, nf, replace=False),
                              rng.choice(nrm, nn, replace=False)])
        rng.shuffle(sel)
        pdf = pdf.iloc[sel].reset_index(drop=True)
        lbl = lbl[sel]

    base = pdf.copy()
    base["Hour"] = base["Time"].astype(str).str.split(":", n=1, expand=True)[0].astype(int)
    base["Amount"] = (base["Amount"].astype(str).str.replace("$", "", regex=False)
                      .str.replace(",", "", regex=False).astype(float))
    base = base.reset_index(drop=True)

    pdf["__row_id__"] = np.arange(len(pdf), dtype=np.int64)
    gp = preprocess_cpu(pdf)  # the (user, card, time) sort that fixes output row order
    rid = gp["__row_id__"].to_numpy(np.int64)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"lbl_{split}.npy"), lbl[rid])
    base.loc[rid, FC].reset_index(drop=True).to_parquet(
        os.path.join(out_dir, f"raw_{split}.parquet"))

    gp = gp[PREP_COLS].reset_index(drop=True)
    gp["__pos__"] = np.arange(len(gp), dtype=np.int64)
    prep_path = os.path.join(out_dir, f"_prep_{split}.parquet")
    gp.to_parquet(prep_path, index=False)
    return {"rows": int(len(gp)), "fraud": int(lbl[rid].sum()), "prep": prep_path}


_ENC = None  # per-worker tokenizer/vocab cache


def encode_txn_batch(batch, merchant_hash: int = 2000, max_length: int = 128):
    """Per-batch mirror of ``pipeline.encode`` on single transactions: 12 token strings
    per row → ``[<bos>, 12 ids, <eos>, <pad>...]`` of length ``max_length``."""
    import numpy as np

    from src.nvidia_tokenizer import FinancialTabularTokenizer
    from src.nvtokenize_cpu import transform_cpu

    global _ENC
    if _ENC is None:
        _ENC = FinancialTabularTokenizer(merchant_hash_size=merchant_hash,
                                         category_hierarchy=True, temporal_encoding=True)
    vocab, unk = _ENC.vocab, _ENC.unk_token_id

    td = transform_cpu(batch, merchant_hash_size=merchant_hash)
    ids = np.full((len(td), max_length), _ENC.pad_token_id, dtype=np.int64)
    ids[:, 0] = _ENC.bos_token_id
    for j, col in enumerate(td.columns, start=1):
        ids[:, j] = td[col].map(vocab).fillna(unk).astype(np.int64).to_numpy()
    ids[:, len(td.columns) + 1] = _ENC.eos_token_id
    return {"__pos__": batch["__pos__"].to_numpy(np.int64), "ids": ids}


class GPUEmbedder:
    """Ray Data actor: forward passes only. One per GPU; CPU workers feed it."""

    def __init__(self, hf_dir: str, merchant_hash: int = 2000, batch_size: int = 1024):
        import os
        import sys
        # This torch build routes some ops (bmm) through triton JIT, which needs a C
        # compiler the GPU workers don't have — fall back to the standard kernels.
        os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
        sys.path.insert(0, ".")
        from src.decoder_inference import HuggingFaceDecoderInference
        from src.nvidia_tokenizer import FinancialTabularTokenizer

        tok = FinancialTabularTokenizer(merchant_hash_size=merchant_hash,
                                        category_hierarchy=True, temporal_encoding=True)
        self.inf = HuggingFaceDecoderInference(model_path=hf_dir, tokenizer=tok,
                                               pooling="last_token")
        self.batch_size = batch_size

    def __call__(self, batch):
        emb = self.inf.extract_embeddings_batched(batch["ids"], batch_size=self.batch_size,
                                                  show_progress=False)
        return {"__pos__": batch["__pos__"], "emb": emb.astype("float32")}


@ray.remote(num_cpus=8)
def _assemble_embed(shards_dir: str, out_path: str, dim: int) -> dict:
    """Restore reference row order (__pos__) and write the embedding matrix."""
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd

    from src.nvsplit import ordered_parquet_files

    df = pd.concat([pd.read_parquet(f) for f in ordered_parquet_files(shards_dir)],
                   ignore_index=True)
    df = df.sort_values("__pos__", kind="mergesort")
    emb = np.stack([np.asarray(e, dtype=np.float32) for e in df["emb"]])
    assert emb.shape[1] == dim
    np.save(out_path, emb)
    return {"rows": int(emb.shape[0]), "dim": int(emb.shape[1])}


def assemble_embeddings(shards_dir: str, out_path: str, prep_path=None,
                        embed_dim: int = 512) -> dict:
    """Public wrapper: restore reference order, write the ``embed_*.npy`` matrix, and
    clean up the temp shard dir + prepared parquet."""
    import shutil
    ray.init(ignore_reinit_error=True)
    meta = ray.get(_assemble_embed.remote(shards_dir, out_path, embed_dim))
    shutil.rmtree(shards_dir)
    if prep_path and os.path.exists(prep_path):
        os.remove(prep_path)
    _wait_for_files([out_path])
    return meta


def embed_splits_distributed(hf_dir: str, split_dir: str, out_dir: str,
                             balanced_train: int = 1_000_000, max_length: int = 128,
                             batch_size: int = 1024, merchant_hash: int = 2000,
                             seed: int = 42, num_gpu_workers: int = 4,
                             use_gpu: bool = True, embed_dim: int = 512) -> dict:
    """The embed stage as a streaming CPU→GPU Ray Data pipeline.

    Same outputs as :func:`embed_splits` (embed_/lbl_/raw_ per split, reference row
    order); execution splits by hardware: seeded preamble + tokenization on CPU
    workers, forward passes on ``num_gpu_workers`` GPU actors that never see a raw
    transaction. This is the headless composition of the pieces Part 5 shows inline.
    """
    import shutil

    import ray.data

    ray.init(ignore_reinit_error=True)
    os.makedirs(out_dir, exist_ok=True)
    files = {"train": None, "val": "val_eval.parquet", "test": "test_eval.parquet"}
    stats = {}
    for split, fname in files.items():
        prep = ray.get(prepare_embed_split.remote(split_dir, fname, out_dir, split,
                                                  balanced_train, seed))
        shards = os.path.join(out_dir, f"_emb_{split}")
        if os.path.isdir(shards):
            shutil.rmtree(shards)
        ray.data.read_parquet(prep["prep"]) \
            .map_batches(lambda b: encode_txn_batch(b, merchant_hash, max_length),
                         batch_format="pandas") \
            .map_batches(GPUEmbedder,
                         fn_constructor_kwargs={"hf_dir": hf_dir,
                                                "merchant_hash": merchant_hash,
                                                "batch_size": batch_size},
                         batch_format="numpy", batch_size=4096,
                         num_gpus=(1 if use_gpu else 0),
                         concurrency=num_gpu_workers) \
            .write_parquet(shards)
        meta = assemble_embeddings(shards, os.path.join(out_dir, f"embed_{split}.npy"),
                                   prep["prep"], embed_dim)
        stats[split] = {"rows": prep["rows"], "fraud": prep["fraud"], **meta}
    return {"embed_dim": embed_dim, "splits": stats}


def embed_splits(hf_dir: str, split_dir: str, out_dir: str, balanced_train: int = 1_000_000,
                 max_length: int = 128, batch_size: int = 1024, merchant_hash: int = 2000,
                 seed: int = 42) -> dict:
    """Embed the train/val/test splits with our HF decoder (GPU task). Reference
    implementation (NVIDIA's single-GPU shape). Requires Ray initialized with the
    template as working dir (the notebook handles this)."""
    import os as _os
    import time as _time

    ray.init(ignore_reinit_error=True)
    meta = ray.get(_embed.remote(hf_dir, split_dir, out_dir, balanced_train, max_length,
                                 batch_size, merchant_hash, seed))
    # NFS/EFS visibility guard (see nvcorpus/nvsplit)
    for split in ("train", "val", "test"):
        for f in (f"embed_{split}.npy", f"lbl_{split}.npy", f"raw_{split}.parquet"):
            p = _os.path.join(out_dir, f)
            t0 = _time.time()
            while not _os.path.exists(p):
                if _time.time() - t0 > 300:
                    raise TimeoutError(f"output not visible: {p}")
                _time.sleep(0.5)
    return meta
