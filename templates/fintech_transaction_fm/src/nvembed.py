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

    tok = FinancialTabularTokenizer(merchant_hash_size=merchant_hash, category_hierarchy=True,
                                    temporal_encoding=True)
    inf = HuggingFaceDecoderInference(model_path=hf_dir, tokenizer=tok, pooling="last_token")
    print(f"[nvembed] model loaded, embed_dim {inf.embedding_dim}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    files = {"train": "train.parquet", "val": "val_eval.parquet", "test": "test_eval.parquet"}
    stats = {}
    for split, fname in files.items():
        gdf = cudf.read_parquet(os.path.join(split_dir, fname))
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


def embed_splits(hf_dir: str, split_dir: str, out_dir: str, balanced_train: int = 1_000_000,
                 max_length: int = 128, batch_size: int = 1024, merchant_hash: int = 2000,
                 seed: int = 42) -> dict:
    """Embed the train/val/test splits with our HF decoder (GPU task). Requires Ray
    initialized with the template as working dir (the notebook handles this)."""
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
