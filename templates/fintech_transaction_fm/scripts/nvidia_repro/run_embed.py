"""Faithful NB04: tokenize NVIDIA's temporal_split with THEIR tokenizer + embed with
THEIR pretrained weights (last-token pooling), on a Ray GPU worker. Produces fresh
train/val/test embeddings. Mirrors 04_inference_embedding_extraction.ipynb."""
import ray, json
ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1, num_cpus=8)
def go():
    import sys, time, numpy as np
    sys.path.insert(0, ".")  # working_dir = their repo (+ cuml shim)
    import cuml.preprocessing  # noqa: F401  (shim)
    import cudf
    from src.tokenizer import FinancialTokenizerPipeline, FinancialTabularTokenizer
    from src.decoder_inference import HuggingFaceDecoderInference

    MODEL = "/mnt/cluster_storage/nvidia_model"
    SD = "/mnt/cluster_storage/nvidia_data/temporal_split"
    ML, BS, HASH, BAL = 128, 1024, 2000, 1_000_000

    tok = FinancialTabularTokenizer(merchant_hash_size=HASH, category_hierarchy=True,
                                    temporal_encoding=True)
    print("tokenizer vocab", tok.vocab_size, flush=True)
    inf = HuggingFaceDecoderInference(model_path=MODEL, tokenizer=tok, pooling="last_token")
    print("model loaded, embed_dim", inf.embedding_dim, flush=True)

    res = {}
    for split, pq in [("train", "train.parquet"), ("val", "val_eval.parquet"),
                      ("test", "test_eval.parquet")]:
        gdf = cudf.read_parquet(f"{SD}/{pq}")
        fr = gdf["Is Fraud?"]
        lbl = ((fr == "Yes") | (fr == "1")).astype("int32").to_pandas().to_numpy()
        if split == "train":
            f = np.where(lbl == 1)[0].tolist(); nrm = np.where(lbl == 0)[0].tolist()
            np.random.seed(42)
            nf = min(len(f), int(BAL * 0.1)); nn = min(len(nrm), BAL - nf)
            s = np.concatenate([np.random.choice(f, nf, replace=False),
                                np.random.choice(nrm, nn, replace=False)])
            np.random.shuffle(s)
            gdf = gdf.iloc[s].reset_index(drop=True); lbl = lbl[s]
        gdf["__row_id__"] = np.arange(len(gdf), dtype=np.int64)
        pip = FinancialTokenizerPipeline(merchant_hash_size=HASH)
        gdf = pip.preprocess(gdf)
        rid = gdf["__row_id__"].to_pandas().to_numpy(np.int64)
        lbl = lbl[rid]
        pip.fit(gdf)
        tdf = pip.transform(gdf)
        ids = pip.encode(tdf, max_length=ML)
        t0 = time.time()
        emb = inf.extract_embeddings_batched(ids, batch_size=BS, show_progress=False)
        print(f"{split}: emb {emb.shape} fraud {int(lbl.sum())}/{len(lbl)} in {time.time()-t0:.0f}s", flush=True)
        np.save(f"/mnt/cluster_storage/nvfresh_embed_{split}.npy", emb)
        np.save(f"/mnt/cluster_storage/nvfresh_lbl_{split}.npy", lbl)
        res[split] = [int(x) for x in emb.shape]
    return res

r = ray.get(go.remote())
json.dump(r, open("/mnt/cluster_storage/nvfresh_done.json", "w"))
print("DONE", r, flush=True)
