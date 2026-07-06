"""Step 1 of re-pretrain: build the pretraining corpus from NVIDIA's temporal train
split using THEIR tokenizer (group by user/card, 315-txn chunks -> 4096-token sequences),
encode to ids, and save. Mirrors their NB02 + clm_data.load_corpus_and_tokenize."""
import ray, json
ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1, num_cpus=8)
def build():
    import sys, time, numpy as np
    sys.path.insert(0, ".")
    import cuml.preprocessing  # shim
    import cudf
    from src.tokenizer import FinancialTokenizerPipeline, FinancialTabularTokenizer
    SD = "/mnt/cluster_storage/nvidia_data/temporal_split"
    HASH, CHUNK, ML = 2000, 315, 4096

    gdf = cudf.read_parquet(f"{SD}/train.parquet")
    print("train rows", len(gdf), flush=True)
    pip = FinancialTokenizerPipeline(merchant_hash_size=HASH)
    gp = pip.preprocess(gdf); pip.fit(gp); tdf = pip.transform(gp)
    gcols = []
    for c in ["user", "User", "cust"]:
        if c in gp.columns: gcols.append(c); break
    for c in ["card", "Card", "card_id"]:
        if c in gp.columns: gcols.append(c); break
    print("group cols", gcols, flush=True)
    # Assemble corpus lines exactly like pipeline.to_corpus_lines, but on CPU (pandas)
    # via Arrow — the cudf groupby->to_pandas path hits a broken cupy JIT (cuda_fp16.h)
    # on this worker. Same output: group by user/card, chunk 315, "<bos> t <sep> ... <eos>".
    import pandas as pd
    tcols = list(tdf.columns)
    tp = tdf.to_arrow().to_pandas()
    gpm = gp[gcols].to_arrow().to_pandas().reset_index(drop=True)
    txn_text = tp[tcols[0]].str.cat([tp[c] for c in tcols[1:]], sep=" ")
    work = gpm.copy()
    work["_txt"] = txn_text.values
    work["_seq"] = work.groupby(gcols).cumcount()
    work["_chunk"] = (work["_seq"] // CHUNK).astype("int32")
    grouped = work.groupby(gcols + ["_chunk"])["_txt"].agg(list)
    lines = grouped.map(lambda l: "<bos> " + " <sep> ".join(l) + " <eos>").tolist()
    print("corpus sequences", len(lines), flush=True)

    tok = FinancialTabularTokenizer(merchant_hash_size=HASH, category_hierarchy=True,
                                    temporal_encoding=True)
    pad = tok.vocab.get("<pad>", 0)
    print("vocab", tok.vocab_size, "pad", pad, flush=True)
    ids = np.full((len(lines), ML), pad, dtype=np.int32)
    t0 = time.time()
    for i, line in enumerate(lines):
        ids[i] = tok.encode(line, max_length=ML)
        if i % 5000 == 0:
            print(f"  encoded {i}/{len(lines)} {time.time()-t0:.0f}s", flush=True)
    attn = (ids != pad).astype(np.int32)
    import os
    os.makedirs("/mnt/cluster_storage/nvpretrain", exist_ok=True)
    np.save("/mnt/cluster_storage/nvpretrain/ids.npy", ids)
    np.save("/mnt/cluster_storage/nvpretrain/attn.npy", attn)
    json.dump({"vocab_size": tok.vocab_size, "seq_length": ML, "pad": pad},
              open("/mnt/cluster_storage/nvpretrain/vocab.json", "w"), indent=2)
    real = int(attn.sum()); tot = attn.size
    print(f"DONE ids {ids.shape} real-token frac {real/tot:.3f}", flush=True)
    return {"n_seq": len(lines), "seq_len": ML, "vocab": tok.vocab_size}

r = ray.get(build.remote())
json.dump(r, open("/mnt/cluster_storage/nvpretrain_built.json", "w"))
print("BUILT", r)
