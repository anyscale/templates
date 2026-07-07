"""Build the next-token pretraining corpus from the temporal-train split using the
vendored NVIDIA tokenizer (``src.nvidia_tokenizer``).

Mirrors NVIDIA's NB02 (+ ``scripts/nvidia_repro/build_corpus.py``): group each card's
transactions in time order, chunk into ``chunk`` transactions per sequence, join as
``"<bos> t1 <sep> t2 ... <eos>"``, and encode to ``seq_len`` tokens. Writes ``ids.npy``
(int32 ``(n_seq, seq_len)``, pad-filled), ``attn.npy`` and ``vocab.json`` — the inputs
Part 4 (pretrain) consumes.

Runs as one cuDF/GPU Ray task. The grouping is done on CPU/pandas via Arrow (the
cuDF groupby→to_pandas path can hit a cupy JIT bug on the worker); output is identical
to the tokenizer's ``to_corpus_lines``.
"""
import json
import os
import time

import ray


def _wait_for_files(file_paths, timeout: float = 300.0) -> None:
    """Block until each path is visible on the caller's node. ``/mnt/cluster_storage`` is
    NFS/EFS-backed, so a file just written by a worker can lag a driver read by a fraction
    of a second (papermill runs cells back-to-back and hits this). Poll to make the
    function's postcondition "outputs are readable here"."""
    for p in file_paths:
        t0 = time.time()
        while not os.path.exists(p):
            if time.time() - t0 > timeout:
                raise TimeoutError(f"output not visible after {timeout}s: {p}")
            time.sleep(0.5)


@ray.remote(num_gpus=1, num_cpus=8)
def _build(train_parquet: str, out_dir: str, seq_len: int, chunk: int,
           merchant_hash: int, max_seq) -> dict:
    import sys
    import time

    import numpy as np

    sys.path.insert(0, ".")  # worker cwd = Ray working_dir (the template) → src.* resolves
    import cudf
    from src.nvidia_tokenizer import FinancialTabularTokenizer, FinancialTokenizerPipeline

    gdf = cudf.read_parquet(train_parquet)
    print(f"[nvcorpus] train rows {len(gdf):,}", flush=True)

    pip = FinancialTokenizerPipeline(merchant_hash_size=merchant_hash)
    gp = pip.preprocess(gdf)
    pip.fit(gp)
    tdf = pip.transform(gp)

    # group by (user, card) — column names as the pipeline emits them
    gcols = []
    for cands in (["user", "User", "cust"], ["card", "Card", "card_id"]):
        for c in cands:
            if c in gp.columns:
                gcols.append(c)
                break
    print(f"[nvcorpus] group cols {gcols}", flush=True)

    tcols = list(tdf.columns)
    tp = tdf.to_arrow().to_pandas()
    gpm = gp[gcols].to_arrow().to_pandas().reset_index(drop=True)
    txn_text = tp[tcols[0]].str.cat([tp[c] for c in tcols[1:]], sep=" ")
    work = gpm.copy()
    work["_txt"] = txn_text.values
    work["_seq"] = work.groupby(gcols).cumcount()
    work["_chunk"] = (work["_seq"] // chunk).astype("int32")
    grouped = work.groupby(gcols + ["_chunk"])["_txt"].agg(list)
    lines = grouped.map(lambda parts: "<bos> " + " <sep> ".join(parts) + " <eos>").tolist()
    if max_seq is not None and len(lines) > max_seq:
        lines = lines[:max_seq]  # CI/mini cap — keep the corpus tiny
    print(f"[nvcorpus] corpus sequences {len(lines):,} (chunk={chunk}, seq_len={seq_len})", flush=True)

    tok = FinancialTabularTokenizer(merchant_hash_size=merchant_hash,
                                    category_hierarchy=True, temporal_encoding=True)
    pad = tok.vocab.get("<pad>", 0)
    ids = np.full((len(lines), seq_len), pad, dtype=np.int32)
    t0 = time.time()
    for i, line in enumerate(lines):
        ids[i] = tok.encode(line, max_length=seq_len)
        if i and i % 5000 == 0:
            print(f"[nvcorpus]   encoded {i:,}/{len(lines):,} ({time.time()-t0:.0f}s)", flush=True)
    attn = (ids != pad).astype(np.int32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ids.npy"), ids)
    np.save(os.path.join(out_dir, "attn.npy"), attn)
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump({"vocab_size": int(tok.vocab_size), "seq_length": int(seq_len), "pad": int(pad)}, f, indent=2)

    return {
        "n_seq": int(len(lines)),
        "seq_len": int(seq_len),
        "vocab_size": int(tok.vocab_size),
        "real_token_frac": float(attn.sum() / attn.size),
    }


def build_corpus(train_parquet: str, out_dir: str, seq_len: int = 4096, chunk: int = 315,
                 merchant_hash: int = 2000, max_seq=None) -> dict:
    """Build the pretrain corpus from ``train_parquet`` into ``out_dir`` (GPU task).

    ``chunk`` = transactions per sequence (NVIDIA uses 315 → ~4096 tokens); ``max_seq``
    caps the number of sequences (mini/CI). Requires Ray initialized with the template as
    the working dir (the notebook's ``ray.init`` handles this) so ``src.nvidia_tokenizer``
    is importable on the worker.
    """
    ray.init(ignore_reinit_error=True)
    meta = ray.get(_build.remote(train_parquet, out_dir, seq_len, chunk, merchant_hash, max_seq))
    _wait_for_files([os.path.join(out_dir, f) for f in ("ids.npy", "attn.npy", "vocab.json")])
    return meta
