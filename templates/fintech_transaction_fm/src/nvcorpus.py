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
    from src.nvsplit import train_parquet_files

    gdf = cudf.read_parquet(train_parquet_files(train_parquet))
    if "__seq__" in gdf.columns:  # sharded split: restore CSV row order, then drop
        gdf = gdf.sort_values("__seq__").drop(columns=["__seq__"]).reset_index(drop=True)
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

    Reference implementation (NVIDIA's single-GPU shape). ``chunk`` = transactions per
    sequence (NVIDIA uses 315 → ~4096 tokens); ``max_seq`` caps the number of sequences
    (mini/CI). Requires Ray initialized with the template as the working dir (the
    notebook's ``ray.init`` handles this) so ``src.nvidia_tokenizer`` is importable on
    the worker.
    """
    ray.init(ignore_reinit_error=True)
    meta = ray.get(_build.remote(train_parquet, out_dir, seq_len, chunk, merchant_hash, max_seq))
    _wait_for_files([os.path.join(out_dir, f) for f in ("ids.npy", "attn.npy", "vocab.json")])
    return meta


# ---------------------------------------------------------------------------
# Ray Data implementation — the corpus build sharded per card on CPU workers.
#
# Tokenization is independent per (User, Card): the vocab is static (6251) and each
# card's sequences depend only on that card's rows. So the 19.5M-row train split
# shuffles into ~6.1K card groups, each group tokenizes + chunks + encodes on a CPU
# worker via the identity-verified pandas mirror (src/nvtokenize_cpu, Stage 0), and a
# final task assembles ids.npy/attn.npy in the reference's (user, card, chunk) order.
# ---------------------------------------------------------------------------

_TOK = None  # per-worker-process tokenizer cache (vocab build is data-free but not free)


def tokenize_card_group(group, seq_len: int = 4096, chunk: int = 315,
                        merchant_hash: int = 2000):
    """One card's rows in (any order) → its encoded pretrain sequences out.

    Mirrors the reference build exactly: restore CSV order (``__seq__``), derive the 12
    token strings per transaction, chunk ``chunk`` consecutive transactions, join as
    ``<bos> t1 <sep> t2 ... <eos>``, encode to ``seq_len`` ids.
    """
    import numpy as np
    import pandas as pd

    from src.nvidia_tokenizer import FinancialTabularTokenizer
    from src.nvtokenize_cpu import preprocess_cpu, transform_cpu

    global _TOK
    if _TOK is None:
        _TOK = FinancialTabularTokenizer(merchant_hash_size=merchant_hash,
                                         category_hierarchy=True, temporal_encoding=True)

    user, card = int(group["User"].iloc[0]), int(group["Card"].iloc[0])
    if "__seq__" in group.columns:
        group = group.sort_values("__seq__", kind="mergesort")
    gp = preprocess_cpu(group)
    td = transform_cpu(gp, merchant_hash_size=merchant_hash)
    txt = td.iloc[:, 0].str.cat([td[c] for c in td.columns[1:]], sep=" ")

    rows = []
    for ci, lo in enumerate(range(0, len(txt), chunk)):
        line = "<bos> " + " <sep> ".join(txt.iloc[lo:lo + chunk]) + " <eos>"
        ids = _TOK.encode(line, max_length=seq_len)
        rows.append({"user": user, "card": card, "chunk": ci,
                     "ids": np.asarray(ids, dtype=np.int32).tolist()})
    return pd.DataFrame(rows)


@ray.remote(num_cpus=8)
def _assemble(rows_dir: str, out_dir: str, seq_len: int, max_seq) -> dict:
    """Collect the per-card sequences, order them exactly as the reference does
    (pandas groupby sorts keys → (user, card, chunk) ascending), write npy + vocab."""
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd

    from src.nvidia_tokenizer import FinancialTabularTokenizer
    from src.nvsplit import ordered_parquet_files

    df = pd.concat([pd.read_parquet(f) for f in ordered_parquet_files(rows_dir)],
                   ignore_index=True)
    df = df.sort_values(["user", "card", "chunk"], kind="mergesort").reset_index(drop=True)
    if max_seq is not None and len(df) > max_seq:
        df = df.iloc[:max_seq]

    tok = FinancialTabularTokenizer(merchant_hash_size=2000, category_hierarchy=True,
                                    temporal_encoding=True)
    pad = tok.vocab.get("<pad>", 0)
    ids = np.stack([np.asarray(r, dtype=np.int32) for r in df["ids"]])
    attn = (ids != pad).astype(np.int32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ids.npy"), ids)
    np.save(os.path.join(out_dir, "attn.npy"), attn)
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump({"vocab_size": int(tok.vocab_size), "seq_length": int(seq_len),
                   "pad": int(pad)}, f, indent=2)
    return {"n_seq": int(len(ids)), "seq_len": int(seq_len),
            "vocab_size": int(tok.vocab_size),
            "real_token_frac": float(attn.sum() / attn.size)}


def fresh_seqs_dir(out_dir: str) -> str:
    """Empty temp dir for the per-card sequence rows (consumed by assemble_corpus)."""
    import shutil
    rows_dir = os.path.join(out_dir, "_seqs_tmp")
    if os.path.isdir(rows_dir):
        shutil.rmtree(rows_dir)
    os.makedirs(rows_dir, exist_ok=True)
    return rows_dir


def assemble_corpus(rows_dir: str, out_dir: str, seq_len: int, max_seq=None) -> dict:
    """Collect the per-card sequences into ``ids.npy``/``attn.npy``/``vocab.json`` in
    the reference's (user, card, chunk) order; removes the temp rows dir."""
    import shutil
    ray.init(ignore_reinit_error=True)
    meta = ray.get(_assemble.remote(rows_dir, out_dir, seq_len, max_seq))
    shutil.rmtree(rows_dir)
    _wait_for_files([os.path.join(out_dir, f) for f in ("ids.npy", "attn.npy", "vocab.json")])
    return meta


def build_corpus_distributed(split_dir: str, out_dir: str, seq_len: int = 4096,
                             chunk: int = 315, merchant_hash: int = 2000,
                             max_seq=None) -> dict:
    """The corpus build as a Ray Data pipeline: shuffle the train split into per-card
    groups, tokenize+encode each on CPU workers, assemble in reference order.

    Same output as :func:`build_corpus` (verified — see PLAN_RAY_DATA.md Stage 2);
    execution is CPU-only and scales with the worker pool. This is the headless
    composition of the pieces Part 3 shows inline: ``groupby(User, Card) →
    map_groups(tokenize_card_group) → assemble_corpus``.
    """
    import ray.data

    from src.nvsplit import train_parquet_files

    ray.init(ignore_reinit_error=True)
    rows_dir = fresh_seqs_dir(out_dir)
    ray.data.read_parquet(train_parquet_files(split_dir)) \
        .groupby(["User", "Card"]) \
        .map_groups(lambda g: tokenize_card_group(g, seq_len, chunk, merchant_hash),
                    batch_format="pandas") \
        .write_parquet(rows_dir)
    return assemble_corpus(rows_dir, out_dir, seq_len, max_seq)
