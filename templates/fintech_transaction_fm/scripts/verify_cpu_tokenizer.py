"""Stage 0 of PLAN_RAY_DATA.md: prove the CPU tokenizer path (src/nvtokenize_cpu.py)
produces byte-identical output to the vendored cuDF/GPU path.

Run from the template root (so Ray ships src/ as the working dir):

    python scripts/verify_cpu_tokenizer.py dump      # one GPU-worker task -> reference dump
    python scripts/verify_cpu_tokenizer.py compare   # local CPU mirror vs the dump

Checks: (1) merchant-name cleaning, (2) murmur3 hashes over ALL distinct merchants,
(3) row order after the (user, card, time_full) sort — the tie-break risk,
(4) all 12 token columns row-by-row, (5) the full token->id vocab.
"""
import json
import os
import subprocess
import sys

import ray

BASE = "/mnt/cluster_storage/transaction-fm"
TRAIN = f"{BASE}/nvsplit/full/train.parquet"
OUT = f"{BASE}/stage0"
SAMPLE_ROWS = 200_000


@ray.remote(num_gpus=1, num_cpus=8)
def _dump() -> dict:
    import sys
    sys.path.insert(0, ".")
    import cudf
    import numpy as np
    from src.nvidia_tokenizer import FinancialTabularTokenizer, FinancialTokenizerPipeline

    os.makedirs(OUT, exist_ok=True)
    gdf = cudf.read_parquet(TRAIN)

    # (A) reference hash of every distinct merchant name, cleaned exactly as preprocess does
    raw = gdf["Merchant Name"].astype(str).unique()
    mdf = cudf.DataFrame({"raw": raw})
    mdf["clean"] = mdf["raw"].str.upper().str.replace(r"[^A-Z0-9\s\-]", "", regex=True)
    mdf["hash"] = mdf["clean"].hash_values()
    mdf.to_pandas().to_parquet(f"{OUT}/merchant_hash_gpu.parquet", index=False)

    # (B) reference preprocess+transform on the first SAMPLE_ROWS rows, row-tagged first
    sub = gdf.head(SAMPLE_ROWS)
    sub["__row_id__"] = cudf.Series(np.arange(len(sub), dtype=np.int64))
    pip = FinancialTokenizerPipeline(merchant_hash_size=2000)
    gp = pip.preprocess(sub)
    pip.fit(gp)
    tdf = pip.transform(gp)

    out = gp[["__row_id__", "user", "card", "time_full"]].to_arrow().to_pandas()
    tp = tdf.to_arrow().to_pandas()
    for c in tdf.columns:
        out[c] = tp[c].values
    out.to_parquet(f"{OUT}/tokens_gpu.parquet", index=False)

    # (C) reference vocab
    tok = FinancialTabularTokenizer(merchant_hash_size=2000, category_hierarchy=True,
                                    temporal_encoding=True)
    with open(f"{OUT}/vocab_gpu.json", "w") as f:
        json.dump({"vocab_size": int(tok.vocab_size), "vocab": tok.vocab}, f)

    return {"train_rows": int(len(gdf)), "distinct_merchants": int(len(mdf)),
            "sample_rows": int(len(out)), "vocab_size": int(tok.vocab_size),
            "token_cols": list(tdf.columns)}


def dump():
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "."})
    meta = ray.get(_dump.remote())
    print(json.dumps(meta, indent=2))


def compare():
    try:
        import mmh3  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mmh3"])

    import numpy as np
    import pandas as pd
    sys.path.insert(0, ".")
    from src.nvtokenize_cpu import (TOKEN_COLS, clean_merchant, merchant_hash,
                                    preprocess_cpu, transform_cpu)

    report = {}

    # (1)+(2) merchant cleaning + murmur3
    m = pd.read_parquet(f"{OUT}/merchant_hash_gpu.parquet")
    cpu_clean = clean_merchant(m["raw"])
    clean_eq = (cpu_clean.to_numpy() == m["clean"].to_numpy())
    report["merchant_clean_equal"] = f"{int(clean_eq.sum())}/{len(m)}"
    cpu_hash = merchant_hash(cpu_clean)
    hash_eq = (cpu_hash == m["hash"].to_numpy().astype(np.uint32))
    report["merchant_hash_equal"] = f"{int(hash_eq.sum())}/{len(m)}"
    if not hash_eq.all():
        bad = np.where(~hash_eq)[0][:5]
        report["merchant_hash_examples"] = [
            {"raw": str(m['raw'].iloc[i]), "clean": str(cpu_clean.iloc[i]),
             "gpu": int(m['hash'].iloc[i]), "cpu": int(cpu_hash[i])} for i in bad]

    # (3)+(4) row order + token columns on the same sample rows
    base = pd.read_parquet(TRAIN).head(SAMPLE_ROWS).copy()
    base["__row_id__"] = np.arange(len(base), dtype=np.int64)
    gp = preprocess_cpu(base)
    td = transform_cpu(gp)

    ref = pd.read_parquet(f"{OUT}/tokens_gpu.parquet")
    order_eq = (gp["__row_id__"].to_numpy() == ref["__row_id__"].to_numpy())
    report["sort_order_equal"] = f"{int(order_eq.sum())}/{len(ref)}"

    # tokens compared aligned by __row_id__ (order-independent correctness)
    cpu_by_rid = td.set_axis(gp["__row_id__"].to_numpy(), axis=0).sort_index()
    gpu_by_rid = ref.set_index("__row_id__").sort_index()
    report["token_cols"] = {}
    for c in TOKEN_COLS:
        eq = (cpu_by_rid[c].to_numpy() == gpu_by_rid[c].astype(str).to_numpy())
        report["token_cols"][c] = f"{int(eq.sum())}/{len(eq)}"
        if not eq.all():
            i = int(np.where(~eq)[0][0])
            report.setdefault("token_examples", {})[c] = {
                "row_id": int(gpu_by_rid.index[i]),
                "gpu": str(gpu_by_rid[c].iloc[i]), "cpu": str(cpu_by_rid[c].iloc[i])}

    # (5) vocab
    from src.nvidia_tokenizer import FinancialTabularTokenizer
    tok = FinancialTabularTokenizer(merchant_hash_size=2000, category_hierarchy=True,
                                    temporal_encoding=True)
    refv = json.load(open(f"{OUT}/vocab_gpu.json"))
    report["vocab_size_cpu"] = int(tok.vocab_size)
    report["vocab_size_gpu"] = int(refv["vocab_size"])
    report["vocab_equal"] = (dict(tok.vocab) == {k: int(v) for k, v in refv["vocab"].items()})

    all_ok = (clean_eq.all() and hash_eq.all() and order_eq.all() and report["vocab_equal"]
              and all(v.split("/")[0] == v.split("/")[1] for v in report["token_cols"].values()))
    report["ALL_IDENTICAL"] = bool(all_ok)

    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    {"dump": dump, "compare": compare}[sys.argv[1]]()
