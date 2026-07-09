"""Stage 1 of PLAN_RAY_DATA.md: prove the Ray Data (CPU) split equals the reference
single-GPU split, byte-for-byte in content and row order.

    python scripts/verify_distributed_split.py run       # build distributed split -> nvsplit_rd/full
    python scripts/verify_distributed_split.py compare   # vs reference nvsplit/full

Checks: cutoff dates, row/fraud counts, val_eval + test_eval frame equality, and full
19.5M-row train equality (values AND order) in a single big-CPU-node task.
"""
import json
import os
import sys
import time

import ray

BASE = "/mnt/cluster_storage/transaction-fm"
CSV = f"{BASE}/source/card_transaction.v1.csv"
REF = f"{BASE}/nvsplit/full"
NEW = f"{BASE}/nvsplit_rd/full"


def run():
    sys.path.insert(0, ".")
    from src.nvsplit import build_temporal_split_distributed
    t0 = time.time()
    meta = build_temporal_split_distributed(CSV, NEW, eval_samples=100_000,
                                            max_users=None, seed=42)
    meta["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(meta, indent=2))
    with open(f"{NEW}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


@ray.remote(num_cpus=16)
def _compare_train(ref_file: str, new_dir: str) -> dict:
    import pandas as pd
    sys.path.insert(0, ".")
    from src.nvsplit import ordered_parquet_files

    ref = pd.read_parquet(ref_file)
    new = pd.concat([pd.read_parquet(f) for f in ordered_parquet_files(new_dir)],
                    ignore_index=True)
    out = {"ref_rows": len(ref), "new_rows": len(new)}
    if len(ref) != len(new):
        out["equal"] = False
        return out
    new = new[list(ref.columns)]
    try:
        pd.testing.assert_frame_equal(ref.reset_index(drop=True), new,
                                      check_dtype=False, check_exact=True)
        out["equal"] = True
    except AssertionError as e:
        out["equal"] = False
        out["detail"] = str(e)[:500]
    return out


def compare():
    import pandas as pd
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "."})
    report = {}

    ref_meta = json.load(open(f"{REF}/split_meta.json"))
    new_meta = json.load(open(f"{NEW}/split_meta.json"))
    report["cutoffs_equal"] = (ref_meta["train_cutoff"] == new_meta["train_cutoff"]
                               and ref_meta["test_cutoff"] == new_meta["test_cutoff"])
    report["cutoffs"] = {"ref": [ref_meta["train_cutoff"], ref_meta["test_cutoff"]],
                         "new": [new_meta["train_cutoff"], new_meta["test_cutoff"]]}
    report["train_counts_equal"] = (ref_meta["train"] == new_meta["train"])
    report["train_counts"] = {"ref": ref_meta["train"], "new": new_meta["train"]}

    for name in ("val_eval", "test_eval"):
        ref = pd.read_parquet(f"{REF}/{name}.parquet")
        new = pd.read_parquet(f"{NEW}/{name}.parquet")
        new = new[list(ref.columns)]
        try:
            pd.testing.assert_frame_equal(ref.reset_index(drop=True),
                                          new.reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
            report[f"{name}_equal"] = True
        except AssertionError as e:
            report[f"{name}_equal"] = False
            report[f"{name}_detail"] = str(e)[:500]

    report["train_frame"] = ray.get(_compare_train.remote(
        f"{REF}/train.parquet", f"{NEW}/train_parquet"))

    report["ALL_IDENTICAL"] = bool(
        report["cutoffs_equal"] and report["val_eval_equal"]
        and report["test_eval_equal"] and report["train_frame"]["equal"])
    with open(f"{NEW}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
