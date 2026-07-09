"""Stage 3 of PLAN_RAY_DATA.md: prove the streaming CPU→GPU embed pipeline equals the
single-GPU reference embed stage.

    python scripts/verify_distributed_embed.py run       # build -> embeddings_rd/full
    python scripts/verify_distributed_embed.py compare   # vs reference embeddings/full

Labels and raw features must be byte-equal (they define the downstream task); the
embedding matrices are compared with allclose + max-abs-diff (same per-row math, but
GPU kernels may differ across batch boundaries at float precision). The decisive
functional check is Stage 4's downstream re-run.
"""
import json
import sys
import time

import ray

BASE = "/mnt/cluster_storage/transaction-fm"
SPLIT = f"{BASE}/nvsplit_rd/full"
HF = f"{BASE}/model_hf/full"
REF = f"{BASE}/embeddings/full"
NEW = f"{BASE}/embeddings_rd/full"


def run():
    sys.path.insert(0, ".")
    from src.nvembed import embed_splits_distributed
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "."})
    t0 = time.time()
    meta = embed_splits_distributed(HF, SPLIT, NEW, balanced_train=1_000_000,
                                    max_length=128, batch_size=1024,
                                    num_gpu_workers=4, use_gpu=True)
    meta["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(meta, indent=2))
    with open(f"{NEW}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


@ray.remote(num_cpus=8)
def _compare_split(split: str) -> dict:
    import numpy as np
    import pandas as pd

    out = {}
    ref_l = np.load(f"{REF}/lbl_{split}.npy")
    new_l = np.load(f"{NEW}/lbl_{split}.npy")
    out["lbl_equal"] = bool(np.array_equal(ref_l, new_l))

    ref_r = pd.read_parquet(f"{REF}/raw_{split}.parquet")
    new_r = pd.read_parquet(f"{NEW}/raw_{split}.parquet")
    for d in (ref_r, new_r):
        for c in d.columns:
            if d[c].dtype == object or str(d[c].dtype) == "string":
                d[c] = d[c].astype("string")
    try:
        pd.testing.assert_frame_equal(ref_r.reset_index(drop=True),
                                      new_r[list(ref_r.columns)].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        out["raw_equal"] = True
    except AssertionError as e:
        out["raw_equal"] = False
        out["raw_detail"] = str(e)[:300]

    ref_e = np.load(f"{REF}/embed_{split}.npy")
    new_e = np.load(f"{NEW}/embed_{split}.npy")
    out["embed_shape"] = [list(ref_e.shape), list(new_e.shape)]
    if ref_e.shape == new_e.shape:
        out["embed_exact"] = bool(np.array_equal(ref_e, new_e))
        out["embed_max_abs_diff"] = float(np.max(np.abs(ref_e - new_e)))
        out["embed_allclose"] = bool(np.allclose(ref_e, new_e, rtol=1e-4, atol=2e-4))
        cos = (np.sum(ref_e * new_e, axis=1)
               / (np.linalg.norm(ref_e, axis=1) * np.linalg.norm(new_e, axis=1) + 1e-9))
        out["embed_min_cosine"] = float(cos.min())
    out["PASS"] = bool(out["lbl_equal"] and out["raw_equal"]
                       and out.get("embed_allclose"))
    return out


def compare():
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "."})
    report = {s: ray.get(_compare_split.remote(s)) for s in ("train", "val", "test")}
    report["ALL_PASS"] = all(report[s]["PASS"] for s in ("train", "val", "test"))
    with open(f"{NEW}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
