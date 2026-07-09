"""Stage 2 of PLAN_RAY_DATA.md: prove the Ray Data (CPU) corpus build equals the
single-GPU reference corpus bit-for-bit.

    python scripts/verify_distributed_corpus.py run       # build -> nvcorpus_rd/full
    python scripts/verify_distributed_corpus.py compare   # vs reference nvcorpus/full

The distributed build reads the Stage-1 distributed split (nvsplit_rd), so a pass here
verifies the chain CSV -> shards -> split -> corpus end-to-end against the reference.
"""
import json
import sys
import time

import ray

BASE = "/mnt/cluster_storage/transaction-fm"
SPLIT = f"{BASE}/nvsplit_rd/full"
REF = f"{BASE}/nvcorpus/full"
NEW = f"{BASE}/nvcorpus_rd/full"


def run():
    sys.path.insert(0, ".")
    from src.nvcorpus import build_corpus_distributed
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "."})
    t0 = time.time()
    meta = build_corpus_distributed(SPLIT, NEW, seq_len=4096, chunk=315,
                                    merchant_hash=2000, max_seq=None)
    meta["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(meta, indent=2))
    with open(f"{NEW}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


@ray.remote(num_cpus=16)
def _compare() -> dict:
    import numpy as np

    out = {}
    ref_ids = np.load(f"{REF}/ids.npy")
    new_ids = np.load(f"{NEW}/ids.npy")
    out["ref_shape"], out["new_shape"] = list(ref_ids.shape), list(new_ids.shape)
    out["vocab_equal"] = (json.load(open(f"{REF}/vocab.json"))
                          == json.load(open(f"{NEW}/vocab.json")))
    out["ids_equal"] = bool(ref_ids.shape == new_ids.shape
                            and np.array_equal(ref_ids, new_ids))
    if not out["ids_equal"] and ref_ids.shape == new_ids.shape:
        row_neq = np.any(ref_ids != new_ids, axis=1)
        out["rows_differing"] = int(row_neq.sum())
        i = int(np.where(row_neq)[0][0])
        j = int(np.where(ref_ids[i] != new_ids[i])[0][0])
        out["first_mismatch"] = {"row": i, "col": j,
                                 "ref": int(ref_ids[i, j]), "new": int(new_ids[i, j])}
        # permutation check: same multiset of rows?
        rh = lambda a: sorted(hash(r.tobytes()) for r in a)
        out["same_row_multiset"] = (rh(ref_ids) == rh(new_ids))
    if out["ids_equal"]:
        ref_at = np.load(f"{REF}/attn.npy")
        new_at = np.load(f"{NEW}/attn.npy")
        out["attn_equal"] = bool(np.array_equal(ref_at, new_at))
    out["ALL_IDENTICAL"] = bool(out["ids_equal"] and out.get("attn_equal")
                                and out["vocab_equal"])
    return out


def compare():
    ray.init(ignore_reinit_error=True, runtime_env={"working_dir": "."})
    report = ray.get(_compare.remote())
    with open(f"{NEW}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
