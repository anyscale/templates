"""Fast collapse test: does the pooling choice cause the embedding collapse?

Loads model/full, embeds a small sample of eval windows with each pooling
({last=eos, last_real=token before eos, mean}) and reports mean pairwise cosine
(->1 = collapse) and feature variance. No XGBoost, no re-pretrain — runs in one
GPU worker task in ~2 min. Decides whether the fix is just pooling or something
deeper (undertraining / trivial task).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ray

from src.paths import artifact_paths, get_demo_base_dir

BASE = get_demo_base_dir()
P = artifact_paths(BASE, "full")
N = 4000


@ray.remote(num_gpus=1, num_cpus=4, memory=24 * 1024 ** 3)
def run():
    import glob
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from src.embed import EmbeddingExtractor

    files = sorted(glob.glob(os.path.join(P["tokenized_eval"], "*.parquet")))[:2]
    tbl = pq.ParquetDataset(files).read(columns=["input_ids", "attention_mask", "label"])
    d = tbl.to_pandas().head(N)
    ids = np.stack(d["input_ids"].to_numpy()).astype(np.int64)
    am = np.stack(d["attention_mask"].to_numpy()).astype(np.int64)
    print(f"[pool] sample {len(d)} windows, real-token counts: "
          f"min={am.sum(1).min()} med={int(np.median(am.sum(1)))} max={am.sum(1).max()}", flush=True)

    ex = EmbeddingExtractor(P["checkpoint"], pooling="last")
    dev = ex.device
    BS = 24

    def collapse(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        s = Xn.sum(0); n = len(X)
        return float((s @ s - n) / (n * (n - 1))), float(X.var(0).mean())

    for pooling in ("last", "last_real", "mean"):
        outs = []
        for i in range(0, len(ids), BS):
            batch = {"input_ids": torch.as_tensor(ids[i:i + BS], device=dev),
                     "attention_mask": torch.as_tensor(am[i:i + BS], device=dev)}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
                outs.append(ex.model.sequence_embedding(batch, pooling=pooling).float().cpu().numpy())
        emb = np.concatenate(outs)
        cos, var = collapse(emb)
        flag = " <-- COLLAPSE" if cos > 0.9 else (" ok" if cos < 0.6 else "")
        print(f"[pool] {pooling:10} mean_pairwise_cos={cos:.3f}  feat_var={var:.4f}{flag}", flush=True)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    ray.get(run.remote())
