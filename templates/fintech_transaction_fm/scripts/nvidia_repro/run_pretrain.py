"""Step 2: Ray-Train OUR Llama on the faithful (NVIDIA-tokenizer) corpus. Uses the
full-scale pretrain recipe (= NVIDIA's: lr 2e-4 cosine, wd 0.077, beta2 0.95, 8 epochs,
8 DDP GPU workers, arch 512/8/8/2). Saves checkpoint to /mnt/cluster_storage/nvpretrain/model."""
import sys, numpy as np, ray
sys.path.insert(0, "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/fintech_transaction_fm")
from src.pretrain import pretrain, save_checkpoint
from src.scale_config import load_scale

ray.init(ignore_reinit_error=True)
ids = np.load("/mnt/cluster_storage/nvpretrain/ids.npy")
attn = np.load("/mnt/cluster_storage/nvpretrain/attn.npy")
print("corpus", ids.shape, flush=True)
dsi = ray.data.from_numpy(ids).rename_columns({"data": "input_ids"})
dsa = ray.data.from_numpy(attn).rename_columns({"data": "attention_mask"})
ds = dsi.zip(dsa)

cfg = load_scale("full")
pt = cfg["pretrain"]
result = pretrain(
    train_ds=ds,
    vocab_path="/mnt/cluster_storage/nvpretrain/vocab.json",
    size="full", max_len=4096, arch=cfg["model"],
    epochs=pt["epochs"], batch_size=pt["batch_size"], lr=pt["lr"],
    num_workers=pt["num_workers"], use_gpu=pt["use_gpu"],
    lr_schedule="cosine", warmup_ratio=pt.get("warmup_ratio", 0.05),
    min_lr_ratio=pt.get("min_lr_ratio", 0.03),
    weight_decay=pt.get("weight_decay", 0.077),
    betas=tuple(pt.get("betas", [0.9, 0.95])),
    storage_base="/mnt/cluster_storage/transaction-fm",  # shared FS — workers must reach it
)
save_checkpoint(result, "/mnt/cluster_storage/nvpretrain/model")
print("PRETRAIN DONE -> /mnt/cluster_storage/nvpretrain/model", flush=True)
