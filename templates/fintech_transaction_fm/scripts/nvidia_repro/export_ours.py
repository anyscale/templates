"""Export our Ray-trained TransactionFM checkpoint -> a HuggingFace dir so the (NVIDIA)
HuggingFaceDecoderInference can load it for embedding. Uses OUR src only (run from template)."""
import json, torch, sys
sys.path.insert(0, "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/fintech_transaction_fm")
from src.model import build_model

CK = "/mnt/cluster_storage/transaction-fm/ray_results/transaction_fm_pretrain/checkpoint_2026-07-07_01-10-15.582557"
cfg = json.load(open(f"{CK}/model_config.json"))
m = build_model(vocab_path=f"{CK}/vocab.json", arch=cfg["arch"], max_len=cfg["max_len"])
sd = torch.load(f"{CK}/model.pt", map_location="cpu")
sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) and "model_state_dict" in sd else sd
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing", len(missing), "unexpected", len(unexpected))
out = "/mnt/cluster_storage/nvpretrain/hf"
m.lm.save_pretrained(out)
print("exported LlamaForCausalLM ->", out)
import os
print(os.listdir(out))
