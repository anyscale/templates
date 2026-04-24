"""
GPU embedding stage using Meta's ESM-2 protein language model.

ESMEmbedder is a callable class — the model loads ONCE at __init__ on GPU,
then encodes batches in __call__. Used with Ray Data map_batches concurrency.

Biology context:
  - ESM-2 is a protein language model trained on millions of protein sequences.
  - It produces per-residue (per-amino-acid) embeddings that capture evolutionary,
    structural, and functional information.
  - We mean-pool over residue embeddings to get a single fixed-size vector per protein.
  - The 650M parameter variant (esm2_t33_650M_UR50D) produces 1280-dim embeddings.
  - The 35M parameter variant (esm2_t12_35M_UR50D) produces 480-dim embeddings (faster for demos).

Performance notes:
  - fp16 inference is numerically safe for embedding extraction and ~2x faster on A10G/L4.
  - HF_HOME env var controls the HuggingFace cache path so all GPU actors share one download.
"""
import os


class ESMEmbedder:
    """
    Ray Data callable class for GPU batch protein embedding generation.

    Loaded once per GPU worker replica. Ray Data manages the lifecycle --
    do not instantiate directly outside of map_batches.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        # Use shared cache on cluster storage to avoid re-downloading per worker
        cache_dir = os.environ.get("HF_HOME", "/mnt/cluster_storage/hf_cache")
        os.environ["HF_HOME"] = cache_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [esm_embedder] Loading {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = self.model.half()  # fp16 — ~2x throughput on A10G/L4
        self.model = self.model.to(self.device).eval()

        self.model_name = model_name
        print(f"  [esm_embedder] Model loaded ({self.device}, fp16)")

    def __call__(self, batch: dict) -> dict:
        """Generate ESM-2 embeddings for a batch of protein sequences.

        Tokenizes sequences, runs a forward pass through ESM-2, and mean-pools
        over valid (non-padding) token positions to produce one embedding per sequence.
        Returns float32 numpy arrays for downstream compatibility.
        """
        import torch
        import numpy as np

        # Decode bytes to strings if needed (numpy batch format returns bytes)
        raw_seqs = list(batch["sequence"])
        seqs = [s.decode("utf-8") if isinstance(s, bytes) else s for s in raw_seqs]

        # Tokenize: ESM-2 tokenizer converts AA letters to token IDs
        # padding=True pads to the longest sequence in this batch
        # truncation=True caps at max_length (1024 tokens for ESM-2)
        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)

        # Mean-pool over valid tokens only (exclude padding positions).
        # attention_mask is 1 for real tokens, 0 for padding.
        # This gives a single vector per sequence that captures the "average"
        # residue-level representation — a standard approach for sequence-level tasks.
        mask = enc["attention_mask"].unsqueeze(-1).half()
        hidden = out.last_hidden_state  # (batch, seq_len, hidden_dim)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Convert back to float32 numpy for Parquet serialization
        batch["embedding"] = pooled.float().cpu().numpy()

        return batch
