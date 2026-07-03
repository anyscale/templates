"""TransactionFM — a compact Llama causal decoder over tokenized transactions.

This is NVIDIA's transaction-FM blueprint architecture: a small Llama decoder
pretrained by **next-token prediction** over the flat token stream from
``flat_tokenizer`` (~12 tokens per transaction, one shared vocab). We build the
decoder from ``transformers`` (LlamaConfig/LlamaForCausalLM) rather than
hand-rolling RoPE/GQA/SwiGLU, with the exact hyperparameters from NVIDIA's
released ``config.json`` (hidden 512, 8 layers, 8 query / 2 KV heads, head_dim
64, SwiGLU intermediate 1408, RMSNorm, rope_theta 5e5).

Two entry points, matching the old field-split model's API so the rest of the
pipeline changes minimally:

* ``forward(batch)`` — training: causal-LM loss from ``input_ids``/``labels``.
* ``sequence_embedding(batch, pooling="last")`` — the customer vector: the last
  non-pad position's hidden state (right-padded, so it's the most recent txn).
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaForCausalLM

# NVIDIA decoder-foundation-model/config.json defaults (overridable via arch).
_LLAMA_DEFAULTS = dict(
    d_model=512, n_layers=8, n_heads=8, n_kv_heads=2, head_dim=64,
    dim_ff=1408, rope_theta=500000.0, rms_eps=1e-5,
)


class TransactionFM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_length: int = 4096,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 64,
        dim_ff: int = 1408,
        rope_theta: float = 500000.0,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        self.config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            intermediate_size=dim_ff,
            hidden_act="silu",
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            max_position_embeddings=max(seq_length, 8192),
            attention_dropout=0.0,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
            tie_word_embeddings=False,
            use_cache=False,  # training; no KV cache
        )
        self.lm = LlamaForCausalLM(self.config)
        # Gradient checkpointing: recompute activations in backward instead of
        # storing all 8 layers' worth. Big memory saver at long context (4096
        # tokens) on smaller GPUs (~15 GiB here), for a modest compute cost.
        self.lm.gradient_checkpointing_enable()

    def forward(self, batch: dict, **_):
        """Causal-LM training step. ``batch`` has input_ids / attention_mask /
        labels (labels = input_ids with pad -> -100; HF shifts internally)."""
        out = self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # (loss, stats) to mirror the old model's return contract.
        return out.loss, {"lm_loss": float(out.loss.detach().item())}

    @torch.no_grad()
    def sequence_embedding(self, batch: dict, pooling: str = "last") -> torch.Tensor:
        """Pool the decoder's last hidden states into one vector per sequence.

        ``last`` (default, and what NVIDIA uses): the final non-pad position —
        the only one that has attended to the whole causal history. Sequences are
        right-padded, so it's the most recent transaction.
        """
        hidden = self.lm.model(  # LlamaModel (no LM head)
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        mask = batch["attention_mask"]
        if pooling == "last":
            last = mask.long().sum(dim=1) - 1  # index of last real token
            return hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def build_model(vocab_path: str, arch: dict | None = None, max_len: int = 4096, **_) -> TransactionFM:
    """Construct the decoder from a written flat-tokenizer vocab.json + arch dims.

    ``arch`` is the ``model:`` block of configs/<scale>.yaml (may override any
    Llama dim); ``**_`` swallows legacy kwargs (e.g. ``objective``) from callers.
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    a = {**_LLAMA_DEFAULTS, **(arch or {})}
    return TransactionFM(
        vocab_size=vocab["vocab_size"],
        seq_length=vocab.get("seq_length", max_len),
        d_model=a["d_model"], n_layers=a["n_layers"], n_heads=a["n_heads"],
        n_kv_heads=a["n_kv_heads"], head_dim=a["head_dim"], dim_ff=a["dim_ff"],
        rope_theta=a["rope_theta"], rms_eps=a["rms_eps"],
    )
