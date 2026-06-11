"""TransactionFM — a compact, field-split transaction encoder.

Architecture (deliberately small — the model is *not* the hard part):

    dynamic field tokens ─ per-field embedding tables ─┐
                                                       ├─ sum ─► per-txn vector
    static field tokens ── per-field embedding tables ─┘        + positional
                                                               + static (broadcast)
                                                                     │
                                                          Transformer encoder
                                                          (bidirectional, MLM)
                                                                     │
                                          ┌──────────────────────────┼─────────────┐
                                   MLM heads (pretrain)        mean-pool (embedding)
                              one classifier per dynamic field   -> customer vector

The encoder is bidirectional because the downstream tasks fintech cares about
(fraud, churn, credit) are fixed-window classification, where masked-feature
modeling beats next-token. Swap the attention mask + a causal head for the
generative/NTP variant.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from .tokenizer import MASK, PAD


class TransactionFM(nn.Module):
    def __init__(
        self,
        field_vocab_sizes: dict,
        dynamic_fields: list,
        static_fields: list,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.dynamic_fields = dynamic_fields
        self.static_fields = static_fields
        self.d_model = d_model

        self.dyn_emb = nn.ModuleDict(
            {f: nn.Embedding(field_vocab_sizes[f], d_model, padding_idx=PAD) for f in dynamic_fields}
        )
        self.static_emb = nn.ModuleDict(
            {f: nn.Embedding(field_vocab_sizes[f], d_model, padding_idx=PAD) for f in static_fields}
        )
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # One MLM classification head per dynamic field.
        self.mlm_heads = nn.ModuleDict(
            {f: nn.Linear(d_model, field_vocab_sizes[f]) for f in dynamic_fields}
        )

    # --- input assembly ---
    def _embed(self, batch: dict):
        any_field = batch[f"d_{self.dynamic_fields[0]}"]
        B, S = any_field.shape
        device = any_field.device

        x = torch.zeros(B, S, self.d_model, device=device)
        for f in self.dynamic_fields:
            x = x + self.dyn_emb[f](batch[f"d_{f}"])

        static_vec = torch.zeros(B, self.d_model, device=device)
        for f in self.static_fields:
            static_vec = static_vec + self.static_emb[f](batch[f"s_{f}"])
        x = x + static_vec.unsqueeze(1)  # broadcast static over all positions

        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = x + self.pos_emb(pos)
        x = self.dropout(self.input_norm(x))

        pad_mask = batch["attention_mask"] == 0  # True where padding
        return x, pad_mask

    def forward(self, batch: dict) -> torch.Tensor:
        x, pad_mask = self._embed(batch)
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def mlm_logits(self, hidden: torch.Tensor) -> dict:
        return {f: head(hidden) for f, head in self.mlm_heads.items()}

    @torch.no_grad()
    def sequence_embedding(self, batch: dict) -> torch.Tensor:
        """Mean-pool final hidden states over valid positions -> customer vector."""
        hidden = self.forward(batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


def build_model(vocab_path: str, size: str = "small", max_len: int = 64) -> TransactionFM:
    """Construct a model from a written vocab.json and a size preset."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    presets = {
        "smoke": dict(d_model=64, n_heads=2, n_layers=2, dim_ff=128),
        "small": dict(d_model=256, n_heads=4, n_layers=4, dim_ff=512),
        "medium": dict(d_model=384, n_heads=6, n_layers=6, dim_ff=1024),
    }
    cfg = presets.get(size, presets["small"])
    return TransactionFM(
        field_vocab_sizes=vocab["field_vocab_sizes"],
        dynamic_fields=vocab["dynamic_fields"],
        static_fields=vocab["static_fields"],
        max_len=max_len,
        **cfg,
    )


def mask_batch(batch: dict, dynamic_fields: list, mask_prob: float = 0.15):
    """Masked-feature-modeling corruption.

    Selects valid (non-pad) positions with probability ``mask_prob``, replaces
    every dynamic field token at those positions with MASK, and returns
    (corrupted_batch, targets, masked_positions) where targets hold the original
    field ids and masked_positions is a [B,S] boolean of supervised positions.
    """
    attn = batch["attention_mask"]
    rand = torch.rand(attn.shape, device=attn.device)
    masked = (rand < mask_prob) & (attn == 1)

    targets = {f: batch[f"d_{f}"].clone() for f in dynamic_fields}
    corrupted = dict(batch)
    for f in dynamic_fields:
        col = batch[f"d_{f}"].clone()
        col[masked] = MASK
        corrupted[f"d_{f}"] = col
    return corrupted, targets, masked
