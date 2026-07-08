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
import torch.nn.functional as F

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
        time_aware: bool = True,
        n_time_buckets: int = 0,
        amount_mode: str = "hard",
        infonce_fields: list | tuple = (),
        infonce_negatives: int = 1024,
        infonce_max_anchors: int = 4096,
        signal_fields: list | tuple = (),
        signal_vocab_sizes: dict | None = None,
    ):
        super().__init__()
        self.dynamic_fields = dynamic_fields
        self.static_fields = static_fields
        self.d_model = d_model
        self.time_aware = time_aware and n_time_buckets > 0
        self.amount_mode = amount_mode  # "hard" | "soft" (soft-binned amount)
        # High-cardinality fields trained with InfoNCE (shared negative sampling)
        # instead of a full-softmax head — TREASURE Alg. 1. Empty in the hashed
        # path, so smoke/CI keeps plain cross-entropy on every field.
        self.infonce_fields = list(infonce_fields)
        self.infonce_negatives = infonce_negatives
        self.infonce_max_anchors = infonce_max_anchors
        # Output-only network-signal fields (e.g. decline/response codes):
        # predicted at every position, never embedded as input (would leak).
        self.signal_fields = list(signal_fields)

        self.dyn_emb = nn.ModuleDict(
            {f: nn.Embedding(field_vocab_sizes[f], d_model, padding_idx=PAD) for f in dynamic_fields}
        )
        self.static_emb = nn.ModuleDict(
            {f: nn.Embedding(field_vocab_sizes[f], d_model, padding_idx=PAD) for f in static_fields}
        )
        self.pos_emb = nn.Embedding(max_len, d_model)        # ordinal: where in the sequence
        if self.time_aware:
            # time-aware: how long since the previous transaction
            self.time_emb = nn.Embedding(n_time_buckets, d_model, padding_idx=PAD)
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

        # Low-cardinality fields: a full-softmax classification head each.
        # High-cardinality (InfoNCE) fields: a d_model->d_model projection whose
        # output is scored against the field's own (tied) embedding table — no
        # giant logit matrix, so a 100k-merchant vocab stays tractable.
        self.mlm_heads = nn.ModuleDict(
            {
                f: nn.Linear(d_model, field_vocab_sizes[f])
                for f in dynamic_fields
                if f not in self.infonce_fields
            }
        )
        self.infonce_proj = nn.ModuleDict(
            {f: nn.Linear(d_model, d_model) for f in self.infonce_fields}
        )
        # Network-signal prediction heads (one per signal field), scored at every
        # valid position against the true current-transaction signal.
        sizes = signal_vocab_sizes or {}
        self.signal_heads = nn.ModuleDict(
            {f: nn.Linear(d_model, sizes[f]) for f in self.signal_fields}
        )
        # Learned per-field homoscedastic log-variance (Kendall & Gal) so the
        # easy 9-way day head and the hard 2002-way merchant head are balanced
        # automatically instead of letting the big head dominate the gradient.
        self.log_var = nn.ParameterDict(
            {f: nn.Parameter(torch.zeros(())) for f in list(dynamic_fields) + self.signal_fields}
        )

    # --- input assembly ---
    def _embed(self, batch: dict):
        any_field = batch[f"d_{self.dynamic_fields[0]}"]
        B, S = any_field.shape
        device = any_field.device

        # .long(): token columns arrive as int32 (storage-efficient at long
        # seq_len); nn.Embedding requires int64 indices.
        x = torch.zeros(B, S, self.d_model, device=device)
        for f in self.dynamic_fields:
            if (
                f == "amount_bucket"
                and self.amount_mode == "soft"
                and "d_amount_frac" in batch
            ):
                # Soft binning: blend the two adjacent bin embeddings so $86.99
                # and $87.01 get near-identical vectors (no hard boundary).
                lo = batch["d_amount_bucket"].long()
                hi = torch.clamp(lo + 1, max=self.dyn_emb[f].num_embeddings - 1)
                frac = batch["d_amount_frac"].unsqueeze(-1).float()  # [B,S,1]
                x = x + (1.0 - frac) * self.dyn_emb[f](lo) + frac * self.dyn_emb[f](hi)
            else:
                x = x + self.dyn_emb[f](batch[f"d_{f}"].long())

        static_vec = torch.zeros(B, self.d_model, device=device)
        for f in self.static_fields:
            static_vec = static_vec + self.static_emb[f](batch[f"s_{f}"].long())
        x = x + static_vec.unsqueeze(1)  # broadcast static over all positions

        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = x + self.pos_emb(pos)
        if self.time_aware and "time_bucket" in batch:
            x = x + self.time_emb(batch["time_bucket"].long())  # inter-txn gap
        x = self.dropout(self.input_norm(x))

        pad_mask = batch["attention_mask"] == 0  # True where padding
        return x, pad_mask

    def encode(self, batch: dict) -> torch.Tensor:
        x, pad_mask = self._embed(batch)
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def forward(
        self,
        batch: dict,
        targets: dict = None,
        masked=None,
        weighting: str = "uncertainty",
        infonce_scale: float = 1.0,
        seq_views: tuple | None = None,
        seq_cl_scale: float = 0.0,
    ):
        """Training forward returns the MLM loss; with ``targets=None`` returns
        the encoder hidden states.

        The heads and ``log_var`` are applied *inside* ``forward`` on purpose, so
        that under DDP every trainable parameter participates in the wrapped
        forward pass and its gradients get all-reduced across workers.

        ``seq_views=(view1, view2, pair_valid)`` adds a CoLES-style
        sequence-level contrastive term (see ``seq_contrastive_loss``), scaled
        by ``seq_cl_scale``. Computed inside this forward so DDP grad sync
        stays correct with the extra encode passes.
        """
        hidden = self.encode(batch)
        if targets is None:
            return hidden
        total, stats = self.field_loss(
            hidden, batch, targets, masked, weighting, infonce_scale
        )
        if seq_views is not None and seq_cl_scale > 0.0:
            loss_cl, acc_cl = self.seq_contrastive_loss(*seq_views)
            total = total + seq_cl_scale * loss_cl
            stats["seq_cl"] = {"ce": float(loss_cl.item()), "acc": float(acc_cl)}
        return total, stats

    def seq_contrastive_loss(self, view1: dict, view2: dict, pair_valid, temperature: float = 0.1):
        """CoLES-style sequence-level InfoNCE (Babaev et al., SIGMOD 2022).

        The two views are disjoint temporal halves of each window — two
        sub-sequences of the same card — with static fields PAD'd out (else
        matching them via the user/card id embedding is a trivial shortcut
        that learns no behavior). Positives: (i, i); negatives: the rest of
        the batch. Headless (loss directly on L2-normalized masked-mean
        pooled states), as in CoLES — no extra parameters, so checkpoints
        stay drop-in compatible with every consumer.

        MLM learns per-position ("local") structure; this term is what makes
        the POOLED embedding carry card-level ("global") behavior — the
        readout the frozen-embeddings-into-XGBoost protocol consumes.
        """

        def pooled(view):
            h = self.encode(view)
            m = view["attention_mask"].unsqueeze(-1).float()
            z = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            return F.normalize(z, dim=-1)

        z1, z2 = pooled(view1), pooled(view2)
        valid = pair_valid.bool()
        if not bool(valid.any()):
            # keep encoder params in the graph so DDP reducers stay in sync
            return (z1.sum() + z2.sum()) * 0.0, 0.0
        sim = z1 @ z2.t() / temperature  # (B, B): halves of row i must find each other
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss = 0.5 * (
            F.cross_entropy(sim[valid], labels[valid])
            + F.cross_entropy(sim.t()[valid], labels[valid])
        )
        with torch.no_grad():
            acc = (sim.argmax(dim=1)[valid] == labels[valid]).float().mean()
        return loss, float(acc.item())

    def field_loss(
        self,
        hidden: torch.Tensor,
        batch: dict,
        targets: dict,
        masked,
        weighting: str = "uncertainty",
        infonce_scale: float = 1.0,
    ):
        """Per-field masked loss over the supervised positions, aggregated.

        Each field is either a full-softmax cross-entropy (low cardinality) or an
        InfoNCE term against its tied embedding table (high cardinality). Masked
        positions are flattened across the batch first, so InfoNCE's negatives
        are shared across all samples and timesteps (TREASURE's memory trick).

        ``weighting="uncertainty"`` applies Kendall & Gal homoscedastic weighting
        (loss_f * exp(-s_f) + 0.5*s_f, with s_f a learned log-variance) so heads
        of very different difficulty/scale stay balanced. ``"mean"`` is the plain
        unweighted average.

        ``infonce_scale`` (0..1) multiplies each InfoNCE field's *entire*
        weighted contribution — a warm-up anneal so the big contrastive head
        doesn't dominate the gradient budget before the fraud-relevant heads
        shape the representation. Scaling the whole term (not just the loss)
        matters: a bare-loss scale of 0 would leave the 0.5*s regularizer
        pushing the learned log-variance down while the head is dormant, then
        amplify it explosively when the ramp ends. Stats report the unscaled
        per-field loss so the merchant head's progress stays observable.

        Returns (total_loss, stats) where stats[field] = {"ce", "acc"}. For
        InfoNCE fields "acc" is the in-batch/negative-pool ranking accuracy (a
        cheap training proxy); the real ranking quality is the recommendation
        eval. These per-field numbers (not the weighted total, which drifts as
        the log-variances learn) are what you watch to confirm training works.
        """
        # ``masked`` is a per-field dict (RUN-2 independent masking) or a
        # single [B,S] bool shared by all fields (legacy whole-row masking).
        per_field = isinstance(masked, dict)
        hid_shared = None if per_field else hidden[masked]
        ce, stats = {}, {}
        for f in self.dynamic_fields:
            mf = masked[f] if per_field else masked
            hid_m = hidden[mf] if per_field else hid_shared
            tg = targets[f][mf]
            if tg.numel() == 0:
                # No positions drew this field's mask (tiny smoke batches):
                # touch the head with zero weight so DDP reducers stay in sync.
                if f in self.infonce_fields:
                    loss_f = self.infonce_proj[f](hidden[:, -1, :]).sum() * 0.0
                else:
                    loss_f = self.mlm_heads[f](hidden[:, -1, :]).sum() * 0.0
                ce[f] = loss_f
                stats[f] = {"ce": 0.0, "acc": 0.0}
                continue
            if f in self.infonce_fields:
                loss_f, acc = infonce_loss(
                    self.infonce_proj[f](hid_m),
                    self.dyn_emb[f].weight,
                    tg,
                    self.infonce_negatives,
                    self.infonce_max_anchors,
                )
            else:
                lg = self.mlm_heads[f](hid_m)
                loss_f = F.cross_entropy(lg, tg)
                with torch.no_grad():
                    acc = (lg.argmax(dim=-1) == tg).float().mean()
            ce[f] = loss_f
            stats[f] = {"ce": float(loss_f.item()), "acc": float(acc.item())}

        # Network-signal heads: supervised at every valid (non-pad) position,
        # independent of MLM masking (the signal is never a model input, so
        # predicting it at unmasked positions can't leak).
        if self.signal_fields:
            valid = batch["attention_mask"] == 1
            hid_v = hidden[valid]
            for f in self.signal_fields:
                tg = batch[f"y_{f}"][valid].long()
                lg = self.signal_heads[f](hid_v)
                loss_f = F.cross_entropy(lg, tg)
                with torch.no_grad():
                    acc = (lg.argmax(dim=-1) == tg).float().mean()
                ce[f] = loss_f
                stats[f] = {"ce": float(loss_f.item()), "acc": float(acc.item())}

        if weighting == "uncertainty":
            total = 0.0
            for f in self.dynamic_fields + self.signal_fields:
                s = self.log_var[f]
                term = torch.exp(-s) * ce[f] + 0.5 * s
                if f in self.infonce_fields:
                    term = term * infonce_scale
                total = total + term
        else:
            total = sum(
                (ce[f] * infonce_scale if f in self.infonce_fields else ce[f])
                for f in ce
            ) / len(ce)
        return total, stats

    @torch.no_grad()
    def target_readout(self, batch: dict, targets: dict):
        """Run-1 readout surgery: target-conditioned features from an MLM.

        ``batch`` must arrive with the LAST position's dynamic fields already
        set to MASK (sequences are right-aligned; last position = the target
        transaction). ``targets`` maps field -> (B,) true token ids at that
        position. Returns:

        * ``h_masked``  — the target-position hidden state under masking: the
          ONLY state this MLM was ever trained to make informative (field_loss
          supervises hidden[masked] exclusively), unlike the unmasked pooled
          readouts that failed.
        * ``surprise`` — (B, n_dynamic_fields) per-field cross-entropy of the
          TRUE target fields under the MLM heads: literally "how anomalous is
          this transaction given this card's history", the quantity a fraud
          model wants. InfoNCE fields score against the full tied table
          (exact softmax — cheap at inference).
        """
        hidden = self.encode(batch)
        h = hidden[:, -1, :]
        cols = []
        for f in self.dynamic_fields:
            tg = targets[f].long()
            if f in self.infonce_fields:
                logits = self.infonce_proj[f](h) @ self.dyn_emb[f].weight.t()
            else:
                logits = self.mlm_heads[f](h)
            cols.append(F.cross_entropy(logits, tg, reduction="none"))
        return h, torch.stack(cols, dim=1)

    @torch.no_grad()
    def sequence_embedding(self, batch: dict, pooling: str = "mean") -> torch.Tensor:
        """Pool final hidden states into one vector per sequence.

        ``"last"`` — the final position's hidden state. Sequences are
        right-aligned (left-padded), so this is the most recent transaction:
        the right readout for per-transaction labels (NVIDIA's blueprint pools
        the last token the same way). Mean-pooling dilutes the target event
        across the window — the longer the window, the worse it gets.

        ``"mean"`` — masked mean over valid positions: a whole-history
        customer summary, the right readout for card-level tasks.

        ``"all"`` — one encode, every readout: dict of {"last", "mean",
        "max"}. The forward pass is 99% of the cost, the reductions are
        free — batch extraction emits all of them so downstream experiments
        never re-run the GPU pass to try a different pooling.
        """
        hidden = self.encode(batch)
        if pooling == "last":
            return hidden[:, -1, :]
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        if pooling == "mean":
            return summed / counts
        # masked max: padded positions -> -inf so they never win
        maxed = hidden.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        if pooling == "max":
            return maxed
        assert pooling == "all", f"unknown pooling: {pooling}"
        return {"last": hidden[:, -1, :], "mean": summed / counts, "max": maxed}


def infonce_loss(
    hidden_a: torch.Tensor,
    embedding: torch.Tensor,
    labels: torch.Tensor,
    n_negative: int,
    max_anchors: int,
):
    """InfoNCE for a high-cardinality categorical field (TREASURE Alg. 1, MLM
    variant). ``hidden_a`` are the projected anchor states at the M supervised
    positions; ``embedding`` is the field's (tied) table.

    The candidate pool for every anchor is **shared**: all M in-batch positives
    (the anchors' own targets) plus ``n_negative`` random ids sampled once for
    the whole batch. Reusing one negative pool across all anchors is the trick
    that keeps a 100k+ vocab tractable — no per-position logit matrix. Anchors
    that share a label (common here: 96% merchant repeat) are masked out of each
    other's negatives so they aren't penalized as false negatives.
    """
    M, device = hidden_a.shape[0], hidden_a.device
    if M == 0:
        return hidden_a.sum() * 0.0, torch.zeros((), device=device)
    if M > max_anchors:
        keep = torch.randperm(M, device=device)[:max_anchors]
        hidden_a, labels, M = hidden_a[keep], labels[keep], max_anchors

    V = embedding.shape[0]
    pos_emb = embedding[labels]                                  # (M, D)
    neg_ids = torch.randint(V, (n_negative,), device=device)
    neg_emb = embedding[neg_ids]                                 # (n_neg, D)

    inbatch = hidden_a @ pos_emb.t()                             # (M, M)
    negs = hidden_a @ neg_emb.t()                                # (M, n_neg)
    same = labels[None, :] == labels[:, None]
    eye = torch.eye(M, dtype=torch.bool, device=device)
    inbatch = inbatch.masked_fill(same & ~eye, float("-inf"))    # drop false negs

    logits = torch.cat([inbatch, negs], dim=1)                   # (M, M + n_neg)
    target = torch.arange(M, device=device)                      # positive = diagonal
    loss = F.cross_entropy(logits, target)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == target).float().mean()
    return loss, acc


def build_model(vocab_path: str, arch: dict, max_len: int = 64, infonce_negatives: int = 1024) -> TransactionFM:
    """Construct a model from a written vocab.json and explicit dims.

    ``arch`` is the `model:` block of configs/<scale>.yaml (d_model / n_heads /
    n_layers / dim_ff); checkpoint consumers read it back from the
    model_config.json saved next to the weights. ``infonce_fields`` is read from
    the vocab so embed/serve rebuild the same module layout as training.
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    cfg = arch
    return TransactionFM(
        field_vocab_sizes=vocab["field_vocab_sizes"],
        dynamic_fields=vocab["dynamic_fields"],
        static_fields=vocab["static_fields"],
        max_len=max_len,
        time_aware=vocab.get("time_aware", False),
        n_time_buckets=vocab.get("n_time_buckets", 0),
        amount_mode=vocab.get("amount_mode", "hard"),
        infonce_fields=vocab.get("infonce_fields", []),
        infonce_negatives=infonce_negatives,
        signal_fields=vocab.get("signal_fields", []),
        signal_vocab_sizes=vocab.get("signal_vocab_sizes"),
        **cfg,
    )


def mask_batch(batch: dict, dynamic_fields: list, mask_prob: float = 0.15):
    """Masked-feature-modeling corruption — INDEPENDENT per-field masks.

    RUN-2 (TEARDOWN.md #3): each dynamic field draws its own mask (prob
    ``mask_prob`` per field per valid position). Whole-transaction masking
    made the pretext solvable from card marginals (identity/position/time
    stayed visible while ALL fields vanished) and forfeited intra-transaction
    conditional structure — with independent masks, predicting a hidden field
    from its visible siblings (P(state|merchant,amount), P(mcc|merchant))
    is most of the task, which is exactly the interaction signal the
    downstream fusion needs.

    Returns (corrupted_batch, targets, masked) where ``masked`` maps
    field -> [B,S] bool of that field's supervised positions.
    """
    attn = batch["attention_mask"]
    targets, masked = {}, {}
    corrupted = dict(batch)
    for f in dynamic_fields:
        m = (torch.rand(attn.shape, device=attn.device) < mask_prob) & (attn == 1)
        targets[f] = batch[f"d_{f}"].clone().long()
        col = batch[f"d_{f}"].clone()
        col[m] = MASK
        corrupted[f"d_{f}"] = col
        masked[f] = m
    # If soft-binned amount is present, zero its blend weight at the amount
    # field's masked positions so the masked input is purely the MASK embedding.
    if "d_amount_frac" in batch and "amount_bucket" in masked:
        frac = batch["d_amount_frac"].clone()
        frac[masked["amount_bucket"]] = 0.0
        corrupted["d_amount_frac"] = frac
    return corrupted, targets, masked
