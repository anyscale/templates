"""Training utilities for embedding fine-tuning with contrastive loss."""

import os
import tempfile
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ray.train import Checkpoint, get_checkpoint, get_context, report

SEED = 42


def build_embedding_trainer(
    records: List[Dict],
    train_result_dir: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    seed: int = SEED,
):
    """Build a Ray :class:`TorchTrainer` configured for contrastive fine-tuning.

    Keeps checkpoint/failure config next to the training loop so the notebook
    only has to say *what* it wants to train, not *how* Ray Train wires it up.
    """
    from ray.train import (
        CheckpointConfig,
        FailureConfig,
        RunConfig,
        ScalingConfig,
    )
    from ray.train.torch import TorchTrainer

    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "base_model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "records": records,
        },
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=torch.cuda.is_available(),
        ),
        run_config=RunConfig(
            name="ecomm_embedding_finetune",
            storage_path=os.path.abspath(train_result_dir),
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="train_loss",
                checkpoint_score_order="min",
            ),
            failure_config=FailureConfig(max_failures=1),
        ),
    )


def save_best_sentence_transformer(result, output_dir: str) -> str:
    """Restore the lowest-loss checkpoint from a Ray Train `Result` and save it
    in SentenceTransformer format — i.e. a folder that `SentenceTransformer(path)`
    can load directly for serving.
    """
    from sentence_transformers import SentenceTransformer

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_checkpoint = result.best_checkpoints[0][0]
    with best_checkpoint.as_directory() as ckpt_dir:
        SentenceTransformer(ckpt_dir).save(output_dir)
    return output_dir


class ContrastivePairDataset(Dataset):
    """
    (anchor, positive, 1.0) for same-category pairs;
    (anchor, negative, -1.0) for cross-category pairs.
    Labels match CosineSimilarityLoss / MSE-on-cosine expectations.
    """

    def __init__(self, records, neg_ratio=0.5, seed=SEED):
        rng = np.random.default_rng(seed)
        cat_to_idx = {}
        for i, r in enumerate(records):
            cat_to_idx.setdefault(r["category"], []).append(i)

        pairs = []
        for indices in cat_to_idx.values():
            for i, a in enumerate(indices):
                for b in indices[i + 1 :]:
                    pairs.append(
                        (records[a]["text_clean"], records[b]["text_clean"], 1.0)
                    )

        cats = list(cat_to_idx.keys())
        n_neg = max(1, int(len(pairs) * neg_ratio))
        for _ in range(n_neg):
            cat_a, cat_b = rng.choice(cats, size=2, replace=False)
            ia = rng.choice(cat_to_idx[cat_a])
            ib = rng.choice(cat_to_idx[cat_b])
            pairs.append((records[ia]["text_clean"], records[ib]["text_clean"], -1.0))

        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        return a, b, torch.tensor(label, dtype=torch.float32)


def _forward_embeddings(model, texts, device):
    """Run a SentenceTransformer forward pass with gradient tracking."""
    features = model.tokenize(texts)
    features = {
        k: v.to(device) for k, v in features.items() if isinstance(v, torch.Tensor)
    }
    return model(features)["sentence_embedding"]


def train_loop_per_worker(config: dict) -> None:
    """Fine-tune the embedding model on each distributed Ray Train worker."""
    from sentence_transformers import SentenceTransformer

    rank = get_context().get_world_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = config["records"]

    pair_dataset = ContrastivePairDataset(records, seed=config["seed"])

    def collate(batch):
        texts_a, texts_b, labels = zip(*batch)
        return list(texts_a), list(texts_b), torch.stack(labels)

    loader = DataLoader(
        pair_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )

    model = SentenceTransformer(config["base_model"], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
    )

    start_epoch = 0
    ckpt = get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            meta = torch.load(
                os.path.join(d, "meta.pt"), map_location="cpu", weights_only=False
            )
            start_epoch = meta.get("epoch", 0) + 1
            model = SentenceTransformer(d, device=device)

    if rank == 0:
        print(
            f"Fine-tuning {config['base_model']} on {device}  "
            f"({config['epochs']} epochs, {len(pair_dataset)} pairs)"
        )

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        for texts_a, texts_b, labels in loader:
            labels = labels.to(device)
            emb_a = _forward_embeddings(model, texts_a, device)
            emb_b = _forward_embeddings(model, texts_b, device)
            loss = F.mse_loss(F.cosine_similarity(emb_a, emb_b), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if rank == 0:
            print(f"  Epoch {epoch+1:2d}/{config['epochs']}  loss={avg_loss:.4f}")

        with tempfile.TemporaryDirectory() as tmpdir:
            if rank == 0:
                model.save(tmpdir)
                torch.save({"epoch": epoch}, os.path.join(tmpdir, "meta.pt"))
                ckpt_out = Checkpoint.from_directory(tmpdir)
            else:
                ckpt_out = None
            report({"epoch": epoch, "train_loss": avg_loss}, checkpoint=ckpt_out)
