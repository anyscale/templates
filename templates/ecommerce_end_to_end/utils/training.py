"""Helpers for the Ray Train fine-tuning stage in ``README.ipynb``.

The training loop, the ``TorchTrainer`` config, and the best-checkpoint save
all live inline in the notebook so the Ray Train APIs are visible. This
module keeps just the framework-agnostic pieces — the contrastive pair
dataset and a tiny SentenceTransformer forward helper — that would only
clutter the demo if pasted into the notebook directly.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

SEED = 42


class ContrastivePairDataset(Dataset):
    """
    (anchor, positive, +1.0) for same-category pairs;
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


def forward_embeddings(model, texts, device):
    """Run a SentenceTransformer forward pass with gradient tracking."""
    features = model.tokenize(texts)
    features = {
        k: v.to(device) for k, v in features.items() if isinstance(v, torch.Tensor)
    }
    return model(features)["sentence_embedding"]
