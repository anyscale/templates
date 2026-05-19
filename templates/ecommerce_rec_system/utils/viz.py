"""Visualization utilities for embedding analysis."""

import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def plot_category_samples(products: List[Dict], categories: List[str], get_image_fn: Callable) -> None:
    """Show one sample product image per category."""
    fig, axes = plt.subplots(1, len(categories), figsize=(15, 3))
    for ax, cat in zip(axes, categories):
        product = next(p for p in products if p["category"] == cat)
        img = get_image_fn(product)
        ax.imshow(img)
        ax.set_title(cat, fontsize=9)
        ax.axis("off")
    plt.suptitle("One sample image per category", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_products_per_category(products: List[Dict]) -> None:
    """Horizontal bar chart of product counts per category."""
    cats = [p["category"] for p in products]
    counts = sorted(Counter(cats).items())
    plt.barh(*zip(*counts))
    plt.xlabel("Products")
    plt.title("Products per category")
    plt.tight_layout()
    plt.show()


def plot_training_loss(metrics_dataframe) -> None:
    """Line plot of training loss across epochs."""
    m = metrics_dataframe
    plt.plot(m["epoch"] + 1, m["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.show()


def plot_recommendations(
    query_img: np.ndarray,
    recommendations: List[Dict],
    products: List[Dict],
    caption: str,
    get_image_fn: Callable,
) -> None:
    """Show query image alongside recommended products."""
    fig, axes = plt.subplots(1, len(recommendations) + 1, figsize=(18, 3))

    axes[0].imshow(query_img)
    axes[0].set_title("Query image", fontsize=9, color="navy", fontweight="bold")
    axes[0].axis("off")

    all_prods = {p["name"]: p for p in products}
    for ax, rec in zip(axes[1:], recommendations):
        prod_name = rec["name"]
        prod = all_prods.get(prod_name, {"name": prod_name, "category": rec["category"]})
        rec_img = get_image_fn({**prod, "category": rec["category"]})
        ax.imshow(rec_img)
        ax.set_title(
            f"{rec['rank']}. {rec['name'][:18]}\n{rec['similarity']:.2f}", fontsize=8
        )
        ax.axis("off")

    plt.suptitle(f'Caption: "{caption}"', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(
    embeddings, labels, n=10, title="Product embedding similarity (top 10)"
):
    """Plot cosine similarity heatmap for the first n embeddings."""
    n = min(n, len(embeddings))
    sub = embeddings[:n]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    sim = (sub / norms) @ (sub / norms).T

    short_labels = [lbl[:20] for lbl in labels[:n]]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def _plot_tsne_side_by_side(
    base_embs: np.ndarray,
    tuned_embs: np.ndarray,
    labels: List[str],
    *,
    base_title: str = "Base model (all-MiniLM-L6-v2)",
    tuned_title: str = "Fine-tuned (contrastive loss)",
) -> None:
    """Side-by-side t-SNE of two embedding matrices, coloured by category labels."""
    from sklearn.manifold import TSNE

    cat_list = sorted(set(labels))
    colors = [cat_list.index(c) for c in labels]
    cmap = plt.cm.get_cmap("tab10", len(cat_list))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, embs, title in [
        (axes[0], base_embs, base_title),
        (axes[1], tuned_embs, tuned_title),
    ]:
        xy = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(embs)
        ax.scatter(
            xy[:, 0], xy[:, 1], c=colors, cmap=cmap, s=80, vmin=0, vmax=len(cat_list)
        )
        ax.set_title(title)
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=9,
            label=c,
        )
        for i, c in enumerate(cat_list)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(cat_list), fontsize=9)
    plt.suptitle("t-SNE: product embeddings coloured by category", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


def plot_tsne_comparison(
    metadata_or_base: Union[List[Dict], np.ndarray],
    embeddings_or_tuned: np.ndarray,
    labels: Optional[List[str]] = None,
    *,
    catalog_records: Optional[List[Dict]] = None,
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Optional[Tuple[np.ndarray, np.ndarray, List[Dict]]]:
    """Compare base MiniLM vs fine-tuned embeddings with a t-SNE plot.

    **High-level (recommended):** pass ``metadata``, ``tuned_embeddings``, and
    ``catalog_records`` (list of dicts with ``product_id`` and ``text_clean``).
    Aligns rows, encodes text with the base model, plots, and returns
    ``(base_embs, tuned_embs, aligned_metadata)`` for downstream metrics.

    **Low-level:** pass ``base_embs``, ``tuned_embs``, and ``labels`` (three
    positional args) to plot precomputed arrays only; returns ``None``.
    """
    if labels is not None:
        _plot_tsne_side_by_side(
            np.asarray(metadata_or_base),
            np.asarray(embeddings_or_tuned),
            labels,
        )
        return None

    from sentence_transformers import SentenceTransformer

    metadata = metadata_or_base  # type: ignore[assignment]
    tuned_embeddings = np.asarray(embeddings_or_tuned)

    if catalog_records is None:
        raise ValueError("catalog_records is required")
    id_to_text = {r["product_id"]: r["text_clean"] for r in catalog_records}

    idx = [i for i, m in enumerate(metadata) if m["product_id"] in id_to_text]
    if not idx:
        raise ValueError("No metadata rows overlap catalog_records product_id values.")

    aligned_metadata = [metadata[i] for i in idx]
    ft_embs = tuned_embeddings[idx]
    texts = [id_to_text[m["product_id"]] for m in aligned_metadata]

    base_embs = SentenceTransformer(base_model_name).encode(
        texts, convert_to_numpy=True
    )
    category_labels = [m["category"] for m in aligned_metadata]
    _plot_tsne_side_by_side(base_embs, ft_embs, category_labels)
    return base_embs, ft_embs, aligned_metadata


def compute_category_precision_at_k(embeddings, metadata, k: int = 5):
    """Share of each product's k nearest neighbors (by cosine similarity) in the same category.

    Often hits 100% when product text already matches its category; use
    ``mean_rank_first_cross_category_neighbor`` or ``mean_intra_inter_sim_gap``
    to compare base vs fine-tuned models in that case.
    """
    n = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = n @ n.T
    np.fill_diagonal(sim, -np.inf)
    cats = [m["category"] for m in metadata]
    return np.mean(
        [
            sum(cats[j] == cats[i] for j in np.argsort(sim[i])[-k:]) / k
            for i in range(len(embeddings))
        ]
    )


def mean_rank_first_cross_category_neighbor(
    embeddings: np.ndarray, metadata: List[Dict]
) -> float:
    """Average rank when others are sorted by similarity: position of the closest other-category item.

    Rank 1 means the most similar product to you is already from another category.
    Higher values mean other categories start further down the similarity-sorted list.
    """
    n = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = n @ n.T
    np.fill_diagonal(sim, -np.inf)
    cats = [m["category"] for m in metadata]
    ranks: List[int] = []
    for i in range(len(embeddings)):
        order = np.argsort(-sim[i])
        for rank_pos, j in enumerate(order, start=1):
            if cats[j] != cats[i]:
                ranks.append(rank_pos)
                break
    return float(np.mean(ranks)) if ranks else float("nan")


def mean_intra_inter_sim_gap(embeddings: np.ndarray, metadata: List[Dict]) -> float:
    """Average gap: similarity to same-category products minus similarity to other categories."""
    n = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = n @ n.T
    np.fill_diagonal(sim, np.nan)
    cats = [m["category"] for m in metadata]
    gaps: List[float] = []
    for i in range(len(embeddings)):
        intra = [
            sim[i, j]
            for j in range(len(embeddings))
            if j != i and cats[j] == cats[i]
        ]
        inter = [sim[i, j] for j in range(len(embeddings)) if cats[j] != cats[i]]
        if not intra or not inter:
            continue
        gaps.append(float(np.nanmean(intra) - np.nanmean(inter)))
    return float(np.mean(gaps)) if gaps else float("nan")


def compute_category_precision_at_5(embeddings, metadata):
    return compute_category_precision_at_k(embeddings, metadata, k=5)


def print_embedding_quality_report(
    base_embs: np.ndarray,
    finetuned_embs: np.ndarray,
    metadata: List[Dict],
) -> None:
    """Print a side-by-side base vs fine-tuned comparison on two robust metrics.

    We deliberately skip the easier "top-k neighbors share the category" score —
    on this demo the product names already leak the category, so the base model
    looks near-perfect there. The two checks below measure geometry in a way
    that still shifts after contrastive training.
    """
    rank_base = mean_rank_first_cross_category_neighbor(base_embs, metadata)
    rank_ft = mean_rank_first_cross_category_neighbor(finetuned_embs, metadata)
    gap_base = mean_intra_inter_sim_gap(base_embs, metadata)
    gap_ft = mean_intra_inter_sim_gap(finetuned_embs, metadata)

    print("1) Sort other products by similarity (best match at the top).")
    print("   How far down is the closest one that is not in your category?")
    print("   (Higher = wrong-category items sit lower in that sorted list.)")
    print(f"   Base model:  {rank_base:.1f}")
    print(f"   Fine-tuned:  {rank_ft:.1f}")
    print(f"   Change:      {rank_ft - rank_base:+.1f}")
    print()
    print("2) How much closer same-category items are than other categories")
    print("   (larger = clearer gap between own category and the rest)")
    print(f"   Base model:  {gap_base:.4f}")
    print(f"   Fine-tuned:  {gap_ft:.4f}")
    print(f"   Change:      {gap_ft - gap_base:+.4f}")
