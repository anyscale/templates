"""
GPU embedding stage for the embedding pipeline.
ProductEmbedder is a callable class — model loads ONCE at __init__ on GPU,
then encodes batches in __call__. Used with Ray Data map_batches concurrency.
"""


class ProductEmbedder:
    """
    Ray Data callable class for GPU batch embedding generation.

    Loaded once per GPU worker replica. Ray Data manages the lifecycle —
    do not instantiate directly outside of map_batches.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [embed] Loading all-MiniLM-L6-v2 on {device}")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        self.device = device

    def __call__(self, batch: dict) -> dict:
        """Generate embeddings for a batch of preprocessed products."""
        texts = batch["combined_text"]
        embeddings = self.model.encode(
            texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
        )
        batch["embedding"] = embeddings.tolist()
        return batch
