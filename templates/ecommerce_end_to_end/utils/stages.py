"""High-level stage runner functions used by the demo notebook."""

import os
import shutil
import time
from pathlib import Path

import ray


# ---------------------------------------------------------------------------
# Stage 1 — Ray Data: Preprocessing
# ---------------------------------------------------------------------------

def run_preprocessing(output_dir: str, batch_size: int = 8) -> int:
    """Generate catalog, preprocess images + text, write Parquet shards.

    Returns the number of rows written.
    """
    from . import PRODUCTS, generate_catalog, preprocess_image_batch, preprocess_text_batch

    t0 = time.time()
    print("=" * 60)
    print("STAGE 1 — RAY DATA: PREPROCESSING")
    print("=" * 60)

    raw_dir = str(Path(output_dir).parent / "raw")

    print("\n[1/4] Generating product catalog …")
    records = generate_catalog(products=PRODUCTS, output_dir=raw_dir)
    ds = ray.data.from_items(records)
    print(f"  Rows: {ds.count()}")

    print("\n[2/4] Preprocessing images (resize + normalise) …")
    ds = ds.map_batches(
        preprocess_image_batch, batch_size=batch_size, num_cpus=1, batch_format="numpy"
    )

    print("\n[3/4] Preprocessing text (clean) …")
    ds = ds.map_batches(
        preprocess_text_batch, batch_size=batch_size, num_cpus=1, batch_format="numpy"
    )

    print(f"\n[4/4] Writing to '{output_dir}' …")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ds.write_parquet(output_dir)

    n = ray.data.read_parquet(output_dir).count()
    print(f"\nPreprocessing complete!  Rows: {n}  ({time.time()-t0:.1f}s)")
    print("=" * 60)
    return n


# ---------------------------------------------------------------------------
# Stage 2 — Ray Train: Embedding Fine-Tuning
# ---------------------------------------------------------------------------

def run_training(
    preprocessed_dir: str,
    model_output_dir: str,
    train_result_dir: str,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    seed: int = 42,
):
    """Fine-tune the embedding model with contrastive loss via TorchTrainer.

    Saves the best checkpoint to *model_output_dir* and returns the Ray Train
    ``Result`` object (includes ``metrics_dataframe`` for plotting).
    """
    import glob

    import torch
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from sentence_transformers import SentenceTransformer

    from .training import train_loop_per_worker

    parquet_files = glob.glob(os.path.join(preprocessed_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in '{preprocessed_dir}'. "
            "Run Stage 1 (preprocessing) first."
        )

    t0 = time.time()
    print("=" * 60)
    print("STAGE 2 — RAY TRAIN: EMBEDDING FINE-TUNING")
    print("=" * 60)

    records = (
        ray.data.read_parquet(preprocessed_dir)
        .select_columns(["product_id", "name", "category", "text_clean"])
        .take_all()
    )
    print(f"\nLoaded {len(records)} records for training")

    Path(train_result_dir).mkdir(parents=True, exist_ok=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "base_model": base_model,
            "epochs": num_epochs,
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

    result = trainer.fit()
    print(f"\nTraining complete! ({time.time()-t0:.1f}s)")

    print("\nSaving fine-tuned model …")
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = result.best_checkpoints[0][0]
    with checkpoint.as_directory() as ckpt_dir:
        model = SentenceTransformer(ckpt_dir)
        model.save(model_output_dir)
    print(f"  Model saved : {model_output_dir}")
    print("=" * 60)
    return result
