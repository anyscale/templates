from typing import List, Optional

CRITEO_NUM_EMBEDDINGS_PER_FEATURE: List[int] = [
    458331,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    292752,
    157217,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    489374,
    113167,
    400945,
    452104,
    12606,
    104,
    35,
]


class RecsysConfig:
    num_workers: int = 1

    train_batch_size: int = 4096
    limit_training_rows: Optional[int] = None

    validation_batch_size: int = 8192
    limit_validation_rows: Optional[int] = None

    # read from cached parquet
    read_from_cached_parquet: bool = False

    # Training
    num_epochs: int = 1
    skip_train_step: bool = False
    # If None, use all days.
    train_day_list: Optional[List[int]] = None

    # Validation
    validate_every_n_steps: int = 1000  # just for testing
    skip_validation_step: bool = False
    skip_validation_at_epoch_end: bool = False

    # Logging
    log_metrics_every_n_steps: int = 100

   # Checkpointing
    start_from_checkpoint_path: Optional[str] = None    # s3 path to the checkpoint to resume from
    save_checkpoint_every_n_epochs: int = 1  # Save checkpoint every N epochs
    save_checkpoint_every_n_mins: int = -1  # Save checkpoint every N minutes (-1 to disable)
    save_checkpoint_at_end: bool = True  # Save final checkpoint at end of training
    checkpoint_dir: Optional[str] = None  # Directory (local or s3) to save checkpoints (if None, uses Ray's default)

    # torchrec config
    embedding_dim: int = 128
    num_embeddings_per_feature: List[int] = CRITEO_NUM_EMBEDDINGS_PER_FEATURE
    over_arch_layer_sizes: List[int] = [1024, 1024, 512, 256, 1]
    dense_arch_layer_sizes: List[int] = [512, 256, 128]
    interaction_type: str = "dcn"
    dcn_num_layers: int = 3
    dcn_low_rank_dim: int = 512

    # Reproducibility
    seed: Optional[int] = None

    # Optimizer Configuration
    learning_rate: float = 0.005
    optimizer_eps: float = 1e-8
