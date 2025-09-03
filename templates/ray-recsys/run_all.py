# Note: we reduce the embedding table size to make it able to run in A10 GPUs.
from configs import RecsysConfig
import os

recsys_config = RecsysConfig()
# We use 2 g5.12xlarge nodes
recsys_config.num_workers = 8
recsys_config.train_step_limit = 5000

# Enable Ray Train V2
os.environ['RAY_TRAIN_V2_ENABLED'] = '1'

import ray
from typing import Dict
from criteo import read_feature_mapping_table, DEFAULT_CAT_NAMES

def build_categorical_to_feature_mapping_refs() -> Dict[str, ray.ObjectRef]:
    return {
        cat_feature: read_feature_mapping_table.remote(cat_feature) for cat_feature in DEFAULT_CAT_NAMES
    }

# After running this, the task `read_feature_mapping_table` will run in the background.
categorical_to_feature_mapping_refs = build_categorical_to_feature_mapping_refs()

import ray
import pyarrow.csv
from criteo import TRAIN_DATASET_PATH, VAL_DATASET_PATH, DEFAULT_COLUMN_NAMES, fill_missing, map_features_to_indices, concat_and_normalize_dense_features
from typing import Tuple

def get_ray_dataset(path: str) -> ray.data.Dataset:
    categorical_to_feature_mapping_refs = build_categorical_to_feature_mapping_refs()
    dataset_path = path
    ds = ray.data.read_csv(
        dataset_path,
        read_options=pyarrow.csv.ReadOptions(column_names=DEFAULT_COLUMN_NAMES),
        parse_options=pyarrow.csv.ParseOptions(delimiter="\t"),
        ray_remote_args={
            # reading is memory intensive
            'memory': 800 * 1024 * 1024,  # 800 MB
        },
        shuffle=(
            "files"
        ),  # coarse file-level shuffle
    )
    ds = ds.map_batches(fill_missing)
    ds = ds.map_batches(map_features_to_indices, fn_kwargs={"categorical_to_feature_mapping_refs": categorical_to_feature_mapping_refs})
    ds = ds.map_batches(concat_and_normalize_dense_features)
    return ds

train_dataset = get_ray_dataset(TRAIN_DATASET_PATH)
val_dataset = get_ray_dataset(VAL_DATASET_PATH)


from torchrec_wrapper import train_loop

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
import logging

logger = logging.getLogger(__name__)

scaling_config = ScalingConfig(
    num_workers=recsys_config.num_workers,
    # reserve CPUs to the training workers can make the training more stable.
    resources_per_worker={"GPU": 1, "CPU": 5},
    use_gpu=True,
)

config_dict = {}
for attr in dir(recsys_config):
    if not attr.startswith('_'):
        value = getattr(recsys_config, attr)
        if not callable(value):
            config_dict[attr] = value

logger.info(f"Starting Ray training with {recsys_config.num_workers} workers")
logger.info(f"Training configuration: {config_dict}")

# Create TorchTrainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop,
    train_loop_config=config_dict,
    scaling_config=scaling_config,
    run_config=RunConfig(
        failure_config=FailureConfig(max_failures=2),
        worker_runtime_env={'env_vars': {"KINETO_USE_DAEMON": "1", "KINETO_DAEMON_INIT_DELAY_S": "5"}},
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
        ),
        storage_path=recsys_config.checkpoint_dir,
    ),
    datasets={
        "train": train_dataset,
        "val": val_dataset,
    },
)

# Run training
logger.info("Starting distributed training...")
result = trainer.fit()

logger.info("Training completed successfully!")
logger.info(f"Final metrics: {result.metrics}")
