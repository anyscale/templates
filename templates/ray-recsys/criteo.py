import boto3
import pyarrow.parquet as pq
import pyarrow as pa
import json
from pathlib import Path
import logging
import ray
import numpy as np
from typing import Dict, List
import pyarrow.compute as pc

logger = logging.getLogger(__name__)


INT_FEATURE_COUNT = 13
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
LOW_FREQUENCY_INDEX = 1  # map low frequency values -> 1

FREQUENCY_THRESHOLD = 3
FEATURE_COUNT_PATH_PATTERN = "criteo/tsv.gz/categorical_feature_value_counts/{}-value_counts.json"
S3_BUCKET = "ray-benchmark-data-internal-us-west-2"
CAT_FEATURE_COUNT = 26
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]

CRITEO_S3_URI = f"s3://{S3_BUCKET}/criteo/tsv.gz"
TRAIN_DATASET_PATH = f"{CRITEO_S3_URI}/train"
VAL_DATASET_PATH = f"{CRITEO_S3_URI}/val"
DEFAULT_LABEL_NAME = "label"
DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]

class DatasetKey:
    TRAIN = "train"
    VALID = "val"

def convert_to_torchrec_batch_format(batch: Dict[str, np.ndarray]) -> "Batch":
    """Convert to a Batch, packaging sparse features as a KJT."""
    import torch

    from torchrec.datasets.utils import Batch
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    dense = batch["dense"]
    sparse = batch["sparse"]
    labels = batch["label"]

    batch_size = len(dense)
    lengths = torch.ones((batch_size * CAT_FEATURE_COUNT,), dtype=torch.int32)
    offsets = torch.arange(0, batch_size * CAT_FEATURE_COUNT + 1, dtype=torch.int32)
    length_per_key: List[int] = [batch_size] * CAT_FEATURE_COUNT
    offset_per_key = [batch_size * i for i in range(CAT_FEATURE_COUNT + 1)]
    index_per_key = {key: i for i, key in enumerate(DEFAULT_CAT_NAMES)}

    # Handle partial batches (last batch).
    # if batch_size == self.batch_size:
    #     length_per_key = self.length_per_key
    #     offset_per_key = self.offset_per_key
    # else:
    #     # handle last batch in dataset when it's an incomplete batch.
    #     length_per_key = CAT_FEATURE_COUNT * [batch_size]
    #     offset_per_key = [batch_size * i for i in range(CAT_FEATURE_COUNT + 1)]

    return Batch(
        dense_features=torch.from_numpy(dense),
        sparse_features=KeyedJaggedTensor(
            keys=DEFAULT_CAT_NAMES,
            # transpose().reshape(-1) introduces a copy
            values=torch.from_numpy(sparse.transpose(1, 0).reshape(-1)),
            lengths=lengths,
            offsets=offsets,
            stride=batch_size,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
        ),
        labels=torch.from_numpy(labels.reshape(-1)),
    )


def fill_missing(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Fill in missing feature values with defaults.
    Default to 0 for dense features, empty string "" for categorical features.
    """
    for feature_name in DEFAULT_INT_NAMES:
        batch[feature_name] = np.nan_to_num(batch[feature_name], nan=0)
    for feature_name in DEFAULT_CAT_NAMES:
        features = batch[feature_name]
        features[np.equal(features, None)] = ""
    return batch


def map_features_to_indices(batch: Dict[str, np.ndarray], categorical_to_feature_mapping_refs: Dict[str, ray.ObjectRef]):

    for cat_feature in DEFAULT_CAT_NAMES:
        feature_mapping_ref = categorical_to_feature_mapping_refs.get(cat_feature, None)
        if feature_mapping_ref:
            feature_mapping = ray.get(feature_mapping_ref)
        else:
            feature_mapping = pa.table({
                'feature_value': [],
                'index': []
            })

        feature_values = batch[cat_feature]

        if len(feature_mapping) > 0:
            # Convert feature values to PyArrow array
            feature_values_array = pa.array(feature_values)

            # Get the lookup array from the mapping table
            lookup_values = feature_mapping['feature_value']

            # Use index_in to find positions in the lookup array
            positions = pc.index_in(feature_values_array, lookup_values)

            # Get the corresponding indices, using null values for not found items
            mapped_indices = pc.take(feature_mapping['index'], positions)

            # Fill null values (not found items) with LOW_FREQUENCY_INDEX
            filled_indices = pc.fill_null(mapped_indices, LOW_FREQUENCY_INDEX)

            # Convert back to numpy array
            batch[cat_feature] = filled_indices.to_numpy().astype(np.int32)
        else:
            # If no mapping available, use default index
            batch[cat_feature] = np.full(len(feature_values), LOW_FREQUENCY_INDEX, dtype=np.int32)

    return batch


def concat_and_normalize_dense_features(
    batch: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Concatenate dense and sparse features together.
    Apply log transformation to dense features."""

    out = {}

    out["dense"] = np.column_stack(
        [batch[feature_name] for feature_name in DEFAULT_INT_NAMES]
    )
    out["sparse"] = np.column_stack(
        [batch[feature_name] for feature_name in DEFAULT_CAT_NAMES]
    )

    out["dense"] += 3  # Prevent log(0)
    out["dense"] = np.log(out["dense"], dtype=np.float32)
    out["label"] = batch["label"]

    return out


@ray.remote(num_cpus=1, memory=21 * 1024 * 1024 * 1024)
def read_feature_mapping_table(feature_name: str) -> pa.Table:
    json_filepath = FEATURE_COUNT_PATH_PATTERN.format(feature_name)
    table = read_parquet_from_s3_or_cache(bucket_name=S3_BUCKET, key=json_filepath, frequency_threshold=FREQUENCY_THRESHOLD)
    # The table returned from ray task will be stored to the object store.
    return table



def read_parquet_from_s3_or_cache(bucket_name, key, frequency_threshold=3):
    # Create cache directory if it doesn't exist
    cache_dir = Path.home() / ".cache" / "ray-recsys" / "s3-cache-parquet" / bucket_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a safe filename from the S3 key (use .parquet extension)
    safe_filename = key.replace("/", "_").replace("\\", "_").replace(".json", ".parquet")
    cache_file_path = cache_dir / f"{safe_filename}"

    # Check if cached parquet file exists and read from it
    if cache_file_path.exists():
        logger.info(f"Reading cached parquet file: {cache_file_path}")
        try:
            table = pq.read_table(cache_file_path)
            return table
        except Exception as e:
            logger.warning(f"Failed to read cached parquet file {cache_file_path}: {e}. Re-downloading from S3.")
            # If cached file is corrupted, remove it and continue to download from S3
            cache_file_path.unlink(missing_ok=True)

    # Download from S3 if not cached or cache is corrupted
    logger.info(f"Downloading JSON from S3: s3://{bucket_name}/{key}")
    s3 = boto3.client("s3")

    # Download object content
    response = s3.get_object(Bucket=bucket_name, Key=key)
    content = response["Body"].read().decode("utf-8")

    # Parse JSON
    value_counts = json.loads(content)

    # Apply frequency filtering and create mapping
    filtered_value_counts = list(filter(lambda x: x[1] >= frequency_threshold, value_counts))

    # Create feature value to index mapping using original logic
    feature_values = []
    indices = []
    for i, (val, _) in enumerate(filtered_value_counts, start=2):
        feature_values.append(val)
        indices.append(i)

    # Create PyArrow table
    table = pa.table({
        'feature_value': feature_values,
        'index': indices
    })

    # Save to cache as parquet
    try:
        pq.write_table(table, cache_file_path)
        logger.info(f"Cached parquet file saved: {cache_file_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache parquet file {cache_file_path}: {e}")

    return table
