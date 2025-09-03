"""
Criteo dataset processing utilities for Ray recommendation system.

This module provides functions for loading, preprocessing, and transforming
Criteo dataset for use with TorchRec recommendation models.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import ray

logger = logging.getLogger(__name__)


# Dataset configuration constants
INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
LOW_FREQUENCY_INDEX = 1  # Index for mapping low frequency values
FREQUENCY_THRESHOLD = 3  # Minimum frequency threshold for categorical features
LOG_OFFSET = 3  # Offset to prevent log(0) in dense feature normalization

# Feature names
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]
DEFAULT_LABEL_NAME = "label"
DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]

# S3 configuration
S3_BUCKET = "ray-benchmark-data-internal-us-west-2"
FEATURE_COUNT_PATH_PATTERN = "criteo/tsv.gz/categorical_feature_value_counts/{}-value_counts.json"
CRITEO_S3_URI = f"s3://{S3_BUCKET}/criteo/tsv.gz"
TRAIN_DATASET_PATH = f"{CRITEO_S3_URI}/train"
VAL_DATASET_PATH = f"{CRITEO_S3_URI}/val"

class DatasetKey:
    """Constants for dataset split names."""
    TRAIN = "train"
    VALID = "val"

def convert_to_torchrec_batch_format(batch: Dict[str, np.ndarray]) -> "Batch":
    """
    Convert a batch dictionary to TorchRec Batch format.
    
    Packages sparse features as a KeyedJaggedTensor and dense features as tensors.
    
    Args:
        batch: Dictionary containing 'dense', 'sparse', and 'label' arrays
        
    Returns:
        TorchRec Batch object with dense_features, sparse_features, and labels
    """
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

    return Batch(
        dense_features=torch.from_numpy(dense),
        sparse_features=KeyedJaggedTensor(
            keys=DEFAULT_CAT_NAMES,
            # Note: transpose().reshape(-1) introduces a copy but is necessary for proper tensor format
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
    """
    Fill in missing feature values with defaults.
    
    Args:
        batch: Dictionary containing feature arrays that may have missing values
        
    Returns:
        Dictionary with missing values filled (0 for dense features, "" for categorical)
    """
    for feature_name in DEFAULT_INT_NAMES:
        batch[feature_name] = np.nan_to_num(batch[feature_name], nan=0.0)
    
    for feature_name in DEFAULT_CAT_NAMES:
        features = batch[feature_name]
        features[np.equal(features, None)] = ""
    
    return batch


def map_features_to_indices(
    batch: Dict[str, np.ndarray], 
    categorical_to_feature_mapping_refs: Dict[str, ray.ObjectRef]
) -> Dict[str, np.ndarray]:
    """
    Map categorical feature values to indices using precomputed mappings.
    
    Args:
        batch: Dictionary containing categorical feature arrays
        categorical_to_feature_mapping_refs: Ray object references to feature mapping tables
        
    Returns:
        Dictionary with categorical features mapped to integer indices
    """
    for cat_feature in DEFAULT_CAT_NAMES:
        feature_mapping_ref = categorical_to_feature_mapping_refs.get(cat_feature, None)
        if feature_mapping_ref:
            feature_mapping = ray.get(feature_mapping_ref)
        else:
            # Create empty mapping table if none exists
            feature_mapping = pa.table({
                'feature_value': [],
                'index': []
            })

        feature_values = batch[cat_feature]

        if len(feature_mapping) > 0:
            # Convert feature values to PyArrow array for efficient lookup
            feature_values_array = pa.array(feature_values)
            lookup_values = feature_mapping['feature_value']

            # Find positions in the lookup array
            positions = pc.index_in(feature_values_array, lookup_values)
            
            # Get corresponding indices, with null for not found items
            mapped_indices = pc.take(feature_mapping['index'], positions)
            
            # Fill null values (not found items) with low frequency index
            filled_indices = pc.fill_null(mapped_indices, LOW_FREQUENCY_INDEX)
            
            # Convert back to numpy array
            batch[cat_feature] = filled_indices.to_numpy().astype(np.int32)
        else:
            # If no mapping available, use default low frequency index
            batch[cat_feature] = np.full(len(feature_values), LOW_FREQUENCY_INDEX, dtype=np.int32)

    return batch


def concat_and_normalize_dense_features(
    batch: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Concatenate and normalize dense features, and organize sparse features.
    
    Applies log transformation to dense features after adding an offset to prevent log(0).
    
    Args:
        batch: Dictionary containing individual feature arrays
        
    Returns:
        Dictionary with 'dense', 'sparse', and 'label' arrays ready for model input
    """
    out = {}

    # Stack dense features into a single array
    out["dense"] = np.column_stack(
        [batch[feature_name] for feature_name in DEFAULT_INT_NAMES]
    )
    
    # Stack sparse features into a single array
    out["sparse"] = np.column_stack(
        [batch[feature_name] for feature_name in DEFAULT_CAT_NAMES]
    )

    # Apply log transformation to dense features (add offset to prevent log(0))
    out["dense"] = out["dense"] + LOG_OFFSET
    out["dense"] = np.log(out["dense"], dtype=np.float32)
    
    # Preserve labels
    out["label"] = batch["label"]

    return out


@ray.remote(num_cpus=1, memory=21 * 1024 * 1024 * 1024)
def read_feature_mapping_table(feature_name: str) -> pa.Table:
    """
    Ray remote task to read feature mapping table for a categorical feature.
    
    Args:
        feature_name: Name of the categorical feature to load mapping for
        
    Returns:
        PyArrow table containing feature value to index mappings
    """
    json_filepath = FEATURE_COUNT_PATH_PATTERN.format(feature_name)
    table = read_parquet_from_s3_or_cache(
        bucket_name=S3_BUCKET, 
        key=json_filepath, 
        frequency_threshold=FREQUENCY_THRESHOLD
    )
    return table


def read_parquet_from_s3_or_cache(
    bucket_name: str, 
    key: str, 
    frequency_threshold: int = FREQUENCY_THRESHOLD
) -> pa.Table:
    """
    Read feature mapping data from S3 with local parquet caching.
    
    Downloads JSON value counts from S3, applies frequency filtering,
    and caches the result as parquet for faster subsequent access.
    
    Args:
        bucket_name: S3 bucket name
        key: S3 object key (path to JSON file)
        frequency_threshold: Minimum frequency for including features
        
    Returns:
        PyArrow table with feature_value and index columns
    """
    # Create cache directory structure
    cache_dir = Path.home() / ".cache" / "ray-recsys" / "s3-cache-parquet" / bucket_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from S3 key
    safe_filename = key.replace("/", "_").replace("\\", "_").replace(".json", ".parquet")
    cache_file_path = cache_dir / safe_filename

    # Try to read from cache first
    if cache_file_path.exists():
        logger.info(f"Reading cached parquet file: {cache_file_path}")
        try:
            return pq.read_table(cache_file_path)
        except Exception as e:
            logger.warning(
                f"Failed to read cached parquet file {cache_file_path}: {e}. "
                "Re-downloading from S3."
            )
            cache_file_path.unlink(missing_ok=True)

    # Download from S3 if not cached or cache is corrupted
    logger.info(f"Downloading JSON from S3: s3://{bucket_name}/{key}")
    s3 = boto3.client("s3")

    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        content = response["Body"].read().decode("utf-8")
        value_counts = json.loads(content)
    except Exception as e:
        logger.error(f"Failed to download or parse S3 object s3://{bucket_name}/{key}: {e}")
        raise

    # Apply frequency filtering
    filtered_value_counts = [
        (val, count) for val, count in value_counts 
        if count >= frequency_threshold
    ]

    # Create feature value to index mapping (starting from index 2)
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

    # Cache the result as parquet
    try:
        pq.write_table(table, cache_file_path)
        logger.info(f"Cached parquet file saved: {cache_file_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache parquet file {cache_file_path}: {e}")

    return table
