from google.cloud import storage
from contextlib import contextmanager
import os
from tempfile import TemporaryDirectory
import boto3
from urllib.parse import urlparse
from ray.data import Dataset
import yaml


def download_files_from_s3(s3_uri, local_dir):
    parsed_uri = urlparse(s3_uri)
    if parsed_uri.scheme != "s3":
        raise ValueError(f"Expected S3 URI, got {s3_uri}")
    bucket = parsed_uri.netloc
    path = parsed_uri.path.lstrip("/")

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=path)
    for page in page_iterator:
        for item in page.get('Contents', []):
            key = item['Key']
            local_path = os.path.join(local_dir, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            print(f"Downloaded {key} to {local_path}")

def download_files_from_gcs(gcs_uri, local_dir):
    parsed_uri = urlparse(gcs_uri)
    if parsed_uri.scheme != "gs":
        raise ValueError(f"Expected GCS URI, got {gcs_uri}")
    bucket_name = parsed_uri.netloc
    prefix = parsed_uri.path.lstrip("/")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        # Skip in case the blob is the root folder
        if blob.name.rstrip("/") == prefix:
            continue
        local_path = os.path.join(local_dir, blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")

def download_files_from_remote(uri, local_dir):
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == "gs":
        download_files_from_gcs(uri, local_dir)
    elif parsed_uri.scheme == "s3":
        download_files_from_s3(uri, local_dir)
    else:
        raise ValueError(f"Expected S3 or GCS URI, got {uri}")


@contextmanager
def get_dataset_file_path(dataset: Dataset):
    """Transforms a Ray `Dataset` into a single temp. JSON file written on disk.
    Yields the path to the file."""
    with TemporaryDirectory() as temp_path:
        dataset.repartition(1).write_json(temp_path)
        assert len(os.listdir(temp_path)) == 1, "The dataset should be written to a single file"
        dataset_file_path = f"{temp_path}/{os.listdir(temp_path)[0]}"
        yield dataset_file_path

def set_key_value_in_config_file(config_path: str, key: str, value: str):
    """Take a path to a .json config file and update the key to a value, then save the file"""
    with open(config_path, 'r') as stream:
        loaded = yaml.safe_load(stream)
    # Modify the fields from the dict
    loaded[key] = value
    # Save it again
    with open(config_path, 'w') as stream:
        yaml.dump(loaded, stream, default_flow_style=False)
