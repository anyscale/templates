from contextlib import contextmanager
import os
from tempfile import TemporaryDirectory
import boto3
from ray.data import Dataset


def download_files_from_bucket(bucket, path, local_dir):
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

@contextmanager
def get_dataset_file_path(dataset: Dataset):
    """Transforms a Ray `Dataset` into a single temp. JSON file written on disk.
    Yields the path to the file."""
    with TemporaryDirectory() as temp_path:
        dataset.repartition(1).write_json(temp_path)
        assert len(os.listdir(temp_path)) == 1, "The dataset should be written to a single file"
        dataset_file_path = f"{temp_path}/{os.listdir(temp_path)[0]}"
        yield dataset_file_path
