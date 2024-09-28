from google.cloud import storage
from contextlib import contextmanager
import os
from tempfile import TemporaryDirectory
import boto3
from urllib.parse import urlparse
from ray.data import Dataset
from openai import OpenAI
from anyscale.llm.dataset import Dataset as AnyscaleDataset
from typing import Dict, Any
import yaml

CLUSTER_STORAGE_PATH = "/mnt/cluster_storage"

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
            file_name = os.path.basename(key)
            local_path = os.path.join(local_dir, file_name)
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
        file_name = os.path.basename(blob.name)
        local_path = os.path.join(local_dir, file_name)
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
    with TemporaryDirectory(dir=CLUSTER_STORAGE_PATH) as temp_path:
        dataset.repartition(1).write_json(temp_path)
        assert len(os.listdir(temp_path)) == 1, "The dataset should be written to a single file"
        dataset_file_path = f"{temp_path}/{os.listdir(temp_path)[0]}"
        yield dataset_file_path


SYSTEM_CONTENT = """
Given a target sentence, construct the underlying meaning representation of the input sentence as a 
single function with attributes and attribute values.

This function should describe the target string accurately and the function must be one of the following

['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation',
'recommend', 'request_attribute'].
    
The attributes must be one of the following:

['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective',
'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier'].
"""


# Query function to call the running service
def query(model: str, prompt: str):
    client = OpenAI(
        base_url="https://e2e-llm-workflows-orange-few-dog-msedw.cld-91sl4yby42b2ivfp.s.anyscaleuserdata.com/v1",
        api_key="your_token_here"
    )

    chat_completions = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": prompt},
        ],
    )

    response = chat_completions.choices[0].message.content
    return response


def to_llm_schema(item):
    messages = [
        {'role': 'system', 'content': SYSTEM_CONTENT},
        {'role': 'user', 'content': item['target']},
        {'role': 'assistant', 'content': item['meaning_representation']}
    ]
    return {'messages': messages}

def update_datasets_in_fine_tuning_config(
    config_path: str,
    train_dataset: AnyscaleDataset,
    validation_dataset: AnyscaleDataset,
):
    # Overwrite `train_path` and `valid_path`
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
        config['train_path'] = train_dataset.storage_uri
        config['valid_path'] = validation_dataset.storage_uri

    # Write the updated configuration back to the file
    with open(config_path, "w") as f:
        f.write(yaml.dump(config))

    # View the training (LoRA) configuration for llama-3-8B
    return config