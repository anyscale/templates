import hashlib
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock

from src.utils.common import MODEL_HOME, init_logger


class DownloadFailedError(Exception):
    pass


logger = init_logger()


def is_remote_path(source_path: str) -> bool:
    scheme = urlparse(source_path).scheme
    if not scheme:
        return False
    return True


def download_to_local(source_path: str):
    """Thread-safe download from remote storage"""
    scheme = urlparse(source_path).scheme
    local_path = get_local_path(source_path)
    if not is_remote_path(source_path):
        logger.info(f"Found local path {source_path}, skipping downloading...")
        return source_path

    elif scheme == "s3":
        download_from_s3(source_path, local_path)
    elif scheme == "gcs":
        download_from_gcs(source_path, local_path)
    else:
        raise DownloadFailedError(f"Invalid remote path: {source_path}")
    return local_path


def get_lock_path(local_path: str):
    path = Path(local_path)
    parent = path.parent
    parent.mkdir(exist_ok=True, parents=True)
    return parent / (path.name + ".lock")


def download_from_s3(remote_path: str, local_path: str):
    lock_file = get_lock_path(local_path)
    with FileLock(lock_file):
        result = subprocess.run(["aws", "s3", "sync", remote_path, local_path])
        if result.returncode != 0:
            raise DownloadFailedError(
                f"Download failed from remote storage {remote_path} with result: {result}"
            )


def download_from_gcs(remote_path: str, local_path: str):
    lock_file = get_lock_path(local_path)
    with FileLock(lock_file):
        result = subprocess.run(
            ["gcloud", "storage", "cp", "--recursive", remote_path, local_path]
        )
        if result.returncode != 0:
            raise DownloadFailedError(
                f"Download failed from remote storage with result: {result}"
            )


def get_local_path(source_path: str):
    checkpoint_path_hash = hashlib.md5(source_path.encode()).hexdigest()
    local_path = os.path.join(MODEL_HOME, f"models--checkpoint--{checkpoint_path_hash}")
    return local_path
