from s3fs import S3FileSystem
import pyarrow.fs
from typing import Optional, Tuple


def get_path_and_fs(storage_path: str) -> Tuple[str, Optional[pyarrow.fs.FileSystem]]:
    """Get the final storage path and corresponding filesystem, if applicable.

    For S3, this will use s3fs.
    For others, this will use the default filesystem(None).
    """
    s3_prefix = "s3://"

    if storage_path.startswith(s3_prefix):
        s3fs_storage_path = storage_path[len(s3_prefix) :]

        s3fs_fs = S3FileSystem()
        pyarrow_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3fs_fs))

        return s3fs_storage_path, pyarrow_fs

    return storage_path, None
