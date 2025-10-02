from s3fs import S3FileSystem
import pyarrow.fs
from typing import Optional, Tuple


def get_path_and_fs(storage_path: str) -> Tuple[str, Optional[pyarrow.fs.FileSystem]]:
    """Get the final storage path and corresponding filesystem, if applicable.

    For S3, this will use s3fs.
    For Azure Data Lake Storage (abfss://), we'll use local storage as a workaround.
    For others, this will use the default filesystem(None).
    """
    s3_prefix = "s3://"
    abfss_prefix = "abfss://"

    if storage_path.startswith(s3_prefix):
        s3fs_storage_path = storage_path[len(s3_prefix) :]

        s3fs_fs = S3FileSystem()
        pyarrow_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3fs_fs))

        return s3fs_storage_path, pyarrow_fs
    
    elif storage_path.startswith(abfss_prefix):
        # Use PyArrow's native ABFSS support (available in PyArrow 20.0.0+)
        # Parse the ABFSS URL to extract account and container info
        # Format: abfss://container@account.dfs.core.windows.net/path
        url_parts = storage_path.replace("abfss://", "").split("/")
        container_account = url_parts[0].split("@")
        
        if len(container_account) == 2:
            container, account = container_account
            account = account.replace(".dfs.core.windows.net", "")
            path = "/" + "/".join(url_parts[1:]) if len(url_parts) > 1 else "/"
            
            # Create AzureFileSystem instance
            azure_fs = pyarrow.fs.AzureFileSystem(account_name=account)
            
            # The container is included in the path for PyArrow
            abfss_storage_path = f"{container}{path}"
            
            print(f"✅ Using PyArrow's native ABFSS support")
            print(f"   Account: {account}")
            print(f"   Container: {container}")
            print(f"   Path: {abfss_storage_path}")
            
            return abfss_storage_path, azure_fs
        else:
            raise ValueError(f"Invalid ABFSS URL format: {storage_path}. Expected: abfss://container@account.dfs.core.windows.net/path")

    return storage_path, None

