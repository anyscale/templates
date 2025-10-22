"""Common utilities for Azure Blob File System (ABFSS) support."""

import logging
import os
import re
import subprocess
from typing import Optional

try:
    from adlfs import AzureBlobFileSystem  # type: ignore
    ADLFS_AVAILABLE = True
except ImportError:
    ADLFS_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_abfss_path(path: str) -> bool:
    """Check if the path is an ABFSS (Azure Blob File System) path."""
    return path.startswith("abfss://") or path.startswith("abfs://")


def create_azure_filesystem(account_name: str) -> AzureBlobFileSystem:
    """Create Azure filesystem using DefaultAzureCredential."""
    if not AZURE_IDENTITY_AVAILABLE:
        raise ImportError(
            "azure-identity is required for Azure authentication. Install it with: pip install azure-identity"
        )

    # Force removal of incomplete service principal environment variables
    # to ensure DefaultAzureCredential uses managed identity
    if "AZURE_CLIENT_ID" in os.environ and "AZURE_CLIENT_SECRET" not in os.environ:
        logger.info("Removing incomplete service principal environment variables to use managed identity")
        os.environ.pop("AZURE_CLIENT_ID", None)
        os.environ.pop("AZURE_TENANT_ID", None)

    try:
        credential = DefaultAzureCredential()
        azure_fs = AzureBlobFileSystem(account_name=account_name, credential=credential)
        logger.info("Successfully created Azure filesystem with DefaultAzureCredential")
        return azure_fs
    except Exception as e:
        logger.error(f"Failed to create Azure filesystem with DefaultAzureCredential: {e}")
        logger.error("DefaultAzureCredential automatically tries multiple authentication methods.")
        logger.error("Common solutions:")
        logger.error("- Ensure you're running in AKS with managed identity configured")
        logger.error("- Or run 'az login' to authenticate with Azure CLI")
        logger.error("- Ensure the identity has proper permissions to access the storage account")
        raise


def upload_to_abfss(
    storage_path: str,
    experiment_name: str,
    local_checkpoint_dir: str = "/mnt/cluster_storage/ray_results/",
) -> None:
    """Upload checkpoints from local storage to ABFSS using Azure CLI.

    Args:
        storage_path: ABFSS path in format abfss://container@account.dfs.core.windows.net/path
        experiment_name: Name of the experiment (used as subdirectory name)
        local_checkpoint_dir: Local directory where checkpoints are stored
    """
    print("Note: Checkpoints are stored locally since Ray Train doesn't support ABFSS storage_path directly.")
    print(f"Local checkpoint directory: {local_checkpoint_dir}")

    # Parse ABFSS path to extract container, account, and path
    # Format: abfss://container@account.dfs.core.windows.net/path
    match = re.match(r"abfss?://([^@]+)@([^.]+)\.dfs\.core\.windows\.net/?(.*)", storage_path)
    if not match:
        print("Warning: Could not parse ABFSS path for upload")
        return

    container = match.group(1)
    account = match.group(2)
    blob_path = match.group(3).rstrip('/')

    print(f"\nUploading checkpoints to ABFSS...")
    print(f"Container: {container}")
    print(f"Account: {account}")
    print(f"Destination path: {blob_path}")

    # Find the experiment directory
    experiment_dir = os.path.join(local_checkpoint_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        print(f"Warning: Experiment directory not found: {experiment_dir}")
        return

    try:
        # First, authenticate with Azure using managed identity
        print("Authenticating with Azure using managed identity...")
        login_cmd = ["az", "login", "--identity"]
        login_result = subprocess.run(login_cmd, capture_output=True, text=True)

        if login_result.returncode != 0:
            print(f"⚠ Warning: az login --identity failed: {login_result.stderr}")
            print("Attempting upload anyway (may work if already authenticated)...")
        else:
            print("✓ Successfully authenticated with managed identity")

        # Use az storage blob upload-batch for reliable uploads
        destination_path = f"{blob_path}/{experiment_name}" if blob_path else experiment_name

        cmd = [
            "az", "storage", "blob", "upload-batch",
            "--account-name", account,
            "--destination", container,
            "--destination-path", destination_path,
            "--source", experiment_dir,
            "--auth-mode", "login",
            "--overwrite"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Successfully uploaded checkpoints to abfss://{container}@{account}.dfs.core.windows.net/{destination_path}")
        else:
            print(f"✗ Failed to upload checkpoints:")
            print(f"Error: {result.stderr}")
            print(f"\nYou can manually upload using:")
            print(f"az login --identity")
            print(f"az storage blob upload-batch --account-name {account} --destination {container} --destination-path {destination_path} --source {experiment_dir} --auth-mode login --overwrite")
    except Exception as e:
        print(f"✗ Error during upload: {e}")
        print(f"\nYou can manually upload using:")
        print(f"az login --identity")
        print(f"az storage blob upload-batch --account-name {account} --destination {container} --destination-path {destination_path} --source {experiment_dir} --auth-mode login --overwrite")


def get_local_storage_path() -> str:
    """Get the local storage path for Ray Train checkpoints."""
    return "/mnt/cluster_storage/ray_results"
