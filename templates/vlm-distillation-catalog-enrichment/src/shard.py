"""
Shard checkpointing — the resumability primitive for the sharded VLM job.

Convention:
    input_dir/shard_NNNN.parquet     — pre-sharded catalog input (one shard per file)
    output_dir/shard_NNNN/           — committed enriched shard (atomic write)
    output_dir/.shard_NNNN.tmp/      — in-flight write (gets renamed on commit)

A shard is "completed" iff a directory named ``shard_NNNN`` exists at
``output_dir`` with at least one parquet inside. Commits use POSIX
``os.rename``, atomic for directories on the same filesystem (works on
NFS too, which is what ``/mnt/shared_storage/walmart-notebooks`` is).
"""
import os
import re
import shutil


SHARD_RE = re.compile(r"shard_(\d{4})$")
TMP_RE = re.compile(r"\.shard_(\d{4})\.tmp$")


def shard_input_path(input_dir: str, shard_id: int) -> str:
    return os.path.join(input_dir, f"shard_{shard_id:04d}.parquet")


def shard_output_path(output_dir: str, shard_id: int) -> str:
    return os.path.join(output_dir, f"shard_{shard_id:04d}")


def shard_tmp_path(output_dir: str, shard_id: int) -> str:
    return os.path.join(output_dir, f".shard_{shard_id:04d}.tmp")


def list_input_shards(input_dir: str) -> list[int]:
    """Return sorted shard IDs present in input_dir."""
    if not os.path.exists(input_dir):
        return []
    ids = []
    for name in os.listdir(input_dir):
        m = re.match(r"shard_(\d{4})\.parquet$", name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def list_completed_shards(output_dir: str) -> set[int]:
    """Return set of shard IDs that have a fully-committed output dir."""
    if not os.path.exists(output_dir):
        return set()
    completed = set()
    for name in os.listdir(output_dir):
        m = SHARD_RE.match(name)
        if not m:
            continue
        shard_path = os.path.join(output_dir, name)
        # Sanity: a committed shard has at least one parquet inside.
        if os.path.isdir(shard_path) and any(
            f.endswith(".parquet") for f in os.listdir(shard_path)
        ):
            completed.add(int(m.group(1)))
    return completed


def cleanup_stale_tmp(output_dir: str) -> int:
    """Remove leftover .shard_NNNN.tmp/ dirs from a previous crashed run.

    Returns the count cleaned. Safe to call before every run.
    """
    if not os.path.exists(output_dir):
        return 0
    cleaned = 0
    for name in os.listdir(output_dir):
        if TMP_RE.match(name):
            shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)
            cleaned += 1
    return cleaned


def commit_shard(output_dir: str, shard_id: int) -> None:
    """Atomically promote a .shard_NNNN.tmp/ dir to shard_NNNN/.

    POSIX rename is atomic for directories on the same filesystem, so an
    observer (e.g. another process scanning for completed shards) will
    always see either the old name or the new name — never a half-state.
    """
    tmp = shard_tmp_path(output_dir, shard_id)
    final = shard_output_path(output_dir, shard_id)
    if not os.path.exists(tmp):
        raise FileNotFoundError(f"No tmp dir to commit at {tmp}")
    if os.path.exists(final):
        # Idempotent: a previous run already committed this shard.
        shutil.rmtree(tmp, ignore_errors=True)
        return
    os.rename(tmp, final)


def remaining_shards(input_dir: str, output_dir: str) -> list[int]:
    """Input shards minus already-committed output shards, sorted."""
    inputs = set(list_input_shards(input_dir))
    completed = list_completed_shards(output_dir)
    return sorted(inputs - completed)
