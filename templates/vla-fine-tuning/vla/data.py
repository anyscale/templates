"""
data.py  —  DROID dataset I/O and Ray Data pipeline functions.

Organized in three layers:

  I/O helpers      read_hdf5_data, iter_video_frames, _iter_rows
                   Low-level utilities for reading GCS-hosted HDF5 and MP4
                   files into Python-native structures.

  flat_map fns     episode_to_rows, episode_to_training_rows
                   Used as Ray Data .flat_map() callables.  Each function
                   receives one episode dict (a parquet row) and yields one
                   dict per timestep.

  map_batches fns  preprocess_batch
                   Used as Ray Data .map_batches() callables.
"""

import io
import itertools
import av
import h5py
import numpy as np
import smart_open

from typing import Any, Dict, Generator, Iterator


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_PATH_COLS = ("hdf5_path", "wrist_mp4_path", "ext1_mp4_path", "ext2_mp4_path")


def resolve_episode_paths(episode: Dict[str, Any], bucket: str, prefix: str) -> Dict[str, Any]:
    """
    Prepend storage URI to relative path columns in an episode dict.

    Paths that already contain '://' are left unchanged, so this function is
    safe to call on a dataset that has not been stripped of its URI prefixes.

    Args:
        bucket: Storage bucket URI including scheme, e.g. "s3://my-bucket"
                or "gs://my-bucket".
        prefix: Path prefix within the bucket, e.g. "droid/1.0.1".
    """
    base = f"{bucket}/{prefix}/"
    resolved = dict(episode)
    for key in _PATH_COLS:
        val = episode.get(key) or ""
        if val and "://" not in val:
            resolved[key] = base + val
    return resolved


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _open(path: str, mode: str = "rb"):
    """
    Open a file from any supported scheme.

    For S3 paths, always uses unsigned (anonymous) access since the DROID
    dataset bucket is publicly readable.  This avoids permission errors
    when the caller has AWS credentials configured for a different account.
    """
    if path.startswith("s3://"):
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config

        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        return smart_open.open(path, mode, transport_params={"client": client})
    return smart_open.open(path, mode)


def read_hdf5_data(hdf5_path: str) -> Dict[str, np.ndarray]:
    """
    Read an HDF5 file from S3 and return every leaf dataset as a flat dict.

    Keys use dot notation (e.g. "action.cartesian_velocity") and values are
    numpy arrays of shape (T, ...).  The file is buffered in memory so that
    h5py can seek freely without multiple S3 round-trips.
    """

    with _open(hdf5_path, "rb") as f:
        buf = io.BytesIO(f.read())

    data = {}
    with h5py.File(buf, "r") as hf:

        def _collect(name, node):
            if isinstance(node, h5py.Dataset):
                data[name.replace("/", ".")] = node[()]

        hf.visititems(_collect)

    return data


def iter_video_frames(mp4_path: str) -> Iterator[np.ndarray]:
    """
    Lazily yield RGB frames (HWC uint8) from an MP4 file on S3.

    The compressed bytes are buffered once in memory so that PyAV can seek
    without additional S3 reads; frames are decoded one at a time.
    """
    with _open(mp4_path, "rb") as f:
        buf = io.BytesIO(f.read())
    with av.open(buf, mode="r") as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def _iter_rows(
    episode: Dict[str, Any],
    hdf5_data: Dict[str, np.ndarray],
    camera_iters: Dict[str, Iterator[np.ndarray]],
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield one dict per timestep, combining:
      - episode metadata  (uuid, task, timestamp)
      - per-step HDF5 values  (scalars or lists)
      - one frame per camera  (keyed as "<camera>_frame")

    Stops at trajectory_length timesteps, or when all iterators are exhausted.
    """
    T = episode.get("trajectory_length")
    base = {k: episode.get(k) for k in ("uuid", "task", "timestamp")}
    cam_names = list(camera_iters.keys())

    for t, cam_frames in enumerate(
        itertools.zip_longest(*camera_iters.values(), fillvalue=None)
    ):
        if T is not None and t >= T:
            break

        row = {**base, "timestep": t}
        for key, arr in hdf5_data.items():
            row[key] = arr[t].tolist() if arr.ndim > 1 else arr[t].item()
        for cam, frame in zip(cam_names, cam_frames):
            row[f"{cam}_frame"] = frame

        yield row


# ---------------------------------------------------------------------------
# flat_map functions  (Ray Data .flat_map())
# ---------------------------------------------------------------------------


def episode_to_rows(episode: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Expand one episode into per-timestep rows, decoding all three cameras.

    Useful for exploration and offline analysis.  For training, prefer
    episode_to_training_rows which decodes only the wrist camera and reads
    only the HDF5 fields required by PI0.5.
    """
    hdf5_path = episode.get("hdf5_path")
    wrist_mp4_path = episode.get("wrist_mp4_path")
    ext1_mp4_path = episode.get("ext1_mp4_path")
    ext2_mp4_path = episode.get("ext2_mp4_path")

    if not hdf5_path:
        return

    hdf5_data = read_hdf5_data(hdf5_path)
    camera_iters = {
        "wrist": iter_video_frames(wrist_mp4_path) if wrist_mp4_path else iter([]),
        "ext1": iter_video_frames(ext1_mp4_path) if ext1_mp4_path else iter([]),
        "ext2": iter_video_frames(ext2_mp4_path) if ext2_mp4_path else iter([]),
    }

    yield from _iter_rows(episode, hdf5_data, camera_iters)


# PI0.5 action/state representation:
#   action  = cartesian_velocity (6,) ++ gripper_velocity (1,) → 7-dim
#   state   = cartesian_position (6,) ++ gripper_position (1,) → 7-dim
#
# NOTE: if the pretrained PI0.5 head was trained on a different action
# dimension (e.g. 14-dim ALOHA), reinitialize action_in_proj /
# action_out_proj before the backbone freeze in train_worker.py.

_HDF5_KEYS_NEEDED = {
    "action/cartesian_velocity",
    "action/gripper_velocity",
    "observation/robot_state/cartesian_position",
    "observation/robot_state/gripper_position",
}


def episode_to_training_rows(
    episode: Dict[str, Any],
) -> Generator[Dict[str, Any], None, None]:
    """
    Expand one episode into per-timestep rows for training.

    Reads only the four HDF5 fields required by the PI0.5 preprocessor and
    decodes only the wrist camera, skipping ext1/ext2 to avoid unnecessary
    GCS reads and video decode work.
    """
    hdf5_path = episode.get("hdf5_path")
    wrist_mp4_path = episode.get("wrist_mp4_path")

    if not hdf5_path:
        return

    all_hdf5 = read_hdf5_data(hdf5_path)
    hdf5_data = {
        k: v for k, v in all_hdf5.items() if k.replace(".", "/") in _HDF5_KEYS_NEEDED
    }

    camera_iters = {
        "wrist": iter_video_frames(wrist_mp4_path) if wrist_mp4_path else iter([]),
    }

    yield from _iter_rows(episode, hdf5_data, camera_iters)


# ---------------------------------------------------------------------------
# map_batches preprocessing  (Ray Data .map_batches(), runs on CPU workers)
# ---------------------------------------------------------------------------


def preprocess_batch(batch) -> dict:
    """
    Convert raw DROID timestep rows into PI0.5-compatible tensor arrays.

    Input keys (produced by episode_to_training_rows):
      wrist_frame                                  (B, H, W, 3)  uint8
      action.cartesian_velocity                    (B, 6)        float64
      action.gripper_velocity                      (B,)          float64
      observation.robot_state.cartesian_position   (B, 6)        float64
      observation.robot_state.gripper_position     (B,)          float64
      task                                         (B,)          str

    Output keys (PI0.5 schema):
      observation.images.base_0_rgb   (B, 3, H, W)  float32
      observation.state               (B, 7)         float32
      action                          (B, 7)         float32
      task                            (B,)           str
    """
    action = np.concatenate(
        [
            np.stack(
                [
                    np.asarray(x, dtype=np.float32)
                    for x in batch["action.cartesian_velocity"]
                ]
            ),
            np.asarray(batch["action.gripper_velocity"], dtype=np.float32).reshape(
                -1, 1
            ),
        ],
        axis=-1,
    )  # (B, 7)

    state = np.concatenate(
        [
            np.stack(
                [
                    np.asarray(x, dtype=np.float32)
                    for x in batch["observation.robot_state.cartesian_position"]
                ]
            ),
            np.asarray(
                batch["observation.robot_state.gripper_position"], dtype=np.float32
            ).reshape(-1, 1),
        ],
        axis=-1,
    )  # (B, 7)

    images = np.stack(
        [
            np.transpose(frame, (2, 0, 1)).astype(np.float32)
            for frame in batch["wrist_frame"]
        ],
        axis=0,
    )  # (B, 3, H, W)

    return {
        "task": batch["task"],
        "observation.images.base_0_rgb": images,
        "observation.state": state,
        "action": action,
    }
