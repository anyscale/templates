"""
stats.py  —  Normalization stats pipeline for DROID.

One-time scan of HDF5 scalars to compute per-feature mean/std.

Three-stage Ray Data pipeline:
  filter        → drop episodes without an HDF5 path
  map           → load the four scalar arrays from HDF5  (one row in, one row out)
  map           → accumulate partial sums per episode    (one row in, one row out)
"""

import numpy as np

from typing import Any, Dict

from vla.data import read_hdf5_data


def extract_episode_action_and_state(episode: Dict[str, Any]) -> Dict[str, Any]:
    """
    map stage: load the four HDF5 scalar arrays needed for stats and attach
    them to the episode dict.

    Separating I/O (this function) from stats accumulation (compute_episode_stats)
    means each concern is independently testable and Ray Data can parallelise
    the two stages with different concurrency settings.
    """
    try:
        hdf5_data = read_hdf5_data(episode["hdf5_path"])
    except Exception:
        return {**episode, "cv": None, "gv": None, "cp": None, "gp": None}
    return {
        **episode,
        "cv": hdf5_data.get("action.cartesian_velocity"),                # (T, 6)
        "gv": hdf5_data.get("action.gripper_velocity"),                  # (T,)
        "cp": hdf5_data.get("observation.robot_state.cartesian_position"),  # (T, 6)
        "gp": hdf5_data.get("observation.robot_state.gripper_position"),    # (T,)
    }


def compute_episode_stats(episode: Dict[str, Any]) -> Dict[str, Any]:
    """
    map stage: reduce a loaded episode to partial sums for online mean/variance.

    Operates directly on the numpy arrays attached by extract_episode_action_and_state,
    avoiding per-timestep Python dict allocations and boxing/unboxing overhead.
    Returns a count=0 zero row if any required array is missing.
    """
    cv = episode.get("cv")
    gv = episode.get("gv")
    cp = episode.get("cp")
    gp = episode.get("gp")

    if any(x is None for x in (cv, gv, cp, gp)):
        zeros = [0.0] * 7
        return {
            "action_sum":    zeros,
            "action_sum_sq": zeros,
            "state_sum":     zeros,
            "state_sum_sq":  zeros,
            "count":         0,
        }

    action = np.concatenate([
        np.asarray(cv, dtype=np.float64),
        np.asarray(gv, dtype=np.float64).reshape(-1, 1),
    ], axis=-1)  # (T, 7)

    state = np.concatenate([
        np.asarray(cp, dtype=np.float64),
        np.asarray(gp, dtype=np.float64).reshape(-1, 1),
    ], axis=-1)  # (T, 7)

    return {
        "action_sum":    action.sum(axis=0).tolist(),
        "action_sum_sq": (action ** 2).sum(axis=0).tolist(),
        "state_sum":     state.sum(axis=0).tolist(),
        "state_sum_sq":  (state ** 2).sum(axis=0).tolist(),
        "count":         int(action.shape[0]),
    }


def compute_stats(stats_ds) -> dict:
    """Reduce the stats pipeline to per-feature mean/std and return as a dict."""
    rows  = stats_ds.to_pandas()
    total = int(rows["count"].sum())

    action_mean, action_std = _finalize(rows, total, "action_sum", "action_sum_sq")
    state_mean,  state_std  = _finalize(rows, total, "state_sum",  "state_sum_sq")

    return {
        "action":            {"mean": action_mean, "std": action_std},
        "observation.state": {"mean": state_mean,  "std": state_std},
    }


def _finalize(rows, total: int, sum_col: str, sq_col: str):
    """Compute per-feature mean and std from accumulated partial sums."""
    s    = np.stack(rows[sum_col].tolist()).sum(axis=0)
    s2   = np.stack(rows[sq_col].tolist()).sum(axis=0)
    mean = s / total
    std  = np.sqrt(np.maximum(s2 / total - mean ** 2, 0)).clip(min=1e-6)
    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()
