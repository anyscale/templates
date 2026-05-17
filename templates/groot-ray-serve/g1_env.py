"""
Isaac Lab G1 environment wrapper for GR00T policy rollout.

Uses `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` which is Isaac Lab's official
G1 loco-manipulation task. The GR00T-N1.6-G1-PnPAppleToPlate checkpoint was
trained on closely related locomanipulation data, so zero-shot should give
something visibly policy-driven (though task success rates vary - fine-tuning
on the exact task distribution is the next step).

IMPORTANT: Isaac Lab must be initialized via its AppLauncher BEFORE any
gymnasium / omni imports. This wrapper handles that. The caller must run
inside an Isaac Lab-enabled Python env (the `./isaaclab.sh -p` wrapper, or
a venv with Isaac Sim's kit kernel on the path).
"""
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np


# Deferred - only set up when first env is created, to avoid double-launching Kit.
_APP_LAUNCHED = False


def _launch_isaac_app(headless: bool = True):
    """Launch the Isaac Sim AppLauncher exactly once per process.

    IMPORTANT: imports pinocchio BEFORE AppLauncher as a workaround for
    NVIDIA IsaacLab bug #4090 (pinocchio pybind11 std::vector<std::string>
    bindings get corrupted after Isaac Lab's URDF importer runs). This is
    the same approach used in Isaac Lab's own test_pink_ik.py tests.
    """
    global _APP_LAUNCHED
    if _APP_LAUNCHED:
        return

    # [WORKAROUND for IsaacLab #4090] Pre-load pinocchio bindings so
    # pybind11 registers the std::vector<std::string> converter before
    # Isaac Lab's C++ extensions load and clobber it.
    import pinocchio  # noqa: F401

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless, enable_cameras=True)
    # Keep a reference so Kit doesn't garbage-collect mid-rollout.
    globals()["_ISAAC_APP"] = app_launcher.app
    _APP_LAUNCHED = True


class G1LocomanipulationEnv:
    """Thin wrapper around the Isaac Lab G1 gym env matching GR00T's obs format.

    The raw Isaac Lab env returns dict observations with joint states and
    camera images. GR00T expects a nested dict keyed by modality (video.*,
    state.*, annotation.*). We translate here.
    """

    def __init__(
        self,
        task_name: str = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
        language_instruction: str = "pick up the apple and place it on the plate",
        headless: bool = True,
        seed: int = 42,
    ):
        _launch_isaac_app(headless=headless)

        # Imports must come AFTER AppLauncher.
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  - registers most Isaac-* task names
        # The locomanipulation subpackage isn't auto-imported by isaaclab_tasks in
        # this Isaac Lab version (2.3.2). Force-import it so the G1 PickPlace task
        # registers itself via its own __init__.py's gym.register() calls.
        from isaaclab_tasks.manager_based.locomanipulation import pick_place  # noqa: F401
        import torch  # noqa: F401  - imported to set CUDA context early

        self.task_name = task_name
        self.language_instruction = language_instruction
        self.seed = seed
        self._step_count = 0

        print(f"[G1Env] Creating {task_name}")
        # Isaac Lab registers tasks with an `env_cfg_entry_point` that's a
        # class, not an instance. We need to instantiate it and pass as cfg=.
        # parse_env_cfg handles this plus common cmdline-style overrides.
        from isaaclab_tasks.utils import parse_env_cfg
        env_cfg = parse_env_cfg(
            task_name,
            device="cuda:0",
            num_envs=1,
            use_fabric=True,
        )
        self.env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
        print(f"[G1Env] obs_space: {self.env.observation_space}")
        print(f"[G1Env] action_space: {self.env.action_space}")

    def reset(self) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=self.seed)
        self._step_count = 0
        return self._format_obs(obs)

    def step(self, action_input, step_idx: int = 0) -> Tuple[Dict[str, Any], float, bool, dict]:
        """Execute one step.

        `action_input` can be either:
          - a dict (GR00T action chunk) — we extract step `step_idx` and flatten.
          - a flat np.ndarray of shape (1, 28) — passed through directly (for
            random-action testing / placeholder).

        GR00T returns 40-step chunks; the run_demo orchestrator decides how many
        to execute before re-querying (typically 8-16 for stability).

        Isaac Lab's env.step() calls action.to(self.device), so we MUST pass a
        torch tensor (not numpy array). Convert here.
        """
        import torch  # deferred for env safety

        if isinstance(action_input, dict):
            raw_action = self._flatten_action(action_input, step_idx=step_idx)
        else:
            raw_action = np.asarray(action_input, dtype=np.float32)
            if raw_action.ndim == 1:
                raw_action = raw_action[None, :]

        # Isaac Lab wants a torch tensor with .to() method
        action_tensor = torch.as_tensor(raw_action, dtype=torch.float32)

        obs, reward, terminated, truncated, info = self.env.step(action_tensor)
        self._step_count += 1
        done = bool(terminated or truncated)
        return self._format_obs(obs), float(reward), done, info

    def _format_obs(self, raw_obs: Any) -> Dict[str, Any]:
        """Translate Isaac Lab's obs dict into GR00T's REAL_G1 nested modality schema.

        Verified schema (via live probe on cluster):
            {
              "video":    {"ego_view":  [B, 2, H, W, 3] uint8},   # 2 frames: -20 & 0
              "state":    {
                  "left_wrist_eef_9d":  [B, 1, 9] float32,   # 3 pos + 6 rot (first 2 rows of R)
                  "right_wrist_eef_9d": [B, 1, 9] float32,
                  "left_hand":          [B, 1, 7] float32,
                  "right_hand":         [B, 1, 7] float32,
                  "left_arm":           [B, 1, 7] float32,
                  "right_arm":          [B, 1, 7] float32,
                  "waist":              [B, 1, 3] float32,
              },
              "language": {"annotation.human.task_description": [[str]]},
            }

        Isaac Lab's PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 task gives us joint
        positions + eef poses. This function slices/packs them into the above.

        KNOWN GAPS (TODO before production use):
          * Joint indexing into Isaac Lab's 29-DoF G1 is a best guess; exact
            ordering depends on the URDF. First run should print raw_obs.keys()
            and actual joint names from env.unwrapped.scene["robot"].joint_names
            to verify.
          * 9D EEF format uses 3 pos + 2 rotation-matrix rows (rows 1&2, 6 dims).
            Isaac Lab exposes eef pose as position + quaternion. We convert
            below via R = quat_to_matrix(q); flat[0:3]=pos, flat[3:9]=R[:2].flatten().
          * Video horizon is 2 frames but we only have 1 current frame, so we
            duplicate current frame for the -20 slot. A proper ring buffer would
            be better but is out of scope for this demo.
        """
        from scipy.spatial.transform import Rotation as _R  # deferred for env safety

        def _np(x, dtype=np.float32):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            elif not isinstance(x, np.ndarray):
                x = np.asarray(x)
            return x.astype(dtype)

        policy_obs = raw_obs.get("policy", raw_obs) if isinstance(raw_obs, dict) else raw_obs

        # ---------- Video ----------
        # Horizon is [-20, 0] so we need TWO frames. We only have current, so
        # duplicate it into both slots. Real implementation would ring-buffer.
        if isinstance(policy_obs, dict) and "rgb" in policy_obs:
            rgb = _np(policy_obs["rgb"], dtype=np.uint8)
        else:
            rgb = np.asarray(self.env.render(), dtype=np.uint8)
        if rgb.ndim == 3:
            rgb = rgb[None, ...]  # add batch dim if missing
        # Now rgb is [B, H, W, 3] or [B, T, H, W, 3]. Need [B=1, T=2, H, W, 3]:
        if rgb.ndim == 4:
            rgb = rgb[:, None].repeat(2, axis=1)  # duplicate across time
        elif rgb.ndim == 5 and rgb.shape[1] == 1:
            rgb = rgb.repeat(2, axis=1)

        # ---------- State ----------
        # Get joint positions (flatten batch dim, Isaac Lab uses shape (1, N)).
        joint_pos = _np(policy_obs.get("joint_pos", np.zeros(29))) if isinstance(policy_obs, dict) \
                    else _np(np.zeros(29))
        if joint_pos.ndim == 2:
            joint_pos = joint_pos[0]  # drop batch dim

        # TODO: these slices are guesses. Verify against Isaac Lab's actual
        # joint ordering on first run.
        left_arm   = joint_pos[6:13]  if joint_pos.shape[0] >= 13 else np.zeros(7, np.float32)
        right_arm  = joint_pos[13:20] if joint_pos.shape[0] >= 20 else np.zeros(7, np.float32)
        left_hand  = joint_pos[20:27] if joint_pos.shape[0] >= 27 else np.zeros(7, np.float32)
        right_hand = joint_pos[27:34] if joint_pos.shape[0] >= 34 else np.zeros(7, np.float32)
        waist      = joint_pos[0:3]   if joint_pos.shape[0] >= 3  else np.zeros(3, np.float32)

        # EEF poses (9D = [xyz, R_row1, R_row2])
        def _pose_to_9d(pos, quat_xyzw):
            """Convert 3D position + quaternion to GR00T's 9D EEF format."""
            R_mat = _R.from_quat(quat_xyzw).as_matrix()  # (3,3)
            return np.concatenate([pos, R_mat[0], R_mat[1]], axis=-1).astype(np.float32)

        def _eef_9d(prefix: str):
            pos_key = f"{prefix}_eef_pos"
            quat_key = f"{prefix}_eef_quat"
            if isinstance(policy_obs, dict) and pos_key in policy_obs and quat_key in policy_obs:
                pos = _np(policy_obs[pos_key])
                quat = _np(policy_obs[quat_key])  # may be wxyz - Isaac Lab uses wxyz
                if pos.ndim == 2: pos = pos[0]
                if quat.ndim == 2: quat = quat[0]
                # Isaac Lab quat is (w, x, y, z); scipy wants (x, y, z, w)
                quat_xyzw = np.roll(quat, -1)
                return _pose_to_9d(pos, quat_xyzw)
            # Fallback: identity at origin
            return np.array([0.3, 0.0, 0.0, 1, 0, 0, 0, 1, 0], dtype=np.float32)

        left_eef_9d  = _eef_9d("left")
        right_eef_9d = _eef_9d("right")

        state = {
            "left_wrist_eef_9d":  left_eef_9d[None, None, :],      # (1, 1, 9)
            "right_wrist_eef_9d": right_eef_9d[None, None, :],
            "left_hand":   left_hand[None, None, :],
            "right_hand":  right_hand[None, None, :],
            "left_arm":    left_arm[None, None, :],
            "right_arm":   right_arm[None, None, :],
            "waist":       waist[None, None, :],
        }

        return {
            "video":    {"ego_view": rgb.astype(np.uint8)},
            "state":    state,
            "language": {
                "annotation.human.task_description": [[self.language_instruction]],
            },
        }

    def _flatten_action(self, action_chunk: Dict[str, np.ndarray], step_idx: int = 0) -> np.ndarray:
        """Flatten one timestep of GR00T's 9-key action chunk into Isaac Lab's
        28-dim flat action.

        GR00T outputs (all with shape [1, 40, D]):
            left_wrist_eef_9d   (1, 40, 9)
            right_wrist_eef_9d  (1, 40, 9)
            left_hand           (1, 40, 7)
            right_hand          (1, 40, 7)
            left_arm            (1, 40, 7)
            right_arm           (1, 40, 7)
            waist               (1, 40, 3)
            base_height_command (1, 40, 1)
            navigate_command    (1, 40, 3)

        Isaac Lab's Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 expects
        a (1, 28) action. From the task cfg inspection earlier, layout is
        upper-body IK targets (probably per-arm 7-DoF EEF + hand joints).

        STRATEGY: we take GR00T's upper-body joint targets (arms + hands)
        which are 7+7+7+7=28. This skips the wrist_eef_9d poses (those drive
        the IK controller; we use the joint-space outputs directly since the
        Isaac Lab cfg uses JointPositionAction).

        28 = left_arm (7) + right_arm (7) + left_hand (7) + right_hand (7).
        (If this doesn't match, env.step will error with a clear shape msg.)

        Args:
            action_chunk: GR00T action output dict
            step_idx: which of the 40 timesteps to execute (0 by default;
                run_demo can iterate 0..N-1 before re-querying).

        Returns:
            np.ndarray shape (1, 28), float32, ready for env.step().
        """
        def _pick(k, n):
            if k not in action_chunk:
                return np.zeros(n, dtype=np.float32)
            arr = np.asarray(action_chunk[k], dtype=np.float32)
            # Expected shape: (B, T, D). Some callers may strip batch.
            if arr.ndim == 3:
                arr = arr[0, step_idx]
            elif arr.ndim == 2:
                arr = arr[step_idx]
            return arr[:n] if arr.shape[-1] >= n else np.pad(arr, (0, n - arr.shape[-1]))

        left_arm   = _pick("left_arm",   7)
        right_arm  = _pick("right_arm",  7)
        left_hand  = _pick("left_hand",  7)
        right_hand = _pick("right_hand", 7)

        flat = np.concatenate([left_arm, right_arm, left_hand, right_hand], axis=-1)
        return flat[None, :].astype(np.float32)  # (1, 28)

    def render_frame(self) -> np.ndarray:
        return np.asarray(self.env.render(), dtype=np.uint8)

    def close(self):
        self.env.close()
