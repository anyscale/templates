"""
GR00T N1.7 Policy Server via Ray Serve.

Wraps NVIDIA's Gr00tPolicy in a Ray Serve deployment so multiple Isaac Lab
sim workers can share inference over a multi-replica, GPU-accelerated backend.

All of the following have been verified to work on the current Anyscale cluster
(Isaac-GR00T main branch, transformers 4.57.6, flash-attn 2.7.4, Python 3.11):

  * Embodiment tag: REAL_G1 (NOT UNITREE_G1 - that's a posttrain-only tag that
    requires a fine-tuned checkpoint; REAL_G1 is what the base N1.7-3B supports)
  * Obs format: nested dict {"video": {...}, "state": {...}, "language": {...}}
  * Action chunk: 9 keys with 40-step horizon:
      left_wrist_eef_9d   (1, 40, 9)
      right_wrist_eef_9d  (1, 40, 9)
      left_hand           (1, 40, 7)
      right_hand          (1, 40, 7)
      left_arm            (1, 40, 7)
      right_arm           (1, 40, 7)
      waist               (1, 40, 3)
      base_height_command (1, 40, 1)
      navigate_command    (1, 40, 3)
  * State format matches action keys except no base_height/navigate:
      left_wrist_eef_9d, right_wrist_eef_9d (9D = 3 pos + 6 rotation),
      left_hand, right_hand, left_arm, right_arm, waist
  * Video: ego_view, horizon [-20, 0] = 2 frames (current + 20 steps ago)

Three runtime patches are required before loading the model:
  1. `VideoInput` moved from transformers.image_utils to transformers.video_utils
     in transformers >=4.54. The Eagle VLM backbone's dynamic processor still
     imports from the old location. We shim it.
  2. Qwen3 VLM asserts _attn_implementation == "flash_attention_2" but
     AutoModel.from_config doesn't propagate that. We monkey-patch it in.
  3. HF_TOKEN must be in the worker's env for gated Cosmos-Reason2-2B access
     (Ray's runtime_env.env_vars is the clean way).
"""
import time
import pickle
import numpy as np
import torch
from fastapi import FastAPI, Request, Response
from ray import serve


# Separate FastAPI apps per deployment (Serve ingress needs a unique app)
_gr00t_app = FastAPI()
_placeholder_app = FastAPI()


def _apply_compat_patches():
    """Idempotent runtime patches required to load N1.7 with current deps."""
    # Patch 1: VideoInput shim
    import transformers.image_utils
    if not hasattr(transformers.image_utils, "VideoInput"):
        try:
            from transformers.video_utils import VideoInput
            transformers.image_utils.VideoInput = VideoInput
        except ImportError:
            pass  # Older transformers may not need the shim

    # Patch 2: Force flash_attention_2 on Qwen3 VLM
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
    if not getattr(_BaseAutoModelClass, "_gr00t_attn_patched", False):
        _orig = _BaseAutoModelClass.from_config.__func__

        def _patched(cls, config, **kwargs):
            if hasattr(config, "text_config"):
                config.text_config._attn_implementation = "flash_attention_2"
            config._attn_implementation = "flash_attention_2"
            if "attn_implementation" not in kwargs:
                kwargs["attn_implementation"] = "flash_attention_2"
            return _orig(cls, config, **kwargs)

        _BaseAutoModelClass.from_config = classmethod(_patched)
        _BaseAutoModelClass._gr00t_attn_patched = True


def _hf_login_if_token():
    """Login to HF if HF_TOKEN env var is set (required for gated Cosmos)."""
    import os
    tok = os.environ.get("HF_TOKEN")
    if tok:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=16,
)
@serve.ingress(_gr00t_app)
class GR00TPolicyServer:
    """Serves GR00T N1.7-3B for G1 (REAL_G1 embodiment, zero-shot)."""

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.7-3B",
        embodiment_tag: str = "REAL_G1",
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.embodiment_tag_name = embodiment_tag
        self.device = device
        self._load_model()

    def _load_model(self):
        print(f"[GR00TServer] Loading {self.model_path} on {self.device}")
        _hf_login_if_token()
        _apply_compat_patches()

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        tag = EmbodimentTag.resolve(self.embodiment_tag_name) \
            if isinstance(self.embodiment_tag_name, str) else self.embodiment_tag_name

        t0 = time.time()
        self.policy = Gr00tPolicy(
            embodiment_tag=tag,
            model_path=self.model_path,
            device=self.device,
        )
        load_time = time.time() - t0

        # Cache modality configs - sim side will use these to validate obs dicts
        self.modality_configs_serializable = {
            mod: {
                "modality_keys": cfg.modality_keys,
                "delta_indices": list(cfg.delta_indices),
            }
            for mod, cfg in self.policy.modality_configs.items()
        }

        num_params = sum(p.numel() for p in self.policy.model.parameters())
        print(f"[GR00TServer] Loaded in {load_time:.1f}s ({num_params/1e9:.2f}B params)")
        print(f"[GR00TServer] Modality config: {self.modality_configs_serializable}")

        self.num_params = num_params
        self.load_time = load_time
        self._call_count = 0
        self._total_latency = 0.0

    @_gr00t_app.post("/predict")
    async def predict_http(self, request: Request):
        """HTTP endpoint: accepts pickled obs dict, returns pickled response."""
        body = await request.body()
        obs_dict = pickle.loads(body)
        result = await self.predict(obs_dict)
        return Response(content=pickle.dumps(result),
                        media_type="application/octet-stream")

    @_gr00t_app.get("/stats")
    async def stats_http(self):
        return await self.get_stats()

    async def predict(self, obs_dict: dict) -> dict:
        """Run one inference step.

        obs_dict MUST be the nested-dict format:
            {
              "video":    {"ego_view":  [B, T_video, H, W, 3] uint8},
              "state":    {"left_arm":  [B, T_state, D] float32, ...},
              "language": {"annotation.human.task_description": [[str]]},
            }

        Returns:
            {"action": dict of 9 np.ndarrays (shape [B, 40, D_k]),
             "latency_ms": float}
        """
        t0 = time.time()
        with torch.no_grad():
            action_chunk, info = self.policy.get_action(obs_dict)

        latency_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_latency += latency_ms

        # Convert all tensors to numpy for clean Ray serialization
        serializable = {}
        for k, v in action_chunk.items():
            if torch.is_tensor(v):
                serializable[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                serializable[k] = v
            else:
                serializable[k] = np.asarray(v)

        return {"action": serializable, "latency_ms": latency_ms}

    async def get_modality_config(self) -> dict:
        """Sim workers can call this to learn what obs/action schema to use."""
        return self.modality_configs_serializable

    async def get_stats(self) -> dict:
        return {
            "model_path": self.model_path,
            "embodiment_tag": self.embodiment_tag_name,
            "num_params_B": self.num_params / 1e9,
            "device": self.device,
            "total_calls": self._call_count,
            "avg_latency_ms": self._total_latency / max(self._call_count, 1),
            "load_time_s": self.load_time,
        }


@serve.deployment(ray_actor_options={"num_gpus": 0})
@serve.ingress(_placeholder_app)
class PlaceholderPolicyServer:
    """Random G1-shaped actions (matching REAL_G1 schema). Lets you test the
    architecture plumbing without loading 3B params."""

    def __init__(self):
        print("[PlaceholderPolicy] Ready (random REAL_G1 actions, no GPU)")
        self._call_count = 0
        self._total_latency = 0.0

    @_placeholder_app.post("/predict")
    async def predict_http(self, request: Request):
        body = await request.body()
        obs_dict = pickle.loads(body)
        result = await self.predict(obs_dict)
        return Response(content=pickle.dumps(result),
                        media_type="application/octet-stream")

    @_placeholder_app.get("/stats")
    async def stats_http(self):
        return await self.get_stats()

    async def predict(self, obs_dict: dict) -> dict:
        t0 = time.time()
        B = 1
        T = 40  # GR00T action horizon
        action_chunk = {
            "left_wrist_eef_9d":  np.random.uniform(-0.01, 0.01, (B, T, 9)).astype(np.float32),
            "right_wrist_eef_9d": np.random.uniform(-0.01, 0.01, (B, T, 9)).astype(np.float32),
            "left_hand":          np.random.uniform(-0.05, 0.05, (B, T, 7)).astype(np.float32),
            "right_hand":         np.random.uniform(-0.05, 0.05, (B, T, 7)).astype(np.float32),
            "left_arm":           np.random.uniform(-0.05, 0.05, (B, T, 7)).astype(np.float32),
            "right_arm":          np.random.uniform(-0.05, 0.05, (B, T, 7)).astype(np.float32),
            "waist":              np.random.uniform(-0.05, 0.05, (B, T, 3)).astype(np.float32),
            "base_height_command":np.full((B, T, 1), 0.72, dtype=np.float32),
            "navigate_command":   np.zeros((B, T, 3), dtype=np.float32),
        }
        latency_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_latency += latency_ms
        return {"action": action_chunk, "latency_ms": latency_ms}

    async def get_modality_config(self) -> dict:
        return {
            "video": {"modality_keys": ["ego_view"], "delta_indices": [-20, 0]},
            "state": {
                "modality_keys": [
                    "left_wrist_eef_9d", "right_wrist_eef_9d",
                    "left_hand", "right_hand",
                    "left_arm", "right_arm", "waist",
                ],
                "delta_indices": [0],
            },
            "language": {
                "modality_keys": ["annotation.human.task_description"],
                "delta_indices": [0],
            },
        }

    async def get_stats(self) -> dict:
        return {
            "model_path": "placeholder-random",
            "embodiment_tag": "REAL_G1",
            "num_params_B": 0.0,
            "device": "cpu",
            "total_calls": self._call_count,
            "avg_latency_ms": self._total_latency / max(self._call_count, 1),
        }
