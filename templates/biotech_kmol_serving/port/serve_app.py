"""Production Ray Serve app for the ported kMoL ensemble (kmolport).

The clean, reusable deployment (as opposed to the P1 benchmark scripts). Serves the
5-model GNN ensemble over HTTP with dynamic batching on fractional GPUs. Same design
as the proven benchmark; this is what a Service would run.

  POST /  {"smiles": "CCO"}                 -> {"logits": [...12], "variance": [...12]}
  POST /  {"smiles": ["CCO", "c1ccccc1"]}   -> [ {...}, {...} ]   (bulk; recommended)

Env knobs: KMOL_NUM_GPUS (0.16 => ~6 replicas/L4), KMOL_MIN_REPLICAS,
KMOL_MAX_REPLICAS, KMOL_MAX_BATCH, KMOL_WAIT_S, KMOL_CONFIG, KMOL_CKPT_DIR.
"""
import asyncio
import os
from typing import Any, Dict, List

from ray import serve
from starlette.requests import Request


@serve.deployment(
    ray_actor_options={"num_gpus": float(os.environ.get("KMOL_NUM_GPUS", "0.16")), "num_cpus": 1},
    autoscaling_config={
        "min_replicas": int(os.environ.get("KMOL_MIN_REPLICAS", "6")),
        "max_replicas": int(os.environ.get("KMOL_MAX_REPLICAS", "6")),
        "target_ongoing_requests": int(os.environ.get("KMOL_TARGET_ONGOING", "8")),
        "upscale_delay_s": 15,
        "downscale_delay_s": 120,
    },
    max_ongoing_requests=int(os.environ.get("KMOL_MAX_ONGOING", "64")),
)
class KmolEnsemble:
    def __init__(self, config_path: str, ckpt_dir: str):
        import torch

        import kmolport
        from kmolport.featurizer import collate

        torch.set_num_threads(1)  # [REC 5] each replica featurizes on its own core
        self._torch = torch
        self._collate = collate
        cfg = kmolport.load_config(config_path)
        # [REC 1] load all 5 checkpoints once, here — not per request. strict=True so a
        # checkpoint that doesn't match the architecture fails startup (never serve a
        # half-random model).
        self.model = kmolport.build_ensemble(cfg, checkpoint_dir=ckpt_dir, device="cuda", strict=True)
        self.feat = kmolport.GraphFeaturizer()
        self.labels = list(cfg.get("loader", {}).get("target_column_names", []))
        self._infer.set_max_batch_size(int(os.environ.get("KMOL_MAX_BATCH", "128")))
        self._infer.set_batch_wait_timeout_s(float(os.environ.get("KMOL_WAIT_S", "0.01")))
        self._infer_sync(["CCO"])  # [REC 6] warm-up forward before healthy

    def _infer_sync(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        torch = self._torch
        data = [self.feat.featurize(s) for s in smiles_list]  # [REC 2] one PyG Batch
        batch = self._collate(data).to("cuda")
        with torch.no_grad():
            out = self.model({"graph": batch})  # [REC 7] 5 sub-models + native mean, one forward
        logits = out["logits"].detach().cpu().numpy()
        var = out["logits_var"].detach().cpu().numpy()
        resp = []
        for i in range(len(smiles_list)):
            r: Dict[str, Any] = {"logits": logits[i].tolist(), "variance": var[i].tolist()}
            if self.labels:
                r["labels"] = self.labels
            resp.append(r)
        return resp

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.01)
    async def _infer(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._infer_sync, smiles_list)

    async def infer(self, smiles):
        """Deployment-handle entrypoint. Accepts a single SMILES or a list."""
        if isinstance(smiles, list):
            return await asyncio.to_thread(self._infer_sync, smiles)
        return await self._infer(smiles)

    async def __call__(self, request: Request) -> Any:
        body = await request.json()
        smiles = body.get("smiles")
        if smiles is None:
            return {"error": "request body must contain a 'smiles' field"}
        if isinstance(smiles, list):
            # bulk: featurize the whole list in one shot (best throughput — see P1b)
            return await asyncio.to_thread(self._infer_sync, smiles)
        return await self._infer(smiles)


def build_app(config_path: str = None, ckpt_dir: str = None) -> serve.Application:
    config_path = config_path or os.environ.get("KMOL_CONFIG", "configs/ensemble_serve.example.json")
    ckpt_dir = ckpt_dir or os.environ.get("KMOL_CKPT_DIR", "checkpoints")
    return KmolEnsemble.bind(config_path, ckpt_dir)


app = build_app()
