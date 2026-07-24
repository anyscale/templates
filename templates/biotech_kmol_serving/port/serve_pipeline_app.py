"""Two-stage Ray Serve app for the ported kMoL ensemble: CPU featurizer tier -> GPU forward tier.

This is `scaled_pipeline.py`'s topology expressed as Ray Serve deployments instead of
raw Ray actors. It exists because the one-deployment design in `serve_app.py` couples
the two stages in a way that caps throughput:

  - In `serve_app.py` each replica featurizes AND forwards, and asks for a GPU slice
    (`num_gpus=0.16`). Needing a GPU slice pins every replica to a GPU node, so the
    featurizer count is capped by that node's vCPU (8 on g6.2xlarge) — even though
    featurization is the bottleneck and needs no GPU at all.
  - Here `Featurizer` requests no GPU, so its replicas schedule onto cheap CPU nodes
    and autoscale independently of the GPU tier. `GpuForward` keeps whole GPUs and
    stays shared. You buy throughput in vCPU, not in L4s.

Data flow: `Ingress` (SMILES in, floats out) -> `Featurizer` (SMILES -> PyG Batch)
-> `GpuForward` (Batch -> logits). The Featurizer calls the GPU tier itself rather than
the Ingress brokering the Batch between them. That was measured, not assumed: routing
the response through the Ingress makes Serve *materialize* the Batch there
(`serialization.py -> pickle.loads -> torch_geometric`), which would force torch onto
the front door and put a deserialize + reserialize of every batch on the one component
all traffic passes through. Here the Batch makes exactly one hop, CPU node -> GPU node,
and the Ingress never imports torch.

  POST /  {"smiles": "CCO"}                 -> {"logits": [...12], "variance": [...12]}
  POST /  {"smiles": ["CCO", "c1ccccc1"]}   -> [ {...}, {...} ]   (bulk; recommended)

Env knobs: KMOL_CHUNK, KMOL_FEAT_MIN/MAX, KMOL_GPU_MIN/MAX, KMOL_NUM_GPUS,
KMOL_CONFIG, KMOL_CKPT_DIR.

NOT YET BENCHMARKED — the numbers in README.md / TAKEDA_BRIEF.md are from
`serve_app.py` (monolithic) and `scaled_pipeline.py` (raw actors). Re-measure with
`scripts/serve_bulk.py` pointed at this app before quoting anything.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request

CHUNK = int(os.environ.get("KMOL_CHUNK", "256"))


@serve.deployment(
    ray_actor_options={"num_cpus": 1},  # no GPU => schedulable on cheap CPU nodes
    autoscaling_config={
        "min_replicas": int(os.environ.get("KMOL_FEAT_MIN", "6")),
        "max_replicas": int(os.environ.get("KMOL_FEAT_MAX", "64")),
        "target_ongoing_requests": 2,  # keep one chunk queued so the core never idles
        "upscale_delay_s": 10,
        "downscale_delay_s": 120,
    },
    # One chunk featurizing while another awaits the GPU. Deeper just adds GIL
    # contention on a single core without adding throughput.
    max_ongoing_requests=2,
)
class Featurizer:
    """SMILES -> PyG Batch -> GPU tier. The bottleneck stage: RDKit, ~1 core per replica.

    `featurize` runs in a worker thread so the replica's event loop stays free to accept
    the next chunk and to await the GPU response — that overlap is what the monolithic
    deployment can't do, since there featurization and the forward share one thread.
    """

    def __init__(self, gpu: DeploymentHandle):
        import torch

        import kmolport
        from kmolport.featurizer import collate

        torch.set_num_threads(1)
        self._feat = kmolport.GraphFeaturizer()
        self._collate = collate
        self._gpu = gpu
        self._featurize(["CCO"])  # warm RDKit before the replica reports healthy

    def _featurize(self, smiles_list: List[str]):
        return self._collate([self._feat.featurize(s) for s in smiles_list])

    async def process(self, smiles_list: List[str]) -> Dict[str, List[List[float]]]:
        batch = await asyncio.to_thread(self._featurize, smiles_list)
        return await self._gpu.forward.remote(batch)


@serve.deployment(
    ray_actor_options={"num_gpus": float(os.environ.get("KMOL_NUM_GPUS", "1")), "num_cpus": 1},
    autoscaling_config={
        "min_replicas": int(os.environ.get("KMOL_GPU_MIN", "1")),
        "max_replicas": int(os.environ.get("KMOL_GPU_MAX", "4")),
        "target_ongoing_requests": 8,
        "upscale_delay_s": 15,
        "downscale_delay_s": 300,
    },
    max_ongoing_requests=32,
)
class GpuForward:
    """Pre-featurized PyG Batch -> ensemble logits + per-target variance.

    No `serve.batch` here: the forward is ~15 ms flat from batch 1 to 1024, so at
    CHUNK=256 one replica already carries ~17k mol/s — far more than the featurizer
    tier can feed it. Coalescing chunks would mean re-collating Batches on the GPU
    node, i.e. spending CPU on the scarce resource to save nothing.
    """

    def __init__(self, config_path: str, ckpt_dir: str):
        import torch

        import kmolport

        torch.set_num_threads(1)
        self._torch = torch
        cfg = kmolport.load_config(config_path)
        # strict=True: a checkpoint that doesn't match the architecture fails startup
        # rather than silently serving a half-random model.
        self.model = kmolport.build_ensemble(
            cfg, checkpoint_dir=ckpt_dir, device="cuda", strict=True
        )
        self._warm_up()

    def _warm_up(self):
        """One real forward before healthy. Builds its own batch so startup doesn't
        depend on the Featurizer tier being up yet."""
        import kmolport
        from kmolport.featurizer import collate

        self._forward_sync(collate([kmolport.GraphFeaturizer().featurize("CCO")]))

    def _forward_sync(self, batch) -> Dict[str, List[List[float]]]:
        torch = self._torch
        batch = batch.to("cuda")
        with torch.no_grad():
            out = self.model({"graph": batch})
        return {
            "logits": out["logits"].cpu().numpy().tolist(),
            "variance": out["logits_var"].cpu().numpy().tolist(),
        }

    async def forward(self, batch) -> Dict[str, List[List[float]]]:
        # In a thread, not on the event loop: a blocking forward would stall the
        # replica from receiving the batches queued behind it.
        return await asyncio.to_thread(self._forward_sync, batch)


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": int(os.environ.get("KMOL_INGRESS_MAX", "4")),
        "target_ongoing_requests": 16,
    },
    max_ongoing_requests=64,
)
class Ingress:
    """HTTP front door + chunk fan-out. Deliberately thin: it never touches torch or
    a featurized Batch, so it doesn't become the next bottleneck."""

    def __init__(self, featurizer: DeploymentHandle, config_path: str):
        self._featurizer = featurizer
        cfg = json.loads(Path(config_path).read_text())
        self._labels = list(cfg.get("loader", {}).get("target_column_names", []))

    async def _predict(self, smiles: List[str]) -> List[Dict[str, Any]]:
        # Dispatch every chunk first so they're all in flight across the featurizer
        # tier, then collect in order. Total latency is the slowest chunk, not the sum.
        pending = [
            self._featurizer.process.remote(smiles[i:i + CHUNK])
            for i in range(0, len(smiles), CHUNK)
        ]
        out: List[Dict[str, Any]] = []
        for response in pending:
            result = await response
            for logits, variance in zip(result["logits"], result["variance"]):
                item: Dict[str, Any] = {"logits": logits, "variance": variance}
                if self._labels:
                    item["labels"] = self._labels
                out.append(item)
        return out

    async def predict(self, smiles):
        """Deployment-handle entrypoint. Accepts a single SMILES or a list."""
        if isinstance(smiles, str):
            return (await self._predict([smiles]))[0]
        return await self._predict(list(smiles))

    async def __call__(self, request: Request) -> Any:
        body = await request.json()
        smiles = body.get("smiles")
        if smiles is None:
            return {"error": "request body must contain a 'smiles' field"}
        return await self.predict(smiles)


def build_app(config_path: str = None, ckpt_dir: str = None) -> serve.Application:
    config_path = config_path or os.environ.get("KMOL_CONFIG", "configs/ensemble_serve.example.json")
    ckpt_dir = ckpt_dir or os.environ.get("KMOL_CKPT_DIR", "checkpoints")
    return Ingress.bind(Featurizer.bind(GpuForward.bind(config_path, ckpt_dir)), config_path)


app = build_app()
