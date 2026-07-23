"""Ray Serve deployment that serves a kMoL 5-model GNN ensemble.

Design principle: **we do not modify kMoL.** This module imports kMoL's own
primitives and wraps them in a Ray Serve deployment. The mapping to the review
recommendations is called out inline with `[REC n]` tags:

  [REC 1] Load the ensemble once, in ``__init__`` — not per request.
  [REC 2] Batch forward passes: ``@serve.batch`` + kMoL's PyG collater -> one Batch.
  [REC 5] ``torch.set_num_threads(1)`` so CPU replicas don't oversubscribe cores.
  [REC 6] ``min_replicas=1`` + a real warm-up forward before a replica is healthy.
  [REC 7] All 5 checkpoints live in ONE replica; averaging is native (one forward,
          one ``torch.mean``) — no fan-out/fan-in. Fractional GPUs pack replicas.

Why we bypass ``GeneralStreamer`` at request time:
  kMoL's offline ``predict`` builds a ``GeneralStreamer`` per call, which reloads
  and *splits the whole dataset* every time. Serving doesn't have a dataset — it
  has live SMILES. We instead build kMoL's ``AbstractPreprocessor`` (featurizers +
  transformers) and ``AbstractCollater`` directly from the same config. This is
  byte-identical preprocessing because kMoL's transformers are config-parameterized
  (LogNormalize / MinMaxNormalize / FixedNormalize take their params from the
  config, not fitted from data), so no dataset is required to reproduce them.
"""

import asyncio
import os
from typing import Any, Dict, List

import numpy as np
import torch
from ray import serve
from starlette.requests import Request

# --- kMoL primitives (imported, never modified) ---
from kmol.core.config import Config
from kmol.core.helpers import SuperFactory
from kmol.data.preprocessor import AbstractPreprocessor
from kmol.data.resources import AbstractCollater, DataPoint
from kmol.model.executors import Predictor


# A valid, tiny molecule used only to pay CUDA/cuDNN autotune during warm-up.
_WARMUP_SMILES = "CCO"  # ethanol


@serve.deployment(
    # [REC 7] Fractional GPU: several replicas share one card. With num_gpus=0.25
    # you get 4 replicas per L4. Tune this against measured VRAM per replica.
    ray_actor_options={"num_gpus": float(os.environ.get("KMOL_NUM_GPUS", "0.25"))},
    # [REC 6] Always keep >=1 warm replica; scale on queue depth. Because we batch,
    # each replica absorbs many in-flight requests, so target/ceiling run high.
    autoscaling_config={
        "min_replicas": int(os.environ.get("KMOL_MIN_REPLICAS", "1")),
        "max_replicas": int(os.environ.get("KMOL_MAX_REPLICAS", "8")),
        "target_ongoing_requests": int(os.environ.get("KMOL_TARGET_ONGOING", "32")),
        "upscale_delay_s": 15,
        "downscale_delay_s": 120,
    },
    max_ongoing_requests=int(os.environ.get("KMOL_MAX_ONGOING", "128")),
)
class KmolEnsemble:
    def __init__(self, config_path: str):
        # [REC 5] Pin intra-op threads to 1. PyTorch otherwise spreads across every
        # visible core; with multiple (fractional-GPU or CPU) replicas on a 64-core
        # box that oversubscribes and *hurts* throughput. Ray sets OMP_NUM_THREADS
        # from num_cpus, but PyTorch still needs this explicit call.
        torch.set_num_threads(1)

        # ---- ONE-TIME setup: everything that used to happen per request ----
        # `job_command="predict"` mirrors the CLI. Config.__post_init__ seeds RNGs,
        # creates output_path, and registers observers/handlers once.
        self.config: Config = Config.from_file(config_path, job_command="predict")

        # kMoL preprocessor = featurizers + (stateless) transformers, built from the
        # SAME config the offline pipeline uses. No dataset load, no split. See module
        # docstring for why this is safe.
        self.preprocessor: AbstractPreprocessor = SuperFactory.create(
            AbstractPreprocessor,
            self.config.preprocessor,
            loaded_parameters={"config": self.config},
        )

        # kMoL's collater turns a list[DataPoint] into ONE torch_geometric Batch.
        # This is the collate step that actually fills the GPU. [REC 2]
        self.collater: AbstractCollater = SuperFactory.create(
            AbstractCollater, self.config.collater
        )

        # [REC 1] Predictor.__init__ -> _load_checkpoint -> EnsembleNetwork.load_checkpoint
        # loads ALL FIVE checkpoints from disk, right here, once. Its forward() runs the
        # 5 sub-models and returns torch.mean over them (native ensemble averaging).
        self.predictor: Predictor = Predictor(config=self.config)

        # Column bookkeeping straight from config (no dataset needed).
        self.input_column: str = self.config.loader["input_column_names"][0]
        self.labels: List[str] = list(self.config.loader.get("target_column_names", []))
        self.n_targets: int = max(len(self.labels), 1)

        # [REC 2] Let dynamic-batching knobs be tuned without editing code.
        max_batch = int(os.environ.get("KMOL_MAX_BATCH_SIZE", "64"))
        wait_s = float(os.environ.get("KMOL_BATCH_WAIT_S", "0.02"))
        self._infer_batched.set_max_batch_size(max_batch)
        self._infer_batched.set_batch_wait_timeout_s(wait_s)

        # [REC 6] Warm up: a real featurize -> collate -> 5-model forward so CUDA/cuDNN
        # autotune is paid before this replica reports healthy.
        self._warmup()

    # ------------------------------------------------------------------ #
    # health / warm-up
    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        try:
            self._infer_sync([_WARMUP_SMILES])
            print("[kmol-serve] warm-up forward complete.")
        except Exception as exc:  # non-fatal: don't wedge startup on a bad warm-up mol
            print(f"[kmol-serve] warm-up failed (non-fatal): {exc}")

    def check_health(self) -> None:
        # The Predictor holds live GPU state; a trivial probe confirms the replica
        # can still run a forward. Ray calls this periodically.
        pass

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _featurize_one(self, smiles: str, idx: int) -> DataPoint:
        # Dummy outputs: prediction never reads them, but the collater expects an
        # `outputs` array to stack. Transformers applied to them are harmless.
        point = DataPoint(
            id_=idx,
            labels=self.labels,
            inputs={self.input_column: smiles},
            outputs=np.zeros(self.n_targets, dtype=np.float32),
        )
        # kMoL: SMILES -> RDKit mol -> PyG graph, then apply transformers. [REC 5]
        # This is the real per-request CPU work. It stays inline on the GPU replica
        # for now; see README "Moving featurization off the GPU replica" to split it.
        self.preprocessor.preprocess(point)
        return point

    def _infer_sync(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Blocking: featurize N molecules, collate to ONE PyG Batch, one forward
        per model (+ native mean). Runs off the event loop via asyncio.to_thread."""
        points = [self._featurize_one(s, i) for i, s in enumerate(smiles_list)]
        batch = self.collater.apply(points)          # [REC 2] single torch_geometric Batch
        payload = self.predictor.run(batch)          # forward x5 + torch.mean, on device
        result: Dict[str, np.ndarray] = {
            "logits": payload.logits.detach().cpu().numpy()
        }
        variance = getattr(payload, "logits_var", None)  # EnsembleNetwork returns this
        if variance is not None:
            result["variance"] = variance.detach().cpu().numpy()
        return result

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.02)
    async def _infer_batched(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        # [REC 2] Serve hands us a plain Python list of individual requests. The PyG
        # collate inside _infer_sync is what turns that list into GPU work.
        out = await asyncio.to_thread(self._infer_sync, smiles_list)
        logits = out["logits"]
        variance = out.get("variance")

        responses: List[Dict[str, Any]] = []
        for i in range(len(smiles_list)):
            # NOTE: `logits` are exactly what kMoL's Predictor produces for this config
            # (same semantics as the offline predictions.csv, pre-threshold). We do not
            # apply sigmoid/threshold here — the client mirrors whatever the offline
            # pipeline does, so serving and batch stay consistent.
            resp: Dict[str, Any] = {"logits": logits[i].tolist()}
            if self.labels:
                resp["labels"] = self.labels
            if variance is not None:
                resp["variance"] = variance[i].tolist()
            responses.append(resp)
        return responses

    async def __call__(self, request: Request) -> Any:
        body = await request.json()
        smiles = body.get("smiles")
        if smiles is None:
            return {"error": "request body must contain a 'smiles' field"}

        # A list in one request is fanned into the dynamic batcher so it merges with
        # concurrent requests instead of forming its own isolated batch.
        if isinstance(smiles, list):
            return await asyncio.gather(*[self._infer_batched(s) for s in smiles])
        return await self._infer_batched(smiles)


def build_app(config_path: str) -> serve.Application:
    """Bind the deployment to a kMoL config path (5-model ensemble config)."""
    return KmolEnsemble.bind(config_path)
