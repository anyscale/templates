"""
Ray Data callable class standing in for a GPU structure-prediction model.

**The scores here are simulated, not predicted.** This template demonstrates the
Ray Data pipeline — CPU feature prep streaming into GPU actors streaming into CPU
post-processing, with no intermediate materialization — and uses a synthetic scorer
so the pipeline runs anywhere, in minutes, with no checkpoint download. Every row it
emits is tagged `scorer="simulated"` so the provenance travels with the data.

To screen for real, replace this one class with a wrapper around your model and keep
everything else. The contract Ray Data depends on:

  __init__(self, weights_path: str)   loads the model once per actor (see pipeline.py:
                                      `fn_constructor_kwargs`, `concurrency=num_gpus`)
  __call__(self, batch: dict) -> dict consumes complex_id, complex_type, boltz_input,
                                      target_len, binder_len, is_valid
                                      emits   complex_id, plddt_mean, iptm, confidence,
                                              num_residues, cif_bytes, runtime_sec,
                                              scorer

Metrics a real predictor would fill in, and which the tiers in postprocess.py assume:
  - pLDDT: per-residue confidence in the fold, averaged. 0-100, higher better;
    >70 generally indicates a reliable fold.
  - ipTM: confidence in the predicted interface between chains. 0-1, higher better;
    >0.8 suggests high confidence in the interaction geometry.
  - confidence: aggregate of the two, 0-1. The primary ranking metric for screening.
"""
import hashlib
import json
import time

import numpy as np

SCORER_NAME = "simulated"


class SimulatedStructureScorer:
    """Ray Data callable class. One actor per GPU.

    Scheduled onto a GPU (pipeline.py passes `num_gpus=1`) because the real
    predictor it stands in for would be — that heterogeneous CPU/GPU split, and the
    autoscaling it drives, is what this template is demonstrating.
    """

    def __init__(self, weights_path: str = ""):
        """Accepts `weights_path` so the constructor signature matches a real
        predictor's; nothing is loaded."""
        self.weights_path = weights_path
        print(f"  {type(self).__name__}: emitting simulated scores, no model loaded")

    @staticmethod
    def _scores(complex_id: str):
        """Draw this complex's scores.

        Seeded from `complex_id` rather than a fixed seed so a given complex scores
        the same no matter which actor gets it or how the data is sharded — a fixed
        per-actor seed makes results depend on arrival order, which is not
        reproducible.

        The distribution mimics a real binder campaign: most random binders score
        low, a few score well.
        """
        digest = hashlib.sha256(complex_id.encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))

        plddt_mean = float(np.clip(rng.beta(2.5, 5.0) * 100, 20, 95))
        iptm = float(np.clip(rng.beta(2.0, 5.0), 0.1, 0.95))
        confidence = float(np.clip(
            0.4 * (plddt_mean / 100) + 0.6 * iptm + rng.normal(0, 0.05), 0.0, 1.0,
        ))
        return plddt_mean, iptm, confidence

    def __call__(self, batch: dict) -> dict:
        n = len(batch["complex_id"])
        outputs = {
            "complex_id": [],
            "plddt_mean": [],
            "iptm": [],
            "confidence": [],
            "num_residues": [],
            "cif_bytes": [],
            "runtime_sec": [],
            "scorer": [],
        }

        for i in range(n):
            complex_id = batch["complex_id"][i]
            if isinstance(complex_id, bytes):
                complex_id = complex_id.decode("utf-8")

            boltz_input_json = batch["boltz_input"][i]
            if isinstance(boltz_input_json, bytes):
                boltz_input_json = boltz_input_json.decode("utf-8")

            target_len = int(batch["target_len"][i])
            binder_len = int(batch["binder_len"][i])
            num_residues = target_len + binder_len

            t0 = time.time()

            # Malformed rows score zero and are dropped by `passed_filter` in
            # postprocess.py, so one bad candidate can't fail the screen. Scoped to
            # the two things that legitimately vary with input data — anything else
            # is a bug and should surface.
            try:
                if not batch["is_valid"][i]:
                    raise ValueError("upstream marked this complex invalid")
                json.loads(boltz_input_json)
                plddt_mean, iptm, confidence = self._scores(complex_id)
                cif_bytes = (
                    f"# placeholder, not a structure ({SCORER_NAME} scorer)\n"
                    f"# {complex_id}: pLDDT={plddt_mean:.1f} ipTM={iptm:.3f}\n"
                ).encode("utf-8")
            except (ValueError, TypeError) as e:
                print(f"  WARNING: skipping {complex_id}: {e}")
                plddt_mean, iptm, confidence, cif_bytes = 0.0, 0.0, 0.0, b""

            outputs["complex_id"].append(complex_id)
            outputs["plddt_mean"].append(plddt_mean)
            outputs["iptm"].append(iptm)
            outputs["confidence"].append(confidence)
            outputs["num_residues"].append(num_residues)
            outputs["cif_bytes"].append(cif_bytes)
            outputs["runtime_sec"].append(time.time() - t0)
            outputs["scorer"].append(SCORER_NAME)

        return outputs
