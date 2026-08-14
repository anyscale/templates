"""
Ray Data callable class wrapping Boltz for GPU structure prediction.

One actor per GPU. Boltz ships a CLI, not a Python inference API, so each actor
writes its batch out as YAML and shells out to `boltz predict` once for the whole
batch. That matters: measured on an L4, a call costs ~31s of fixed startup (model
load) plus ~11s per complex, so a batch of 32 pays the startup once instead of 32
times. Keep `batch_size` high for the same reason.

Weights are ~5.5GB and are cached on shared cluster storage so a node downloads
them at most once. `ensure_weights()` runs on the driver before the pipeline
starts -- see the note there about why "the file exists" is not good enough.

Metrics come straight from Boltz's confidence JSON:
  - pLDDT: per-residue confidence in the fold, averaged. Boltz reports 0-1;
    scaled to 0-100 here, the convention the rest of the pipeline and the
    literature use. >70 generally indicates a reliable fold.
  - ipTM: confidence in the predicted interface between chains. 0-1, higher
    better; >0.8 suggests high confidence in the interaction geometry.
  - confidence: Boltz's aggregate score, 0-1. The primary ranking metric.
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

BOLTZ_VERSION = "2.2.1"
SCORER_NAME = f"boltz-{BOLTZ_VERSION}"
DEFAULT_CACHE = "/mnt/cluster_storage/boltz_cache"
_SENTINEL = ".boltz_cache_complete"


def ensure_weights(cache_dir: str = DEFAULT_CACHE) -> None:
    """Download Boltz's weights once, on the driver, before any actor starts.

    A shared cache turns a killed job into a landmine: Boltz only checks whether
    a checkpoint file exists, so a run interrupted mid-download leaves a truncated
    .ckpt that every later run happily loads and dies on
    (`PytorchStreamReader failed reading zip archive`). The sentinel is written
    only after a prediction has actually succeeded against these weights, so a
    partial download is never mistaken for a complete one.
    """
    cache = Path(cache_dir)
    if (cache / _SENTINEL).exists():
        print(f"  Boltz weights ready in {cache}")
        return

    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob("*.ckpt"):
        print(f"  Discarding possibly-truncated checkpoint: {stale.name}")
        stale.unlink()

    print(f"  Fetching Boltz weights into {cache} (~5.5GB, once per cluster)...")
    with tempfile.TemporaryDirectory() as tmp:
        warm = Path(tmp) / "in"
        warm.mkdir()
        (warm / "warmup.yaml").write_text(yaml.safe_dump({
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKQLEDKVEELLSKNYHLENEVARLKKLV", "msa": "empty"}},
            ],
        }, sort_keys=False))
        _run_boltz(warm, Path(tmp) / "out", cache)

    (cache / _SENTINEL).write_text(f"{SCORER_NAME}\n")
    print("  Boltz weights ready")


def _run_boltz(in_dir: Path, out_dir: Path, cache: Path) -> Path:
    """Run one `boltz predict` over a directory of YAMLs; return the predictions dir."""
    subprocess.run(
        [
            "boltz", "predict", str(in_dir),
            "--out_dir", str(out_dir),
            "--cache", str(cache),
            "--accelerator", "gpu",
            "--devices", "1",
            "--output_format", "mmcif",
            "--diffusion_samples", "1",
            # cuequivariance kernels ship in the `boltz[cuda]` extra and measured
            # no faster on an L4; skipping them keeps the dep list smaller and the
            # template runnable on older GPUs.
            "--no_kernels",
            "--override",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_dir / f"boltz_results_{in_dir.name}" / "predictions"


class BoltzPredictor:
    """Ray Data callable class. One actor per GPU."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE):
        self.cache = Path(cache_dir)
        os.environ.setdefault("BOLTZ_CACHE", str(self.cache))

    def __call__(self, batch: dict) -> dict:
        n = len(batch["complex_id"])
        outputs = {
            "complex_id": [], "plddt_mean": [], "iptm": [], "confidence": [],
            "num_residues": [], "cif_bytes": [], "runtime_sec": [], "scorer": [],
        }

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            in_dir, out_dir = Path(tmp) / "in", Path(tmp) / "out"
            in_dir.mkdir()

            ids, residues = [], {}
            for i in range(n):
                complex_id = _as_str(batch["complex_id"][i])
                if not batch["is_valid"][i]:
                    print(f"  WARNING: skipping {complex_id}: upstream marked it invalid")
                    continue
                spec = json.loads(_as_str(batch["boltz_input"][i]))
                (in_dir / f"{complex_id}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
                ids.append(complex_id)
                residues[complex_id] = int(batch["target_len"][i]) + int(batch["binder_len"][i])

            if not ids:
                return outputs
            preds = _run_boltz(in_dir, out_dir, self.cache)

            # Read inside the TemporaryDirectory — everything below lives under it.
            # Boltz drops a complex it cannot process rather than failing the batch,
            # so score the missing ones zero; postprocess filters them out.
            per_complex = (time.time() - t0) / len(ids)
            for complex_id in ids:
                d = preds / complex_id
                conf_file = d / f"confidence_{complex_id}_model_0.json"
                cif_file = d / f"{complex_id}_model_0.cif"
                if conf_file.exists():
                    c = json.loads(conf_file.read_text())
                    plddt = float(c["complex_plddt"]) * 100.0  # Boltz reports 0-1
                    iptm = float(c["iptm"])
                    confidence = float(c["confidence_score"])
                    cif = cif_file.read_bytes() if cif_file.exists() else b""
                else:
                    print(f"  WARNING: boltz produced no confidence for {complex_id}")
                    plddt, iptm, confidence, cif = 0.0, 0.0, 0.0, b""

                outputs["complex_id"].append(complex_id)
                outputs["plddt_mean"].append(plddt)
                outputs["iptm"].append(iptm)
                outputs["confidence"].append(confidence)
                outputs["num_residues"].append(residues[complex_id])
                outputs["cif_bytes"].append(cif)
                outputs["runtime_sec"].append(per_complex)
                outputs["scorer"].append(SCORER_NAME)

        return outputs


def _as_str(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)
