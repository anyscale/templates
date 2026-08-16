"""
Ray Data callable class wrapping Boltz for GPU structure prediction.

One actor per GPU. Boltz ships a CLI, not a Python inference API, so each actor
writes its batch out as YAML and shells out to `boltz predict` once for the whole
batch. That matters: measured on an L4, a call costs ~31s of fixed startup (model
load) plus ~11s per complex, so a batch of 32 pays the startup once instead of 32
times. Keep `batch_size` high for the same reason.

Weights are ~6GB and are cached on shared cluster storage so a cluster downloads
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
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

BOLTZ_VERSION = "2.2.1"
SCORER_NAME = f"boltz-{BOLTZ_VERSION}"
DEFAULT_CACHE = "/mnt/cluster_storage/boltz_cache"
_SENTINEL = ".boltz_cache_complete"
# What a complete cache holds. `mols.tar` is deliberately absent: it is the
# archive `mols/` is extracted from, and keeping it costs 1.9GB for nothing.
#
# `boltz2_aff.ckpt` is absent too, and not fetched at all. Boltz only loads the
# affinity model when the input YAML carries a `properties: affinity` block, and
# this template's inputs never do -- it screens for fold and interface confidence
# (pLDDT, ipTM), which come from the confidence model. Fetching it anyway cost
# 2.06GB of the 6.2GB, a third of the download, for a checkpoint nothing opened.
_ARTIFACTS = ("boltz2_conf.ckpt", "mols")

# Boltz's own downloader fetches these serially, one urlretrieve after another.
# Overlapping them is worth a little; the real saving is the third of the bytes
# that no longer move at all.
_HF = "https://huggingface.co/boltz-community/boltz-2/resolve/main"
_DOWNLOADS = ((f"{_HF}/boltz2_conf.ckpt", "boltz2_conf.ckpt"), (f"{_HF}/mols.tar", "mols.tar"))


def _fetch(url: str, dest: Path) -> None:
    """Download one artifact, into `.part` first.

    The rename lands only on success, so an interrupted fetch cannot leave
    something at the real name for a later run to trust.
    """
    part = Path(f"{dest}.part")
    urllib.request.urlretrieve(url, str(part))  # noqa: S310
    part.replace(dest)


def ensure_weights(cache_dir: str = DEFAULT_CACHE) -> None:
    """Populate the shared Boltz cache once, on the driver, before any actor starts.

    Downloads only -- no warm-up prediction. The driver runs on the head node,
    which this template's compute config leaves GPU-less (`CPU: 0`, m5.2xlarge),
    so there is nothing here for `boltz predict` to run on.

    Fetches the artifacts directly rather than calling `boltz.main.download_boltz2`,
    which always pulls the affinity checkpoint this template never opens and fetches
    the files one after another. Boltz still finds what it needs -- it locates
    weights by path, and its own downloader is a no-op once they are there.

    Populating the cache measured 1216s for 6.17GB, and the prints below break that
    into transfer and extraction so the next run says which half to attack.

    A shared cache turns a killed job into a landmine: whoever fetches decides by
    checking whether each file exists, so a run interrupted mid-download leaves a
    truncated .ckpt that every later run happily loads and dies on
    (`PytorchStreamReader failed reading zip archive`). The sentinel goes in last,
    after every artifact is present, so a partial cache is discarded and refetched
    rather than trusted.
    """
    cache = Path(cache_dir)
    if (cache / _SENTINEL).exists():
        print(f"  Boltz weights ready in {cache}")
        return

    for stale in (*cache.glob("*.ckpt"), *cache.glob("*.part"), cache / "mols.tar", cache / "mols"):
        if stale.is_dir():
            print(f"  Discarding possibly-incomplete {stale.name}/")
            shutil.rmtree(stale)
        elif stale.exists():
            print(f"  Discarding possibly-truncated {stale.name}")
            stale.unlink()
    cache.mkdir(parents=True, exist_ok=True)

    print(f"  Fetching Boltz weights into {cache} (~4.2GB, once per cluster)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(_DOWNLOADS)) as pool:
        for future in [pool.submit(_fetch, url, cache / name) for url, name in _DOWNLOADS]:
            future.result()
    fetched = sum((cache / name).stat().st_size for _, name in _DOWNLOADS)
    print(f"  Downloaded {fetched / 1e9:.1f}GB in {time.time() - t0:.0f}s")

    t0 = time.time()
    with tarfile.open(cache / "mols.tar", "r") as tar:
        tar.extractall(cache)
    print(f"  Extracted the CCD archive in {time.time() - t0:.0f}s")

    missing = [name for name in _ARTIFACTS if not (cache / name).exists()]
    if missing:
        raise RuntimeError(
            f"Boltz cache in {cache} is incomplete after download: {', '.join(missing)}"
        )

    (cache / "mols.tar").unlink(missing_ok=True)
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
