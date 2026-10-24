"""
One-time setup: download the dataset, the published benchmark, scored results and weights.

    python scripts/setup_data.py                 # data and scored results (about 500 MB, 1 to 3 min)
    python scripts/setup_data.py --with-weights  # also Boltz-2 and OpenFold3 (about 15 GB, 10 to 20 min)
    python scripts/setup_data.py --with-weights --skip-of3   # Boltz-2 only (about 6 GB)

The default is enough to open every notebook with USE_GPUS = False: all tables,
figures and the before/after comparison read the scored results downloaded here.
Weights are only needed to predict or fine-tune, which is USE_GPUS = True.

Everything lands under $AIDDD_SHARED (default /mnt/user_storage/aiddd), which
every node of a Workspace or Job can read. Each step is skipped if its output
already exists, so the script is safe to rerun.

Sources (all public):
    OpenBind EV-A71 2A data release     Zenodo 10.5281/zenodo.22142262 (CC0)
    OpenBind benchmark results          github.com/OpenBind-Consortium/EV-A71_2A_benchmark (Apache-2.0)
    Boltz-2 weights + CCD               downloaded by the boltz package
    OpenBind-0 and fine-tuned OF3       openfold3-data S3 bucket
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))

RELEASE_URL = "https://zenodo.org/api/records/22142262/files/OpenBind_EV-A71_2A.zip/content"
BENCHMARK_RAW = ("https://raw.githubusercontent.com/OpenBind-Consortium/EV-A71_2A_benchmark/main/"
                 "structure/processed_outputs/")
BENCHMARK_FILES = ["final_cofolding_pose_data.parquet", "fragment_followon_similarity.csv",
                   "annotated_complexes.csv"]
# Scored benchmark results for every model and target, plus the fine-tune loss curve
# and the derived Mpro metadata (about 2 MB). Published alongside this template.
RESULTS_BUNDLE = os.environ.get(
    "AIDDD_RESULTS_BUNDLE",
    "https://github.com/miss-classification/boltz2-openbind-ray/releases/download/"
    "data-v1/aiddd-results-bundle.tar.gz")

OF3_BASE = "https://openfold3-data.s3.amazonaws.com/openfold3-parameters/"
OF3_CKPTS = {
    "of3-ob-2025-06-30-174k.pt": OF3_BASE + "of3-ob-2025-06-30-174k.pt",      # OpenBind-0
    "of3p2-ft-ev2a.ckpt": OF3_BASE + "openbind/of3p2-ft-ev2a.ckpt",           # fine-tuned on EV-A71 2A
}


def download(url: str, dest: Path):
    """Download to a temp file first, so an interrupted run never leaves a partial file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    subprocess.run(["curl", "-fSL", "--retry", "3", "-o", str(tmp), url], check=True)
    tmp.rename(dest)


def step(name: str, done: bool):
    print(f"[{'skip' if done else 'run '}] {name}")
    return not done


def setup_release():
    target = SHARED / "raw" / "OpenBind_EV-A71_2A"
    if step("OpenBind EV-A71 2A release (structures, metadata, MSA)", (target / "EV-A71_2A_metadata.csv").exists()):
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "release.zip"
            download(RELEASE_URL, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(SHARED / "raw")
    n = sum(1 for _ in (target / "structures").iterdir())
    print(f"       {n} ligand folders in {target / 'structures'}")


def setup_benchmark():
    out = SHARED / "EV-A71_2A_benchmark" / "structure" / "processed_outputs"
    for f in BENCHMARK_FILES:
        if step(f"published benchmark: {f}", (out / f).exists()):
            download(BENCHMARK_RAW + f, out / f)


def setup_results():
    """Scored results for every model and target, so the notebooks display without a GPU."""
    marker = SHARED / "results" / "ev71-2a"
    if step("scored benchmark results (about 2 MB)", marker.exists()):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "results.tar.gz"
            download(RESULTS_BUNDLE, bundle)
            with tarfile.open(bundle) as t:
                t.extractall(SHARED)
    targets = sorted(p.name for p in (SHARED / "results").iterdir() if p.is_dir())
    print(f"       scored targets: {', '.join(targets)}")


def setup_tune_summaries():
    """Ray Tune sweep summaries for Appendix A of notebook 02, shipped in docs/data (about 1 KB)."""
    src = Path(__file__).resolve().parents[1] / "docs" / "data"
    for f in sorted(src.rglob("summary.csv")):
        dest = SHARED / f.relative_to(src)
        if step(f"Ray Tune sweep summary: {dest.parent.name}", dest.exists()):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)


def setup_boltz():
    cache = SHARED / "boltz_cache"
    if step("Boltz-2 weights and CCD", (cache / "boltz2_conf.ckpt").exists() and (cache / "boltz2_aff.ckpt").exists()):
        from boltz.main import download_boltz2
        cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)


def setup_of3():
    cache = SHARED / "of3_cache"
    for name, url in OF3_CKPTS.items():
        if step(f"OpenFold3 checkpoint {name}", (cache / name).exists()):
            download(url, cache / name)
    # OpenFold3 reads its default checkpoint location from this file.
    (cache / "ckpt_root").write_text(str(cache))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-weights", action="store_true",
                    help="also download Boltz-2 and OpenFold3 weights (about 15 GB), needed only to predict")
    ap.add_argument("--skip-of3", action="store_true", help="skip the two OpenFold3 checkpoints (about 9 GB)")
    args = ap.parse_args()

    print(f"Shared data directory: {SHARED}")
    SHARED.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(SHARED).free / 1e9
    print(f"Free space: {free_gb:.0f} GB")

    setup_release()
    setup_benchmark()
    setup_results()
    setup_tune_summaries()
    if args.with_weights:
        setup_boltz()
        if not args.skip_of3:
            setup_of3()
    else:
        print("[skip] model weights. Add --with-weights to predict or fine-tune on GPUs.")
    print("\nSetup complete. Open 01_ray_and_offline_inference.ipynb and run the cells in order.")


if __name__ == "__main__":
    main()
