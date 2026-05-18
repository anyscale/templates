#!/usr/bin/env bash
# setup_workers.sh - Install GR00T into anaconda env alongside Isaac Lab.
#
# Strategy:
#   The Anyscale container has Python 3.11 with Isaac Lab 0.54.3 + Isaac Sim
#   5.1.0 + torch 2.7.0+cu128 already installed. GR00T's pyproject.toml
#   pins `requires-python = "==3.10.*"` which blocks install on 3.11.
#
#   GR00T doesn't actually need 3.10 — the pin is conservative. We patch it
#   to allow 3.11, then install GR00T and the few missing deps.
#
# Usage:
#     bash setup_workers.sh

set -e

echo "=================================================="
echo "Installing GR00T into anaconda (Python 3.11)"
echo "=================================================="

python - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)

# Prebuilt flash-attn wheel (Python 3.11, torch 2.7, cu12, cxx11abiFALSE)
FLASH_ATTN_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

# GR00T's supporting packages. We relax all exact pins because the anaconda
# env already has compatible-but-newer versions. --no-deps on gr00t itself
# means pip won't re-resolve these.
EXTRA_DEPS = [
    "accelerate",
    "diffusers",
    "peft",
    "deepspeed",
    "einops",
    "tyro",
    "wandb",
    "av",
    "opencv-python-headless",
    "imageio",
    "omegaconf",
    "hydra-core",
    "albumentations",
    "msgpack-numpy",      # was missing in previous run
    "jsonlines",          # was missing in previous run
    "lmdb",               # was missing in previous run
    "datasets",
]


@ray.remote(num_gpus=1)
def install_on_worker():
    import subprocess, os, pathlib, re

    PIP = "/home/ray/anaconda3/bin/pip"
    PY  = "/home/ray/anaconda3/bin/python"

    # ----------------------------------------------------------------
    # 1. Ensure Isaac-GR00T clone exists (previous runs may have deleted it).
    # ----------------------------------------------------------------
    subprocess.run(
        "test -d /home/ray/Isaac-GR00T || "
        "git clone --recurse-submodules "
        "https://github.com/NVIDIA/Isaac-GR00T /home/ray/Isaac-GR00T",
        shell=True, check=True, executable="/bin/bash",
    )

    # ----------------------------------------------------------------
    # 2. Patch Isaac-GR00T's pyproject.toml to allow Python 3.11.
    #    The pin is conservative — code works fine on 3.11.
    # ----------------------------------------------------------------
    pp = pathlib.Path("/home/ray/Isaac-GR00T/pyproject.toml")
    text = pp.read_text()

    # Relax requires-python from "==3.10.*" to ">=3.10,<3.13".
    new_text, n1 = re.subn(
        r'requires-python\s*=\s*"==3\.10\.\*"',
        'requires-python = ">=3.10,<3.13"',
        text,
    )
    # Also relax if it's written as >=3.10,<3.11
    new_text, n2 = re.subn(
        r'requires-python\s*=\s*">=3\.10,\s*<3\.11"',
        'requires-python = ">=3.10,<3.13"',
        new_text,
    )
    if n1 + n2 > 0:
        pp.write_text(new_text)
        print(f"[patch] relaxed requires-python ({n1+n2} change)")
    else:
        # Check if it's already relaxed or differently written.
        current_pin = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
        print(f"[patch] requires-python currently: {current_pin.group(1) if current_pin else 'unset'}")

    # ----------------------------------------------------------------
    # 3. Install flash-attn prebuilt (Python 3.11 wheel).
    # ----------------------------------------------------------------
    subprocess.run([PIP, "install", FLASH_ATTN_WHEEL_URL], check=True)

    # ----------------------------------------------------------------
    # 4. Install extra deps.
    # ----------------------------------------------------------------
    subprocess.run([PIP, "install"] + EXTRA_DEPS, check=True)

    # ----------------------------------------------------------------
    # 5. Install gr00t itself, editable, ignoring pinned deps.
    # ----------------------------------------------------------------
    subprocess.run(
        [PIP, "install", "--no-deps", "-e", "/home/ray/Isaac-GR00T"],
        check=True,
    )

    # ----------------------------------------------------------------
    # 6. Smoke check.
    # ----------------------------------------------------------------
    smoke = """
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version.split()[0]}")

errors = []
for mod in ["torch", "transformers", "accelerate", "flash_attn",
            "gr00t", "isaaclab", "isaacsim", "gymnasium",
            "ray", "huggingface_hub", "diffusers", "peft"]:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  OK   {mod} {ver}")
    except Exception as e:
        errors.append((mod, str(e)))
        print(f"  FAIL {mod}: {e}")

import torch
print(f"\\ntorch CUDA: {torch.version.cuda}, GPU avail: {torch.cuda.is_available()}")

if errors:
    print(f"\\n{len(errors)} FAILURES")
    sys.exit(1)
print("\\nAll imports clean.")
"""
    with open("/tmp/smoke.py", "w") as f:
        f.write(smoke)
    result = subprocess.run(
        [PY, "/tmp/smoke.py"],
        capture_output=True, text=True,
    )
    return f"host={os.uname().nodename}\n{result.stdout}{result.stderr}"


num_gpu_nodes = int(ray.cluster_resources().get("GPU", 0))
print(f"Running install on {num_gpu_nodes} GPU worker(s)...")

refs = [install_on_worker.remote() for _ in range(num_gpu_nodes)]
results = ray.get(refs)
for i, r in enumerate(results):
    print(f"--- Worker {i} ---")
    print(r)
PY

echo "=================================================="
echo "Worker setup complete."
echo ""
echo "All python scripts run with anaconda's python (the default here)."
echo "Do NOT source any venv."
echo ""
echo "Next steps:"
echo "  1. python test_g1_sim.py"
echo "  2. python test_groot_standalone.py"
echo "  3. python run_demo.py --placeholder"
echo "  4. python run_demo.py"
echo "=================================================="
