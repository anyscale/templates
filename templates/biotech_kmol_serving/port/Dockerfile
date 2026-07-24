# =============================================================================
# Ray Serve image for the PORTED kMoL molecule ensemble (kmolport).
#
# Contrast with ../Dockerfile (the pre-port approach): that one had to layer Ray
# onto kMoL's frozen conda stack (CUDA 11.7 / py3.9 / torch 1.13.1) whose install
# compiles a CUDA extension needing a GPU at BUILD time. The port removes all of
# that — the molecule path is standard torch + PyG + rdkit, so this is an ordinary
# pip layer on a stock modern Anyscale Ray image. No conda, no GPU-at-build, no
# vendored CUDA kernels.
# =============================================================================

# Stock Anyscale Ray image: Python 3.11 + CUDA 12. Match your Anyscale cloud's
# supported Ray version.
ARG BASE_IMAGE=anyscale/ray:2.56.0-py311-cu121
FROM ${BASE_IMAGE}

# The ported serving stack. The default PyPI linux torch wheel is a CUDA 12.x
# build and runs on L4 (sm_89) — validated at 60k mol/s forward on one L4.
RUN pip install --no-cache-dir \
      torch==2.5.1 \
      torch_geometric==2.6.1 \
      rdkit==2024.3.5 \
      "numpy<2"

# App: the minimal ported package + the Serve deployment + configs + scripts.
WORKDIR /home/ray/app
COPY kmolport /home/ray/app/kmolport
COPY configs /home/ray/app/configs
COPY scripts /home/ray/app/scripts
COPY serve_app.py /home/ray/app/

ENV PYTHONPATH=/home/ray/app

# Self-contained demo weights (SYNTHETIC — correct architecture/format, random values).
# For production, replace with real trained checkpoints: mount them at
# /home/ray/app/checkpoints, or COPY them in and rebuild. Prediction values are only
# meaningful with real weights; throughput is identical either way.
RUN python scripts/make_synthetic_checkpoints.py configs/ensemble_serve.example.json checkpoints
# Each Serve replica featurizes on its own core; keep torch intra-op threads at 1.
ENV OMP_NUM_THREADS=1

# Serve entrypoint (for local `serve run`); on Anyscale, deploy via service.image.yaml.
#   serve run serve_app:app
# =============================================================================
