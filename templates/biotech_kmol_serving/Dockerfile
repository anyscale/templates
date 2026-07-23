# =============================================================================
# kMoL + Ray Serve image.
#
# kMoL is a conda stack (CUDA 11.7 / Python 3.9 / PyTorch 1.13.1) and its install
# compiles a CUDA extension that needs a GPU visible at BUILD time (see kMoL's own
# docker/build_docker.sh). That makes a clean from-scratch pip build fragile in CI.
#
# PRIMARY PATH (recommended): layer Ray onto the image Takeda ALREADY builds when
# running kMoL (`make build-docker` -> elix-kmol:<version>). It has the `kmol` conda
# env with kMoL installed; we only add Ray Serve + Anyscale on top.
#
# The molecule GNN + ensemble + collater path we serve never imports the OpenFold
# CUDA kernel (only the protein/MSA path does), so at RUNTIME no GPU-compiled
# extension is exercised.
# =============================================================================

# ---- PRIMARY: base on kMoL's prebuilt runtime image -------------------------
ARG KMOL_IMAGE=elix-kmol:latest
FROM ${KMOL_IMAGE}

# Ray must match the env's Python (3.9). 2.9.x supports py3.9 + torch 1.13.
# Confirm the Ray version is one your Anyscale cloud supports before deploying.
ARG RAY_VERSION=2.9.3
RUN bash -lc 'conda run -n kmol pip install --no-cache-dir \
      "ray[serve]==${RAY_VERSION}" anyscale fastapi "uvicorn[standard]" requests'

# Make `import kmol` and Ray resolve to the kmol conda env by default.
ENV PATH=/opt/conda/envs/kmol/bin:$PATH
ENV OMP_NUM_THREADS=4

# =============================================================================
# FALLBACK (no prebuilt kMoL image): build the conda env and use PYTHONPATH so
# `import kmol` works WITHOUT compiling the CUDA extension. Uncomment to use.
# -----------------------------------------------------------------------------
# FROM mambaorg/micromamba:1.5.8
# USER root
# RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
#     && rm -rf /var/lib/apt/lists/*
# ARG KMOL_REF=c7f8833
# RUN git clone https://github.com/elix-tech/kmol.git /opt/kmol && \
#     cd /opt/kmol && git checkout ${KMOL_REF}
# RUN micromamba create -y -n kmol -f /opt/kmol/environment.yml && micromamba clean -a -y
# RUN micromamba run -n kmol pip install --no-cache-dir "ray[serve]==2.9.3" anyscale fastapi "uvicorn[standard]" requests
# # PYTHONPATH avoids the GPU-at-build-time CUDA extension compile. If import fails on
# # a vendored compiled module, fall back to: micromamba run -n kmol pip install -e /opt/kmol
# ENV PYTHONPATH=/opt/kmol/src
# ENV PATH=/opt/conda/envs/kmol/bin:$PATH
# ENV OMP_NUM_THREADS=4
# =============================================================================
