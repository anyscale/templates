# =============================================================================
# Isaac Sim 5.1.0 + Isaac Lab + GR00T on Anyscale Ray
# Anyscale builder-safe version:
# - FROM uses Anyscale base image
# - no CMD
# - no ENTRYPOINT
# - no build-breaking flash-attn install
# - all GR00T runtime deps baked in (av, diffusers, peft, etc.)
# =============================================================================

FROM anyscale/ray:2.53.0-slim-py311-cu128

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Isaac / Omniverse headless
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV DISPLAY=""
ENV OMNI_KIT_RENDERING_MODE=headless
ENV __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

USER root

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    wget \
    build-essential \
    pkg-config \
    libgl1 \
    libegl1 \
    libegl-mesa0 \
    libgles2 \
    libglvnd0 \
    libglvnd-dev \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxkbcommon0 \
    libx11-6 \
    libxext6 \
    libxt6 \
    libglu1-mesa \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    libnss3 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Vulkan ICD stub
# -----------------------------------------------------------------------------
RUN mkdir -p /etc/vulkan/icd.d && \
    printf '%s\n' \
    '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}' \
    > /etc/vulkan/icd.d/nvidia_icd.json

USER ray
WORKDIR /home/ray

# -----------------------------------------------------------------------------
# Python packaging tools
# -----------------------------------------------------------------------------
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# PyTorch CUDA 12.8
# -----------------------------------------------------------------------------
RUN python -m pip install \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128

# -----------------------------------------------------------------------------
# Isaac Sim 5.1.0
# -----------------------------------------------------------------------------
RUN python -m pip install \
    "isaacsim[all]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

# -----------------------------------------------------------------------------
# Isaac Lab
# -----------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git /home/ray/IsaacLab

RUN cd /home/ray/IsaacLab && \
    python -m pip install \
    -e source/isaaclab \
    -e source/isaaclab_tasks \
    -e source/isaaclab_rl

# -----------------------------------------------------------------------------
# Isaac GR00T
# -----------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git /home/ray/Isaac-GR00T

RUN cd /home/ray/Isaac-GR00T && \
    sed -i 's|requires-python = "==3\.10\.\*"|requires-python = ">=3.10"|' pyproject.toml && \
    python -m pip install --no-deps -e .

# -----------------------------------------------------------------------------
# GR00T runtime deps (transitive deps that --no-deps skipped)
# These versions are pinned to match GR00T's pyproject.toml
# -----------------------------------------------------------------------------
RUN python -m pip install \
    "av==16.1.0" \
    "diffusers==0.35.1" \
    "peft==0.17.1" \
    "albumentations==1.4.18" \
    "tyro==0.9.17" \
    "einops==0.8.1" \
    "msgpack-numpy==0.4.8" \
    "dm-tree==0.1.8" \
    "lmdb==1.7.5" \
    "termcolor==3.2.0" \
    "pyzmq==27.0.1" \
    "torchcodec==0.4.0" \
    "imageio" \
    "imageio-ffmpeg" \
    "accelerate" \
    "deepspeed" \
    "huggingface_hub" \
    "datasets" \
    "opencv-python-headless"

# -----------------------------------------------------------------------------
# Additional training/runtime deps
# -----------------------------------------------------------------------------
RUN python -m pip install \
    pillow \
    wandb \
    gymnasium \
    packaging \
    ninja \
    numpy \
    pandas \
    matplotlib \
    scipy \
    requests \
    fastapi \
    uvicorn

# -----------------------------------------------------------------------------
# flash-attn (best-effort).
# This compiles from source if a prebuilt wheel for py311+torch2.7+cu12 isn't
# available. If your build host is too small, comment out the next block and
# install flash-attn manually post-build with the matching prebuilt wheel:
#
#   pip install \
#     https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
#     flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# -----------------------------------------------------------------------------
# RUN python -m pip install flash-attn==2.7.4.post1 --no-build-isolation

# -----------------------------------------------------------------------------
# PYTHONPATH so Isaac Lab + Isaac-GR00T are importable everywhere
# -----------------------------------------------------------------------------
ENV PYTHONPATH=/home/ray/IsaacLab/source/isaaclab:/home/ray/IsaacLab/source/isaaclab_tasks:/home/ray/IsaacLab/source/isaaclab_rl:/home/ray/Isaac-GR00T:${PYTHONPATH}

WORKDIR /home/ray/default
