# =============================================================================
# Isaac Sim 5.1.0 + Isaac Lab + GR00T on Anyscale Ray
#
# Layered on the Anyscale Ray base image, which already sets:
#   USER     = ray (with NOPASSWD sudo)
#   WORKDIR  = /home/ray/default  (ray-owned)
#
# Follow the repo's BYOD convention (see templates/{unstructured_data_ingestion,
# object-detection-video-processing,mcp-ray-serve}/Dockerfile): no USER /
# WORKDIR directives — use sudo for root-only steps.
# =============================================================================

FROM anyscale/ray:2.55.1-slim-py311-cu129

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

# System dependencies: EGL / Vulkan / X11 libs Isaac Sim links against, + build tools
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    bash ca-certificates curl git wget build-essential pkg-config \
    libgl1 libegl1 libegl-mesa0 libgles2 libglvnd0 libglvnd-dev \
    libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers \
    libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 \
    libx11-6 libxext6 libxt6 libglu1-mesa \
    libsm6 libice6 libfontconfig1 libglib2.0-0 libusb-1.0-0 libnss3 libasound2 \
    && sudo rm -rf /var/lib/apt/lists/*

# Vulkan ICD stub (NVIDIA container runtime usually mounts this; fallback here)
RUN sudo mkdir -p /etc/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.277"}}' \
    | sudo tee /etc/vulkan/icd.d/nvidia_icd.json > /dev/null

# Python packaging tools
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch (pinned to Isaac Lab's recommended version, cu128 wheels run on cu129)
RUN python -m pip install \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim 5.1.0
RUN python -m pip install \
    "isaacsim[all]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

# Isaac Lab (editable, into /home/ray)
RUN git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git /home/ray/IsaacLab && \
    cd /home/ray/IsaacLab && \
    python -m pip install \
        -e source/isaaclab \
        -e source/isaaclab_tasks \
        -e source/isaaclab_rl

# Isaac-GR00T (editable, --no-deps to skip the upstream py-version pin)
RUN git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git /home/ray/Isaac-GR00T && \
    cd /home/ray/Isaac-GR00T && \
    sed -i 's|requires-python = "==3\.10\.\*"|requires-python = ">=3.10"|' pyproject.toml && \
    python -m pip install --no-deps -e .

# GR00T runtime deps (the --no-deps above skipped these; pinned to GR00T's pyproject.toml)
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

# Training + serving runtime deps (PPO sweep + Ray Serve VLA)
RUN python -m pip install \
    pillow wandb gymnasium packaging ninja \
    numpy pandas matplotlib scipy requests \
    fastapi uvicorn

# flash-attn is intentionally NOT installed; GR00T falls back to eager attention cleanly.
# To enable later, install the matching prebuilt wheel post-build:
#   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
#     flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# PYTHONPATH so Isaac Lab + Isaac-GR00T are importable everywhere
ENV PYTHONPATH=/home/ray/IsaacLab/source/isaaclab:/home/ray/IsaacLab/source/isaaclab_tasks:/home/ray/IsaacLab/source/isaaclab_rl:/home/ray/Isaac-GR00T:${PYTHONPATH}
