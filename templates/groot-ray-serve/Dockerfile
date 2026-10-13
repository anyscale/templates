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

FROM anyscale/ray:2.59.0-slim-py311-cu129

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
# Pinned to v2.3.2 (last release for the Isaac Sim 5.1.0 stack, and the last release
# to support py3.10/3.11). Upstream's default branch has since moved to release/3.0.0-beta2,
# which pairs with a newer Isaac Sim and pins `python_requires=">=3.12"` — cloning an unpinned
# `--depth 1` snapshot would silently break the build (pip errors "isaaclab requires >=3.12").
# v2.3.2 pins `flatdict==4.0.1`, which has no PyPI wheel; its sdist's setup.py imports
# `pkg_resources`, removed from modern isolated-build-env setuptools (81+). Widen the pin so
# pip can pick a version that ships a wheel (flatdict 4.1.0 does).
RUN git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /home/ray/IsaacLab && \
    cd /home/ray/IsaacLab && \
    sed -i -E 's|"flatdict==4\.0\.1"|"flatdict>=4.0.1"|' source/isaaclab/setup.py && \
    python -m pip install \
        -e source/isaaclab \
        -e source/isaaclab_tasks \
        -e source/isaaclab_rl

# Isaac-GR00T (editable, --no-deps to skip the upstream py-version pin)
# NOTE: Isaac-GR00T upstream is unpinned and periodically retightens its
# requires-python (was `==3.10.*`, later `>=3.12,<3.13`); rewrite whatever
# it currently declares so the build stays on the base image's Python.
# Pinned to 51d4c89f72 (Aug 20, 2026). The Oct 7, 2026 "Sync GR00T N1.7
# staging package to Isaac-GR00T" commit (d2b7e75b93) made Gr00tN1d7Config
# incompatible with newer transformers' PretrainedConfig base and raises
# `TypeError: non-default argument 'diffusion_model_cfg' follows default
# argument` at import time, which crashes the Ray Serve replica.
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /home/ray/Isaac-GR00T && \
    cd /home/ray/Isaac-GR00T && \
    git checkout 51d4c89f72 && \
    sed -i -E 's|^requires-python = .*|requires-python = ">=3.10"|' pyproject.toml && \
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

# Force-pin protobuf back to a Ray-compatible version. The Isaac Sim / Isaac Lab
# / GR00T installs above transitively pull in protobuf 7.x, which breaks Ray
# Serve at runtime with:
#   AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label'
# in ray/serve/_private/config.py::_proto_to_dict. Ray itself declares
# `protobuf<7`; keep it aligned with the base image's 4.25.x line.
RUN python -m pip install --no-deps "protobuf==4.25.8"

# Force-pin pyarrow back to a numpy-1.x-compatible version. Ray 2.59's base image
# ships pyarrow==23.0.1 (built against numpy 2.0 and raises `ImportError: pyarrow
# requires NumPy 2.0 or newer` at import), but Isaac Sim 5.1.0 above downgrades
# numpy to 1.26.0. Ray 2.59 also added an eager pyarrow import in
# ray/util/serialization_addons.py (via _register_custom_datasets_serializers)
# whose except clause only catches ModuleNotFoundError, so an incompatible
# pyarrow blows up the first `ray.get(...)`. pyarrow 17.0.0 is the last line
# built against numpy<2 and still satisfies Ray's declared `pyarrow>=17.0.0`.
RUN python -m pip install --no-deps "pyarrow==17.0.0"

# Force-align torchaudio with the cu128 torch wheel installed above. Later pip
# installs (notably `isaacsim[all]` and the GR00T runtime step) pull a torchaudio
# wheel from PyPI that was compiled against CUDA 12.6; the first time Ray Serve
# initializes a replica importing gr00t -> transformers.AutoProcessor it dies
# with "PyTorch and TorchAudio were compiled with different CUDA versions"
# (PyTorch 12.8 vs TorchAudio 12.6). Reinstalling against the cu128 index pins
# the matching 2.7.0 build last, so nothing can displace it.
RUN python -m pip install --no-deps --force-reinstall \
    "torchaudio==2.7.0" \
    --index-url https://download.pytorch.org/whl/cu128

# Pin transformers to the version Isaac-GR00T 51d4c89f72 declares in its
# pyproject.toml. Newer transformers (>=4.58) turned PretrainedConfig into a
# dataclass with non-default fields, which breaks `@dataclass class
# Gr00tN1d7Config(PretrainedConfig)` with
# `TypeError: non-default argument 'diffusion_model_cfg' follows default argument`
# at module import. The Ray 2.59 base image happens to resolve transformers
# higher than this, so the pin has to be applied last.
RUN python -m pip install --no-deps "transformers==4.57.3" "tokenizers==0.22.2" "huggingface-hub==0.36.2"

# flash-attn is intentionally NOT installed; GR00T falls back to eager attention cleanly.
# To enable later, install the matching prebuilt wheel post-build:
#   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
#     flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# PYTHONPATH so Isaac Lab + Isaac-GR00T are importable everywhere
ENV PYTHONPATH=/home/ray/IsaacLab/source/isaaclab:/home/ray/IsaacLab/source/isaaclab_tasks:/home/ray/IsaacLab/source/isaaclab_rl:/home/ray/Isaac-GR00T:${PYTHONPATH}
