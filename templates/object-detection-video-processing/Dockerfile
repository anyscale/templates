# Use base image.
FROM anyscale/ray:2.59.0-slim-py312-cu129
# For open source-ray, use: rayproject/ray:2.59.0-py312-cu129.

# Install system dependencies.
RUN sudo apt update && sudo apt install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && sudo rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip install --no-cache-dir \
    imageio-ffmpeg==0.6.0 \
    opencv-python-headless==5.0.0.93 \
    pycocotools==2.0.11 \
    torch==2.9.1 \
    torchvision==0.24.1 \
    xmltodict==1.0.4 \
    torchmetrics==1.9.0 \
    decord==0.6.0
