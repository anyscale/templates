# Optional custom image.
#
# The default `anyscale/ray-llm:2.57.0-py312-cu130` image already ships
# vLLM 0.25.1 (>= the model's required 0.18.1) with the built-in "nemotron_v3"
# reasoning parser, so you normally do NOT need this Dockerfile.
#
# Use it only if you need a newer/pinned vLLM, or to bundle the model's
# standalone reasoning parser as a fallback for older engines. To use it, set
# `containerfile: ./Dockerfile` in service.yaml (instead of `image_uri`).
FROM anyscale/ray-llm:2.57.0-py312-cu130

# Example: pin a specific vLLM build.
# RUN pip install --no-cache-dir vllm==0.25.1

# Optional fallback: fetch NVIDIA's standalone reasoning parser plugin.
# RUN curl -fsSL -o /home/ray/super_v3_reasoning_parser.py \
#     https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8/raw/main/super_v3_reasoning_parser.py
