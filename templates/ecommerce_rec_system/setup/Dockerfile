FROM anyscale/ray:2.55.0-slim-py312-cu129

# Hugging Face cache: base image runs builds as `ray`, which cannot write under /root.
ENV HF_HOME=/home/ray/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Inline deps: remote image builds disallow COPY from local paths (see setup/requirements.txt for the list).
RUN python3 -m pip install --no-cache-dir \
    torch \
    transformers \
    sentence-transformers \
    hf_transfer \
    streamlit

# Bake public weights into the image so Serve cold start skips Hub downloads.
RUN python3 - <<'PY'
from huggingface_hub import snapshot_download

for repo_id in (
    "Salesforce/blip-image-captioning-base",
    "sentence-transformers/all-MiniLM-L6-v2",
):
    snapshot_download(repo_id, token=False)
PY
