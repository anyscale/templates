#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill==2.7.0 nbconvert==7.16.6 ipykernel==6.29.5

set +x  # don't echo the resolved secret under xtrace
export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
    --secret-id anyscale_hf_token --query SecretString --output text)
set -x

# Only small-size-llm runs in CI; the other 6 sub-notebooks need multi-GPU/H100.
# Deploy locally and wait for the model to load before the notebook's query cell.
cd small-size-llm
serve run serve_llama_3_1_8b:app --non-blocking
trap 'serve shutdown -y || true' EXIT
for _ in $(seq 1 60); do curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done

jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags=skip-in-ci \
    --output /tmp/small-size-llm.ci.ipynb
papermill /tmp/small-size-llm.ci.ipynb /tmp/small-size-llm.out.ipynb --log-output --kernel python3 --cwd .
