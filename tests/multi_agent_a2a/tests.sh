#!/usr/bin/env bash
set -euxo pipefail
# Deploy locally, drive the notebook walkthrough, then run the shipped suite directly:
# the notebook's `!python tests/run_all.py` swallows its exit code, so CI runs it for real.

pip install -r requirements.txt
pip install "papermill==2.7.0" "nbconvert==7.16.6" "ipykernel==6.29.5"

# Research/travel agents use the web-search MCP; key is optional but makes them deterministic.
BRAVE_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id brave-search-api-key \
  --query SecretString \
  --output text)
export BRAVE_API_KEY

serve run serve_multi_config.yaml --non-blocking
trap 'serve shutdown -y || true' EXIT

# Block until the LLM app finishes loading on the L4 worker (the slowest deployment).
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/llm/v1/models >/dev/null 2>&1 && break
  sleep 5
done

jupyter nbconvert --to notebook README.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output /tmp/multi_agent_a2a.ci.ipynb
papermill /tmp/multi_agent_a2a.ci.ipynb /tmp/multi_agent_a2a.out.ipynb --log-output --kernel python3 --cwd .

python tests/run_all.py
