#!/usr/bin/env bash
set -euxo pipefail

bash build.sh
pip install --no-cache-dir "papermill==2.7.0" "jupyter==1.1.1" "nbconvert==7.16.6"

set +x  # don't echo the resolved secret under xtrace
BRAVE_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id brave-search-api-key \
  --query SecretString \
  --output text)
export BRAVE_API_KEY
set -x

run_nb() {
  local nb="$1" base
  base="$(basename "${nb}" .ipynb)"
  jupyter nbconvert --to notebook "${nb}" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output "/tmp/${base}.ci.ipynb"
  papermill "/tmp/${base}.ci.ipynb" "/tmp/${base}.out.ipynb" --log-output --kernel python3 --cwd .
}

run_nb "01 Deploy_custom_mcp_in_streamable_http_with_ray_serve.ipynb"
run_nb "02 Build_mcp_gateway_with_existing_ray_serve_apps.ipynb"

# NB03 queries a service the customer deploys from a terminal; replicate that here.
# `serve run` doesn't propagate the runner's shell env to Ray Serve replicas (customers
# set BRAVE_API_KEY via the workspace Dependencies tab in prod) — pass it via runtime-env.
# python (not jq — the workspace image doesn't ship it) builds the JSON safely.
set +x  # hide the secret from xtrace
RUNTIME_ENV_JSON=$(python -c 'import json,os; print(json.dumps({"env_vars": {"BRAVE_API_KEY": os.environ["BRAVE_API_KEY"]}}))')
serve run --non-blocking --runtime-env-json "$RUNTIME_ENV_JSON" brave_mcp_ray_serve:brave_search_tool
set -x
trap 'serve shutdown -y || true' EXIT
for _ in $(seq 1 60); do curl -sf http://localhost:8000/tools >/dev/null 2>&1 && break; sleep 5; done
run_nb "03 Deploy_single_mcp_stdio_docker_image_with_ray_serve.ipynb"
serve shutdown -y

# NB04 deploys + queries locally on its own; the EXIT trap tears it down. NB05 is markdown-only (skipped).
run_nb "04 Deploy_multiple_mcp_stdio_docker_images_with_ray_serve.ipynb"
