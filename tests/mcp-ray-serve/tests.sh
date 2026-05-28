#!/usr/bin/env bash
set -euxo pipefail

# NB01 (custom HTTP MCP via Ray Serve) + NB02 (MCP gateway, HTTP-only) cover the
# workspace-friendly half of the template. NB03/NB04 are excluded: they shell out
# to `podman run docker.io/mcp/<image>` and Anyscale workspaces don't permit
# nested container network namespaces — even `sudo podman run hello-world` fails
# with "failed to create namespace: open /proc/.../ns/net: permission denied".
# TODO(@kunling): document the workspace incompatibility in the template README
# OR migrate the stdio-docker examples to an HTTP MCP pattern that runs in workspaces.

bash build.sh
pip install --no-cache-dir "papermill==2.7.0" "jupyter==1.1.1" "nbconvert==7.16.6"

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
