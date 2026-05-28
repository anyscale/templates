#!/usr/bin/env bash
set -euxo pipefail

# Anyscale workspaces run with CapEff=0 (no Linux capabilities → nested containers
# cannot create network namespaces, mount overlayfs, or fork-exec the OCI init).
# So the stdio-podman MCP pattern in NB03/NB04 can't actually launch containers
# here. We swap in a CI-only `podman` shim (lives only in this test dir — the
# user-facing template is untouched) that maps `docker.io/mcp/<image>` invocations
# to the native MCP CLI equivalents:
#   docker.io/mcp/brave-search → npx @modelcontextprotocol/server-brave-search
#   docker.io/mcp/fetch        → uvx mcp-server-fetch
# The shim is picked up by the Ray Serve replicas via PATH injected through
# --runtime-env-json. Customers running the template outside workspaces (where
# they have docker/podman privileges) hit the real container path unchanged.

bash build.sh
pip install --no-cache-dir "papermill==2.7.0" "jupyter==1.1.1" "nbconvert==7.16.6"

# Node 20 (brave-search MCP uses optional chaining — needs >=14; Debian's apt
# nodejs is 12.x in this image).
NODE_DIR=/tmp/node20
if [ ! -x "$NODE_DIR/bin/node" ]; then
  curl -fsSL https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz -o /tmp/node20.tar.xz
  mkdir -p "$NODE_DIR" && tar -xJf /tmp/node20.tar.xz -C "$NODE_DIR" --strip-components=1
fi

# Podman shim. The Ray Serve replicas exec this when they think they're calling
# the system podman; we forward stdio MCPs to their native CLI equivalents.
mkdir -p /tmp/bin
cat > /tmp/bin/podman <<'PODMAN_SHIM'
#!/usr/bin/env bash
# tests-only shim — see tests/mcp-ray-serve/tests.sh header.
if [ "$1" != "run" ]; then
  exec /usr/bin/podman "$@"
fi
shift
env_vars=()
image=""
cmd_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -i|-t|-it|--rm|-d) shift ;;
    -e|--env) env_vars+=("$2"); shift 2 ;;
    --net=*|--security-opt=*|--privileged|--userns=*) shift ;;
    docker.io/mcp/*) image="$1"; shift; cmd_args+=("$@"); break ;;
    *) shift ;;
  esac
done
for ev in "${env_vars[@]}"; do export "$ev"; done
case "$image" in
  docker.io/mcp/brave-search) exec /tmp/node20/bin/npx -y --quiet @modelcontextprotocol/server-brave-search "${cmd_args[@]}" ;;
  docker.io/mcp/fetch)        exec uvx --quiet mcp-server-fetch "${cmd_args[@]}" ;;
  *) echo "podman-shim: unsupported image: $image" >&2; exit 1 ;;
esac
PODMAN_SHIM
chmod +x /tmp/bin/podman

set +x  # hide BRAVE_API_KEY from xtrace
BRAVE_API_KEY=$(aws --region=us-west-2 secretsmanager get-secret-value \
  --secret-id brave-search-api-key --query SecretString --output text)
export BRAVE_API_KEY
RUNTIME_ENV_JSON=$(python -c "import json,os; print(json.dumps({'env_vars': {'BRAVE_API_KEY': os.environ['BRAVE_API_KEY'], 'PATH': '/tmp/bin:/tmp/node20/bin:/home/ray/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'}}))")
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

trap 'serve shutdown -y >/dev/null 2>&1 || true' EXIT

# NB01 + NB02: HTTP-only MCPs, no containers involved.
run_nb "01 Deploy_custom_mcp_in_streamable_http_with_ray_serve.ipynb"
run_nb "02 Build_mcp_gateway_with_existing_ray_serve_apps.ipynb"

# NB03: brave_mcp_ray_serve invokes `podman run -e BRAVE_API_KEY=... docker.io/mcp/brave-search`.
# Shim → npx @modelcontextprotocol/server-brave-search.
set +x
serve run --non-blocking --runtime-env-json "$RUNTIME_ENV_JSON" brave_mcp_ray_serve:brave_search_tool
set -x
for _ in $(seq 1 60); do curl -sf http://localhost:8000/tools >/dev/null 2>&1 && break; sleep 5; done
run_nb "03 Deploy_single_mcp_stdio_docker_image_with_ray_serve.ipynb"
serve shutdown -y
sleep 3

# NB04: multi_mcp_ray_serve runs brave-search + fetch. Shim → npx + uvx.
set +x
serve run --non-blocking --runtime-env-json "$RUNTIME_ENV_JSON" multi_mcp_ray_serve:app
set -x
for _ in $(seq 1 60); do curl -sf http://localhost:8000/brave_search/tools >/dev/null 2>&1 && break; sleep 5; done
run_nb "04 Deploy_multiple_mcp_stdio_docker_images_with_ray_serve.ipynb"
