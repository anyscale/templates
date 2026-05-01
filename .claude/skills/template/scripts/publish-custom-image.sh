#!/usr/bin/env bash
set -euo pipefail

REGISTRY="us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates"

if [ $# -ne 3 ]; then
  cat >&2 <<EOF
Usage:   $0 <dockerfile-dir> <image-name> <ray-version>
Example: $0 templates/deployment-serve-llm/small-size-llm template_deployment-serve-llm 2.54.1
EOF
  exit 1
fi

dockerfile_dir="$1"
image_name="$2"
ray_version="$3"
tag="${REGISTRY}/${image_name}:${ray_version}"

if [ ! -f "$dockerfile_dir/Dockerfile" ]; then
  echo "Error: $dockerfile_dir/Dockerfile not found" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not reachable. Ensure docker is running and retry." >&2
  exit 1
fi

if [ -n "${GCP_TEMPLATE_REGISTRY_SA_KEY:-}" ]; then
  DOCKER_CONFIG=$(mktemp -d)
  export DOCKER_CONFIG
  trap 'rm -rf "$DOCKER_CONFIG"' EXIT
  printf '%s' "$GCP_TEMPLATE_REGISTRY_SA_KEY" | docker login -u _json_key --password-stdin https://us-docker.pkg.dev
fi

docker build --platform linux/amd64 -t "$tag" "$dockerfile_dir"

if ! docker push "$tag"; then
  cat >&2 <<EOF
Push failed. Inspect the docker error above:
  - Auth (denied/unauthorized): ask the user to set GCP_TEMPLATE_REGISTRY_SA_KEY (JSON content) or run \`gcloud auth configure-docker us-docker.pkg.dev\`
  - Transient (network/timeout): retry
  - Else: stop and report the error
EOF
  exit 1
fi

echo "Published: $tag"
