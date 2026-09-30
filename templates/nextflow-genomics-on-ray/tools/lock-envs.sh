#!/usr/bin/env bash
# Write tools/env.<name>.lock (micromamba --explicit, with md5s) for each env.<name>.yml, read out
# of a built image so the lock is linux-64 wherever this runs:  bash tools/lock-envs.sh <image>
set -euo pipefail

IMAGE="${1:?usage: lock-envs.sh <image>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${CONTAINER_ENGINE:-$(command -v docker || command -v podman || true)}"
[[ -n "$ENGINE" ]] || { echo "lock-envs.sh: need docker or podman on PATH" >&2; exit 1; }

for spec in "$HERE"/env.*.yml; do
    name="$(basename "$spec" .yml)"
    name="${name#env.}"
    lock="$HERE/env.$name.lock"
    tmp="$(mktemp)"
    "$ENGINE" run --rm --platform linux/amd64 --entrypoint /opt/nf-tools/bin/micromamba \
        "$IMAGE" env export --explicit --md5 -p "/opt/nf-tools/envs/$name" > "$tmp"
    # An empty export means a wrong prefix; don't let it replace a good lock.
    if ! grep -q '^https://' "$tmp"; then
        echo "lock-envs.sh: $IMAGE has no packages in /opt/nf-tools/envs/$name" >&2
        cat "$tmp" >&2
        exit 1
    fi
    mv "$tmp" "$lock"
    printf '%s: %d packages\n' "$lock" "$(grep -c '^https://' "$lock")"
done
