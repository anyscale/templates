#!/usr/bin/env bash
#
# Write tools/env.<name>.lock for every tools/env.<name>.yml, from a built image.
#
#   bash tools/lock-envs.sh nextflow-genomics-on-ray:local-test
#
# The lock is micromamba's `--explicit` export: one package URL per line, with its
# md5. build_tools.sh prefers a lock over its .yml when both exist, and installs
# from it with no solve, so the next build gets exactly the files this image has
# rather than whatever the solver would pick on the day.
#
# Read out of an image rather than solved here because the environment has to be
# a linux-64 one, and the machine running this is often a Mac. Nothing is built or
# downloaded: the image already holds the environment, this only lists it.
#
# Commit the lock only after building from it, and say so in the commit: a lock
# nobody has installed from is one more claim about the toolchain, not a check
# of it.
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
    # An export that lists no packages means the prefix was wrong or empty; do not
    # let it replace a good lock.
    if ! grep -q '^https://' "$tmp"; then
        echo "lock-envs.sh: $IMAGE has no packages in /opt/nf-tools/envs/$name" >&2
        cat "$tmp" >&2
        exit 1
    fi
    mv "$tmp" "$lock"
    printf '%s: %d packages\n' "$lock" "$(grep -c '^https://' "$lock")"
done
