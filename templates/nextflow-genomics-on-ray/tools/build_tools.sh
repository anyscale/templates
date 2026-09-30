#!/usr/bin/env bash
# Install the toolchain into a prefix: manifest.toml's sha256-checked artifacts, then one conda
# env per env.*.yml. Standalone, so it rebuilds the toolchain on any Linux box without an image.
set -euo pipefail

PREFIX="${1:?usage: build_tools.sh <prefix>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PREFIX/bin" "$PREFIX/envs" "$PREFIX/share"

log() { printf '\n=== %s\n' "$*" >&2; }

log "manifest artifacts"

# Stdlib only (tomllib): nothing is installed yet.
python3 - "$HERE/manifest.toml" "$PREFIX" <<'PY'
import hashlib
import os
import sys
import tomllib
import urllib.request

manifest_path, prefix = sys.argv[1], sys.argv[2]
with open(manifest_path, "rb") as fh:
    manifest = tomllib.load(fh)

for name, spec in manifest.get("tools", {}).items():
    dest = os.path.join(prefix, spec["install"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  {name} {spec['version']} -> {dest}", file=sys.stderr)

    digest = hashlib.sha256()
    with urllib.request.urlopen(spec["url"]) as response, open(dest, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)

    actual = digest.hexdigest()
    if actual != spec["sha256"]:
        # Fatal: catches a release artifact re-cut under the same tag.
        os.unlink(dest)
        raise SystemExit(
            f"sha256 mismatch for {name}\n"
            f"  url      {spec['url']}\n"
            f"  expected {spec['sha256']}\n"
            f"  actual   {actual}"
        )
    if spec.get("mode"):
        os.chmod(dest, int(spec["mode"], 8))
PY

if ! command -v micromamba >/dev/null 2>&1; then
    echo "micromamba not found on PATH" >&2
    exit 1
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PREFIX/mamba}"

for spec in "$HERE"/env.*.yml; do
    env_name="$(basename "$spec" .yml)"
    env_name="${env_name#env.}"
    target="$PREFIX/envs/$env_name"
    lock="$HERE/env.$env_name.lock"

    if [[ -f "$lock" ]]; then
        # An --explicit lock (from lock-envs.sh) installs exact package URLs, with no solve.
        log "conda env '$env_name' (from lock)"
        micromamba create -y -p "$target" --file "$lock"
    else
        log "conda env '$env_name' (solving; no lock file)"
        micromamba create -y -p "$target" --file "$spec"
    fi

    micromamba clean --all --yes >/dev/null 2>&1 || true
done

# No shared bin/ of every env: only envs/main/bin goes on PATH (last), so no env's python can
# shadow the image's. Any other env is reached by absolute path.
log "verifying the main env"
for exe in nextflow bwa-mem2 samtools bcftools fastp gatk rtg multiqc; do
    if [[ -x "$PREFIX/envs/main/bin/$exe" || -x "$PREFIX/bin/$exe" ]]; then
        printf '  ok      %s\n' "$exe" >&2
    else
        printf '  MISSING %s\n' "$exe" >&2
        missing=1
    fi
done
[[ -z "${missing:-}" ]] || { echo "toolchain incomplete" >&2; exit 1; }

log "done: $PREFIX"
