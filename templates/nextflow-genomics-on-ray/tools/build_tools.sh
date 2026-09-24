#!/usr/bin/env bash
#
# Install the pipeline's toolchain into a prefix.
#
#   bash tools/build_tools.sh /opt/nf-tools
#
# Two delivery mechanisms, for two different jobs:
#
#   manifest.toml   binary artifacts fetched by URL and verified by sha256.
#                   Just Nextflow -- see the file for why it is not from conda.
#
#   env.*.yml       conda environments. Everything else. A solve pins a whole
#                   dependency tree at once, and DeepVariant in particular *must*
#                   have its own environment because it pins python <3.11 while
#                   the cluster runs 3.12.
#
# Run by the Dockerfile, but standalone on purpose: `bash tools/build_tools.sh
# ~/nf-tools` reproduces the toolchain on any Linux box without building an image,
# which is how you check whether a tool version is the problem.
set -euo pipefail

PREFIX="${1:?usage: build_tools.sh <prefix>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PREFIX/bin" "$PREFIX/envs" "$PREFIX/share"

log() { printf '\n=== %s\n' "$*" >&2; }

# --- binary artifacts from manifest.toml -------------------------------------

log "manifest artifacts"

# tomllib is stdlib from 3.11, so this needs no dependency of its own -- which
# matters because this script runs before anything is installed.
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
        # Fatal, not a warning. This exists to catch a release artifact re-cut
        # under the same tag, which happens, and which would otherwise become a
        # silent toolchain change between two builds of "the same" image.
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

# --- conda environments -------------------------------------------------------

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
        # An --explicit lock is byte-reproducible: exact URLs plus hashes, no
        # solver involved. Preferred when present; tools/lock-envs.sh writes it.
        log "conda env '$env_name' (from lock)"
        micromamba create -y -p "$target" --file "$lock"
    else
        log "conda env '$env_name' (solving; no lock file)"
        micromamba create -y -p "$target" --file "$spec"
    fi

    micromamba clean --all --yes >/dev/null 2>&1 || true
done

# --- PATH ---------------------------------------------------------------------
#
# The Dockerfile puts $PREFIX/bin first on PATH and $PREFIX/envs/main/bin last,
# after the image's own /home/ray/anaconda3/bin, and nothing else. Last because
# the main env carries its own python, which must not shadow the image's. The
# other environments are reached by absolute path:
#
#   deepvariant  pins python <3.11, so it must never shadow the cluster's 3.12.
#                Called through tools/run_deepvariant.sh.
#
# That is the whole quarantine, and it is why this script does not symlink each
# environment's executables into a shared bin/. Putting every env on PATH would
# work right up until something resolved `python` to 3.10 and a Ray worker
# refused to start -- a failure that surfaces as a task timeout, several
# processes into a scatter, with nothing in the log naming the interpreter.
# The one wrapper that has to cross the quarantine boundary goes on PATH, so a
# process script can call it by name. It is a shim, not a tool: it unsets the
# caller's Python environment, asserts the interpreter it lands on really is 3.10,
# and execs DeepVariant.
install -m 0755 "$HERE/run_deepvariant.sh" "$PREFIX/bin/run_deepvariant.sh"

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
