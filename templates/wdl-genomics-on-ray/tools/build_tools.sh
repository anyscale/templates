#!/usr/bin/env bash
# Install the bioinformatics toolchain described by manifest.toml into a prefix.
#
#   usage: build_tools.sh [PREFIX]        (default: /opt/wdl-tools)
#
# The Dockerfile runs this to bake the toolchain into the template's image, which is what
# `--container-runtime none` needs: that mode runs each WDL task's command directly in the Ray
# worker's environment, so every tool must already be on every node. Nothing is fetched at run
# time and nothing is installed per task.
#
# Versions, URLs and hashes come from manifest.toml and appear nowhere else. The *build
# recipes* for the two `kind = "source"` tools live here, in build_one(), because they are
# genuinely bespoke -- a `build =` string in the manifest would only be this shell code moved
# somewhere it can't be read as shell.
#
# Every download is checksum-verified before it is unpacked. A mismatch is fatal and leaves
# nothing behind.
set -euo pipefail

PREFIX="${1:-/opt/wdl-tools}"
MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manifest.toml"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$PREFIX/bin" "$PREFIX/lib"

# Emit one `name|kind|url|sha256|strip` record per tool. Reading TOML with the stdlib keeps
# this dependency-free: tomllib is in the standard library from Python 3.11, and the base
# image is 3.12.
records() {
  python3 - "$MANIFEST" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    manifest = tomllib.load(fh)
for name, spec in manifest["tools"].items():
    print("|".join([
        name,
        spec["kind"],
        spec.get("url", ""),
        spec.get("sha256", ""),
        str(spec.get("strip", 0)),
    ]))
PY
}

# Space-separated `pathInsidePayload=nameOnPath` pairs for one tool.
provides_of() {
  python3 - "$MANIFEST" "$1" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    manifest = tomllib.load(fh)
for on_path, inside in manifest["tools"][sys.argv[2]].get("provides", {}).items():
    print(f"{inside}={on_path}")
PY
}

requirements_of() {
  python3 - "$MANIFEST" "$1" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    manifest = tomllib.load(fh)
print(" ".join(manifest["tools"][sys.argv[2]].get("requirements", [])))
PY
}

fetch() {
  local url="$1" sha="$2" dest="$3"
  echo "  fetch $url"
  curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"
  echo "${sha}  ${dest}" | sha256sum -c - >/dev/null \
    || { echo "FATAL: sha256 mismatch for $url" >&2; exit 1; }
}

# Unpack into a fresh directory, honouring the manifest's `strip`.
unpack() {
  local archive="$1" into="$2" strip="$3"
  mkdir -p "$into"
  tar -xf "$archive" -C "$into" --strip-components="$strip"
}

build_one() {
  local name="$1" kind="$2" url="$3" sha="$4" strip="$5"
  local src="$WORK/$name" archive="$WORK/$name.archive"

  echo "== $name ($kind)"

  case "$kind" in
    binary)
      fetch "$url" "$sha" "$archive"
      unpack "$archive" "$PREFIX/lib/$name" "$strip"
      ;;

    source)
      fetch "$url" "$sha" "$archive"
      unpack "$archive" "$src" "$strip"
      case "$name" in
        samtools)
          # No --with-htslib flag: configure's default search finds the htslib-1.21 source
          # bundled inside this exact release tarball, which is the version-matched copy the
          # manifest's comment describes. (There is no "builtin" keyword -- naming one makes
          # configure look for a directory literally called that, and fail.) curses is only
          # needed by `samtools tview`, which no task calls, and it drags in a dev package.
          ( cd "$src" \
            && ./configure --prefix="$PREFIX/lib/$name" --without-curses \
            && make -j"$(nproc)" \
            && make install )
          ;;
        bcftools)
          # Same shape as samtools: the release tarball bundles the matching htslib and
          # configure's default search builds it. --disable-bcftools-plugins because only
          # `norm` is called, and the plugins want a dlopen path resolved at run time that a
          # relocatable payload cannot promise.
          ( cd "$src" \
            && ./configure --prefix="$PREFIX/lib/$name" --disable-bcftools-plugins \
            && make -j"$(nproc)" \
            && make install )
          ;;
        flye)
          # Flye is a Python distribution with C++ submodules. Installing it with pip puts
          # `flye` on the environment's PATH directly, so no shim is needed -- but the shim
          # loop below is still driven off `provides`, so point it at what pip produced.
          ( cd "$src" && pip install --no-cache-dir . )
          mkdir -p "$PREFIX/lib/$name/bin"
          ln -sf "$(command -v flye)" "$PREFIX/lib/$name/bin/flye"
          ;;
        *)
          echo "FATAL: no build recipe for source tool '$name'" >&2
          exit 1
          ;;
      esac
      ;;

    pypi)
      # shellcheck disable=SC2046  # deliberate word splitting: one pip arg per requirement
      pip install --no-cache-dir $(requirements_of "$name")
      mkdir -p "$PREFIX/lib/$name/bin"
      ;;

    *)
      echo "FATAL: unknown kind '$kind' for '$name'" >&2
      exit 1
      ;;
  esac

  # Put each provided name on PATH. For `pypi` the target is already in the environment's
  # bin, so resolve it there rather than inside a payload that doesn't exist.
  while IFS='=' read -r inside on_path; do
    [ -n "$inside" ] || continue
    local target
    if [ "$kind" = "pypi" ]; then
      target="$(command -v "$(basename "$inside")" || true)"
    else
      target="$PREFIX/lib/$name/$inside"
    fi
    if [ -z "$target" ] || [ ! -e "$target" ]; then
      echo "FATAL: $name declares '$on_path' but $inside is missing after the build" >&2
      exit 1
    fi
    chmod +x "$target" 2>/dev/null || true
    ln -sf "$target" "$PREFIX/bin/$on_path"
  done < <(provides_of "$name")
}

while IFS='|' read -r name kind url sha strip; do
  build_one "$name" "$kind" "$url" "$sha" "$strip"
done < <(records)

# Verify every declared name actually runs. A tool that installed but cannot execute -- a
# missing shared library is the usual cause -- would otherwise surface as a failed WDL task
# an hour into a run.
echo "== verify"
export PATH="$PREFIX/bin:$PATH"
for exe in minimap2 "paftools.js" samtools bcftools flye quast; do
  command -v "$exe" >/dev/null || { echo "FATAL: $exe not on PATH" >&2; exit 1; }
  echo "  $exe -> $(command -v "$exe")"
done
minimap2 --version >/dev/null
samtools --version >/dev/null
bcftools --version >/dev/null
flye --version >/dev/null
quast --version >/dev/null

echo "toolchain installed to $PREFIX"
