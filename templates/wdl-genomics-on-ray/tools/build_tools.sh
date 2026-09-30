#!/usr/bin/env bash
# Install the toolchain in manifest.toml into PREFIX (default /opt/wdl-tools), checksum-verified.
# The Dockerfile runs it: under --container-runtime none every tool must be on every node.
set -euo pipefail

PREFIX="${1:-/opt/wdl-tools}"
MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/manifest.toml"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$PREFIX/bin" "$PREFIX/lib"

# One name|kind|url|sha256|strip record per tool, read with the stdlib's tomllib.
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
          # No --with-htslib: the default search builds the bundled htslib ("builtin" would
          # name a directory). curses is only for `samtools tview`.
          ( cd "$src" \
            && ./configure --prefix="$PREFIX/lib/$name" --without-curses \
            && make -j"$(nproc)" \
            && make install )
          ;;
        bcftools)
          # --disable-bcftools-plugins: only norm is used, and plugins need a run-time dlopen path.
          ( cd "$src" \
            && ./configure --prefix="$PREFIX/lib/$name" --disable-bcftools-plugins \
            && make -j"$(nproc)" \
            && make install )
          ;;
        flye)
          # pip puts flye on PATH itself; the symlink only feeds the provides loop below.
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

  # pypi targets are already in the environment's bin, not in a payload.
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

# A tool that cannot execute (a missing shared library, say) would otherwise fail a task mid-run.
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
