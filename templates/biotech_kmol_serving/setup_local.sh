#!/usr/bin/env bash
# Local dev setup: clone kMoL (NOT vendored) and make it importable.
#
# kMoL's supported install is its conda env (see kmol/environment.yml). This script
# clones the pinned commit and sets PYTHONPATH so `import kmol` resolves without
# compiling the CUDA extension — enough to iterate on the wrapper locally.
#
# For a full runtime (RDKit/PyG/openbabel), create kMoL's conda env:
#   cd kmol && make create-env && conda activate kmol
set -euo pipefail

KMOL_REF="${KMOL_REF:-c7f8833}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${HERE}/kmol" ]; then
  echo "-> cloning elix-tech/kmol @ ${KMOL_REF}"
  git clone https://github.com/elix-tech/kmol.git "${HERE}/kmol"
  git -C "${HERE}/kmol" checkout "${KMOL_REF}"
else
  echo "-> kmol/ already present, skipping clone"
fi

echo
echo "Add kMoL to your PYTHONPATH for this shell:"
echo "  export PYTHONPATH=${HERE}/kmol/src:${HERE}"
echo
echo "Then run locally (needs kMoL's conda env active + your 5 checkpoints in place):"
echo "  serve run serve_app:app"
