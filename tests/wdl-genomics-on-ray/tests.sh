#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system 'papermill==2.7.0'

# 2 Mbp per sample. The three assemblies need max_nodes: 3; with fewer workers they queue.
export WDL_DEMO_SCALE=quick

python "$(dirname "$0")/test_readout_cells.py"

python "$(dirname "$0")/test_backend_units.py"

papermill README.ipynb /tmp/wdl-genomics-on-ray.out.ipynb --log-output --kernel python3 --cwd .
