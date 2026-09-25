#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system 'papermill==2.7.0'

# CI runs the notebook at `quick` scale, 2 Mbp per sample. The three assemblies need the compute
# config's max_nodes: 3; with fewer workers they queue, and a timeout may mean the cluster did not
# scale rather than a slower pipeline.
export WDL_DEMO_SCALE=quick

# Checks the readout cells against a synthetic outputs.json in seconds, before any assembly runs.
python "$(dirname "$0")/test_readout_cells.py"

papermill README.ipynb /tmp/wdl-genomics-on-ray.out.ipynb --log-output --kernel python3 --cwd .
