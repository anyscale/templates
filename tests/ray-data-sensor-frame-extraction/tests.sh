#!/usr/bin/env bash
# CI test for ray-data-sensor-frame-extraction: runs the notebook top to bottom, as a user does.
#
# The notebook asserts two signs and no magnitudes: binary(N) writes faster than list<uint8>
# (fixture cell) and reads faster end to end (A/B cell). The measured ratios, with Ray version
# and run counts, are in the README. A threshold here would depend on the hardware and go
# stale; an inversion means the fixture, the codec or the hardware changed enough to re-measure.
#
# CI runs each arm once, which gives only a direction. measure_layout.py takes replicates for
# magnitudes and is not run here.
set -euo pipefail

uv pip install -q --system papermill

# Change the frame count, not the geometry: every README figure uses 16.7 MB frames. Exported
# so an interactive run of the notebook reads the same variables.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/sensor-fixture}"
export FRAMES="${FRAMES:-24}"
export FILES="${FILES:-4}"

papermill README.ipynb /tmp/ray-data-sensor-frame-extraction.out.ipynb \
    --log-output --kernel python3 --cwd .
