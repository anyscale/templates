#!/usr/bin/env bash
# CI test for ray-data-sensor-frame-extraction.
#
# Runs the notebook top to bottom, the way a user does.
#
# The template's thesis -- that the READ binds, and that the recommended Parquet layout wins --
# is asserted INSIDE the notebook, and as a DIRECTION rather than a wall-clock number. A
# threshold would be fleet-dependent and would go stale; the shape is what the template
# actually claims. So this fails loudly in exactly the case that matters: when the shape
# inverts on the CI fleet and the template would be teaching something false.
set -euo pipefail

uv pip install -q --system papermill

# The fixture is ~16.7 MB a row by design -- that is the regime where per-value Parquet
# bookkeeping dominates, and it is where the whole lesson lives. Shrink the frame COUNT for
# CI, never the geometry.
#
# These are exported rather than inlined because the notebook reads the same variables for an
# interactive run: this is the template's configuration, not test scaffolding.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/sensor-fixture}"
export FRAMES="${FRAMES:-24}"
export FILES="${FILES:-4}"

papermill README.ipynb /tmp/ray-data-sensor-frame-extraction.out.ipynb \
    --log-output --kernel python3 --cwd .
