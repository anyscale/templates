#!/usr/bin/env bash
set -euxo pipefail

# Pinned: an unpinned test dependency makes "the template broke" and "papermill changed"
# the same red build, and only one of those is worth waking up for.
uv pip install -q --system 'papermill==2.7.0'

# The notebook's default is `standard` (three samples x 10 Mbp, about 17 min end to end), which
# is what a reader gets. CI runs the same notebook over a 2 Mbp region per sample instead to
# keep the test short: same pipeline, tools, resource requests and cohort shape, fewer bases.
# The Buildkite step allows BUILD.yaml's timeout_in_sec plus a 30 min startup allowance.
#
# The cohort is still three concurrent assemblies, so this leans on the compute config's
# max_nodes: 3. With fewer nodes the assemblies queue behind each other (13m16s at max_nodes 2,
# against 8m05s at 3); if this starts timing out, check the cluster scaled before assuming the
# pipeline slowed.
export WDL_DEMO_SCALE=quick

# Offline first, and deliberately before anything expensive. This runs the notebook's readout
# cells against a synthetic outputs.json in the shape ONTAssembleCohort.wdl declares, so a
# mismatch between a WDL output name and what the notebook reads fails in seconds rather than
# after the assemblies.
python "$(dirname "$0")/test_readout_cells.py"

papermill README.ipynb /tmp/wdl-genomics-on-ray.out.ipynb --log-output --kernel python3 --cwd .
