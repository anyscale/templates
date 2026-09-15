#!/usr/bin/env bash
set -euxo pipefail

# Pinned: an unpinned test dependency makes "the template broke" and "papermill changed"
# the same red build, and only one of those is worth waking up for.
uv pip install -q --system 'papermill==2.7.0'

# The notebook's default is `standard` -- three samples x 10 Mbp, 60-90 minutes -- which is what
# a reader gets and is deliberately more than CI can spend. The Buildkite step for a template
# test is capped at 75 minutes regardless of BUILD.yaml's timeout_in_sec, so CI runs the same
# notebook over a 2 Mbp region per sample instead. Same pipeline, same tools, same resource
# requests, same cohort shape: only the number of bases differs.
#
# The cohort is still three concurrent assemblies, so this leans on the compute config's
# max_nodes: 3. With fewer nodes the samples assemble serially and the step gets close to the
# cap -- if this starts timing out, check the cluster scaled before assuming the pipeline slowed.
export WDL_DEMO_SCALE=quick

# Offline first, and deliberately before anything expensive. This runs the notebook's readout
# cells against a synthetic outputs.json in the shape ONTAssembleCohort.wdl declares, so a
# mismatch between a WDL output name and what the notebook reads fails in seconds rather than
# after the assemblies. The QUAST metric-name bug this guards against survived undetected
# because CI never reached the cell that would have raised it.
python "$(dirname "$0")/test_readout_cells.py"

papermill README.ipynb /tmp/wdl-genomics-on-ray.out.ipynb --log-output --kernel python3 --cwd .
