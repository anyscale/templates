#!/usr/bin/env bash
# CI test for ray-data-multimodel-frame-enrichment: papermill README.ipynb, minus the cells tagged
# skip-in-ci.
#
# No shared token. SAM 3 and DINOv3 are gated on Hugging Face and their terms are accepted per
# account by whoever runs the template. Do not add an organisation secret here, and do not copy
# tests/vla-fine-tuning/tests.sh, which reads one for its own gated model.
#
# Keep the gated cells out by tag. Do not replace the tag with "skip the gated stages when
# HF_TOKEN is unset": an inferred skip is an untested branch that ends up being what runs, and
# the job stays green while asserting nothing. pipeline.py --ungated-only is a flag and never
# reads a credential.
#
# Covers:
#   - packing arithmetic under --strict: one actor of each of this template's stages holds
#     5.53 GiB of a 22.03 GiB L4, and no stage is over-committed. No GPU, no weights. (The
#     cost-ordering check does not run on the measured table.)
#   - stage answers, from tests/test_packing.py and tests/test_pipeline.py.
#   - scheduling and output shapes, from the four-stage --stub run.
#   - two stages sharing one GPU at the shipped fractional reservations: SigLIP2's vision tower
#     with real weights (ungated; anonymous config.json and model.safetensors both 200,
#     measured 2026-08-19) and the weightless metrics stage. The actors run on the GPU worker,
#     which never ran the driver's install, so a pass also shows the runtime env delivered the
#     lock.
#
# Does not cover:
#   - the four-stage run. It needs a token whose terms only a person can accept; the fit rests
#     on the arithmetic above plus the reader's own run.
#   - the >=26.5% co-residency margin. One run per arm cannot separate 26% from a
#     session-warmup artifact; measure_packing.py measures it with replicates.
#
# CI runs two of the four stages only because the other two are licensed per account. Do not
# trim stages to make the test cheaper; shrink the frame count or geometry instead.
#
# The pass records and the timeout measurement are in BUILD.yaml.
set -euxo pipefail

# Shrink the run. The notebook reads these, and its own defaults are the demo size.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/frames-fixture}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cluster_storage/frames-enriched}"
export FRAMES=24 FILES=4 WIDTH=640 HEIGHT=480

# One actor per stage fits an L4; pipeline.py's defaults need more than one GPU (see
# packing.py). The notebook applies these with setdefault, so what is exported here wins.
export DETECTOR_ACTORS=1 EMB_ACTORS=1 METRICS_ACTORS=1
export DETECTOR_BATCH=2 EMB_BATCH=8 METRICS_BATCH=8 METRICS_SUBBATCH=2

# Do not install the template's lock here. The notebook installs it partway through, as a user
# would, and the cells above that point must run on the bare image: numpy, pyarrow and ray, no
# torch, torchvision or transformers (measured on 2.57.0; the 2.58.0 freeze lists the same).
# Installing it here would test something the user never runs.
#
# Use uv pip, never bare pip. A workspace tracks a bare install and appends it unpinned to every
# actor's pip list; one unhashed entry puts pip in hash-checking mode against the hashed lock,
# and every runtime env for this template then fails to build. check-dep-delivery.py bare-pip
# enforces this.
uv pip install -q --system papermill "nbconvert==7.16.6" ipykernel

# Strip the gated cells into CI's own copy; the committed notebook keeps them all. This
# JSON-list spelling of remove_cell_tags, and the bare-string form, both checked on nbconvert
# 7.16.6 and 7.17.1, remove the tagged cells and leave untagged cells alone.
jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb

# If the strip silently no-ops, this job goes at gated weights with no token and fails on a 401
# that reads like an infrastructure error. An over-strip would pass without touching the GPU.
# Check for both.
python - <<'PY'
import json
import sys

nb = json.load(open("/tmp/ray-data-multimodel-frame-enrichment.ci.ipynb"))
left = [i for i, c in enumerate(nb["cells"]) if "skip-in-ci" in c["metadata"].get("tags", [])]
if left:
    sys.exit(f"skip-in-ci strip did not run: cells {left} still carry the tag. CI holds no "
             "token for SAM 3 or DINOv3 and must not execute those cells.")
code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
if code < 6:
    sys.exit(f"only {code} code cells survived the strip; the ungated GPU run is missing.")
print(f"stripped notebook: {code} code cells, no skip-in-ci tags left")
PY

# --cwd . so the notebook's relative paths resolve; --log-output streams to the CI log, and the
# .out.ipynb is what you read afterwards.
#
# Before believing a pass, check that the log's `Executing Cell N` and `Ending Cell N` counts
# match: rayapp has reported success for a run whose SSH session dropped mid-notebook.
papermill /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb \
    /tmp/ray-data-multimodel-frame-enrichment.out.ipynb \
    --log-output --kernel python3 --cwd .
