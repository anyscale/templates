#!/usr/bin/env bash
# CI test for ray-data-multimodel-frame-enrichment.
#
# THERE IS NO SHARED TOKEN FOR THIS TEMPLATE, AND THAT IS A LICENSING FACT RATHER THAN A GAP.
# SAM 3 and DINOv3 are gated on Hugging Face and their terms are accepted PER ACCOUNT by the
# person who runs the template. A service account cannot agree to a licence on a reader's
# behalf, so this script does not fetch an organisation secret, and `templates/vla-fine-tuning`
# -- which does, for its own gated model -- is deliberately NOT the pattern followed here.
#
# So CI runs the half of the notebook that needs nobody's permission, and the gated cells are
# removed by TAG rather than skipped by inference. That distinction is the whole design:
#
#   * DECLARED  the cells are tagged `skip-in-ci` in README.ipynb and stripped below. What CI
#               runs is visible in the notebook and in this file.
#   * INFERRED  "if HF_TOKEN is unset, skip the gated stages" -- which is what NOT to do. An
#               inferred skip is an untested branch that silently becomes the only branch that
#               ever runs, and the job goes green forever while asserting nothing about the
#               thesis. `pipeline.py --ungated-only` is likewise a declared flag: it never
#               consults a credential.
#
# WHAT THIS RUN PROVES
#
#   * the packing arithmetic, under `--strict`: four models CAN co-reside on a 22.03 GiB L4
#     at 5.53 GiB, no stage is over-committed, and the relative-cost ordering holds. That is
#     the four-model claim's feasibility half, and it needs no GPU or weights at all.
#   * the stage answers, from the two unit-test files: per-frame scatter, the degenerate
#     empty-detection path, the SAM 3 prompt shape against a recording fake, and the
#     `get_image_features` return type that shipped broken once.
#   * the DAG schedules and every output shape is right, from the four-stage `--stub` run.
#   * REAL two-model co-residency on one card: SigLIP2's vision tower (ungated, 200
#     anonymously -- measured) plus the weightless metrics stage, with the fractional GPU
#     reservations the template ships, real weights, real embeddings, finite scores. This
#     also proves the runtime env reached the actors, which the driver install cannot.
#
# WHAT THIS RUN DOES NOT PROVE, AND NOTHING HERE CAN
#
#   * four models on one GPU, measured. The gated pair is absent. That claim rests on the
#     arithmetic above plus the reader's own run, and `README.ipynb` carries the cells for it.
#   * the >=26.5% co-residency margin. Never asserted in CI in any case: one run per arm
#     cannot separate 26% from a session-warmup artifact -- two of the three measurement
#     rounds were NOT SEPARABLE on their own full data for exactly that reason. Assert
#     answers in CI; measure ratios where you can replicate. `measure_packing.py` is the
#     harness and it ships with the template.
#
# THE RESIDENT-MODEL COUNT DROPS TO TWO HERE, AND THAT IS NEW. An earlier version of this file
# forbade exactly that: "the NUMBER OF RESIDENT MODELS may not [shrink] -- dropping to two
# stages to fit a budget would make the template demonstrate something it does not claim."
# That prohibition was about SHRINKING TO FIT A BUDGET and it still stands. This is not that.
# The count drops because two of the four models are licensed to the reader and not to CI, and
# the run announces which pair it got rather than quietly reporting a smaller number under the
# same name. Do not use this as precedent for trimming stages to make a test cheaper.
set -euxo pipefail

# CI shrink. The notebook's defaults are the demo: 48 frames, and the shipped actor counts
# from packing.py. The notebook reads these from the environment, so nothing in it is edited
# to run smaller. The frame COUNT and the frame GEOMETRY are what may shrink.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/frames-fixture}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cluster_storage/frames-enriched}"
export FRAMES=24 FILES=4 WIDTH=640 HEIGHT=480

# One actor per stage, which is what fits an L4. The shipped counts (10 detectors) need the
# 48 GiB class -- see packing.py. The notebook applies these with setdefault, so what is
# exported here wins.
export DETECTOR_ACTORS=1 EMB_ACTORS=1 METRICS_ACTORS=1
export DETECTOR_BATCH=2 EMB_BATCH=8 METRICS_BATCH=8 METRICS_SUBBATCH=2

# The ONLY install this script does, and it deliberately adds no torch. The notebook installs
# the template's own python_depset.lock partway through, at the point a user would -- and the
# cells above that point run on the BARE IMAGE, which is load-bearing rather than incidental:
# the base image ships numpy, pyarrow and ray and no torch, torchvision or transformers at
# all (measured; `-cu129` is the CUDA runtime, not PyTorch). Installing the lock here instead
# would destroy that property and test something the user never runs.
#
# `uv pip`, never a bare `pip install`: a workspace tracks a bare install and appends it
# UNPINNED to every actor's pip list, and one unhashed entry puts pip in hash-checking mode
# against a hashed lock, which fails every runtime env for this template.
# scripts/hooks/check-dep-delivery.py bare-pip enforces it.
uv pip install -q --system papermill "nbconvert==7.16.6" ipykernel

# Strip the gated cells. The JSON-list spelling of the tag, which is what 13 of the 16
# tests.sh using this mechanism write; both it and the bare-string form were verified on
# nbconvert 7.16.6 and 7.17.1 to remove exactly the tagged cell and nothing else, with a cell
# tagged `s` left in place as the control against character-wise coercion.
#
# The committed notebook keeps every cell. This strip is CI's copy only, so a reader still
# gets the four-model path.
jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb

# Belt and braces: if the tag ever stops matching -- renamed, mistyped, or the preprocessor
# flag silently ignored -- the strip becomes a no-op and this job would try to pull gated
# weights with no token, then fail on a 401 that reads like an infrastructure problem. Refuse
# now, with the reason.
python - <<'PY'
import json
import sys

nb = json.load(open("/tmp/ray-data-multimodel-frame-enrichment.ci.ipynb"))
left = [i for i, c in enumerate(nb["cells"]) if "skip-in-ci" in c["metadata"].get("tags", [])]
if left:
    sys.exit(f"the skip-in-ci strip did not run: cells {left} still carry the tag. CI must "
             "not execute the gated cells -- SAM 3 and DINOv3 are licensed per account and "
             "this job holds no token.")
code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
if code < 6:
    sys.exit(f"only {code} code cells survived the strip; the ungated run is missing, so a "
             "pass here would assert nothing on a GPU.")
print(f"stripped notebook: {code} code cells, no skip-in-ci tags left")
PY

# Mimic the user: the whole (stripped) notebook, top to bottom. `--cwd .` runs it from the
# template dir so its relative paths (packing.py, python_depset.lock, tests/) resolve;
# `--log-output` streams to the CI log and the .out.ipynb is what you read afterwards.
#
# Check the log's `Executing Cell N` and `Ending Cell N` counts MATCH before believing a
# pass: rayapp has reported success for a run whose SSH session dropped mid-notebook.
papermill /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb \
    /tmp/ray-data-multimodel-frame-enrichment.out.ipynb \
    --log-output --kernel python3 --cwd .
