#!/usr/bin/env bash
# CI test for ray-data-multimodel-frame-enrichment.
#
# Runs README.ipynb with papermill, minus the cells tagged skip-in-ci.
#
# NO SHARED TOKEN. SAM 3 and DINOv3 are gated on Hugging Face and their terms are accepted per
# account by whoever runs the template. Do not add an organisation secret here, and do not copy
# tests/vla-fine-tuning/tests.sh, which reads one for its own gated model.
#
# Do not replace the tag with "skip the gated stages when HF_TOKEN is unset". An inferred skip
# is an untested branch that becomes the only branch that runs, and the job stays green while
# asserting nothing. pipeline.py --ungated-only is a flag and never reads a credential.
#
# WHAT THIS RUN COVERS
#   - packing arithmetic under --strict: four models co-reside at 5.53 GiB on a 22.03 GiB L4,
#     no stage over-committed, relative-cost ordering intact. No GPU, no weights.
#   - stage answers, from tests/test_packing.py and tests/test_pipeline.py.
#   - scheduling and output shapes, from the four-stage --stub run.
#   - two models on one card with real weights: SigLIP2's vision tower (ungated; anonymous
#     config.json and model.safetensors both 200, measured 2026-08-19) and the weightless
#     metrics stage, at the fractional GPU reservations the template ships. This is also the
#     only path that shows the runtime env reaching the actors.
#
# WHAT IT DOES NOT COVER
#   - four models on one GPU. The gated pair is absent. That claim rests on the arithmetic
#     above plus the reader's own run.
#   - the >=26.5% co-residency margin. One run per arm cannot separate 26% from a
#     session-warmup artifact; two of the three measurement rounds came out NOT SEPARABLE on
#     their own full data. measure_packing.py is the harness, with replicates.
#
# The resident-model count drops to two here because two models are licensed to the reader and
# not to CI. Do not read that as licence to trim stages to make a test cheaper: make_fixture.py
# still forbids shrinking the count to fit a budget.
set -euxo pipefail

# CI shrink. The notebook reads these; its defaults are the demo. Frame count and geometry are
# what may shrink.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/frames-fixture}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cluster_storage/frames-enriched}"
export FRAMES=24 FILES=4 WIDTH=640 HEIGHT=480

# One actor per stage, which fits an L4. The shipped counts need the 48 GiB class; see
# packing.py. The notebook applies these with setdefault, so what is exported here wins.
export DETECTOR_ACTORS=1 EMB_ACTORS=1 METRICS_ACTORS=1
export DETECTOR_BATCH=2 EMB_BATCH=8 METRICS_BATCH=8 METRICS_SUBBATCH=2

# Do not install the template's lock here. The notebook installs it partway through, where a
# user would, and the cells above that point run on the bare image: it ships numpy, pyarrow and
# ray, and no torch, torchvision or transformers (measured; -cu129 is the CUDA runtime, not
# PyTorch). Installing it here would test something the user never runs.
#
# Use uv pip, never bare pip. A workspace tracks a bare install and appends it unpinned to every
# actor's pip list; one unhashed entry puts pip in hash-checking mode against a hashed lock and
# every runtime env for this template then fails to build. check-dep-delivery.py bare-pip
# enforces this.
uv pip install -q --system papermill "nbconvert==7.16.6" ipykernel

# Strip the gated cells into CI's own copy. The committed notebook keeps them all.
#
# The JSON-list spelling of remove_cell_tags below was checked on nbconvert 7.16.6 and 7.17.1;
# so was the bare-string form. Both remove the tagged cell and leave untagged cells alone.
jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb

# A strip that silently no-ops would send this job at gated weights with no token, and it would
# fail on a 401 that reads like an infrastructure error. An over-strip would pass while touching
# no GPU. Check for both.
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

# --cwd . so the notebook's relative paths resolve; --log-output streams to the CI log and the
# .out.ipynb is what you read afterwards.
#
# Before believing a pass, check the log's `Executing Cell N` and `Ending Cell N` counts match.
# rayapp has reported success for a run whose SSH session dropped mid-notebook.
papermill /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb \
    /tmp/ray-data-multimodel-frame-enrichment.out.ipynb \
    --log-output --kernel python3 --cwd .
