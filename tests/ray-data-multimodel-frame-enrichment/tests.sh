#!/usr/bin/env bash
# CI: papermill README.ipynb minus the skip-in-ci cells. SAM 3 and DINOv3 are licensed per
# account, so CI holds no token and skips the four-stage run and the co-residency measurement.
# It runs the unit tests, packing.py --strict, the --stub DAG, and SigLIP2 (ungated: anonymous
# downloads return 200, checked 2026-08-19) with metrics on a GPU worker, which also shows the
# runtime env delivered the lock. Keep the gated cells out by tag, not by a skip when HF_TOKEN
# is unset, and add no organisation secret.
set -euxo pipefail

# Shrink the run by frames, never by stages. The notebook reads these.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/frames-fixture}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cluster_storage/frames-enriched}"
export FRAMES=24 FILES=4 WIDTH=640 HEIGHT=480

# One actor per stage fits an L4. The notebook applies these with setdefault.
export DETECTOR_ACTORS=1 EMB_ACTORS=1 METRICS_ACTORS=1
export DETECTOR_BATCH=2 EMB_BATCH=8 METRICS_BATCH=8 METRICS_SUBBATCH=2

# Not the lock: the notebook installs it partway through, as a user would. uv, never bare pip,
# which a workspace appends unpinned to every actor's pip list (check-dep-delivery.py bare-pip).
uv pip install -q --system papermill "nbconvert==7.16.6" ipykernel

# Strip the gated cells into CI's own copy; the committed notebook keeps them.
jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb

# Fail if the strip no-ops (the gated cells would 401) or over-strips (no GPU run left).
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

# Trust a pass only if the log's `Executing Cell N` and `Ending Cell N` counts match: rayapp has
# reported success for a run whose SSH session dropped mid-notebook.
papermill /tmp/ray-data-multimodel-frame-enrichment.ci.ipynb \
    /tmp/ray-data-multimodel-frame-enrichment.out.ipynb \
    --log-output --kernel python3 --cwd .
