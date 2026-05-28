#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill nbconvert==7.16.6

# NB4 cells 13/15/17 (`anyscale service deploy/status` + the leaked-token prod
# query) are tagged skip-in-ci. The local `serve run` test path (cells 7/9/11)
# still runs.
for nb in 1.object_detection_train 2.object_detection_batch_inference_eval 3.video_processing_batch_inference 4.object_detection_serve; do
  jupyter nbconvert --to notebook "${nb}.ipynb" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output "/tmp/${nb}.ci.ipynb"
  papermill "/tmp/${nb}.ci.ipynb" "/tmp/${nb}.out.ipynb" --log-output --kernel python3 --cwd .
done
