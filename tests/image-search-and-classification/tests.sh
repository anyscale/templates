#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill

for nb in 01-Batch-Inference 02-Distributed-Training 03-Online-Serving; do
  papermill "notebooks/${nb}.ipynb" "/tmp/${nb}.out.ipynb" --log-output --kernel python3 --cwd notebooks
done
