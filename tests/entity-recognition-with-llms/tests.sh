#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill
papermill README.ipynb /tmp/entity-recognition-with-llms.out.ipynb --log-output --kernel python3 --cwd .
