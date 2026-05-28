#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill nbconvert==7.16.6

# Strip the `# client_anyscale_service.py` cell (tagged skip-in-ci) — it requires a
# real prod endpoint + auth token from `anyscale service deploy`, not the localhost
# serve.run path the test exercises.
jupyter nbconvert --to notebook README.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output /tmp/model-composition-recsys.ci.ipynb
papermill /tmp/model-composition-recsys.ci.ipynb /tmp/model-composition-recsys.out.ipynb --log-output --kernel python3 --cwd .
