#!/usr/bin/env bash
set -euo pipefail
pip install papermill

# Use a fresh service name per CI run. Anyscale services are cloud-pinned by
# their first-created name; reusing a stale name keeps the service stuck on
# its original cloud (which may have a different cert / lifecycle than the
# workspace's current default). A fresh name forces the new service onto the
# workspace's current cloud.
TIMESTAMP=$(date +%s)
SERVICE_NAME="my_service_${TIMESTAMP}"
trap "anyscale service terminate --name=${SERVICE_NAME} >/dev/null 2>&1 || true" EXIT

# Rewrite the service name in the files the notebook reads. .bak files let us
# restore originals after, so the on-disk state matches what users see in
# production.
sed -i.bak "s/^name: my_service$/name: ${SERVICE_NAME}/" service.yaml
sed -i.bak "/SERVICE_NAME = /s/my_service/${SERVICE_NAME}/" README.ipynb

papermill README.ipynb output.ipynb -k python3 --log-output

mv service.yaml.bak service.yaml
mv README.ipynb.bak README.ipynb
