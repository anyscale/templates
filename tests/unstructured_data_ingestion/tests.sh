#!/usr/bin/env bash
set -euo pipefail

python nb2py.py README.ipynb unstructured_data_ingestion.py
python unstructured_data_ingestion.py
rm unstructured_data_ingestion.py
