#!/bin/bash

set -euxo pipefail

# Only test the core RAG data ingestion pipeline (NB01, NB02)
# Notebooks 03-07 require deploying a large LLM (4 L4 GPUs) which is too expensive for CI

ordered_notebook_names=(
  "01_(Optional)_Regular_Document_Processing_Pipeline"
  "02_Scalable_RAG_Data_Ingestion_with_Ray_Data"
)

for nb in "${ordered_notebook_names[@]}"; do
  python nb2py.py "notebooks/${nb}.ipynb" "notebooks/${nb}.py"
  (cd notebooks && python "${nb}.py")
  (cd notebooks && rm "${nb}.py")
done
