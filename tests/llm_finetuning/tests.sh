#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill nbconvert==7.16.6 ipykernel

# Heavy llamafactory-cli train cells are tagged skip-in-ci; CI runs install + data prep.
# %%bash cells reference ../train-configs and ../dataset-configs, so run from notebooks/.
cd notebooks
for nb in dpo_qlora.ipynb kto_lora.ipynb sft_lora_deepspeed.ipynb cpt_deepspeed.ipynb; do
  base="${nb%.ipynb}"
  jupyter nbconvert --to notebook "${nb}" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags=skip-in-ci \
    --output "/tmp/${base}.ci.ipynb"
  papermill "/tmp/${base}.ci.ipynb" "/tmp/${base}.out.ipynb" --log-output --kernel python3 --cwd .
done
