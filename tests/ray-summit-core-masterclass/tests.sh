#!/usr/bin/env bash
set -euo pipefail

# nbconvert (last) is needed for the NB6 tag-strip below.
pip install papermill torch torchvision xgboost scikit-learn pillow tqdm matplotlib pandas pyarrow requests nbconvert

# Notebooks 1-5 run end-to-end as-is.
for nb in \
  Ray_Core_1_Remote_Functions \
  Ray_Core_2_Remote_Objects \
  Ray_Core_3_Remote_Classes_part_1 \
  Ray_Core_4_Remote_Classes_part_2 \
  Ray_Core_5_Best_Practices; do
  papermill "${nb}.ipynb" "/tmp/${nb}.out.ipynb" --log-output --kernel python3 --cwd .
done

# NB6's BONUS #2 GPU-GPU cells need cross-node NCCL on 2 single-GPU nodes —
# unstable on Ray 2.55.1. Strip those skip-in-ci cells, then run the rest.
jupyter nbconvert --to notebook Ray_Core_6_ADAG_experimental.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output-dir /tmp --output nb6.ci
papermill /tmp/nb6.ci.ipynb /tmp/nb6.out.ipynb --log-output --kernel python3 --cwd .

echo "Tests passed successfully!"
