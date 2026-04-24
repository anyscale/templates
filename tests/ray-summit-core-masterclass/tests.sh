#!/usr/bin/env bash
set -euo pipefail

# Install dependencies needed by the template
pip install papermill torch torchvision xgboost scikit-learn pillow tqdm matplotlib pandas pyarrow requests

# Run the first notebook to validate Ray Core remote functions work
papermill Ray_Core_1_Remote_Functions.ipynb output_1.ipynb --log-output

# Run a notebook covering Ray Actors to validate stateful functionality
papermill Ray_Core_3_Remote_Classes_part_1.ipynb output_3.ipynb --log-output

echo "Tests passed successfully!"
