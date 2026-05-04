#!/usr/bin/env bash
set -euo pipefail

# Convert and run the notebook
python nb2py.py "README.ipynb" "README.py"
python "README.py"
rm "README.py"
