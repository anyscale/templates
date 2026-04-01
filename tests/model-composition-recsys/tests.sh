#!/usr/bin/env bash
set -euo pipefail

python nb2py.py "README.ipynb" "README.py"
python "README.py"
rm "README.py"
