#!/bin/bash

set -exo pipefail

python nb2py.py "README.ipynb" "README.py" --ignore-cmds
python "README.py"
rm "README.py"
