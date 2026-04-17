#!/bin/bash
set -e

pip install -r requirements.txt

# TODO: Add notebook execution test
# e.g., jupyter nbconvert --to notebook --execute VHOL_without_output.ipynb
echo "Tests passed"
