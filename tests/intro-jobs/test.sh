#!/bin/bash

set -euo pipefail

pip install pytest==9.0.2 nbmake==1.5.5

pytest --nbmake . -s -vv
