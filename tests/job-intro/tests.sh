#!/usr/bin/env bash
set -euo pipefail

uv pip install -q --system emoji
pip install nbmake==1.5.5 pytest==9.0.2
pytest --nbmake . -s -vv
