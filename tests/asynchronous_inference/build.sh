#!/bin/bash

set -exo pipefail

uv pip install -r ../../templates/asynchronous_inference/python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
