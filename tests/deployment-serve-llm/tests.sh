#!/usr/bin/env bash
set -euo pipefail
# Test script for deployment-serve-llm template.
# Add template-specific smoke tests here.
echo "Running tests for deployment-serve-llm template..."

for nb in \
  "small-size-llm/notebook" \
  "medium-size-llm/notebook" \
  "large-size-llm/notebook" \
  "vision-llm/notebook" \
  "reasoning-llm/notebook" \
  "hybrid-reasoning-llm/notebook" \
  "gpt-oss/notebook"
do
  python nb2py.py "${nb}.ipynb" "${nb}.py" --ignore-cmds
  python "${nb}.py"
  rm "${nb}.py"
done
