#!/usr/bin/env bash
set -euo pipefail
# Test script for deployment-serve-llm template.
# Add template-specific smoke tests here.
echo "Running tests for deployment-serve-llm template..."

for nb in \
  "small-size-llm/README" \
  "medium-size-llm/README" \
  "large-size-llm/README" \
  "vision-llm/README" \
  "reasoning-llm/README" \
  "hybrid-reasoning-llm/README" \
  "gpt-oss/README"
do
  python nb2py.py "${nb}.ipynb" "${nb}.py" --ignore-cmds
  python "${nb}.py"
  rm "${nb}.py"
done
