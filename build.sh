#!/bin/bash

set -euo pipefail

echo "--- Building maketmpl"
mkdir -p ci/bin
(cd ci; go build -o bin/maketmpl ./maketmpl/maketmpl)

echo "--- Building template releases"
rm -rf _build
exec ci/bin/maketmpl
