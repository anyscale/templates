#!/bin/bash

set -euo pipefail

wget https://github.com/ray-project/rayci/releases/download/v0.25.0/rayapp-linux-amd64
chmod +x rayapp-linux-amd64
mv rayapp-linux-amd64 "$HOME/.local/bin/rayapp" && echo "Rayapp downloaded and installed"
