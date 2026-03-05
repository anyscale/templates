#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${1:-configs/basic.yaml}"

if [ ! -f "$PROJECT_DIR/$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found in $PROJECT_DIR"
    echo "Usage: $0 [config_file]"
    echo "Available configs:"
    ls -1 "$PROJECT_DIR/configs/"
    exit 1
fi

echo "======================================"
echo "Deploying ResNet50 with Ray Serve"
echo "======================================"
echo "Config: $CONFIG_FILE"
echo ""

cd "$PROJECT_DIR"

echo "Starting Ray Serve deployment..."
serve run "$CONFIG_FILE"

echo ""
echo "Deployment complete!"
echo "Service endpoint: http://127.0.0.1:8000"
echo ""
echo "Test with:"
echo "curl -X POST http://127.0.0.1:8000 -H 'Content-Type: application/json' -d '{\"uri\": \"https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg\"}'"

