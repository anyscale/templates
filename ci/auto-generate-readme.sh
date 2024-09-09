#!/bin/bash
echo "Auto-generating README files..."

# Search for notebook files named README.ipynb in the ../templates directory
NOTEBOOK_FILES=($(find ../templates -name "README.ipynb"))

# Loop through each notebook file
for NOTEBOOK_FILE in "${NOTEBOOK_FILES[@]}"; do
    # Exclude specific notebooks from conversion
    if [[ "$NOTEBOOK_FILE" != "../templates/templates/getting-started/README.ipynb" && "$NOTEBOOK_FILE" != "../templates/templates/e2e-llm-workflows/README.ipynb" ]]; then
        if ! grep -q "Time to complete" "$NOTEBOOK_FILE" ; then
            echo "**********"
            echo "LINT ERROR: $NOTEBOOK_FILE must include 'Time to complete' statement, failing."
            echo "**********"
            exit 1
        fi
    fi

    if [[ "$NOTEBOOK_FILE" != "../templates/templates/e2e-llm-workflows/README.ipynb" ]]; then
        # Convert notebook file to README.md using nbconvert
        jupyter nbconvert --to markdown "$NOTEBOOK_FILE" --embed-images --output-dir "$(dirname "$NOTEBOOK_FILE")"
    else
        echo "Skipping README generation for $NOTEBOOK_FILE"
    fi
done
