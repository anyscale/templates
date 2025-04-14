#!/bin/bash

set -euo pipefail

echo "Auto-generating README files..."

REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ "$(pwd)" != "${REPO_ROOT}" ]]; then
	echo "Must run this script at repo's root directory".
	exit 1
fi

# Search for notebook files named README.ipynb in the ../templates directory
TEMPLATES_DIRS=($(find "templates" -mindepth 1 -maxdepth 1 -type d))

# Define the repo prefix
REPO_PREFIX="https://raw.githubusercontent.com/anyscale/templates/main"

# Loop through each notebook file
for TMPL in "${TEMPLATES_DIRS[@]}"; do
	echo "===== Processing ${TMPL}"

	if [[ ! -f "${TMPL}/README.ipynb" ]]; then
		echo "README.ipynb file not found; skipping notebook conversion and checking."
	else
        # Exclude specific notebooks from conversion
        TMPL_NAME="$(basename "${TMPL}")"
        NOTEBOOK_FILE="${TMPL}/README.ipynb"

        if [[ "${TMPL_NAME}" == "getting-started" || "${TMPL_NAME}" == "e2e-llm-workflows" || "${TMPL_NAME}" == "ray-summit-multi-modal-search" || "${TMPL_NAME}" == "image-search-and-classification" || "${TMPL_NAME}" == "entity-recognition-with-llms" ]]; then
            echo "Skip 'Time to complete' checking for ${TMPL_NAME}"
        elif ! grep -q "Time to complete" "${NOTEBOOK_FILE}" ; then
            echo "**********"
            echo "LINT ERROR: ${NOTEBOOK_FILE} must include 'Time to complete' statement, failing."
            echo "**********"
            exit 1
        fi

        if [[ "${TMPL_NAME}" != "e2e-llm-workflows" && "${TMPL_NAME}" != "image-search-and-classification" && "${TMPL_NAME}" != "entity-recognition-with-llms" ]]; then
            # Convert notebook file to README.md using nbconvert
            jupyter nbconvert --to markdown "${NOTEBOOK_FILE}" --output-dir "${TMPL}"
        else
            echo "Skipping README generation for ${NOTEBOOK_FILE}"
        fi
    fi

	# Post-processing on README markdown files
	README_FILE="${TMPL}/README.md"
    if [[ ! -f "${TMPL}/README.md" ]]; then
		echo "README.md file not found; skipping markdown processing."
	else
        # Check the operating system
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS system
            sed -i '' "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${TMPL}/\1\"|g" "$README_FILE"
            sed -i '' "s|!\[.*\](\(assets/.*\))|<img src=\"${REPO_PREFIX}/${TMPL}/\1\"/>|g" "$README_FILE"
        else
            # Assuming Linux
            sed -i "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${TMPL}/\1\"|g" "$README_FILE"
            sed -i "s|!\[.*\](\(assets/.*\))|<img src=\"${REPO_PREFIX}/${TMPL}/\1\"/>|g" "$README_FILE"
        fi
    fi
done
