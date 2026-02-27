#!/bin/bash

set -euo pipefail

echo "Auto-generating README files..."

REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ "$(pwd)" != "${REPO_ROOT}" ]]; then
	echo "Must run this script at repo's root directory".
	exit 1
fi

# Determine which notebooks to process
if [[ $# -gt 0 ]]; then
	# Extract unique template directories (templates/<name>) from the provided file paths
	TMPL_DIRS=($(printf '%s\n' "$@" | { grep '^templates/' || true; } | cut -d'/' -f1-2 | sort -u))

	if [[ ${#TMPL_DIRS[@]} -eq 0 ]]; then
		echo "No template directories affected; nothing to do."
		exit 0
	fi

	# Find README.ipynb files within the affected template directories
	NOTEBOOK_FILES=()
	for TMPL_DIR in "${TMPL_DIRS[@]}"; do
		NOTEBOOK_FILES+=($(find "${TMPL_DIR}" -name "README.ipynb" -type f))
	done

	if [[ ${#NOTEBOOK_FILES[@]} -eq 0 ]]; then
		echo "No README.ipynb files found in affected template directories; nothing to do."
		exit 0
	fi
else
	# No arguments provided; process all notebooks
	NOTEBOOK_FILES=($(find "templates" -name "README.ipynb" -type f))
fi

# Define the repo prefix
REPO_PREFIX="https://raw.githubusercontent.com/anyscale/templates/main"

# Maximum number of concurrent notebook processing jobs
MAX_JOBS=${MAX_JOBS:-8}

# Loop through each notebook file
PIDS=()
for NOTEBOOK_FILE in "${NOTEBOOK_FILES[@]}"; do
	(
		# Get the directory containing the notebook
		NOTEBOOK_DIR="$(dirname "${NOTEBOOK_FILE}")"

		echo "===== Processing ${NOTEBOOK_FILE}"

		# Get the template name (first directory after templates/)
		TMPL_NAME="$(echo "${NOTEBOOK_DIR}" | cut -d'/' -f2)"

		# Exclude specific notebooks from conversion
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
			jupyter nbconvert --to markdown "${NOTEBOOK_FILE}" --output-dir "${NOTEBOOK_DIR}"
		else
			echo "Skipping README generation for ${NOTEBOOK_FILE}"
		fi

		# Post-processing on README markdown files
		README_FILE="${NOTEBOOK_DIR}/README.md"
		if [[ ! -f "${README_FILE}" ]]; then
			echo "README.md file not found; skipping markdown processing."
		else
			# Check the operating system
			if [[ "$OSTYPE" == "darwin"* ]]; then
				# macOS system
				sed -i '' "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${NOTEBOOK_DIR}/\1\"|g" "$README_FILE"
				sed -i '' "s|!\[.*\](\(assets/.*\))|<img src=\"${REPO_PREFIX}/${NOTEBOOK_DIR}/\1\"/>|g" "$README_FILE"
			else
				# Assuming Linux
				sed -i "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${NOTEBOOK_DIR}/\1\"|g" "$README_FILE"
				sed -i "s|!\[.*\](\(assets/.*\))|<img src=\"${REPO_PREFIX}/${NOTEBOOK_DIR}/\1\"/>|g" "$README_FILE"
			fi
		fi
	) &
	PIDS+=($!)

	# Limit concurrent jobs
	if (( ${#PIDS[@]} >= MAX_JOBS )); then
		wait "${PIDS[0]}" || exit 1
		PIDS=("${PIDS[@]:1}")
	fi
done

# Wait for remaining background jobs and check their exit codes
for PID in "${PIDS[@]}"; do
	wait "${PID}" || exit 1
done
