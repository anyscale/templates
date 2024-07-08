#!/bin/bash
set -e

echo "Auto-generating README files..."

# Function to convert notebook to README.md
convert_notebook() {
    local notebook_file=$1
    local output_dir=$(dirname "$notebook_file")
    jupyter nbconvert --to markdown "$notebook_file" --output-dir "$output_dir"
    git add "$output_dir/README.md"
}

# Convert all README.ipynb files to README.md, excluding specific ones
find templates -name "README.ipynb" | while read notebook_file; do
    if [[ "$notebook_file" != "templates/templates/e2e-llm-workflows/README.ipynb" ]]; then
        convert_notebook "$notebook_file"
    else
        echo "Skipping README generation for $notebook_file"
    fi
done

# Define the repo prefix
REPO_PREFIX="https://raw.githubusercontent.com/anyscale/templates/main"

# Update image paths in README.md files
find templates -name "README.md" | while read readme_file; do
    readme_dir=$(dirname "$readme_file" | sed "s|templates/||")

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS system
        sed -i '' "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${readme_dir}/\1\"|g" "$readme_file"
        sed -i '' "s|!\[.*\](\(assets/.*\))|<img src=\"${REPO_PREFIX}/${readme_dir}/\1\"/>|g" "$readme_file"
    else
        # Assuming Linux
        sed -i "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${readme_dir}/\1\"|g" "$readme_file"
        sed -i "s|!\[.*\](\(assets/.*\))|<img src=\"${REPO_PREFIX}/${readme_dir}/\1\"/>|g" "$readme_file"
    fi
done

# Stage all modified files
git add -A
