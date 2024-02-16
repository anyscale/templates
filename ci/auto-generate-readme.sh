#!/bin/bash

# Search for notebook files named "demo" in the ../templates directory
notebook_files=$(find ../templates -name "README.ipynb")

# Loop through each notebook file
for notebook_file in $notebook_files; do
    # Convert notebook file to README.md using nbconvert
    jupyter nbconvert --to markdown "$notebook_file" --output-dir "$(dirname "$notebook_file")"
done

# Define the repo prefix
REPO_PREFIX="https://raw.githubusercontent.com/anyscale/templates/main"

# Search for README.md in the ../templates directory
readme_files=$(find ../templates -name "README.md")

# Loop through each readme files
for readme_file in $readme_files; do
    # Extract the path of the directory containing the README file, relative to the repository root
    readme_dir=$(dirname "$readme_file" | sed "s|\.\./templates/||")

    # Check the operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS system
        sed -i '' "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${readme_dir}/\1\"|g" "$readme_file"
    else
        # Assuming Linux
        sed -i "s|<img src=\"\([^\"http://][^\":/][^\"].*\)\"|<img src=\"${REPO_PREFIX}/${readme_dir}/\1\"|g" "$readme_file"
    fi
done
