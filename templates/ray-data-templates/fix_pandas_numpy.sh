#!/bin/bash
# Fix pandas/numpy compatibility issue

echo "Fixing pandas/numpy compatibility issue..."

# Method 1: Update conda packages
echo "Attempting conda update..."
conda update -y numpy pandas pyarrow

# Method 2: If conda fails, use pip
if [ $? -ne 0 ]; then
    echo "Conda update failed, trying pip..."
    pip install --upgrade numpy pandas pyarrow
fi

# Method 3: Force reinstall if still failing
if [ $? -ne 0 ]; then
    echo "Force reinstalling pandas..."
    pip uninstall -y pandas numpy
    pip install numpy pandas pyarrow
fi

echo "Attempting to test pandas import..."
python -c "import pandas as pd; print('Pandas import successful!')"

if [ $? -eq 0 ]; then
    echo "Pandas/numpy issue fixed!"
    echo "You can now run: python upload_benchmark_data.py"
else
    echo "Issue persists. Try:"
    echo "1. conda create -n fresh_env python=3.9"
    echo "2. conda activate fresh_env" 
    echo "3. pip install boto3 pandas pyarrow requests yfinance"
    echo "4. python upload_benchmark_data.py"
fi
