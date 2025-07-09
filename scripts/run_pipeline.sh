#!/bin/bash
echo "🚀 Starting Xinfluencer AI pipeline..."

# Define the project directory
PROJECT_DIR="/home/ubuntu/Xinfluencer"

# Activate virtual environment using an absolute path
source "$PROJECT_DIR/xinfluencer_env/bin/activate"

# Add project root to PYTHONPATH to ensure correct module resolution
export PYTHONPATH="$PROJECT_DIR"

# Run the main pipeline script as a module
python -m src.main

# Deactivate virtual environment
deactivate

echo "✅ Pipeline script finished." 