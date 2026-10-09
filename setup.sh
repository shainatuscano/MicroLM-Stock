#!/bin/bash
# Setup script for MicroLM
set -e
cd "$(dirname "$0")"

echo "=== MicroLM Environment Setup ==="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
./.venv/bin/pip install --upgrade pip -q
./.venv/bin/pip install -r requirements.txt -q

# Verify installation
echo "Verifying environment..."
./.venv/bin/python check_env.py

echo ""
echo "=== Setup complete! ==="
echo "Activate environment: source .venv/bin/activate"
echo "Launch web demo:      python app.py"
echo "Train model:          python train.py"
