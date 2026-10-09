#!/bin/bash
# Run training with MicroLM virtual environment
cd "$(dirname "$0")"

if [ -f "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
else
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" train.py "$@"
