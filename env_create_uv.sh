#!/bin/bash
set -e

echo "Creating Python environment with uv..."

# Parse optional demo extras flag
PACKAGE_SPEC="."
if [ "$1" = "--demo" ] || [ "$1" = "demo" ] || [ "$1" = "-d" ]; then
  echo "Including demo optional dependencies (extra: demo)"
  PACKAGE_SPEC=".[demo]"
fi

# Create virtual environment with Python 3.11
uv venv --python 3.11
source .venv/bin/activate
echo "Installing PyTorch (CUDA 11.8, sm_61 compatible)..."
uv pip install --index-url https://download.pytorch.org/whl/cu118 "torch==2.0.1+cu118"

echo "Installing the project in editable mode..."
# Install the project in editable mode (optionally with demo extras)
uv pip install -e "$PACKAGE_SPEC"

echo "Installing ipykernel and creating a Jupyter kernel bound to this venv..."
# Install ipykernel into the environment without triggering a full project re-resolve
uv pip install ipykernel
python -m ipykernel install --user --name=UniversalUnconsciousness --display-name=UniversalUnconsciousness

echo "Environment created successfully!"
echo "To start Jupyter in Cursor/VSCode, click the kernel selecter in the top right corner."
echo "Then select the 'Select Another Kernel...' --> 'Jupyter Kernel...' --> 'UniversalUnconsciousness'"
echo "In the 'Jupyter Kernel...' selector, you may need to click the refresh button in the top right corner to see the new kernel."
echo "Note: If you do not see the new kernel, you may need to reload (i.e. close and reopen) your IDE window."
echo "To activate the environment, from inside this directory, run: source .venv/bin/activate"
echo "To change the Python interpreter in Cursor/VSCode, use the keyboard shortcut: Cmd+Shift+P --> 'Python: Select Interpreter' --> Enter interpreter path..."
echo "Then, navigate to this directory and select the file: .venv/bin/python as the interpreter."