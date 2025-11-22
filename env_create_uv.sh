#!/bin/bash
set -e

echo "Creating Python environment with uv..."

# Create virtual environment with Python 3.11
uv venv --python 3.11
source .venv/bin/activate

echo "Installing the project and dependencies (including PyTorch from custom index)..."
# uv sync respects the [tool.uv.sources] configuration in pyproject.toml
# It will automatically install torch from the PyTorch custom index
# Include dev dependencies (ipykernel) and optionally demo extras
if [ "$1" = "--demo" ] || [ "$1" = "demo" ] || [ "$1" = "-d" ]; then
  echo "Including demo optional dependencies (extra: demo)"
  uv sync --group dev --extra demo 2>/dev/null || uv sync --group dev
else
  uv sync --group dev
fi

echo "Creating Jupyter kernel..."
python -m ipykernel install --user --name=UniversalUnconsciousness --display-name=UniversalUnconsciousness

echo "Environment created successfully!"
echo "To start Jupyter in Cursor/VSCode, click the kernel selecter in the top right corner."
echo "Then select the 'Select Another Kernel...' --> 'Jupyter Kernel...' --> 'UniversalUnconsciousness'"
echo "In the 'Jupyter Kernel...' selector, you may need to click the refresh button in the top right corner to see the new kernel."
echo "Note: If you do not see the new kernel, you may need to reload (i.e. close and reopen) your IDE window."
echo "To activate the environment, from inside this directory, run: source .venv/bin/activate"
echo "To change the Python interpreter in Cursor/VSCode, use the keyboard shortcut: Cmd+Shift+P --> 'Python: Select Interpreter' --> Enter interpreter path..."
echo "Then, navigate to this directory and select the file: .venv/bin/python as the interpreter."