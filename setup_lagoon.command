#!/bin/bash
# LAGOON setup for macOS: installs uv (if needed), Python 3.12 and LAGOON, runs the tests,
# then starts LAGOON. Double-click this file in Finder. Output is also written to setup_lagoon.log.
HERE="$(cd "$(dirname "$0")" && pwd)"
exec > >(tee "$HERE/setup_lagoon.log") 2>&1
cd "$HERE" || exit 1
export PATH="$HOME/.local/bin:$PATH"

echo "== 1/4 uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || { echo "ERROR: uv could not be installed"; read -r -p "Press Return to close"; exit 1; }

echo "== 2/4 Python 3.12 environment (.venv)"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
source .venv/bin/activate
python --version

echo "== 3/4 install LAGOON"
uv pip install -e ./lagoon-format -e ".[dev]" || { echo "ERROR: install failed"; read -r -p "Press Return to close"; exit 1; }

echo "== 4/4 tests"
python -m pytest -q

echo "== starting LAGOON (close the window to finish)"
lagoon
echo "== done"
