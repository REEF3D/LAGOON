#!/bin/bash
# LAGOON setup for macOS: installs uv (if needed), Python 3.12 and LAGOON, runs the tests,
# then starts LAGOON. Run it with:  bash ~/Codelite/LAGOON/setup_lagoon.command [case folder]
# Output is also written to setup_lagoon.log; the application log goes to lagoon.log.
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
python -c "import pygfx, wgpu; print('pygfx', pygfx.__version__, '| wgpu', wgpu.__version__); print(wgpu.gpu.request_adapter_sync(power_preference='high-performance').info)" 2>&1 | tail -3

echo "== 4/4 tests"
export LAGOON_TEST_OUTPUT="$HERE/.test_output"
python -m pytest -q -rs

# macOS: LAGOON.app, so the menu bar and the Dock say LAGOON (double-click works too)
LAUNCH=(lagoon)
if [ "$(uname)" = "Darwin" ] && bash "$HERE/tools/make_macos_app.sh"; then
  LAUNCH=("$HERE/LAGOON.app/Contents/MacOS/LAGOON")
fi

echo "== starting LAGOON (close the window to finish)"
export LAGOON_LOG="$HERE/lagoon.log"
if [ -n "$LAGOON_DEV" ]; then
  # developer testing: Claude asks the running app for screenshots via .dev_remote/
  "${LAUNCH[@]}" --dev-remote "$HERE/.dev_remote" "$@"
else
  "${LAUNCH[@]}" "$@"
fi
echo "== done"
