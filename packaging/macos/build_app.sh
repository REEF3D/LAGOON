#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
#
# The standalone LAGOON.app in a DMG, for Macs with the same chip as this one:
#
#   bash packaging/macos/build_app.sh
#
# makes dist-mac/LAGOON-<version>-macos-<arm64|x86_64>.dmg (not in dist/, where the
# Python packages go and twine checks every file). It needs uv (as
# setup_lagoon.command) and Xcode's command line tools (iconutil, codesign, hdiutil).
#
# Signing (RELEASING.md, "The macOS app"):
#   LAGOON_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)"  sign for Gatekeeper
#   LAGOON_NOTARY_PROFILE=lagoon-notary   then notarize the DMG with Apple and staple it
# Without them the app is signed ad hoc: it runs, but a downloaded copy opens the
# first time only with right-click > Open.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/src/lagoonviz/__init__.py")"
ARCH="$(uname -m)"
BUILD="$ROOT/build/macos"
DIST="$ROOT/dist-mac"
DMG="$DIST/LAGOON-$VERSION-macos-$ARCH.dmg"

say() { printf '\n== %s\n' "$*"; }
command -v uv >/dev/null || { echo "uv is needed: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v iconutil >/dev/null || { echo "Xcode's command line tools are needed: xcode-select --install"; exit 1; }

say "LAGOON $VERSION for macOS ($ARCH)"
rm -rf "$BUILD"
mkdir -p "$BUILD" "$DIST"

say "a clean environment with what the app needs (no test or docs tools)"
uv venv --quiet --python 3.12 "$BUILD/venv"
PY="$BUILD/venv/bin/python"
uv pip install --quiet --python "$PY" "$ROOT/lagoon-format" "$ROOT" "pyinstaller>=6.10"

say "the icon"
"$PY" "$HERE/make_icon.py" "$ROOT/logo/icon_dark.png" "$BUILD/LAGOON.iconset"
iconutil -c icns "$BUILD/LAGOON.iconset" -o "$BUILD/LAGOON.icns"

say "the app (PyInstaller)"
LAGOON_ICON="$BUILD/LAGOON.icns" "$PY" -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath "$BUILD/dist" --workpath "$BUILD/work" "$HERE/LAGOON.spec"
APP="$BUILD/dist/LAGOON.app"
LAGOON="$APP/Contents/MacOS/LAGOON"

say "checks: the app runs, renders and converts a small run on its own"
"$LAGOON" --version
CASE="$BUILD/check-run"
"$PY" -c "import sys; sys.path.insert(0, '$ROOT/tests'); \
from pathlib import Path; from reef3d_fixtures import make_run; \
make_run(Path('$CASE'), frames=3, print_interval=0.5, series=True)"
"$LAGOON" render "$CASE" "$BUILD/check.png" --size 320x180
test -s "$BUILD/check.png" || { echo "the app drew no picture"; exit 1; }
"$LAGOON" convert "$CASE" --out "$BUILD/check.lagoon"
test -d "$BUILD/check.lagoon" || { echo "the app wrote no store"; exit 1; }
for part in lagoonviz/render/blender_cycles.py lagoonviz/render/assets/sea_clear_sky.hdr \
            imageio_ffmpeg/binaries; do
    find "$APP" -path "*$part*" -print -quit | grep -q . || { echo "missing in the app: $part"; exit 1; }
done
codesign --verify --deep --strict "$APP"
codesign -dv "$APP" 2>&1 | grep -E "Signature|Authority|flags" || true

say "the DMG"
STAGE="$BUILD/dmg"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -quiet -volname "LAGOON $VERSION" -srcfolder "$STAGE" -format UDZO "$DMG"

if [ -n "${LAGOON_SIGN_IDENTITY:-}" ]; then
    say "signing the DMG"
    codesign --sign "$LAGOON_SIGN_IDENTITY" --timestamp "$DMG"
    if [ -n "${LAGOON_NOTARY_PROFILE:-}" ]; then
        say "notarizing with Apple (a few minutes)"
        xcrun notarytool submit "$DMG" --keychain-profile "$LAGOON_NOTARY_PROFILE" --wait
        xcrun stapler staple "$DMG"
        spctl --assess --type open --context context:primary-signature -v "$DMG" || true
    fi
fi

say "done"
du -h "$DMG"
echo "$DMG"
if [ -z "${LAGOON_SIGN_IDENTITY:-}" ]; then
    echo "Signed ad hoc: on another Mac, open it the first time with right-click > Open,"
    echo "or run: xattr -dr com.apple.quarantine /Applications/LAGOON.app"
fi
