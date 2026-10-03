#!/bin/bash
# Build LAGOON.app next to this repository's .venv, so macOS shows "LAGOON" (name and
# icon) in the menu bar and the Dock instead of "Python". Called by setup_lagoon.command.
#
# The app holds no copy of LAGOON: Contents/MacOS/python is a link to the venv's
# interpreter, and Contents/pyvenv.cfg plus Contents/lib make Python use the venv's
# packages (an editable install, so code changes show up at the next start).
# macOS takes the app name from the bundle the running executable sits in, which is
# why the interpreter is started through the link inside the bundle.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HERE/LAGOON.app"
VENV="$HERE/.venv"
[ -x "$VENV/bin/python" ] || { echo "no .venv: run setup_lagoon.command first"; exit 1; }
REAL_PYTHON="$("$VENV/bin/python" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
VERSION="$("$VENV/bin/python" -c 'import lagoonviz; print(lagoonviz.__version__)')"

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$VENV/pyvenv.cfg" "$APP/Contents/pyvenv.cfg"
ln -sfn "$VENV/lib" "$APP/Contents/lib"
ln -sfn "$REAL_PYTHON" "$APP/Contents/MacOS/python"

cat > "$APP/Contents/MacOS/LAGOON" <<EOF
#!/bin/bash
# started by macOS (double-click) or by setup_lagoon.command
exec "$APP/Contents/MacOS/python" -m lagoonviz.gui.app "\$@"
EOF
chmod +x "$APP/Contents/MacOS/LAGOON"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LAGOON</string>
  <key>CFBundleDisplayName</key><string>LAGOON</string>
  <key>CFBundleIdentifier</key><string>org.reef3d.lagoon</string>
  <key>CFBundleExecutable</key><string>LAGOON</string>
  <key>CFBundleIconFile</key><string>LAGOON</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSRequiresAquaSystemAppearance</key><false/>
</dict>
</plist>
EOF

# icon: the 256 px PNG as an .icns (sips and iconutil come with macOS)
ICON="$HERE/src/lagoonviz/gui/resources/icon.png"
if command -v iconutil >/dev/null 2>&1 && [ -f "$ICON" ]; then
  SET="$(mktemp -d)/LAGOON.iconset"
  mkdir -p "$SET"
  for size in 16 32 128 256; do
    sips -z $size $size "$ICON" --out "$SET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    [ $double -le 256 ] && sips -z $double $double "$ICON" --out "$SET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$SET" -o "$APP/Contents/Resources/LAGOON.icns" || true
fi

# check that the app's Python finds LAGOON
"$APP/Contents/MacOS/python" -c "import lagoonviz.gui.app" && echo "LAGOON.app ready: $APP"
