# -*- mode: python -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
# PyInstaller recipe of the standalone LAGOON.app (macOS). Run through
# packaging/macos/build_app.sh, which sets up a clean environment, makes the icon,
# signs, checks and packs the app into a DMG. Environment variables read here:
#   LAGOON_ICON            the .icns file (made by build_app.sh)
#   LAGOON_SIGN_IDENTITY   "Developer ID Application: ..." to sign for notarization;
#                          unset: an ad-hoc signature (runs after right-click > Open)
import os
import re
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).resolve().parents[1]  # noqa: F821 (set by PyInstaller)
VERSION = re.search(
    r'^__version__ = "([^"]+)"', (ROOT / "src/lagoonviz/__init__.py").read_text(), re.M
).group(1)
SHORT = re.match(r"\d+\.\d+\.\d+", VERSION).group(0)
IDENTITY = os.environ.get("LAGOON_SIGN_IDENTITY") or None


def _not(*names):
    return lambda module: not any(part in module for part in names)


# LAGOON's own files that are not Python modules: the sky picture, the window's
# icons, and the script Blender runs (read as a file, never imported)
datas = collect_data_files("lagoonviz")
datas += [(str(ROOT / "src/lagoonviz/render/blender_cycles.py"), "lagoonviz/render")]
# pygfx's shaders and fonts, wgpu's native library, matplotlib's styles and fonts
for package in ("pygfx", "wgpu", "rendercanvas", "matplotlib"):
    datas += collect_data_files(package)
binaries = collect_dynamic_libs("wgpu")
# the ffmpeg imageio-ffmpeg brings: a program, signed like the rest
import imageio_ffmpeg  # noqa: E402

ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
binaries += [(str(ffmpeg), "imageio_ffmpeg/binaries")]

hiddenimports = collect_submodules("lagoonviz", filter=_not("blender_cycles"))
hiddenimports += collect_submodules("lagoon_format", filter=_not("paraview"))
hiddenimports += collect_submodules("wgpu", filter=_not("glfw", "jupyter", "wx"))
hiddenimports += collect_submodules(
    "rendercanvas", filter=_not("glfw", "jupyter", "wx", "pyqt", "pyside2")
)
hiddenimports += ["matplotlib.backends.backend_agg", "matplotlib.backends.backend_pdf",
                  "matplotlib.backends.backend_svg", "matplotlib.backends.backend_ps"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging/macos/lagoon_app.py")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "PyQt5", "PyQt6", "PySide2", "bpy"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LAGOON",
    console=False,
    argv_emulation=False,
    codesign_identity=IDENTITY,
    entitlements_file=str(ROOT / "packaging/macos/entitlements.plist"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="LAGOON")  # noqa: F821
app = BUNDLE(  # noqa: F821
    coll,
    name="LAGOON.app",
    icon=os.environ.get("LAGOON_ICON") or None,
    bundle_identifier="org.reef3d.lagoon",
    version=SHORT,
    info_plist={
        "CFBundleName": "LAGOON",
        "CFBundleDisplayName": "LAGOON",
        "CFBundleShortVersionString": SHORT,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHumanReadableCopyright": "LAGOON, the post-processor of REEF3D. GPL-3.0-or-later.",
    },
)
