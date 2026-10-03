# SPDX-License-Identifier: GPL-3.0-or-later
"""macOS: show "LAGOON" instead of "Python" in the menu bar and the Dock.

LAGOON runs as a Python script, not as an app bundle, so macOS would name it after
the Python executable. Before the QApplication exists, this renames the process and
sets the bundle name through the Objective-C runtime (plain ctypes, no PyObjC).
Everything is guarded: on any surprise it does nothing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys

log = logging.getLogger("lagoon.gui")


def set_app_name(name: str) -> None:
    if sys.platform != "darwin":
        return
    try:
        _set_app_name(name)
    except Exception as error:  # noqa: BLE001 (cosmetic only)
        log.info("could not set the macOS app name: %s", error)


def _set_app_name(name: str) -> None:
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    vp = ctypes.c_void_p
    # objc_msgSend must be called with the exact prototype on arm64
    send0 = ctypes.CFUNCTYPE(vp, vp, vp)(address)
    send_str = ctypes.CFUNCTYPE(vp, vp, vp, ctypes.c_char_p)(address)
    send1 = ctypes.CFUNCTYPE(vp, vp, vp, vp)(address)
    send2 = ctypes.CFUNCTYPE(vp, vp, vp, vp, vp)(address)
    responds = ctypes.CFUNCTYPE(ctypes.c_bool, vp, vp, vp)(address)

    def cls(text: str) -> int:
        return objc.objc_getClass(text.encode())

    def sel(text: str) -> int:
        return objc.sel_registerName(text.encode())

    def ns(text: str) -> int:
        return send_str(cls("NSString"), sel("stringWithUTF8String:"), text.encode())

    value = ns(name)
    process = send0(cls("NSProcessInfo"), sel("processInfo"))
    if process and responds(process, sel("respondsToSelector:"), sel("setProcessName:")):
        send1(process, sel("setProcessName:"), value)
    bundle = send0(cls("NSBundle"), sel("mainBundle"))
    if not bundle:
        return
    key = ns("CFBundleName")
    for getter in ("localizedInfoDictionary", "infoDictionary"):
        info = send0(bundle, sel(getter))
        # only a real bundle's dictionary (it has the key) is known to be mutable;
        # writing to an immutable one would raise an Objective-C exception
        if info and send1(info, sel("objectForKey:"), key):
            send2(info, sel("setObject:forKey:"), value, key)
