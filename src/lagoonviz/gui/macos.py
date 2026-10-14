# SPDX-License-Identifier: GPL-3.0-or-later
"""macOS: show "LAGOON" instead of "Python" in the menu bar and the Dock, and make
LAGOON a foreground application however it was started.

LAGOON runs as a Python script, not as an app bundle, so macOS would name it after
the Python executable. Before the QApplication exists, this renames the process and
sets the bundle name through the Objective-C runtime (plain ctypes, no PyObjC).

Started from a script in Terminal (setup_lagoon.command runs LAGOON.app's program
directly, not through Finder), macOS does not register LAGOON as an application of
its own: it gets no menu bar and no Dock icon, and runs as a background process,
which macOS slows down (App Nap). Once the QApplication exists, ``bring_forward``
makes it a regular application, brings it to the front and keeps App Nap off while
it runs. Everything is guarded: on any surprise it does nothing.
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


#: NSApplicationActivationPolicyRegular: a menu bar and a Dock icon
REGULAR = 0
#: NSActivityUserInitiatedAllowingIdleSystemSleep: no App Nap, the Mac may still sleep
USER_INITIATED = 0x00FFFFFF
#: the App Nap activity, kept while LAGOON runs (ending it would let App Nap back)
_activity = None


def bring_forward() -> None:
    """After the QApplication exists: a regular, frontmost application without App
    Nap, also when started from Terminal."""
    if sys.platform != "darwin":
        return
    try:
        _bring_forward()
    except Exception as error:  # noqa: BLE001 (on any surprise: as macOS left it)
        log.info("could not make LAGOON a foreground application: %s", error)


class _Runtime:
    """objc_msgSend with the exact prototypes (arm64 needs them)."""

    def __init__(self) -> None:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc = objc
        address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        vp = ctypes.c_void_p
        self.send0 = ctypes.CFUNCTYPE(vp, vp, vp)(address)
        self.send_str = ctypes.CFUNCTYPE(vp, vp, vp, ctypes.c_char_p)(address)
        self.send1 = ctypes.CFUNCTYPE(vp, vp, vp, vp)(address)
        self.send2 = ctypes.CFUNCTYPE(vp, vp, vp, vp, vp)(address)
        self.responds = ctypes.CFUNCTYPE(ctypes.c_bool, vp, vp, vp)(address)
        self.send_long = ctypes.CFUNCTYPE(ctypes.c_bool, vp, vp, ctypes.c_long)(address)
        self.get_long = ctypes.CFUNCTYPE(ctypes.c_long, vp, vp)(address)
        self.send_bool = ctypes.CFUNCTYPE(None, vp, vp, ctypes.c_bool)(address)
        self.send_u64_id = ctypes.CFUNCTYPE(vp, vp, vp, ctypes.c_uint64, vp)(address)

    def cls(self, text: str) -> int:
        return self.objc.objc_getClass(text.encode())

    def sel(self, text: str) -> int:
        return self.objc.sel_registerName(text.encode())

    def ns(self, text: str) -> int:
        return self.send_str(self.cls("NSString"), self.sel("stringWithUTF8String:"), text.encode())


def _bring_forward() -> None:
    global _activity
    rt = _Runtime()
    sel = rt.sel
    app = rt.send0(rt.cls("NSApplication"), sel("sharedApplication"))
    if not app:
        return
    if rt.get_long(app, sel("activationPolicy")) != REGULAR:
        log.info("started as a background process: made a regular application")
        rt.send_long(app, sel("setActivationPolicy:"), REGULAR)
        # to the front once the event loop runs (before, macOS may not take it)
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: rt.send_bool(app, sel("activateIgnoringOtherApps:"), True))
    if _activity is None:
        process = rt.send0(rt.cls("NSProcessInfo"), sel("processInfo"))
        reason = rt.ns("LAGOON shows and renders results")
        activity = rt.send_u64_id(
            process, sel("beginActivityWithOptions:reason:"), USER_INITIATED, reason
        )
        if activity:
            _activity = rt.send0(activity, sel("retain"))


def _set_app_name(name: str) -> None:
    rt = _Runtime()
    send0, send1, send2, responds = rt.send0, rt.send1, rt.send2, rt.responds
    cls, sel, ns = rt.cls, rt.sel, rt.ns

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
