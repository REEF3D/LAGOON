# SPDX-License-Identifier: GPL-3.0-or-later
"""Developer remote: drive a running LAGOON from files, for testing on real screens.

Started only with ``lagoon --dev-remote DIR``. A tool (or a person) writes
``DIR/request.json``; LAGOON runs the listed actions, writes ``DIR/result.json`` and
deletes the request. Only the actions below exist; there is no way to run code.

    {"actions": [
        {"open_case": "~/REEF3D_NHFLOW_VTP_FSF"},
        {"frame": 120},                      # go to a frame (or {"time": 60.0})
        {"set": ["free_surface-1", "exaggeration", 5.0]},
        {"visible": ["bathymetry-1", false]},
        {"select": "free_surface-1"},
        {"view": "top"},                     # perspective, top, front, right, ...
        {"play": true},
        {"wait": 2.0},                       # seconds
        {"screenshot": "after.png"},         # waits until the frame is loaded
        {"resize": [1400, 860]},
        {"fps": 30},                         # playback speed
        {"clear_cache": true},               # forget loaded frames (timing tests)
        {"measure_play": 5.0}                # play for 5 s and report frames per second
    ]}
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer

log = logging.getLogger("lagoon.devremote")

POLL_MS = 500
LOAD_TIMEOUT = 20.0


class DevRemote(QObject):
    def __init__(self, window, folder: str | Path):
        super().__init__(window)
        self.window = window
        self.folder = Path(folder).expanduser()
        self.folder.mkdir(parents=True, exist_ok=True)
        self._queue: list[dict[str, Any]] = []
        self._results: list[str] = []
        self._wait_until = 0.0
        self._started = 0.0
        self._measure = (0.0, 0)
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        log.info("developer remote listening in %s", self.folder)

    # ------------------------------------------------------------- loop
    def _tick(self) -> None:
        if not self._queue:
            self._read_request()
        while self._queue:
            if time.monotonic() < self._wait_until:
                return
            action = self._queue[0]
            if "screenshot" in action and self.window.viewport.busy:
                if time.monotonic() - self._started < LOAD_TIMEOUT:
                    return  # wait for the frame to arrive
                self._results.append("timeout waiting for frames; screenshot taken anyway")
            self._queue.pop(0)
            self._started = time.monotonic()
            try:
                self._results.append(self._run(action))
            except Exception as error:  # noqa: BLE001
                self._results.append(f"ERROR {action}: {error}\n{traceback.format_exc()}")
        if self._results:
            self._write_result()

    def _read_request(self) -> None:
        request = self.folder / "request.json"
        if not request.is_file():
            return
        try:
            actions = json.loads(request.read_text(encoding="utf-8")).get("actions", [])
        except (OSError, ValueError) as error:
            actions = []
            self._results.append(f"ERROR bad request: {error}")
        request.unlink(missing_ok=True)
        self._queue = [a for a in actions if isinstance(a, dict)]
        self._started = time.monotonic()

    def _write_result(self) -> None:
        window = self.window
        state = {
            "results": self._results,
            "scene": window.session.scene.to_dict(),
            "viewport": getattr(window.viewport, "info", None) and window.viewport.info.text(),
            "status": window.statusBar().currentMessage(),
            "timeline": window.timeline.label.text(),
            "cache": {
                "frames": len(window.session.data.cache),
                "megabytes": round(window.session.data.cache.nbytes / 1e6, 1),
                "hits": window.session.data.cache.hits,
                "misses": window.session.data.cache.misses,
            },
        }
        (self.folder / "result.json").write_text(json.dumps(state, indent=2, default=str))
        self._results = []

    # ------------------------------------------------------------- actions
    def _run(self, action: dict[str, Any]) -> str:
        window = self.window
        session = window.session
        (name, value), *_ = action.items()
        if name == "open_case":
            start = time.monotonic()
            case_id = window.open_case(Path(value).expanduser())
            return f"open_case -> {case_id} in {time.monotonic() - start:.2f} s"
        if name == "new_session":
            window.ask_before_discard = False
            window.new_session()
            window.ask_before_discard = True
            return "new session"
        if name == "frame":
            window.timeline.go_to_frame(int(value))
            return f"frame {value}"
        if name == "time":
            session.set_time(float(value))
            return f"time {value}"
        if name == "set":
            item_id, key, new = value
            session.set_property(item_id, key, new)
            return f"set {item_id}.{key} = {new!r}"
        if name == "visible":
            session.set_visible(value[0], bool(value[1]))
            return f"visible {value}"
        if name == "select":
            window.scene_tree.select(value)
            return f"select {value}"
        if name == "view":
            window.viewport.frame(str(value))
            return f"view {value}"
        if name == "play":
            window.timeline.play_button.setChecked(bool(value))
            return f"play {value}"
        if name == "wait":
            self._wait_until = time.monotonic() + float(value)
            return f"wait {value}"
        if name == "resize":
            window.resize(int(value[0]), int(value[1]))
            return f"resize {value}"
        if name == "fps":
            window.timeline.fps.setValue(int(value))
            return f"fps {value}"
        if name == "clear_cache":
            session.data.cache.clear()
            return "cache cleared"
        if name == "measure_play":
            self._measure = (time.monotonic(), window.timeline.frame_of(session.scene.time))
            window.timeline.play_button.setChecked(True)
            self._wait_until = time.monotonic() + float(value)
            self._queue.insert(0, {"_measure_end": True})
            return f"playing for {value} s"
        if name == "_measure_end":
            window.timeline.play_button.setChecked(False)
            start, first = self._measure
            frames = window.timeline.frame_of(session.scene.time) - first
            seconds = time.monotonic() - start
            return f"played {frames} frames in {seconds:.2f} s = {frames / seconds:.1f} fps"
        if name == "screenshot":
            path = self.folder / Path(str(value)).name
            ok = window.save_screenshot(path)
            return f"screenshot {path} {'ok' if ok else 'FAILED'}"
        return f"ERROR unknown action {name!r}"
