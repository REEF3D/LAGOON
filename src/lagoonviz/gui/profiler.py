# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Help → Record a performance report: where LAGOON spends its time while you use
it, for finding what makes it sluggish.

A thread looks every few milliseconds at what every thread of LAGOON is doing (the
window's own and the workers': reading frames, making maps) and counts it; stopped,
it writes a plain text report: per thread, the functions it was in most (on their
own and with what they called), and the window thread's busiest moments. Nothing
is slowed down noticeably, nothing leaves the computer.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

#: seconds between two looks
INTERVAL = 0.005
#: frames of a stack kept (from the innermost)
DEPTH = 40
#: the window's thread doing nothing: Qt waiting for events (its innermost Python
#: frame is then in gui/app.py, where Qt's event loop was started)
IDLE = "gui/app.py:"


def _where(frame) -> str:
    code = frame.f_code
    path = code.co_filename
    marker = "lagoonviz"
    short = path[path.rfind(marker) :] if marker in path else Path(path).name
    return f"{code.co_name} ({short}:{frame.f_lineno})"


class Recorder:
    """Samples every thread's stack until stopped."""

    def __init__(self, interval: float = INTERVAL) -> None:
        self.interval = interval
        self.samples = 0
        self.started = 0.0
        self.seconds = 0.0
        self.own: dict[str, Counter] = defaultdict(Counter)  # thread -> innermost
        self.within: dict[str, Counter] = defaultdict(Counter)  # thread -> on the stack
        self.stacks: dict[str, Counter] = defaultdict(Counter)  # thread -> stack
        self.seen: Counter = Counter()  # thread -> samples
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._main = threading.main_thread().ident

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="lagoon-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.seconds = time.perf_counter() - self.started

    def _run(self) -> None:
        me = threading.get_ident()
        names = {}
        while not self._stop.wait(self.interval):
            if len(names) != threading.active_count():
                names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                name = "window" if ident == self._main else names.get(ident, str(ident))
                stack = []
                while frame is not None and len(stack) < DEPTH:
                    stack.append(_where(frame))
                    frame = frame.f_back
                if not stack:
                    continue
                self.seen[name] += 1
                self.own[name][stack[0]] += 1
                for place in set(stack):
                    self.within[name][place] += 1
                self.stacks[name][tuple(reversed(stack[:12]))] += 1
            self.samples += 1

    def report(self) -> str:
        """The report as text."""
        lines = [
            "LAGOON performance report",
            f"{self.seconds:.1f} s recorded, {self.samples} looks every "
            f"{1000 * self.interval:.0f} ms",
            "",
        ]
        for name, count in self.seen.most_common():
            share = 100.0 * count / max(self.samples, 1)
            lines.append(f"=== thread {name}: seen in {share:.0f} % of the looks")
            idle = sum(n for place, n in self.own[name].items() if IDLE in place)
            if name == "window":
                lines.append(f"    waiting for events (idle): {100.0 * idle / max(count, 1):.0f} %")
            lines.append("    most often in (the function itself):")
            for place, n in self.own[name].most_common(15):
                lines.append(f"      {100.0 * n / max(count, 1):5.1f} %  {place}")
            lines.append("    most often under (with what it called):")
            for place, n in self.within[name].most_common(25):
                lines.append(f"      {100.0 * n / max(count, 1):5.1f} %  {place}")
            lines.append("    busiest call chains:")
            for stack, n in self.stacks[name].most_common(5):
                lines.append(f"      {100.0 * n / max(count, 1):5.1f} %")
                for place in stack:
                    lines.append(f"          {place}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.write_text(self.report(), encoding="utf-8")
        return path
