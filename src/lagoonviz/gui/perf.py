# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the time goes while playing: cheap timers the developer remote reads out."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class Perf:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.counts[name] += 1

    def count(self, name: str) -> None:
        self.counts[name] += 1

    def summary(self) -> str:
        parts = []
        for name in sorted(self.counts, key=lambda n: -self.seconds.get(n, 0.0)):
            n = self.counts[name]
            total = self.seconds.get(name, 0.0)
            mean = f" {1000 * total / max(n, 1):.1f} ms" if total else ""
            parts.append(f"{name} {n}x{mean}")
        return "; ".join(parts)


PERF = Perf()
