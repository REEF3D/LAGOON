"""Garbage collection on the GUI thread only (a crash when a loader thread collected
while the window deleted widgets)."""

import gc
import threading

from lagoonviz.gui.gc_guard import MainThreadGarbageCollector


class _Cycle:
    def __init__(self):
        self.me = self


def test_collection_happens_on_the_gui_thread_only(qapp):
    collector = MainThreadGarbageCollector(qapp, interval_ms=10)
    try:
        assert not gc.isenabled()
        threads = []
        gc.callbacks.append(lambda phase, info: threads.append(threading.current_thread()))
        try:

            def allocate():  # a loader thread making cyclic garbage
                for _ in range(20000):
                    _Cycle()

            worker = threading.Thread(target=allocate)
            worker.start()
            worker.join()
            assert not threads  # nothing collected in the loader thread
            collector.check()
            assert threads and all(t is threading.main_thread() for t in threads)
        finally:
            gc.callbacks.pop()
    finally:
        collector.stop()
    assert gc.isenabled()
