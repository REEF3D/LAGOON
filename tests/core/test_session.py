"""The design test: what the GUI does must be reproducible headless from a script or file."""

import subprocess
import sys
import textwrap

from lagoonviz.core import Session


def build(s: Session) -> None:
    fs = s.add_item("free_surface", colormap="balance")
    s.add_item("bathymetry")
    g = s.add_item("gauge", name="Gauge 1", x=4.5)
    s.set_property(fs, "exaggeration", 8.0)
    s.set_visible(g, False)
    s.rename(fs, "Interface")
    sl = s.add_item("slice")
    s.remove_item(sl)
    s.add_item("slice")
    s.undo()  # undone commands are not part of the script


def test_script_replay_rebuilds_scene():
    s = Session()
    build(s)
    replayed = Session.from_script(s.to_script())
    assert replayed.scene.to_dict() == s.scene.to_dict()


def test_script_starting_from_saved_session(tmp_path):
    s = Session()
    build(s)
    path = tmp_path / "a.lagoon-session"
    s.save(path)
    loaded = Session.open(path)
    loaded.add_item("slice", normal="x")
    replayed = Session.from_script(loaded.to_script())
    assert replayed.scene.to_dict() == loaded.scene.to_dict()


def test_log_lines():
    lines = []
    s = Session()
    s.subscribe_log(lines.append)
    fs = s.add_item("free_surface")
    s.set_property(fs, "opacity", 0.5, merge=True)
    s.set_property(fs, "opacity", 0.4, merge=True)
    s.stack.end_merge()
    s.undo()
    assert lines == [
        "session.add_item('free_surface')",
        "session.set_property('free_surface-1', 'opacity', 0.5)",
        "session.set_property('free_surface-1', 'opacity', 0.4)  # final value",
        "# undo: Set opacity",
    ]


def test_core_runs_without_qt(tmp_path):
    """A recorded script runs in a fresh interpreter without importing Qt."""
    s = Session()
    build(s)
    script = tmp_path / "replay.py"
    script.write_text(
        s.to_script()
        + textwrap.dedent("""
        import sys
        assert "PySide6" not in sys.modules, "core imported Qt"
        print(len(session.scene))
    """)
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(len(s.scene))
