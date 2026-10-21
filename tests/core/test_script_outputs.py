"""The recorded script makes the session's figures (and films, screenshots) again."""

import pytest
from reef3d_fixtures import make_run, show_plots

pytest.importorskip("matplotlib")

from lagoonviz.core import Session  # noqa: E402
from lagoonviz.core.figure_template import FigureTemplate, Line, Panel  # noqa: E402
from lagoonviz.render.script import (  # noqa: E402
    cycles_still_code,
    figure_code,
    film_code,
    screenshot_code,
    template_code,
)

SETTINGS = {
    "style": "Journal (serif)",
    "width_mm": 90.0,
    "height_mm": None,
    "font_size": 8.0,
    "font_family": "serif",
    "dpi": 150,
    "usetex": False,
    "title": "",
}


@pytest.fixture
def session(tmp_path):
    run = make_run(tmp_path / "run", frames=4, print_interval=0.5, series=True, profiles=True)
    s = Session()
    s.open_case(run)
    show_plots(s)
    s.set_time(1.0)
    return s


def _run(script: str) -> Session:
    namespace = {"__name__": "__main__"}
    exec(compile(script, "<script>", "exec"), namespace)
    return namespace["session"]


def test_figures_in_the_script(session, tmp_path):
    items = session.data.series_items()
    assert items
    traces = session.data.traces(items[0])
    case = next(item for item in session.scene if item.kind == "case")
    key = next(iter(session.data.case(case.id).series))
    lab = tmp_path / "lab.txt"
    lab.write_text(
        "time WG1\n"
        + "".join(f"{t} {v}\n" for t, v in zip(traces[0].time, traces[0].values, strict=True))
    )
    out = tmp_path / "out"
    out.mkdir()
    session.record_output(
        figure_code(
            "series",
            out / "series.pdf",
            SETTINGS,
            time=1.0,
            items=items,
            window=(0.0, 1.5),
        )
    )
    session.record_output(
        figure_code(
            "spectra", out / "spectra.pdf", {**SETTINGS, "start": 0.2}, time=1.0, items=items
        )
    )
    session.record_output(
        figure_code(
            "validation",
            out / "validation.pdf",
            {**SETTINGS, "measured": str(lab), "measured_column": "WG1"},
            time=1.0,
            line=(items[0], traces[0].name),
        )
    )
    session.record_output(figure_code("profiles", out / "profiles.pdf", SETTINGS, time=1.0))
    template = FigureTemplate(panels=[Panel(lines=[Line(key, traces[0].name, color="#c4302b")])])
    session.record_output(
        template_code(template.to_dict(), out / "template.pdf", dpi=150, time=1.0)
    )
    script = session.to_script()
    compile(script, "<script>", "exec")
    assert script.count("if MAKE_OUTPUTS:") == 5
    # rebuilding the scene does not write the figures
    Session.from_script(script)
    assert not any(out.iterdir())
    # running the script does
    replayed = _run(script)
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(
        ["series.pdf", "spectra.pdf", "validation.pdf", "profiles.pdf", "template.pdf"]
    )
    assert all(p.stat().st_size > 1000 for p in out.iterdir())
    assert replayed.scene.time == 1.0


def test_outputs_follow_their_place_in_the_history(session, tmp_path):
    session.record_output(figure_code("profiles", tmp_path / "a.pdf", SETTINGS, time=1.0))
    session.add_item("slice", normal="y", position=1.0)
    session.record_output(figure_code("profiles", tmp_path / "b.pdf", SETTINGS, time=1.0))
    session.add_item("slice", normal="x", position=1.0)
    script = session.to_script()
    a, b = script.index("a.pdf"), script.index("b.pdf")
    first, second = script.index("normal='y'"), script.index("normal='x'")
    assert a < first < b < second
    # undone past an output: it moves to the end, under a note
    session.undo()
    session.undo()
    script = session.to_script()
    assert "normal='y'" not in script
    assert script.index("# made with changes that were undone") < script.index("b.pdf")
    assert script.index("a.pdf") < script.index("# made with changes")
    # a new session forgets them
    session.reset()
    assert session.outputs == [] and "MAKE_OUTPUTS" not in session.to_script()


def test_film_and_screenshot_code(tmp_path):
    camera = {"position": [1.0, 2.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0], "fov": 50.0}
    film = film_code(
        tmp_path / "film.mp4",
        start=0,
        end=3,
        every=1,
        size=(3840, 2160),
        fps=60.0,
        camera=camera,
        plan=(False, "top"),
        overlay=True,
    )
    shot = screenshot_code(tmp_path / "shot.png", size=(1600, 900), time=1.0, camera=camera)
    for code in (film, shot):
        compile(code, "<code>", "exec")
        assert code.startswith("if MAKE_OUTPUTS:")
        assert "session.set_camera(" in code
    assert "write_movie(" in film and "(3840, 2160)" in film and "legend=True" in film
    assert "render_session(" in shot and "write_png(" in shot


def test_image_code(tmp_path):
    from lagoonviz.render.script import image_code

    code = image_code(
        tmp_path / "view.tif", size=(3840, 2160), scale=3.2, time=2.0, legend=True, dpi=300
    )
    compile(code, "<code>", "exec")
    assert "render_image(" in code and "(3840, 2160)" in code and "scale=3.2" in code
    assert "legend=True" in code and "dpi=300.0" in code
    plain = image_code(tmp_path / "view.png", size=(800, 450))
    assert "scale=" not in plain and "legend" not in plain and "dpi" not in plain


def test_write_image(tmp_path):
    """PNG, JPEG (on white) and TIFF, the print resolution kept."""
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    from lagoonviz.render.offscreen import write_image

    image = np.zeros((20, 30, 4), np.uint8)
    image[..., 0] = 200
    image[:10, ..., 3] = 255  # the lower half transparent
    for name in ("a.png", "a.jpg", "a.tif"):
        write_image(tmp_path / name, image, dpi=300)
        with Image.open(tmp_path / name) as read:
            assert read.size == (30, 20) and round(read.info["dpi"][0]) == 300
    with Image.open(tmp_path / "a.jpg") as read:
        assert read.mode == "RGB" and read.getpixel((5, 15))[1] > 200  # on white
    with pytest.raises(ValueError):
        write_image(tmp_path / "a.bmp", image)


def test_path_traced_film_code(session, tmp_path):
    film = film_code(
        tmp_path / "showcase.mp4",
        start=0,
        end=3,
        size=(1920, 1080),
        fps=25.0,
        style="underwater",
        engine="cycles",
        samples=256,
    )
    compile(film, "<code>", "exec")
    assert "path_trace(" in film and "'underwater'" in film and "256" in film
    assert "write_movie" not in film and "with Blender" in film


def test_cycles_still_code(tmp_path):
    camera = {"position": [1.0, 2.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0], "fov": 50.0}
    code = cycles_still_code(
        tmp_path / "still.png", size=(1920, 1080), time=1.5, style="wave_flume", camera=camera
    )
    compile(code, "<code>", "exec")
    assert "session.set_time(1.5)" in code and "session.set_camera(" in code
    assert "path_trace(" in code and "look='wave_flume'" in code and "samples=64" in code
