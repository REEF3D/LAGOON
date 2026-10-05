# Scripting

Everything the window does goes through one object, `lagoonviz.core.Session`: the
GUI calls its methods, and so can a script, a notebook or a batch job on a
cluster, without Qt and without a screen.

## A session in Python

```python
from lagoonviz.core import Session

session = Session()
case = session.open_case("~/runs/flume")         # the case, its free surface and bed
session.set_property("free_surface-1", "colormap", "ocean")
session.set_exaggeration(case, 8.0)               # the case's vertical scale
session.add_slice("y", 2.0)                       # a slice through the volume at y = 2 m
session.set_time(12.5)
session.save("flume.lagoon-session")              # open it in the window later

mesh = session.data.load("free_surface-1")        # the current output as NumPy arrays
print(mesh.points.shape, mesh.point_data["eta"].max())
```

Item ids are the kind and a number (`free_surface-1`, `slice-1`); `session.scene`
lists them. Every change is a command: `session.undo()` and `session.redo()` work
as in the window.

## The window writes the script

The **Python console** (a panel of the window) shows every command the window
runs, and runs Python with `session` in scope. *File → Export script…* writes the
whole session as a script:

```python
# LAGOON session script, recorded from the command history
from lagoonviz.core import Session

# figures, films and screenshots: made again only when run as a script
MAKE_OUTPUTS = __name__ == "__main__"

session = Session()
session.open_case('/Users/me/runs/flume')
session.set_property('free_surface-1', 'colormap', 'ocean')
if MAKE_OUTPUTS:  # File > Export animation
    from lagoonviz.render.movie import frame_times, write_movie
    times = frame_times(session, None, None, 1)
    write_movie(session, '/Users/me/runs/flume/flume.mp4', times, (1920, 1080), 25.0, 'studio')
```

The figures, films and screenshots made in the window are in it, each after the
change it followed. `python flume.py` makes them all again, without the window;
for another run of the same set-up, change the `open_case` path.
`Session.from_script(text)` rebuilds only the scene (the outputs are left out).

## Pictures and films

```python
from lagoonviz.render.offscreen import render_session, write_png
from lagoonviz.render.movie import frame_times, write_movie
from lagoonviz.render.figure import FigureSpec, save_figure, series_figure
from lagoonviz.render.pathtrace import path_trace

write_png("view.png", render_session(session, (1920, 1080), style="ocean"))
write_movie(session, "flume.mp4", frame_times(session), (1920, 1080), 25.0)

spec = FigureSpec.from_style("Journal (serif)", width_mm=90)
save_figure(series_figure(session, spec), "gauges.pdf", spec)

session.orbit_camera(degrees=360)                  # a camera path around the scene
path_trace(session, "showcase.mp4", size=(1920, 1080), look="open_ocean", samples=64)
```

The [Session API](reference/api.md) lists all methods, and the
[command line](reference/cli.md) does the common cases without Python.
