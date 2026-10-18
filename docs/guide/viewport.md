# The 3D view

## Moving around

| | Mouse or trackpad | Blender style |
| --- | --- | --- |
| Turn | left drag | middle drag |
| Move | right drag | ++shift++ + middle drag |
| Zoom | wheel or pinch | ++ctrl++ + middle drag |

The view turns about the middle of what it shows, all the way round. Keys (with
the mouse over the view):

| Key | View |
| --- | --- |
| ++1++, ++3++, ++7++ | front, right side, top (with ++ctrl++: back, left, bottom) |
| ++5++ | perspective or orthographic |
| ++n++ | north up |
| ++p++ | Probe: a click lists every value at a point (++escape++ ends it) |
| ++2++ | the 2D view (see below) |
| ++home++ or ++period++ | frame everything |
| arrow keys | move the view (with ++shift++: further) |
| ++space++ | play or pause |

![The flume from the side, in perspective](../images/view-studio.png)

## The 2D view

++2++ switches to a flat, orthographic view without rotation, where any drag
moves the view. In it, ++1++, ++3++ and ++7++ choose the side it looks from: from
the top every item is drawn flat (the free surface as a plane coloured by its
values, over the bed) but lit as in the 3D view, in the same look and light, so
waves and the bed's relief stay shaded; from the front or side it is an elevation
of the shapes.
++2++ again goes back to 3D. The buttons in the corner of the view do the same.

*North up* in the view's bar (++n++) turns the view so that north points up the
screen, in 2D and in 3D (the tilt kept): north of the case's georeference
([Maps](maps.md)), else the model's y. A *Background map* under the case follows the 2D view, fetched
again in finer detail when you zoom in and for the new area when you pan; a bar
above the view counts its tiles while they come.

![The 2D view from the top](../images/view-top.png)

## Looks

*View → Viewport look* sets how the 3D view is drawn:

- **Studio** (light grey, the default) and **Ocean** (dark blue): plain, even
  lighting for reading values.
- **Open ocean**, **Wave flume**, **Underwater**, **Diorama** and **Diorama
  classic**: the cinematic looks, with physically based materials, reflections
  and, under water, fog. Open ocean, Wave flume and Diorama show what Blender
  renders: ÆgirFrames' sky picture around the model (lighting it and reflected by
  the water, its sun where the look's sun is) and structures in ÆgirFrames' red
  (in their default colour); the bed stays LAGOON's light sand (path traced: dark
  rock). Diorama classic keeps the earlier Diorama: a blue backdrop. See
  [Cinematic looks and path tracing](cinematic.md).

## The sun

*View → Sun...* moves the light: **Direction** turns it around the vertical
(0° from +x, 90° from +y) and **Height** lifts it from just over the horizon
(long shadows) to straight overhead. The 3D view follows while the sliders move.
Your sun is saved with the session and used by films and by path tracing too;
**Look's own** gives the sun back to the look. In a script:

```python
session.set_sun(azimuth=200, elevation=20)  # degrees
session.set_sun()                           # the look's own sun again
```

*View → Theme* switches the window between light and dark.

![The Ocean look](../images/view-ocean.png)

## Compare

*View → Compare side by side* (++ctrl+shift+2++) splits the view in two, one case
in each (with two cases open). The cameras and the time move together, so the
same wave can be compared in two runs.

## Full screen

*View → Full screen viewer* (++ctrl+shift+f++) shows only the 3D view and the
player on the whole screen, for presentations; ++escape++ goes back.

## Images and screenshots

*File → Export image…* (++ctrl+shift+i++) draws the 3D view alone again, off
screen, as large as you choose: as on the screen, twice as sharp, Full HD, 4K or
8K wide, a width in pixels of your own, or a width in centimetres at a print
resolution (17 cm at 300 dpi: a journal's page width; the resolution is kept in
the file). The image frames what the view shows, and lines, markers and text keep
their share of it: a 4K image is the view drawn sharper, not smaller. The time and
the colour bars go where the legend layout puts them (*Legend…*). PNG, JPEG or
TIFF, by the file's end; the GPU draws up to 8192 pixels a side. In a script:

```python
from lagoonviz.render.offscreen import render_image
render_image(session, "view.tif", (2008, 1130), scale=2.5, legend=True, dpi=300)
```

*File → Save screenshot…* (++f12++) saves a quick picture of the whole window with
the 3D view in it. For a paper figure with axes, see *File → Export figure… → 3D
view*.

## Without a GPU

Without a usable GPU (or over a remote desktop without one), the window opens
with a note in place of the 3D view; plots, figures and scripts still work, and
`lagoon-render` can draw with Mesa's software driver on Linux.

## Speed or memory

*View → Memory use* chooses how much LAGOON keeps in memory to be fast:

| | Frames kept | Read ahead while playing | |
| --- | --- | --- | --- |
| **Performance (in memory)** | a quarter of the computer's memory (2 – 16 GB) | the whole play range, 6 threads | runs from memory from the second time round; a volume's cells kept for slices and iso-surfaces |
| **Balanced** (the default) | 1 GB | 8 frames, 4 threads | as LAGOON always did |
| **Low memory** | 256 MB | 2 frames, 2 threads | the cells read again for every slice: large volumes fit, playing is slower |
| **Streaming (very large cases)** | 768 MB | 6 frames, 4 threads | surfaces of more than 2.5 million points shown coarser while playing or dragging the timeline, at full detail as soon as the time stands still |

Frames played once come back from memory without reading the files again, as
long as they fit; films read ahead in the same measure. Whatever the mode, LAGOON
never reads further ahead than the frames fit (they would push out each other
before being shown). The choice is kept for the next start and applies at once
(frames beyond the new budget are let go). For `lagoon-render` and scripts,
`LAGOON_MEMORY=low` (or `performance`, `streaming`) does the same.

### Very large cases

Cases of 100 million cells and more (a free surface of 10 million points and
more) open in a moment: LAGOON reads a frame when it is shown, not before.

- **Streaming** is for playing them. While the timeline plays or is dragged, a
  free surface from a LAGOON store is read at every 2nd, 3rd, ... grid line, to
  about 2.5 million points; when it stops, the frame comes in at full detail. The
  bed and other surfaces that do not change are always at full detail. Films and
  stills are always made at full detail, whatever the mode. Opening a case of
  more than 5 million points a frame in another mode, the status bar says so.
- Items that are **switched off** (the volume a case opens with, for one) are
  not read at all, neither while playing nor for films: tick them to see them.
- Fields that are the **same in every output** (FNPF's still water depth, its
  coastline) are read once.
- The **automatic colour range** of a large surface comes from tiles spread over
  it in a few outputs, not from whole outputs.

