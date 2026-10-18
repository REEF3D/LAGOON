# First steps

This walks through one run from opening it to a figure and a film. Any REEF3D
case folder does; the pictures show a small wave flume with three wave gauges and
a moored floating box.

## Open the run

Start LAGOON and choose *File → Open case…* (++ctrl+shift+o++), then the folder
REEF3D wrote its `REEF3D_*` folders into (the folder REEF3D ran in; choosing its
`REEF3D_Case` folder opens it too). A single output folder, such as
`REEF3D_FNPF_WSF`, opens only that output. From a terminal:

```bash
lagoon ~/runs/flume
```

LAGOON finds what the run wrote and fills the **Scene** panel on the left: the
case, its free surface and its bed, and, if the run wrote them, the wave gauges,
free surface lines, floating bodies and mooring lines.

![The flume in the 3D view](../images/view-studio.png)

## Look around

In the 3D view, drag with the left button to turn, with the right button to move,
and scroll to zoom. The number keys give the standard views (1 front, 3 side,
7 top, 5 perspective or orthographic) and ++home++ frames everything.

Click an item in the Scene panel to see its settings in the **Properties** panel
on the right. Set *Colour by* of the free surface to `velocity`, or raise the
*Vertical scale* of the case to make small waves visible. *Probe* in the view's
bar (++p++), then a click on the surface, lists every value there in a box beside
the view; they follow the time.

## Play

The **Timeline** under the 3D view plays the outputs (++space++ in the 3D view),
steps through them and scrubs with the slider. Frames are read in the background
and kept, so going back is instant.

## Plot

Wave gauges, forces and motions appear in the **Plots** panel, all on one time
axis. The orange line is the current time: click or drag in a plot to move the 3D
view to that moment.

## Make a figure

*File → Export figure…* (++ctrl+shift+p++) draws the plots again as a paper
figure: choose a style (REEF3D, a serif journal, slides), a width (single, 1.5 or
double column) and a file (PDF, SVG, EPS, PNG, TIFF).

![Time series as a paper figure](../images/figure-series.png)

## Make a film

*File → Export animation…* (++ctrl+shift+e++) films the 3D view as it is over the
outputs you choose, into an MP4. For a moving camera, add camera keyframes on the
timeline first ([Timeline and camera paths](timeline.md)).

## Keep it

*File → Save session* (++ctrl+s++) keeps everything (the scene, the settings, the
view, the time) in a `.lagoon-session` file. *File → Export script…* writes it as
a Python script that does the same without the window:

```bash
lagoon-render flume.lagoon-session view.png      # the saved view as an image
python flume.py                                   # the script: figures and films again
```
