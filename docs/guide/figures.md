# Paper figures

LAGOON draws figures for papers again with matplotlib, not as screenshots: at the
width of a journal column, with fonts in points, axis labels with units, vector
output, and LaTeX in labels.

## Export figure

*File → Export figure…* (++ctrl+shift+p++) makes one figure of:

| Figure of | |
| --- | --- |
| Plots (time series) | the rows of the Plots panel, stacked, sharing the time axis |
| Wave spectra of the plots | a spectrum per row, $H_{m0}$ and $T_p$ in the legend, over a time window |
| Compare with measurements | a computed line against a column of a measured file |
| Free surface lines | the free surface lines at the current time |
| 3D view | the 3D view as it is, rendered at the figure's resolution, with its colour bars |

Then:

- **Style**: *REEF3D* (sans serif, the group's colours), *Journal (serif)* (black
  first, serif text), *Presentation* (large text, thick lines, a light grid).
- **Width**: single column (90 mm), 1.5 columns (140 mm), double column (190 mm)
  or any; the height follows the content unless set.
- **Font size**, **Font**, **Resolution** (for PNG and TIFF, and the 3D view's
  pixels), **Labels through LaTeX** (with a LaTeX installation), **Title**.
- **File**: `.pdf`, `.svg`, `.eps`, `.png`, `.tif`.

![Wave spectra](../images/figure-spectra.png)

Labels may hold `$\eta$`-style mathematics without LaTeX installed.

## The figure editor

The figure editor builds figures of several panels. It is the window's
[2D plots](plots2d.md) mode (*View → 2D plots*, or *File → Figure editor…*), which
also draws tables of data in x-y panels:

- **Left**: what can be drawn: the time-series files and free surface lines of the open
  cases, measured files added with *Add measured file*, and the columns of the
  figure's tables (*Add table*).
- **Middle**: the figure's panels (time series, spectra, free surface lines, 3D views)
  and their lines. Drag a column onto a panel, or double-click it, to add it; onto
  empty space for a new panel. Under it, the settings of what is selected: of the
  figure (style, size, a grid of columns, panel letters (a), (b), ...), of a panel
  (title, how many columns it spans, notes, its moment in time; a 3D view panel
  takes the 3D view as it is when asked), of a line.
- **Right**: the figure as it will be written, drawn again after each change; the
  *Data* tab shows the tables and adds columns from expressions.

Notes are placed with `x, y: text` (fractions of the panel) and may hold
`$LaTeX$`.

**Legends.** A panel of lines has its *Legend*: inside where it covers least
(by default only with several lines), in a corner or at a side inside the axes,
outside right, above or below the axes, or off; its columns, text size and a
frame. A line's *In the legend* leaves it out. A 3D view panel puts its
*Colour bars* under the image, right of it, or in a corner of it, with their
*Bar length*; each bar's title, numbers and ticks are the item's
([Colours](scene.md#colours)).

![Free surface lines](../images/figure-profiles.png)

## Templates

The editor keeps a figure as a **template** (`*.lagoonfig`): what it shows (by
file and column names, such as `nhflow_wsf_hg` and `P1`), not the run it came
from. Open the next run of the same set-up, open the template, and the same
figure is drawn for it. In a script:

```python
from lagoonviz.core.figure_template import FigureTemplate
from lagoonviz.render.figure import save_figure, template_figure, template_spec

template = FigureTemplate.load("gauges.lagoonfig")
spec = template_spec(template, dpi=300)
save_figure(template_figure(session, template, spec), "gauges.pdf", spec)
```

Every figure made in the window is also in the session's script
([Scripting](../scripting.md)), so it can be made again without the window.
