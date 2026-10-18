# 2D plots

*View → 2D plots* (++ctrl+shift+d++, the third mode button next to *Set-up* and
*Results*) fills the window with the figure editor: the figures of a project made
in LAGOON rather than in a separate plotting program. The panels make way; the
timeline stays, so free surface lines and 3D views in the figure follow its time.
*Set-up* or *Results* bring back the 3D view with the panels as they were. The
figure shown is kept for the next start.

What is drawn:

- the time series of the open runs (gauges, forces, motions), with measured files
  next to them;
- **tables** of columns: data files (measurements, published results, a
  spreadsheet), a time-series file as columns, the free surface lines at a moment,
  or values collected over all open runs (one row per run);
- new columns from **expressions** of the others;
- **x-y panels**: any column against any other, as lines, points, steps, bars or a
  filled area, a right y axis, logarithmic axes, and fitted curves with their
  equation and $R^2$ in the legend.

The figure is written as for a paper ([Paper figures](figures.md)): journal
widths, fonts in points, vector PDF/SVG/EPS or PNG/TIFF, and kept as a
[template](figures.md#templates) that draws the same figure for the next run.

## Tables

*Add table* (under the list of sources, left) takes:

| Table from | |
| --- | --- |
| Data file | text columns (spaces, tabs, commas or semicolons; a header line of names, units in brackets such as `Fz [N]`; `#` comments; decimal commas with semicolons) or an Excel file (`.xlsx`, the first sheet) |
| Time-series file | a REEF3D time-series file of the run (`nhflow_wsf_hg`, the forces, ...): its time and columns |
| Free surface lines | the lines at the current time: x and one column per line |
| Values over all open runs | one row per open run, a value per line written `name = file: expression`, e.g. `Hm0 P3 = nhflow_wsf_hg: hm0(P3)` or `Fz max = nhflow_6dof_forces_0: max(Fz)` |

The table's columns are then in the list of sources: drag one onto the figure (or
double-click it) and it is drawn against the table's first column on a new x-y
panel. The line's settings choose another *x column*, how it is drawn (*Draw as*:
line, scatter, step, bar, fill), the *y axis* (left or right) and a *Fit* (linear,
quadratic, cubic, exponential, power, logarithmic).

A data file's path is kept relative to the template when it is saved, so a folder
of template and data can be moved together.

## The Data tab

Beside the figure, the *Data* tab shows a table's rows. **Add column** takes
`name = expression` in the table's columns by name:

```text
eta_cm = 100 * P1
U = sqrt(u**2 + v**2)
H = max(eta) - min(eta)
late = where(t > 20, eta, nan)
smoothed = smooth(Fz, 11)
Kr2 = Kr**2
```

The expressions know `+ - * / ** %`, comparisons, `&`, `|`, `x if c else y`,
the constants `pi`, `e`, `g` (9.81), `nan`, `inf`, and the functions `sin`, `cos`,
`tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `sinh`, `cosh`, `tanh`, `exp`,
`log`, `log10`, `sqrt`, `abs`, `sign`, `floor`, `ceil`, `round`, `deg`, `rad`,
`where`, `clip`, `isnan`, `minimum`, `maximum`, `interp`, `cumsum`, `diff`,
`gradient`, `smooth`, `mean`, `median`, `std`, `var`, `min`, `max`, `rms`, `hm0`.
A name with spaces or brackets is written `col("Fz [N]")`. A column that cannot be
computed is named under the table with the reason.

*Remove column* takes away a computed column, *Remove table* the table and its
lines, *Copy* puts the table on the clipboard (tab separated, for a spreadsheet).

## Live

*Live* (under the figure) draws the figure again when its data change: a table's
file written again, the time series of a running simulation. With
[Follow running simulation](runs.md) the plots show a run as it goes.

## New figures

*New* starts an empty figure and keeps the tables, which are often drawn again;
*Open template…* and *Save template…* switch between figures.

## In scripts

```python
from lagoonviz.core.tables import TableSpec, fit, read_table

flume = TableSpec("flume", "file:flume.dat", formulas={"Kr2": "Kr**2"}).load(session)
line = fit(flume.columns["T"], flume.columns["Kr"], "linear")
print(line.equation, line.r2)
```

`lagoonviz.core.tables.evaluate(expression, columns)` evaluates an expression on a
dictionary of arrays.
