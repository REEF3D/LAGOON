# Plots and free surface lines

## Time series

Each **Time series** item is a row in the **Plots** panel under the 3D view, all
rows on one time axis: wave gauges (`WSF`), probes, 6DOF motions and velocities,
forces, run-up and bed gauges. A case opened with such files gets them as items by
itself, switched off: tick them in the scene to show their rows and their gauges in
3D. *Add → Time series* adds more. They are named as REEF3D's User Guide names
the print-outs:

| Name | Print-out | File |
| --- | --- | --- |
| Wave gauges, Wave gauges (theory) | P 51, P 50 | `REEF3D_*_WSF/...-WSF-HG(-THEORY).dat` |
| Velocity probe n, ... (theory) | P 65, P 66 | `REEF3D_*_ProbePoint/...-Vel-Probe(-Theory)-n.dat` |
| Point probe n | P 61 | `REEF3D_CFD_ProbePoint/...-Probe-Point-n.dat` |
| Depth-averaged point probe n | P 63 | `REEF3D_SFLOW_ProbePoint/...` |
| Pressure probe n | P 64 | `REEF3D_CFD_PressureProbe`, `REEF3D_NHFLOW_ProbePoint/...-Probe_Press-n.dat` |
| Force box n | P 81 | `REEF3D_*_Force/..._Force-n.dat` |
| Morison force n | P 85 | `REEF3D_*_Force_ALE/..._ALE_Force-n.dat` |
| Run-up gauges, Maximum run-up gauges | P 133, P 134 | `REEF3D_NHFLOW_RUNUP` |
| Cylinder run-up n | P 140 | `REEF3D_FNPF_Runup` |
| Bed level gauges, Bed shear stress gauges, Maximum ... | P 121, P 125, P 122, P 126 | `REEF3D_*_Sediment` |
| Theoretical wave time series n | P 58 | `REEF3D_Log-Wave` (not shown by itself: REEF3D always writes one) |
| Body n forces, Body n motion, Mooring line n | 6DOF | `REEF3D_*_6DOF` |

Tables headed `t` (the probes), headed without the time (the run-up gauges) or not
headed at all (CFD's pressure probes) are read too: the time is their first
column. In the item's properties:

- **File**: which file of the case (gauges, a body's forces or motion, ...).
- **Columns**: which of its columns, e.g. `P1, P4` or `Fx, Fz` (empty: all).
- **Gauges in 3D**: the gauges as staffs in the 3D view, with a dot at the water
  surface that follows the time.

The orange line is the current time. In a plot:

| | |
| --- | --- |
| click or drag | move the time (and the 3D view) there |
| wheel | zoom the time axis |
| right or middle drag | move along the time axis |
| double-click | show everything |

The legend shows each line's value at the current time. A point picked on a
surface in the 3D view ([Reading values](scene.md#reading-values)) is plotted
too, in orange.

A time series item says how its legend looks: *Line names* (`P1 = gauge 1;
P4 = gauge 4`; also in the figures made from the plots), *Legend* above the
plot, in the plot (its upper right corner) or off, and *Values at the cursor*.

## Measured data

A time series item takes a **Measured file** (*…* to choose it; a path relative to
the case folder is kept relative): a text file with the time in the first column
and one column per gauge, separated by spaces, tabs, commas or semicolons; lines
starting with `#` or `%` and a header line of names are allowed. Its columns are
drawn **dashed** in the colour of the line they go with: the column of the same
name (`P1`), else in order (the first value column for the first line plotted).
*Measured columns* says otherwise (`P1 = 2; P4 = G4`), *Measured time shift* moves
the measured times (the moment the wave maker started).

In the row's lower left corner each line is compared with its measurement **over
the time shown** (zoom in to compare a part): RMSE, bias (computed minus measured),
correlation r, Willmott's index of agreement d (1: perfect) and the lag (the shift
of the computed line that matches best; positive: it comes later). Figures made
from the plots draw the measured lines dashed too.

## Free surface lines

**Free surface line** items show the surface elevation along REEF3D's free surface
lines, at the current time: lines along x (P 52, `WSFLINE`), along y (P 56,
`WSFLINE_Y`, *Free surface lines in y*) and NHFLOW's time-averaged lines (P 146,
P 147: one item per line, at the end of its averaging window). A case opens with
them switched off; ticked, they are drawn in the **Free surface lines** panel (a tab
beside the plots), and as lines in the 3D view (*Lines in 3D*). *Lines* picks
some of the lines (`1, 3`); with *Wave theory* on, the theory FNPF wrote is drawn
dashed. The value axis stays fixed while playing, so the waves move through a
steady frame.

## Wave analysis

*View → Wave analysis* works over a time window (the Plots panel's when it opens;
*From*, *to*) with Welch's method (*segments*, half overlapping):

- **Spectra**: the lines ticked (measured ones dashed) and a table of their wave
  parameters: $H_{m0} = 4\sqrt{m_0}$, $T_p$, $T_{m01} = m_0/m_1$,
  $T_{m02} = \sqrt{m_0/m_2}$ from the spectrum; $H_{1/3}$, $H_{max}$, $T_{1/3}$ and
  the number of waves from zero down-crossings.
- **Reflection**: two or three gauges along the wave direction (their x from the
  gauge file), and the water depth (the case's still water level to start with):
  the incident and reflected spectra fitted per frequency (Mansard and Funke; with
  two gauges Goda and Suzuki), $K_r$ and the incident and reflected $H_{m0}$.
  Frequencies at which a gauge spacing is near a multiple of half a wave length
  cannot be separated: shaded, and left out of $K_r$.
- **Response (RAO)**: a wave gauge in, a motion out (a 6DOF column): gain
  $|S_{xy}|/S_{xx}$, phase (positive: the motion comes later) and coherence; the
  gain is drawn strong where the coherence reaches the value asked for, and the
  largest RAO is given with its period.

*Copy table* puts the numbers on the clipboard (tab separated, for a spreadsheet),
*Save figure* writes the tab's figure as PDF, SVG or PNG. In a script:
`lagoonviz.core.waves` (`zero_crossing`, `reflection`, `transfer`) and
`lagoonviz.core.analysis` (`spectrum`, `wave_parameters`, `compare`), drawn with
`lagoonviz.render.wave_figures`.

Spectra and comparisons with measured data are also made as paper figures: see
[Paper figures](figures.md).
