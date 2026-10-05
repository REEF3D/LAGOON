# Plots and wave profiles

## Time series

Each **Time series** item is a row in the **Plots** panel under the 3D view, all
rows on one time axis: wave gauges (`WSF`), 6DOF motions and velocities, forces.
A case opened with such files gets them as items by itself; *Add → Time series*
adds more. In the item's properties:

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

## Wave profiles

**Wave profile** items show the surface elevation along the lines of REEF3D's
`WSFLINE` output, at the current time, in the **Wave profiles** panel (a tab
beside the plots), and as lines in the 3D view (*Profile in 3D*). *Lines* picks
some of the lines (`1, 3`); with *Wave theory* on, the theory FNPF wrote is drawn
dashed. The value axis stays fixed while playing, so the waves move through a
steady frame.

## Spectra and comparisons

The spectra of the plotted series (with $H_{m0}$ and $T_p$), and comparisons with
measured data, are made as paper figures: see [Paper figures](figures.md).
