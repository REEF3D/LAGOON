# Scene items

Everything shown is an item in the **Scene** panel: the cases, the surfaces and
volumes drawn from them, the time series and wave profiles plotted from them. The
*Add* menu (and the toolbar, for the common ones) adds items; ++delete++ removes
the selected one (a case with all the items that show it), and *Edit → Undo*
(++ctrl+z++) takes back any change. The check
box beside an item shows or hides it.

Select an item to change it in the **Properties** panel. Each change runs as a
command: it can be undone, it shows in the Python console, and it is in the
script the session writes ([Scripting](../scripting.md)).

## Kinds of items

| Item | Shows |
| --- | --- |
| Case | a REEF3D run; its *Vertical scale* applies to all its items |
| Case set-up | the input files' domain, sides, relaxation zones, gauges, probes, bodies ([Case set-up](setup.md)) |
| DIVEMesh grid | the grid lines on the faces of the domain, the MPI partition ([Case set-up](setup.md)) |
| geo.dat points | the points of `geo.dat`, coloured by height, with their number and spacing ([Case set-up](setup.md#the-points-of-geodat)) |
| Background map | a map or satellite image round a georeferenced case, under the set-up and the results ([Maps](maps.md#background-maps)) |
| Free surface | the water surface, coloured by a variable or plain |
| AMR surface | the surface of an adaptive-mesh run, finer patches over coarser ones, with the patch outlines |
| Bed | the bathymetry (sand coloured by default) |
| Volume | the outside of the volume output |
| Slice | the volume cut by an x, y or z plane |
| Iso-surface | where a variable of the volume has a given value |
| Streamlines | lines along a vector field, through the volume, in a plane or on the free surface |
| Time series | wave gauges, 6DOF motions or forces, plotted under the 3D view |
| Wave profile | the surface along the lines of `WSFLINE`, with the wave theory (FNPF) |
| Bodies, Mooring lines, Structures | floating bodies, their moorings (coloured by tension), fixed structures |
| DEM elements, Ice floes, Rod trees, Particles | the objects of those modules, with trails for DEM and particles |

## Colours

Surfaces, slices and objects share these properties:

- **Colour by**: a variable of the output (a vector's magnitude, or one of its
  components, `velocity.x`), or `none` for a plain colour.
- **Colour map**: `reef3d`, `ocean`, `tension`, `lagoon_blue`, `balance`,
  `viridis`, `rainbow` (Google's Turbo), `sand`, `greys`, `white`.
- **Automatic range** is found over a sample of the outputs, so a colour means the
  same value at every time (slices and iso-surfaces widen it to each output
  shown). *Range from*: the 1st to the 99th percentile, symmetric around zero
  when the values change sign (the default: strong colours, still water white in
  the diverging maps, a single spike does not wash them out), or the lowest to the
  highest value, as ParaView (every extreme in range, e.g. the deepest scour). Switched off, *Range min* and
  *Range max* hold. *Fit range to this frame* sets them to the current output.
- **Range [%]** scales the automatic range: 100 % runs from the lowest to the
  highest value, 95 % from 0.95 × min to 0.95 × max (values beyond take the colours
  at the ends), 50 % brings out the small waves. *Fit range to this frame* uses it
  too.
- **Opacity**: a transparent free surface shows the bed under it.
- **Contour lines of** a variable, and how many.
- **Vectors of** a vector field: arrows spread over the surface, how many, and
  their length.

Bodies, structures and the other objects have a **Material** (steel, concrete,
rock, ice, wood): its colour, and how the [cinematic looks](cinematic.md) draw it.

## Vertical scale

Waves are small next to the length of a domain. The case's *Vertical scale*
stretches the heights of all its items about the still water level (surface, bed,
bodies, slices, gauges alike). With *Sync* off on an item, that item gets its own
scale. The free surface can leave out dry land (*Hide dry land*), the bed what is
under still water (*Hide wet land*).

## Reading values

Click on a surface in the 3D view to read its values there (in the status bar):
the orange marker stays where it was put and its values follow the time, and the
Plots panel adds them over time. Click on empty space to remove it.

## Slices, iso-surfaces, streamlines

*Add → Slice* cuts the volume in the middle; set the *Normal* (x, y or z) and the
*Position* in metres; *Grid lines* (dark or white) draws the edges of the cells it
cuts, so the slice shows the grid the solver computed on (on a σ-grid the lines
follow the bed and the free surface). *Add → Iso-surface* draws where a variable
has a *Value*.
*Add → Streamlines* starts lines on a seed plane (*Seeds on plane*, *Plane at*)
and follows a vector field through the volume, only in the seed plane, or on the
free surface; *Lines* sets about how many. New streamlines start where you are
looking: in the plane of a selected slice, or on a selected free surface.
