# Scene items

Everything shown is an item in the **Scene** panel: the cases, the surfaces and
volumes drawn from them, the time series and free surface lines plotted from them. The
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
| AMR surface | the surface of an adaptive-mesh run, finer patches over coarser ones, with the patch outlines; *AMR patches* off shows the base grid (level 0) alone, *Grid lines* draws the cells of every level |
| Bed | the bathymetry (sand coloured by default) |
| Volume | the outside of the volume output |
| Slice | the volume cut by an x, y or z plane |
| Iso-surface | where a variable of the volume has a given value; on the free surface or the bed, the line where it has it |
| Sea state | SEASTATE's map of the sea state: Hs, periods, direction, with arrows where the waves go ([Opening runs](runs.md)) |
| Iso-volume | the part of the volume where a variable lies in a range (the water of a CFD run: phi from 0 up); on the free surface or the bed, the part of the surface in the range |
| Streamlines | lines along a vector field, through the volume, in a plane or on the free surface |
| Time series | wave gauges, 6DOF motions or forces, plotted under the 3D view |
| Free surface line | the surface along the lines of `WSFLINE`, with the wave theory (FNPF) |
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
- **Colour bar**: whether the item has one, under the 3D view, on films and in
  figures. *Bar title* (empty: name · variable; `$\eta$ [m]` for symbols), *Bar
  numbers* (auto, integer, 1 to 3 decimals, scientific) and *Bar ticks* (0: the
  ends under the view and on films, round numbers in figures; 1: the ends and the
  middle; 2 to 11: that many, evenly spaced) are the bar's in all of them.
- **Contour lines of** a variable, and how many.
- **Vectors of** a vector field: arrows spread over the surface, how many, and
  their length.

Bodies, structures and the other objects have a **Material** (steel, concrete,
rock, ice, wood): its colour, and how the [cinematic looks](cinematic.md) draw it.

## Vertical scale

Waves are small next to the length of a domain. The case's *Vertical scale*
stretches the heights of all its items about the still water level (surface, bed,
bodies, slices, gauges alike). With *Sync* off on an item, that item gets its own
scale. The free surface can leave out dry land (*Hide dry land*, on by default),
so only the wet free surface shows: FNPF's surface where it has no water depth,
NHFLOW's and SFLOW's over their dry cells (`wetdry`), the AMR surface likewise.
The bed can leave out what is under still water (*Hide wet land*).

## Reading values

Click on a surface in the 3D view to read its values there (in the status bar):
the orange marker stays where it was put and its values follow the time, and the
Plots panel adds them over time. Click on empty space to remove it.

## Slices, iso-surfaces, streamlines

With several cases open, a new slice, iso-surface, iso-volume or streamlines cut
the case of the item selected in the scene tree (else the first case that has a
volume); *Case* in its properties puts it on another case, through the middle of
that case's volume if its plane would miss it. They are made of the data of the
item selected: with a volume, a slice or an iso item selected, of that volume (a
case may have several); with the free surface or the bed selected, an iso-surface
or iso-volume of that surface. *Dataset* in the properties chooses another (slices
and streamlines: the volumes; iso items: the volumes and the surfaces).

Or choose the data as you add the item: *Add → Iso-surface* (iso-volume, free
surface, bed, slice, streamlines) lists *Automatic* (as above) and every output of
the open cases the item can show: an iso-line on SFLOW's bed, an iso-volume of the
NHFLOW volume, the SFLOW free surface beside the NHFLOW one. In the toolbar, the
arrow beside the button opens the same list. In a script:
`session.add_of("iso_surface", "sflow_bed")` (`case=` with several cases open).

On a surface an **iso-surface** is the line where the variable has the *Value*
(eta = 0: the still water line on the free surface; a depth contour on the bed),
coloured as the item says, and an **iso-volume** the part of the surface where the
variable lies between *From* and *To* (the shallows of the bed, the crests of the
waves). In 2D they lie just over their surface.

*Add → Slice* cuts the volume in the middle; set the *Normal* (x, y or z) and the
*Position* in metres; *Grid lines* (dark or white) draws the edges of the cells it
cuts, so the slice shows the grid the solver computed on (on a σ-grid the lines
follow the bed and the free surface). *Add → Iso-surface* draws where a variable
has a *Value*. *Add → Iso-volume* (and its button in the toolbar) draws the part of
the volume where a variable lies between *From* and *To*, as a closed body: for a
CFD run it starts with the level set phi from 0 up, the water itself; else the
upper half of the velocity. The free surface, the bed and the volume have *Grid
lines* too: the edges of their cells (the triangles REEF3D writes, two per cell,
without the diagonals), dark or white, drawn just off the surface.
*Add → Streamlines* starts lines on a seed plane (*Seeds on plane*, *Plane at*)
and follows a vector field through the volume, only in the seed plane, or on the
free surface; *Lines* sets about how many. New streamlines start where you are
looking: in the plane of a selected slice, or on a selected free surface.
