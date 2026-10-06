# Cinematic looks and path tracing

For showcase pictures rather than for reading values.

## Cinematic looks

*View → Viewport look* has four cinematic looks besides Studio and Ocean:

=== "Open ocean"

    A sky with a low sun over the sea, reflected by the water.

    ![Open ocean](../images/look-open-ocean.png)

=== "Wave flume"

    A lab hall with rows of ceiling lights; neutral and bright.

    ![Wave flume](../images/look-wave-flume.png)

=== "Underwater"

    The camera below the surface: blue depth fog, light from above.

    ![Underwater](../images/look-underwater.png)

=== "Diorama"

    The model on a stand against a blue backdrop, in soft light: the bed a solid
    block of sand, the water see-through with blue sides down to the bed (path
    traced: a clear blue body of water).

    ![Diorama](../images/look-diorama.png)

They use physically based materials: smooth water that reflects its
surroundings, sand and rock with a fine grain, concrete, steel, ice and wood (the
*Material* of bodies, structures and other objects). The surroundings are made by
LAGOON; nothing is downloaded. Water coloured by a variable keeps its colours;
coloured by nothing, it takes the look's water colour.

## Path tracing with Blender

[Blender](https://www.blender.org)'s Cycles path tracer renders the same scene with
light that bounces: reflections and refraction through the water, soft shadows,
light through the surface onto the bed. LAGOON runs Blender in the background as
a separate program; Blender's window never opens.

Clear water (a free surface coloured by nothing) goes to Blender as a body of
water: the surface, walls along the sides of the domain and the shore, and a
floor. Above water it is rendered as [ÆgirFrames](https://github.com/NTNU-MCS/aegir-frames)
renders the ocean: glass in the look's water colour that refracts and reflects
what the camera sees, while light, shadows and reflections pass through it, so the
bed and the structures under it are lit as in air and cast their shadows on the
bed; a faint haze fills it. It is subdivided when rendered, for smooth crests
(twice, unless its mesh is already as fine as the picture's pixels). In the Open
ocean look it carries small wind ripples that break the reflections into
glitter. Under water the water absorbs light the deeper it goes and scatters
it. Water coloured by a variable stays an opaque, coloured surface, so its colours
read. Where the water thins out to nothing at the shore it fades away, so the sand
shows instead of a sheet of glass.

The **Open ocean**, **Wave flume** and **Diorama** looks are path traced as
ÆgirFrames shows a case: the domain as a model floating in ÆgirFrames' sky
picture, a panorama of a clear sky over the sea with a low sun, which lights the
model and is seen behind it. The picture is turned so that its sun stands in the
sun's direction (*View → Sun...*); its sun's height is the picture's own, about
30°. A bed in its default colour is ÆgirFrames' dark rock in the Wave flume and
Diorama looks, and structures and bodies in their default colours are
ÆgirFrames' red (half metallic, matte); colours you choose are kept.

In the **Open ocean** look a coast looks like one: the bed (coloured by nothing)
is dressed by its height above the still water level and its slope, with sand
under water, wet sand at the water line, dry sand, grass and heath higher up, and
rock where it is steep. At the sides of the domain the land is cut off in rock
down to a base, and the water ends in clear walls, with small wind ripples on it
and broken foam just off the beach.

1. Install Blender 4.2 or newer. LAGOON finds it in Applications (macOS), on the
   `PATH`, or where the environment variable `LAGOON_BLENDER` points.
2. Choose a cinematic look (with Studio or Ocean, Cycles uses Wave flume or Open
   ocean).
3. *File → Render still with Cycles…* for one picture of the 3D view as it is, or
   *File → Export animation… → Renderer: Blender Cycles* for a film.

```bash
lagoon-render flume.lagoon-session still.png --engine cycles --style open_ocean
lagoon-render flume.lagoon-session film.mp4 --engine cycles --samples 64 --size 1920x1080
```

### Foam, spray and bubbles

Path traced films and stills show white water where the free surface breaks:
spray thrown off the crests, foam drifting on the surface and fading, bubbles
under it rising to the top. It is born where REEF3D flags breaking (NHFLOW's
`breaking`), on sharp crests whose water runs forward fast, and where the
surface flow converges and traps air, and it moves with the free surface's
velocity from output to output. Its size and lifetime scale with the case (the
cells and the water depth), so a flume and a coast look alike. A still runs the
foam from a few wave periods before its time, so it has formed.

It needs the free surface's velocity on its grid (NHFLOW and FNPF write it) and
the output times in seconds. *Foam and spray* in the export dialog, or
`--no-foam` on the command line, turns it off.

### Time

Path tracing takes from seconds to minutes per frame, depending on the size, the
samples and the GPU. The progress dialog shows what Blender is doing (its first
run on a computer compiles its GPU code, which takes a few minutes once).

- **Samples** per pixel (default 64): the images are denoised, so 32 to 64 are
  usually enough; 16 for a quick look.
- **Size**: HD takes half the time of Full HD, 4K four times as long.
- Blender uses the GPU (Metal, OptiX, CUDA, HIP or oneAPI) when there is one;
  the first lines of its log say which.

A stopped film goes on where it stopped: the scene and frames are kept in
`<film>.cycles/` beside the film, and when the film is made again, frames whose
data, camera and settings have not changed are kept. Blender's own output is in
`<film>.cycles/blender.log`.
