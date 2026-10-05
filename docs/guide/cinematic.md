# Cinematic looks and path tracing

For showcase pictures rather than for reading values.

## Cinematic looks

*View → Viewport look* has three cinematic looks besides Studio and Ocean:

=== "Open ocean"

    A sky with a low sun over the sea, reflected by the water.

    ![Open ocean](../images/look-open-ocean.png)

=== "Wave flume"

    A lab hall with rows of ceiling lights; neutral and bright.

    ![Wave flume](../images/look-wave-flume.png)

=== "Underwater"

    The camera below the surface: blue depth fog, light from above.

    ![Underwater](../images/look-underwater.png)

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
floor, filled with water that absorbs light on its way through (shallow water is
clear, deep water takes the look's water colour) and, under water, scatters it.
Its surface reflects the sky and the sun and carries small wind ripples that break
the reflections into glitter. Water coloured by a variable stays an opaque,
coloured surface, so its colours read.

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
