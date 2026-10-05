# Films

*File → Export animation…* (++ctrl+shift+e++) films the 3D view over the
timeline into an MP4. Frames are drawn off screen at the size of the film, so the
film is smooth whatever the window does. A progress dialog shows how far it is;
*Stop* ends the film early and keeps what is done.

| Setting | |
| --- | --- |
| File | where the `.mp4` goes (by default next to the case) |
| Size | the 3D view's shape at Full HD, HD or 4K width, or 16:9 sizes up to 4K |
| Outputs | from, to, and every how many outputs |
| Frames per second | of the film |
| Show | the time and colour bars on the frames |
| Renderer | LAGOON's own (fast, as the 3D view draws), or Blender Cycles (path traced, see [below](#path-traced-films)) |

The film shows what the 3D view shows: its items, look and camera, or the camera
path when it is on ([Timeline and camera paths](timeline.md)). Pick a size with
the 3D view's shape so the film frames exactly what the view shows; a different
shape shows more at the sides or above and below.

From 2560 pixels wide, a Mac encodes the film with its video hardware, which
keeps 4K films fast. Without `imageio-ffmpeg`, numbered PNG frames are written
instead of the MP4.

## Without a screen

```bash
lagoon-render flume.lagoon-session flume.mp4 --size 1920x1080 --fps 25
lagoon-render ~/runs/flume flume.mp4 --start 100 --end 400 --every 2 --style ocean
```

A saved session brings its items, view and camera path; a case folder is shown as
it opens. See [Command line](../reference/cli.md#lagoon-render).

## Path-traced films

With *Renderer: Blender Cycles*, each frame is path traced by Blender: light
reflected and refracted by the water, soft shadows. This takes minutes per frame
instead of a fraction of a second; see
[Cinematic looks and path tracing](cinematic.md#path-tracing-with-blender).
