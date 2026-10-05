# Timeline and camera paths

## Playing

The **Timeline** panel steps through the outputs of the open cases:

- the buttons go to the first, previous, next and last output, and play or pause
  (++space++ in the 3D view);
- the slider jumps to where it is clicked and follows the mouse while held;
- the number box sets the playback speed in frames per second (up to 240).

The label shows the time in seconds when it is known, and the output number.
Playing waits for each frame to be read, and reads the next ones ahead, so it
never skips a frame; a long run plays as fast as the disk allows.

## Camera keyframes

A film can move the camera. A **camera keyframe** keeps the 3D view as it is at
the current time:

1. Go to a time, set up the view, and press the key button on the timeline (or
   *View → Camera path → Add camera keyframe*, ++ctrl+k++).
2. Go to a later time, change the view, add another keyframe. And so on.

The slider marks the keyframes with small diamonds. With the **camera path** on
(the button beside the key button, or *View → Camera path → Follow camera path*),
the 3D view, and every film made from it, follows a smooth path through the
keyframes: it eases in at the first, passes the others without a jolt, and comes
to rest at the last. Before the first keyframe and after the last, the camera
stays put.

*Remove camera keyframe* takes away the one at the current time, *Clear camera
keyframes* all of them. Keyframes are saved with the session and written into its
script.

## Orbit

*View → Camera path → Orbit around the scene* makes the keyframes of one turn
around the middle of what is shown, from the first output to the last, starting
from the view as it is: a quick showcase film. In a script,
`session.orbit_camera(degrees=90)` turns by any angle.
