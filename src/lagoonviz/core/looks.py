# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""What the core knows of the looks: where each one puts its sun.

The looks themselves (colours, materials, surroundings) are the renderer's
(``lagoonviz.render.cinematic``); the sun's place is shared, so a session can
start from it and scripts know it without a renderer.
"""

#: the unit-free direction towards the sun of each cinematic look, z up
LOOK_SUNS: dict[str, tuple[float, float, float]] = {
    "open_ocean": (-0.75, -0.35, 0.56),  # low, from the side: long shadows, a glint
    "wave_flume": (0.0, 0.25, 1.0),  # the ceiling lights, nearly overhead
    "underwater": (-0.2, -0.1, 1.0),  # light from above through the surface
    "diorama": (-0.5, -0.6, 1.0),  # high and to the front: soft, short shadows
}

#: where the engineering looks (studio, ocean) put the sun
DEFAULT_SUN: tuple[float, float, float] = (-0.9, -0.4, 0.6)
