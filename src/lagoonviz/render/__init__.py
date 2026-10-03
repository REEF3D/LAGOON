# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON render layer: pygfx scenes now; matplotlib figures and path tracing later.

* :mod:`lagoonviz.render.scene3d`: the 3D scene drawn with pygfx (Metal, Vulkan, D3D12)
* :mod:`lagoonviz.render.offscreen`: the same scene rendered to an image, headless

This layer may import the core, never Qt. Importing this package does not import
pygfx, so the application can start (with a placeholder viewport) without a GPU.
"""
