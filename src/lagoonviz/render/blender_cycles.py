# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Blender's side of LAGOON's path tracing. Blender runs this; LAGOON never imports it.

    blender -b --factory-startup -P blender_cycles.py -- <folder> [--resume] [--frames 0:10]

Reads the folder LAGOON wrote (``lagoonviz.render.pathtrace.export_scene``):
``scene.json``, a ``frame_NNNNN.npz`` of triangles per frame and the surroundings
``environment.hdr``. Sets up Cycles (the GPU when there is one), the world (the
surroundings lighting and reflected, a plain gradient behind the scene where the
look has one, fog in a box around the scene under water), the sun and the
materials, then per frame builds the meshes and the camera and renders the frame
into the PNG named in ``scene.json``. Prints ``LAGOON frame i/n`` after each.
With ``--resume`` a frame whose image is there with the same key
(``frame_NNNNN.key``: its data, camera and settings) is kept.

Only Blender's Python and NumPy (Blender brings both); Blender 4.2 or newer
(older ones mostly work: the names of changed node inputs are tried in turn).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy  # Blender's own module: only there inside Blender
import numpy as np

COLLECTION = "LAGOON frame"
#: geometry node groups drawing particles, one per material (made once)
POINT_GROUPS: dict = {}
COLOR_ATTRIBUTE = "lagoon_color"
DEPTH_ATTRIBUTE = "lagoon_depth"  # how deep the water is, per corner of its surface
RADIUS_ATTRIBUTE = "lagoon_radius"  # the size of each particle of white water


# ======================================================================== helpers
def parse(argv: list[str]) -> tuple[Path, bool, tuple[int, int] | None]:
    """The folder, --resume, and --frames first:last (end exclusive) after "--"."""
    args = argv[argv.index("--") + 1 :] if "--" in argv else []
    if not args:
        raise SystemExit("usage: blender -b -P blender_cycles.py -- <folder> [--resume]")
    folder = Path(args[0])
    resume = "--resume" in args
    frames = None
    if "--frames" in args:
        first, _, last = args[args.index("--frames") + 1].partition(":")
        frames = (int(first or 0), int(last) if last else 1 << 30)
    return folder, resume, frames


def set_input(node, names, value) -> bool:
    """Set the first of ``names`` the node has (input names changed between Blender
    versions: "Transmission" became "Transmission Weight", ...)."""
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    return False


def rgba(rgb, alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))


def use_nodes(owner) -> None:
    """Node shading on (always on, and the switch deprecated, from Blender 5)."""
    try:
        owner.use_nodes = True
    except (AttributeError, TypeError):
        pass


def principled(tree):
    """The material's Principled BSDF (made if the material has none)."""
    node = tree.nodes.get("Principled BSDF")
    if node is None:
        node = tree.nodes.new("ShaderNodeBsdfPrincipled")
        output = tree.nodes.get("Material Output") or tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(node.outputs["BSDF"], output.inputs["Surface"])
    return node


def connect(tree, value, socket) -> None:
    """A node socket linked from ``value`` (a socket) or set to it (a number or colour)."""
    if hasattr(value, "is_output"):
        tree.links.new(value, socket)
    else:
        socket.default_value = value


def map_range(tree, value, low, high, to_low=0.0, to_high=1.0):
    """``value`` from low..high onto to_low..to_high, clamped (low > high turns it round)."""
    node = tree.nodes.new("ShaderNodeMapRange")
    node.clamp = True
    connect(tree, value, node.inputs[0])
    for index, number in ((1, low), (2, high), (3, to_low), (4, to_high)):
        node.inputs[index].default_value = float(number)
    return node.outputs[0]


def calc(tree, operation: str, a, b=0.0, c=0.0):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for index, value in enumerate((a, b, c)):
        if index < len(node.inputs):
            connect(tree, value, node.inputs[index])
    return node.outputs[0]


def mix_color(tree, factor, a, b):
    """Colour a, or b as far as ``factor`` says."""
    try:
        node = tree.nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        sockets = node.inputs[0], node.inputs[6], node.inputs[7], node.outputs[2]
    except (RuntimeError, AttributeError, TypeError):  # older Blender
        node = tree.nodes.new("ShaderNodeMixRGB")
        sockets = node.inputs[0], node.inputs[1], node.inputs[2], node.outputs[0]
    for value, socket in zip((factor, a, b), sockets[:3], strict=True):
        connect(tree, value, socket)
    return sockets[3]


def noise(tree, scale: float, detail: float = 4.0, roughness: float = 0.5):
    """Noise (0..1) over the object's own coordinates (the scene's, in metres)."""
    coords = tree.nodes.new("ShaderNodeTexCoord")
    node = tree.nodes.new("ShaderNodeTexNoise")
    set_input(node, ("Scale",), float(scale))
    set_input(node, ("Detail",), float(detail))
    set_input(node, ("Roughness",), float(roughness))
    tree.links.new(coords.outputs["Object"], node.inputs["Vector"])
    return node.outputs["Fac"]


# ======================================================================== render
def use_gpu(scene) -> str:
    """Render on the GPU when Blender finds one (Metal, OptiX, CUDA, HIP, oneAPI)."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return "CPU"
    for kind in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except (TypeError, ValueError):
            continue
        try:
            prefs.refresh_devices()
        except AttributeError:
            prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == kind]
        if devices:
            for device in prefs.devices:
                device.use = device.type == kind
            scene.cycles.device = "GPU"
            return kind
    try:
        prefs.compute_device_type = "NONE"
    except (TypeError, ValueError):
        pass
    return "CPU"


def setup_render(scene, settings: dict) -> str:
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x, render.resolution_y = (int(v) for v in settings["size"])
    render.resolution_percentage = 100
    render.film_transparent = False
    render.use_persistent_data = True  # the next frame reuses what it can
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGB"
    render.image_settings.color_depth = "8"
    cycles = scene.cycles
    cycles.samples = int(settings["samples"])
    # fast settings for these scenes: a few large smooth surfaces, one water sheet
    # (light crosses it once or twice), no glass objects; denoised, so few samples do
    for name, value in (
        ("use_adaptive_sampling", True),
        ("adaptive_threshold", 0.03),  # stop sampling a pixel once it is this clean
        ("adaptive_min_samples", 8),
        ("use_denoising", True),
        ("denoising_use_gpu", True),
        ("max_bounces", 6),
        ("diffuse_bounces", 2),
        ("glossy_bounces", 3),
        ("transmission_bounces", 4),
        ("transparent_max_bounces", 4),
        ("volume_bounces", 1),  # haze in the water under water
        ("sample_clamp_indirect", 5.0),  # no fireflies from the bright sky and panels
        ("caustics_reflective", False),
        ("caustics_refractive", False),
        ("blur_glossy", 1.0),
    ):
        try:
            setattr(cycles, name, value)
        except (AttributeError, TypeError):
            pass
    for transform in (settings.get("view_transform", "AgX"), "AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = transform
            break
        except (TypeError, ValueError):
            continue
    look = settings.get("look_contrast", "Medium High Contrast")
    if look:  # AgX's own contrast: named "AgX - <look>" in Blender 4 (none: AgX as it is)
        for name in (f"AgX - {look}", look):
            try:
                scene.view_settings.look = name
                break
            except (TypeError, ValueError):
                continue
    try:
        scene.view_settings.exposure = float(settings.get("exposure", 0.0))
    except (AttributeError, TypeError):
        pass
    return use_gpu(scene)


# ======================================================================== world
def setup_world(scene, folder: Path, world: dict) -> None:
    """The surroundings: they light the scene and are reflected; behind the scene a
    plain gradient where the look has one (camera rays only)."""
    node_world = bpy.data.worlds.new("LAGOON")
    scene.world = node_world
    use_nodes(node_world)
    tree = node_world.node_tree
    nodes, links = tree.nodes, tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    light = nodes.new("ShaderNodeBackground")
    light.inputs["Strength"].default_value = float(world.get("strength", 1.0))
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(folder / world["environment"]))
    links.new(environment.outputs["Color"], light.inputs["Color"])
    if world.get("rotation"):  # a sky picture turned about z: its sun where the sun is
        coords = nodes.new("ShaderNodeTexCoord")
        turn = nodes.new("ShaderNodeMapping")
        turn.inputs["Rotation"].default_value = (0.0, 0.0, float(world["rotation"]))
        links.new(coords.outputs["Generated"], turn.inputs["Vector"])
        links.new(turn.outputs["Vector"], environment.inputs["Vector"])
    shader = light.outputs["Background"]
    sky = physical_sky(nodes, world.get("sky"))
    if sky is not None:
        # the physical sky above the horizon; below it (black in the physical sky) the
        # sea of LAGOON's own surroundings, blended over a narrow band
        above = nodes.new("ShaderNodeBackground")
        above.inputs["Strength"].default_value = float(world["sky"].get("strength", 0.35))
        links.new(sky.outputs["Color"], above.inputs["Color"])
        coords = nodes.new("ShaderNodeTexCoord")
        split = nodes.new("ShaderNodeSeparateXYZ")
        links.new(coords.outputs["Generated"], split.inputs["Vector"])  # the ray's direction
        band = nodes.new("ShaderNodeMath")
        band.operation = "MULTIPLY_ADD"  # 0 a little below the horizon, 1 a little above
        band.use_clamp = True
        band.inputs[1].default_value = 40.0
        band.inputs[2].default_value = 0.5
        links.new(split.outputs["Z"], band.inputs[0])
        blend = nodes.new("ShaderNodeMixShader")
        links.new(band.outputs[0], blend.inputs[0])
        links.new(shader, blend.inputs[1])
        links.new(above.outputs["Background"], blend.inputs[2])
        shader = blend.outputs["Shader"]
    if world.get("background"):
        bottom, top = world["background"]
        coords = nodes.new("ShaderNodeTexCoord")
        split = nodes.new("ShaderNodeSeparateXYZ")
        links.new(coords.outputs["Generated"], split.inputs["Vector"])  # the ray's direction
        half = nodes.new("ShaderNodeMath")
        half.operation = "MULTIPLY_ADD"  # z / 2 + 1 / 2
        half.inputs[1].default_value = 0.5
        half.inputs[2].default_value = 0.5
        links.new(split.outputs["Z"], half.inputs[0])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.35
        ramp.color_ramp.elements[0].color = rgba(bottom)
        ramp.color_ramp.elements[1].position = 0.65
        ramp.color_ramp.elements[1].color = rgba(top)
        links.new(half.outputs[0], ramp.inputs["Fac"])
        plain = nodes.new("ShaderNodeBackground")
        links.new(ramp.outputs["Color"], plain.inputs["Color"])
        path = nodes.new("ShaderNodeLightPath")
        mix = nodes.new("ShaderNodeMixShader")
        links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
        links.new(shader, mix.inputs[1])
        links.new(plain.outputs["Background"], mix.inputs[2])
        shader = mix.outputs["Shader"]
    links.new(shader, output.inputs["Surface"])


def physical_sky(nodes, sky: dict | None):
    """Blender's physical sky (Nishita; "multiple scattering" in newer versions) with
    the sun where LAGOON's sun is, its disc off (the sun lamp is the sun); None when
    not wanted or this Blender has none."""
    if not sky:
        return None
    node = nodes.new("ShaderNodeTexSky")
    for kind in ("NISHITA", "MULTIPLE_SCATTERING", "SINGLE_SCATTERING"):
        try:
            node.sky_type = kind
            break
        except (TypeError, ValueError):
            continue
    else:
        nodes.remove(node)
        return None
    for name, value in (
        ("sun_disc", False),
        ("sun_elevation", float(sky["elevation"])),
        ("sun_rotation", float(sky["rotation"])),
        ("altitude", 0.0),
        ("air_density", 1.0),
        ("dust_density", 1.5),  # a little haze over the sea
        ("ozone_density", 1.0),
    ):
        try:
            setattr(node, name, value)
        except (AttributeError, TypeError):
            pass
    return node


def add_fog(collection, fog: dict) -> None:
    """Fog in a box around the scene (under water): clear up to the scene, thicker
    with the distance through it."""
    low, high = np.asarray(fog["low"], float), np.asarray(fog["high"], float)
    corners = np.array(
        [[x, y, z] for z in (low[2], high[2]) for y in (low[1], high[1]) for x in (low[0], high[0])]
    )
    faces = np.array(
        [
            (0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
            (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5),
        ]
    )  # fmt: skip
    mesh = make_mesh("LAGOON fog", corners, faces, None)
    material = bpy.data.materials.new("LAGOON fog")
    use_nodes(material)
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    set_input(volume, ("Color",), rgba(fog["color"]))
    set_input(volume, ("Density",), float(fog["density"]))
    set_input(volume, ("Absorption Color",), rgba([c * 0.6 for c in fog["color"]]))
    tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
    mesh.materials.append(material)
    obj = bpy.data.objects.new("LAGOON fog", mesh)
    collection.objects.link(obj)


def add_sun(collection, sun: dict) -> None:
    light = bpy.data.lights.new("LAGOON sun", "SUN")
    light.energy = float(sun["strength"])
    light.color = tuple(float(c) for c in sun["color"])
    light.angle = math.radians(float(sun.get("angle", 1.0)))
    if not sun.get("shadows", True):
        for owner, name in ((light, "use_shadow"), (getattr(light, "cycles", None), "cast_shadow")):
            try:
                setattr(owner, name, False)
            except (AttributeError, TypeError):
                pass
    obj = bpy.data.objects.new("LAGOON sun", light)
    collection.objects.link(obj)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = tuple(float(v) for v in sun["rotation_wxyz"])


# ======================================================================== materials
def terrain(tree, spec: dict):
    """A coast's colour, roughness and fine relief from the height above the still
    water level and the slope: sand under water, wet sand at the water line, dry sand,
    grass with patches of heath, rock where steep. Wide, ragged transitions (noise at
    two sizes), so it reads as land seen from the air, not as the bands of a map.
    Returns the colour, the roughness and the normal."""
    land = spec["terrain"]
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    position = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Position"], position.inputs[0])
    normal = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Normal"], normal.inputs[0])
    scale = float(land["scale"])
    patches = noise(tree, scale, 5.0, 0.55)
    ragged = noise(tree, 8.0 * scale, 6.0, 0.6)
    # the height above the water, as a fraction of the land's rise, ragged
    height = calc(tree, "SUBTRACT", position.outputs["Z"], float(land["level"]))
    height = calc(tree, "DIVIDE", height, float(land["relief"]))
    height = calc(tree, "ADD", height, calc(tree, "MULTIPLY_ADD", patches, 0.14, -0.07))
    height = calc(tree, "ADD", height, calc(tree, "MULTIPLY_ADD", ragged, 0.05, -0.025))
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    tree.links.new(map_range(tree, height, -0.05, 1.0), ramp.inputs["Fac"])
    under = spec.get("color") or land["sand"]
    stops = (
        (-0.05, under),
        (0.0, land["wet"]),
        (0.02, land["sand"]),
        (0.035, land["sand"]),
        (0.12, land["grass"]),
        (0.45, land["grass"]),
        (0.8, land["heath"]),
    )
    elements = ramp.color_ramp.elements
    for _ in range(len(stops) - len(elements)):
        elements.new(1.0)
    for element, (h, color) in zip(elements, stops, strict=False):
        element.position = (h + 0.05) / 1.05
        element.color = rgba(color)
    # patches of heath in the grass
    other = noise(tree, 2.5 * scale, 4.0, 0.5)
    heath = calc(
        tree,
        "MULTIPLY",
        map_range(tree, other, 0.55, 0.72),
        map_range(tree, height, 0.06, 0.15),
    )
    color = mix_color(tree, heath, ramp.outputs["Color"], rgba(land["heath"]))
    # rock where it is steep (the normal leans more than about 30 to 50 degrees)
    upright = calc(tree, "ADD", normal.outputs["Z"], calc(tree, "MULTIPLY_ADD", ragged, 0.2, -0.1))
    steep = map_range(tree, upright, 0.86, 0.62)
    color = mix_color(tree, steep, color, rgba(land["rock"]))
    shade = tree.nodes.new("ShaderNodeHueSaturation")  # lighter and darker patches
    connect(tree, color, shade.inputs["Color"])
    connect(tree, map_range(tree, patches, 0.3, 0.7, 0.85, 1.1), shade.inputs["Value"])
    # wet sand and the bed under water shine a little; dry land is matt
    rough = map_range(tree, height, 0.004, 0.03, 0.4, 0.95)
    # fine relief: a few pixels across, so the land is not smooth clay
    detail = float(land.get("detail") or 40.0 * scale)
    bump = tree.nodes.new("ShaderNodeBump")
    set_input(bump, ("Strength",), 0.35)
    set_input(bump, ("Distance",), 0.3 / max(detail, 1e-9))
    tree.links.new(noise(tree, detail, 10.0, 0.6), bump.inputs["Height"])
    return shade.outputs["Color"], rough, bump.outputs["Normal"]


def shore(tree, spec: dict) -> None:
    """Water thinning out at the shore: see-through where it is only a film (no sheet
    of glass over the sand) and, under the open sky, patchy foam just off the water
    line. Needs the surface's depth per corner (``lagoon_depth``)."""
    band = spec["shore"]
    output = tree.nodes.get("Material Output")
    links = output.inputs["Surface"].links
    if not links:
        return
    shader = links[0].from_socket
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = DEPTH_ATTRIBUTE
    depth = attribute.outputs["Fac"]
    fade, reach = float(band["fade"]), float(band.get("foam") or 0.0)
    if reach > fade:
        foam = principled_node(tree, (0.85, 0.88, 0.88), 0.45)
        # broken: only where the noise is high, never a continuous line
        patches = map_range(tree, noise(tree, float(band["scale"]), 6.0, 0.65), 0.5, 0.68)
        amount = calc(tree, "MULTIPLY", map_range(tree, depth, fade, reach, 0.7, 0.0), patches)
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(amount, mix.inputs[0])
        tree.links.new(shader, mix.inputs[1])
        tree.links.new(foam.outputs["BSDF"], mix.inputs[2])
        shader = mix.outputs["Shader"]
    clear = tree.nodes.new("ShaderNodeBsdfTransparent")
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(map_range(tree, depth, 0.0, max(fade, 1e-9)), mix.inputs[0])
    tree.links.new(clear.outputs["BSDF"], mix.inputs[1])
    tree.links.new(shader, mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])


def principled_node(tree, color, roughness: float):
    node = tree.nodes.new("ShaderNodeBsdfPrincipled")
    set_input(node, ("Base Color",), rgba(color))
    set_input(node, ("Roughness",), float(roughness))
    return node


def make_material(key: str, spec: dict):
    material = bpy.data.materials.new(f"LAGOON {key}")
    use_nodes(material)
    tree = material.node_tree
    bsdf = principled(tree)
    if spec.get("color") is None:  # coloured per vertex (a variable, or own colours)
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_name = COLOR_ATTRIBUTE
        tree.links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        set_input(bsdf, ("Base Color",), rgba(spec["color"]))
    set_input(bsdf, ("Roughness",), float(spec["roughness"]))
    if spec.get("terrain"):
        try:
            color, rough, relief = terrain(tree, spec)
            tree.links.new(color, bsdf.inputs["Base Color"])
            tree.links.new(rough, bsdf.inputs["Roughness"])
            tree.links.new(relief, bsdf.inputs["Normal"])
        except (AttributeError, KeyError, IndexError, TypeError, RuntimeError) as error:
            print(f"LAGOON warning: no land colours ({error!r})", flush=True)
    set_input(bsdf, ("Metallic",), float(spec["metallic"]))
    set_input(bsdf, ("IOR",), float(spec["ior"]))
    set_input(bsdf, ("Transmission Weight", "Transmission"), float(spec["transmission"]))
    if spec.get("water") and not spec.get("volume"):
        # a water sheet has no depth: looking straight down, much of the bed shows
        # through; at a slant (a longer way through the water) little does and the
        # water's own colour takes over. Transmission = t (1 - facing).
        weight = bsdf.inputs.get("Transmission Weight") or bsdf.inputs.get("Transmission")
        if weight is not None:
            facing = tree.nodes.new("ShaderNodeLayerWeight")
            set_input(facing, ("Blend",), 0.5)
            falloff = tree.nodes.new("ShaderNodeMath")
            falloff.operation = "MULTIPLY_ADD"
            falloff.inputs[1].default_value = -float(spec["transmission"])
            falloff.inputs[2].default_value = float(spec["transmission"])
            tree.links.new(facing.outputs["Facing"], falloff.inputs[0])
            tree.links.new(falloff.outputs[0], weight)
    if spec.get("coat"):
        set_input(bsdf, ("Coat Weight", "Clearcoat"), float(spec["coat"]))
        set_input(bsdf, ("Coat Roughness", "Clearcoat Roughness"), 0.1)
    if float(spec.get("alpha", 1.0)) < 1.0:
        set_input(bsdf, ("Alpha",), float(spec["alpha"]))
    if spec.get("camera_only"):
        # ÆgirFrames' water: only the camera sees it (refracting and reflecting); light,
        # shadows and reflections pass through it, so the bed under it is lit as in air
        path = tree.nodes.new("ShaderNodeLightPath")
        tree.links.new(path.outputs["Is Camera Ray"], bsdf.inputs["Alpha"])
    elif float(spec["transmission"]) > 0:
        # sunlight passes the water surface to the bed: shadow rays see it as a
        # tinted window (the refracted light itself, caustics, would take far longer)
        output = tree.nodes.get("Material Output") or tree.nodes.new("ShaderNodeOutputMaterial")
        path = tree.nodes.new("ShaderNodeLightPath")
        window = tree.nodes.new("ShaderNodeBsdfTransparent")
        tint = np.asarray(spec.get("color") or (1.0, 1.0, 1.0), float) ** 0.25
        set_input(window, ("Color",), rgba(0.92 * tint))
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(path.outputs["Is Shadow Ray"], mix.inputs[0])
        tree.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
        tree.links.new(window.outputs["BSDF"], mix.inputs[2])
        tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    volume = spec.get("volume")
    if volume:
        # the water's body: light absorbed on its way through (and scattered, the
        # haze under water)
        output = tree.nodes.get("Material Output") or tree.nodes.new("ShaderNodeOutputMaterial")
        color = rgba(volume["color"])
        if volume.get("absorption") is not None:  # a principled volume as given
            medium = tree.nodes.new("ShaderNodeVolumePrincipled")
            set_input(medium, ("Color",), color)
            set_input(medium, ("Absorption Color",), rgba(volume["absorption"]))
            set_input(medium, ("Density",), float(volume["density"]))
        elif float(volume.get("scatter", 0.0)) > 0:
            # mostly absorbing; a little scattering (dark), so the haze stays blue
            medium = tree.nodes.new("ShaderNodeVolumePrincipled")
            dim = float(volume["scatter"])
            set_input(medium, ("Color",), rgba([c * dim for c in volume["color"]]))
            set_input(medium, ("Absorption Color",), color)
            set_input(medium, ("Density",), float(volume["density"]))
        else:
            medium = tree.nodes.new("ShaderNodeVolumeAbsorption")
            set_input(medium, ("Color",), color)
            set_input(medium, ("Density",), float(volume["density"]))
        tree.links.new(medium.outputs["Volume"], output.inputs["Volume"])
    ripples = spec.get("ripples")
    if ripples:
        # wind ripples on the water: they break the reflections of the sky and the sun
        # into glitter, as on real water
        coords = tree.nodes.new("ShaderNodeTexCoord")
        wave = tree.nodes.new("ShaderNodeTexNoise")
        set_input(wave, ("Scale",), float(ripples["scale"]))
        set_input(wave, ("Detail",), 3.0)
        set_input(wave, ("Roughness",), 0.5)
        tree.links.new(coords.outputs["Object"], wave.inputs["Vector"])
        bump = tree.nodes.new("ShaderNodeBump")
        set_input(bump, ("Strength",), float(ripples["strength"]))
        set_input(bump, ("Distance",), 0.2 / max(float(ripples["scale"]), 1e-6))
        tree.links.new(wave.outputs["Fac"], bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        swell = spec.get("swell")
        if swell:
            # long, low swell under the ripples (the sea beyond the domain)
            under = tree.nodes.new("ShaderNodeBump")
            set_input(under, ("Strength",), float(swell["strength"]))
            set_input(under, ("Distance",), 0.3 / max(float(swell["scale"]), 1e-6))
            height = noise(tree, float(swell["scale"]), 2.0, 0.4)
            tree.links.new(height, under.inputs["Height"])
            tree.links.new(under.outputs["Normal"], bump.inputs["Normal"])
    if spec.get("shore"):
        try:
            shore(tree, spec)
        except (AttributeError, KeyError, IndexError, TypeError, RuntimeError) as error:
            print(f"LAGOON warning: no shore ({error!r})", flush=True)
    grain = spec.get("grain")
    if grain:
        # the grain as bump: noise over the object's own coordinates (metres)
        coords = tree.nodes.new("ShaderNodeTexCoord")
        wave = tree.nodes.new("ShaderNodeTexNoise")
        set_input(wave, ("Scale",), 4.0 / max(float(grain["tile"]), 1e-6))
        set_input(wave, ("Detail",), 8.0)
        set_input(wave, ("Roughness",), 0.6)
        tree.links.new(coords.outputs["Object"], wave.inputs["Vector"])
        bump = tree.nodes.new("ShaderNodeBump")
        set_input(bump, ("Strength",), min(float(grain["relief"]) / 45.0, 1.0) * 0.6)
        set_input(bump, ("Distance",), 0.02 * float(grain["tile"]))
        tree.links.new(wave.outputs["Fac"], bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


# ======================================================================== meshes
def make_mesh(name: str, positions, triangles, colors, smooth=None, attributes=None):
    """A triangle mesh, smooth shaded (or where ``smooth`` says), with linear vertex
    colours if given and per corner numbers (``attributes``: name -> values)."""
    positions = np.ascontiguousarray(positions, np.float32).reshape(-1, 3)
    triangles = np.ascontiguousarray(triangles, np.int32).reshape(-1, 3)
    n, m = len(positions), len(triangles)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(n)
    mesh.vertices.foreach_set("co", positions.ravel())
    mesh.loops.add(3 * m)
    mesh.loops.foreach_set("vertex_index", triangles.ravel())
    mesh.polygons.add(m)
    mesh.polygons.foreach_set("loop_start", np.arange(0, 3 * m, 3, dtype=np.int32))
    try:  # Blender before 4.0 needs the sizes too (later: read-only, from the starts)
        mesh.polygons.foreach_set("loop_total", np.full(m, 3, np.int32))
    except (AttributeError, TypeError, RuntimeError):
        pass
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)
    faces = len(mesh.polygons)
    flags = np.ones(faces, dtype=bool)
    if smooth is not None and len(smooth) == faces:  # validate() removed no faces
        flags = np.asarray(smooth, dtype=bool)
    mesh.polygons.foreach_set("use_smooth", flags)
    if colors is not None:
        values = np.ones((n, 4), np.float32)
        values[:, :3] = np.asarray(colors, np.float32).reshape(-1, 3)
        attribute = mesh.color_attributes.new(COLOR_ATTRIBUTE, "FLOAT_COLOR", "POINT")
        attribute.data.foreach_set("color", values.ravel())
    for key, values in (attributes or {}).items():
        attribute = mesh.attributes.new(key, "FLOAT", "POINT")
        attribute.data.foreach_set("value", np.asarray(values, np.float32).ravel())
    mesh.update()
    return mesh


def subdivide(obj, mesh, levels: int = 1) -> None:
    """Catmull-Clark subdivision when rendering (smooth crests, no facets); the edges
    of flat faces (walls, floors, the sides of a block) creased, so they keep their
    shape."""
    try:
        edges = np.zeros(len(mesh.loops), np.int32)
        mesh.loops.foreach_get("edge_index", edges)
        smooth = np.zeros(len(mesh.polygons), bool)
        mesh.polygons.foreach_get("use_smooth", smooth)
        totals = np.zeros(len(mesh.polygons), np.int32)
        mesh.polygons.foreach_get("loop_total", totals)
        crease = np.zeros(len(mesh.edges), np.float32)
        crease[edges[~np.repeat(smooth, totals)]] = 1.0
        if crease.any():
            attribute = mesh.attributes.get("crease_edge")
            if attribute is None:
                attribute = mesh.attributes.new("crease_edge", "FLOAT", "EDGE")
            attribute.data.foreach_set("value", crease)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        print(f"LAGOON warning: no creases ({error!r})", flush=True)
    modifier = obj.modifiers.new("LAGOON smooth", "SUBSURF")
    modifier.levels = 0  # the viewport (never opened) stays light
    modifier.render_levels = int(levels)
    try:
        modifier.subdivision_type = "CATMULL_CLARK"
    except (AttributeError, TypeError):
        pass


def points_group(material):
    """A geometry node group drawing a mesh's vertices as spheres (a point cloud,
    which Cycles renders directly) of radius ``lagoon_radius``, in ``material``."""
    group = bpy.data.node_groups.new(f"LAGOON points {material.name}", "GeometryNodeTree")
    interface = getattr(group, "interface", None)
    if interface is not None:  # Blender 4
        interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    else:  # Blender 3
        group.inputs.new("NodeSocketGeometry", "Geometry")
        group.outputs.new("NodeSocketGeometry", "Geometry")
    nodes, links = group.nodes, group.links
    source = nodes.new("NodeGroupInput")
    sink = nodes.new("NodeGroupOutput")
    to_points = nodes.new("GeometryNodeMeshToPoints")
    radius = nodes.new("GeometryNodeInputNamedAttribute")
    radius.data_type = "FLOAT"
    radius.inputs["Name"].default_value = RADIUS_ATTRIBUTE
    value = radius.outputs.get("Attribute") or next(o for o in radius.outputs if o.enabled)
    paint = nodes.new("GeometryNodeSetMaterial")
    paint.inputs["Material"].default_value = material
    links.new(source.outputs[0], to_points.inputs["Mesh"])
    links.new(value, to_points.inputs["Radius"])
    links.new(to_points.outputs["Points"], paint.inputs["Geometry"])
    links.new(paint.outputs["Geometry"], sink.inputs[0])
    return group


def make_points(name: str, positions, radius, material, groups: dict):
    """Particles (foam, spray, bubbles) as an object: a point cloud of spheres by
    geometry nodes, or small tetrahedra where those fail."""
    positions = np.ascontiguousarray(positions, np.float32).reshape(-1, 3)
    radius = np.ascontiguousarray(radius, np.float32).reshape(-1)
    try:
        group = groups.get(material.name)
        if group is None:
            group = groups[material.name] = points_group(material)
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(len(positions))
        mesh.vertices.foreach_set("co", positions.ravel())
        attribute = mesh.attributes.new(RADIUS_ATTRIBUTE, "FLOAT", "POINT")
        attribute.data.foreach_set("value", radius)
        mesh.update()
        mesh.materials.append(material)
        obj = bpy.data.objects.new(name, mesh)
        modifier = obj.modifiers.new("LAGOON points", "NODES")
        modifier.node_group = group
        return obj
    except (AttributeError, KeyError, RuntimeError, StopIteration, TypeError) as error:
        print(f"LAGOON warning: particles as tetrahedra ({error!r})", flush=True)
    corners = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], np.float32)
    points = (positions[:, None, :] + radius[:, None, None] * corners).reshape(-1, 3)
    faces = np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
    triangles = (np.arange(len(positions))[:, None, None] * 4 + faces).reshape(-1, 3)
    mesh = make_mesh(name, points, triangles, None, np.zeros(len(triangles), bool))
    mesh.materials.append(material)
    return bpy.data.objects.new(name, mesh)


def clear(collection) -> None:
    for obj in list(collection.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and getattr(data, "users", 0) == 0:
            bpy.data.meshes.remove(data)


def build_frame(collection, frame: dict, arrays, materials: dict) -> int:
    """The frame's objects; returns the number of triangles."""
    clear(collection)
    count = 0
    for index, entry in enumerate(frame["objects"]):
        if entry.get("points"):  # white water: a cloud of small spheres
            obj = make_points(
                entry["name"],
                arrays[f"points_{index}"],
                arrays[f"radius_{index}"],
                materials[entry["material"]],
                POINT_GROUPS,
            )
            collection.objects.link(obj)
            count += len(arrays[f"points_{index}"])
            continue
        positions = arrays[f"positions_{index}"]
        triangles = arrays[f"triangles_{index}"]
        colors = arrays[f"colors_{index}"] if entry.get("colors") else None
        smooth = arrays[f"smooth_{index}"] if f"smooth_{index}" in arrays else None
        attributes = {
            key: arrays[f"attribute_{index}_{key}"] for key in entry.get("attributes", ())
        }
        mesh = make_mesh(entry["name"], positions, triangles, colors, smooth, attributes)
        mesh.materials.append(materials[entry["material"]])
        obj = bpy.data.objects.new(entry["name"], mesh)
        collection.objects.link(obj)
        if entry.get("subdivide"):
            subdivide(obj, mesh, int(entry["subdivide"]))
        count += len(triangles)
    return count


def set_camera(obj, camera: dict) -> None:
    data = obj.data
    data.type = camera["type"]
    data.sensor_fit = "VERTICAL"
    if camera["type"] == "PERSP":
        data.angle_y = float(camera["angle_y"])
    else:
        data.ortho_scale = float(camera["ortho_scale"])
    data.clip_start = float(camera.get("clip_start", 0.01))
    data.clip_end = float(camera.get("clip_end", 1e5))
    obj.location = tuple(float(v) for v in camera["position"])
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = tuple(float(v) for v in camera["rotation_wxyz"])


# ======================================================================== main
def main(argv: list[str]) -> int:
    folder, resume, selection = parse(argv)
    print(f"LAGOON reading {folder}", flush=True)
    settings = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    device = setup_render(scene, settings)
    print(f"LAGOON Cycles on {device}, {settings['samples']} samples", flush=True)
    setup_world(scene, folder, settings["world"])
    lights = bpy.data.collections.new("LAGOON lights")
    scene.collection.children.link(lights)
    add_sun(lights, settings["sun"])
    if settings["world"].get("fog"):
        add_fog(lights, settings["world"]["fog"])
    materials = {key: make_material(key, spec) for key, spec in settings["materials"].items()}
    print(f"LAGOON scene set up: {len(materials)} materials", flush=True)
    collection = bpy.data.collections.new(COLLECTION)
    scene.collection.children.link(collection)
    camera = bpy.data.objects.new("LAGOON camera", bpy.data.cameras.new("LAGOON camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    frames = settings["frames"]
    numbers = list(range(len(frames)))
    if selection is not None:
        numbers = numbers[selection[0] : selection[1]]
    for done, number in enumerate(numbers, start=1):
        frame = frames[number]
        image = folder / frame["image"]
        data = folder / frame["data"]
        stamp = image.with_suffix(".key")  # the frame's key, written once its image is
        if resume and image.exists() and stamp.exists():
            if stamp.read_text(encoding="utf-8").strip() == frame.get("key"):
                print(f"LAGOON frame {done}/{len(numbers)} (kept)", flush=True)
                continue
        started = time.perf_counter()
        print(f"LAGOON building frame {done}/{len(numbers)}", flush=True)
        with np.load(data) as arrays:
            triangles = build_frame(collection, frame, arrays, materials)
        set_camera(camera, frame["camera"])
        scene.render.filepath = str(image)
        stamp.unlink(missing_ok=True)
        print(f"LAGOON rendering frame {done}/{len(numbers)} ({triangles} triangles)", flush=True)
        bpy.ops.render.render(write_still=True)
        if frame.get("key"):
            stamp.write_text(frame["key"], encoding="utf-8")
        seconds = time.perf_counter() - started
        print(
            f"LAGOON frame {done}/{len(numbers)} {triangles} triangles {seconds:.1f} s", flush=True
        )
    return 0


if __name__ == "__main__":
    main(sys.argv)  # an error fails Blender with --python-exit-code
