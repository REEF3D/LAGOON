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
COLOR_ATTRIBUTE = "lagoon_color"


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
    if float(spec["transmission"]) > 0:
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
        if float(volume.get("scatter", 0.0)) > 0:
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
        noise = tree.nodes.new("ShaderNodeTexNoise")
        set_input(noise, ("Scale",), float(ripples["scale"]))
        set_input(noise, ("Detail",), 3.0)
        set_input(noise, ("Roughness",), 0.5)
        tree.links.new(coords.outputs["Object"], noise.inputs["Vector"])
        bump = tree.nodes.new("ShaderNodeBump")
        set_input(bump, ("Strength",), float(ripples["strength"]))
        set_input(bump, ("Distance",), 0.2 / max(float(ripples["scale"]), 1e-6))
        tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    grain = spec.get("grain")
    if grain:
        # the grain as bump: noise over the object's own coordinates (metres)
        coords = tree.nodes.new("ShaderNodeTexCoord")
        noise = tree.nodes.new("ShaderNodeTexNoise")
        set_input(noise, ("Scale",), 4.0 / max(float(grain["tile"]), 1e-6))
        set_input(noise, ("Detail",), 8.0)
        set_input(noise, ("Roughness",), 0.6)
        tree.links.new(coords.outputs["Object"], noise.inputs["Vector"])
        bump = tree.nodes.new("ShaderNodeBump")
        set_input(bump, ("Strength",), min(float(grain["relief"]) / 45.0, 1.0) * 0.6)
        set_input(bump, ("Distance",), 0.02 * float(grain["tile"]))
        tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


# ======================================================================== meshes
def make_mesh(name: str, positions, triangles, colors, smooth=None):
    """A triangle mesh, smooth shaded (or where ``smooth`` says), with linear vertex
    colours if given."""
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
    mesh.update()
    return mesh


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
        positions = arrays[f"positions_{index}"]
        triangles = arrays[f"triangles_{index}"]
        colors = arrays[f"colors_{index}"] if entry.get("colors") else None
        smooth = arrays[f"smooth_{index}"] if f"smooth_{index}" in arrays else None
        mesh = make_mesh(entry["name"], positions, triangles, colors, smooth)
        mesh.materials.append(materials[entry["material"]])
        obj = bpy.data.objects.new(entry["name"], mesh)
        collection.objects.link(obj)
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
