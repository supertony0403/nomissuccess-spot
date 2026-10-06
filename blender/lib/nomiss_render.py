"""Render setup shared by all nomissuccess shots (Blender 5.2, EEVEE).

* square 1920x1920 at 60 fps (both delivery formats are crops of it),
* ``Standard`` view transform so brand colours stay exact,
* motion blur with shutter 0.5 (180 degrees),
* subtle bloom through the compositor Glare node (5.x node-group API),
* PNG RGB 8-bit to ``renders/<shot>/####.png``.

Every shot script calls :func:`parse_args` and :func:`setup`; the CLI is the
same for all shots (see ``blender/render.sh``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[2]
RENDERS = REPO / "renders"


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments after ``--`` on the blender command line."""
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="shot", description="nomissuccess Blender shot")
    ap.add_argument("--res", type=int, default=1920, help="square resolution in px")
    ap.add_argument("--samples", type=int, default=0, help="EEVEE samples (0 = shot default)")
    ap.add_argument("--frames", default="all", help="'all', '1-120', or '1,mid,last,ev:logo_fold'")
    ap.add_argument("--out", default="", help="output directory (default renders/<shot>)")
    ap.add_argument("--matte", default="", help="render only this object as white alpha matte")
    ap.add_argument("--logo-mask", default="", help="s8: write frontal orthographic logo alpha to this PNG")
    ap.add_argument("--no-mb", action="store_true", help="disable motion blur")
    ap.add_argument("--no-glare", action="store_true", help="disable compositor bloom")
    ap.add_argument("--timeline", default="", help="explicit timeline.json")
    ap.add_argument("--save-blend", default="", help="also save the built scene to this .blend")
    return ap.parse_args(argv)


def frame_list(spec: str, total: int, events: dict[str, int] | None = None) -> list[int]:
    """Expand a frame spec into sorted 1-based frame numbers."""
    spec = spec.strip()
    if spec in ("", "all"):
        return list(range(1, total + 1))
    out: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "mid":
            out.add((total + 1) // 2)
        elif tok == "last":
            out.add(total)
        elif tok.startswith("ev:"):
            name, _, off = tok[3:].partition("+")
            if not events or name not in events:
                raise KeyError(f"unbekanntes Event in --frames: {name}")
            out.add(events[name] + (int(off) if off else 0))
        elif "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(f for f in out if 1 <= f <= total)


# --------------------------------------------------------------------------
# scene setup
# --------------------------------------------------------------------------

def engine_id() -> str:
    items = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    for cand in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if cand in items:
            return cand
    raise RuntimeError(f"kein EEVEE in {items}")


def clean_scene() -> bpy.types.Scene:
    """Empty the factory-startup scene (cube, light, camera)."""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.curves):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)
    return bpy.context.scene


def setup(
    scene: bpy.types.Scene,
    shot_id: str,
    frames: int,
    *,
    res: int = 1920,
    samples: int = 64,
    motion_blur: bool = True,
    glare: bool = True,
    out_dir: str | os.PathLike | None = None,
    transparent: bool = False,
) -> Path:
    """Configure engine, colour, motion blur, compositor and output."""
    r = scene.render
    r.engine = engine_id()
    r.resolution_x = res
    r.resolution_y = res
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.fps = 60
    r.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = frames
    r.film_transparent = transparent
    r.filter_size = 1.5
    r.dither_intensity = 1.0  # breaks banding of dark gradients in 8-bit

    r.use_motion_blur = motion_blur
    r.motion_blur_shutter = 0.5
    r.motion_blur_position = "CENTER"

    ee = scene.eevee
    ee.taa_render_samples = samples
    ee.motion_blur_steps = 1
    ee.motion_blur_depth_scale = 100.0
    ee.motion_blur_max = 64
    ee.use_bokeh_jittered = True
    ee.bokeh_overblur = 12.0
    ee.bokeh_max_size = 160.0
    ee.bokeh_threshold = 1.0
    ee.bokeh_neighbor_max = 10.0
    ee.use_raytracing = True
    ee.ray_tracing_method = "SCREEN"
    ee.use_shadows = True
    ee.shadow_ray_count = 2
    ee.shadow_step_count = 8
    ee.use_fast_gi = False
    ee.clamp_surface_indirect = 10.0

    vs = scene.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0
    scene.display_settings.display_device = "sRGB"

    img = r.image_settings
    img.file_format = "PNG"
    img.color_mode = "RGBA" if transparent else "RGB"
    img.color_depth = "8"
    img.compression = 15
    r.use_file_extension = True
    r.use_overwrite = True
    r.use_placeholder = False

    out = Path(out_dir) if out_dir else RENDERS / shot_id
    out.mkdir(parents=True, exist_ok=True)
    r.filepath = str(out) + "/"

    if glare:
        compositor_glow(scene)
    else:
        scene.compositing_node_group = None
    return out


def compositor_glow(
    scene: bpy.types.Scene,
    *,
    threshold: float = 0.85,
    strength: float = 0.22,
    size: float = 0.62,
    saturation: float = 1.0,
) -> bpy.types.NodeTree:
    """Render Layers -> Glare (Bloom) -> Group Output (Blender 5.x)."""
    tree = bpy.data.node_groups.get("NomissKomp") or bpy.data.node_groups.new("NomissKomp", "CompositorNodeTree")
    tree.nodes.clear()
    if not any(i.in_out == "OUTPUT" for i in tree.interface.items_tree):
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)
    gl = tree.nodes.new("CompositorNodeGlare")
    gl.name = "Glow"
    gl.location = (-100, 0)
    gl.inputs["Type"].default_value = "Bloom"
    gl.inputs["Quality"].default_value = "High"
    gl.inputs["Threshold"].default_value = threshold
    gl.inputs["Smoothness"].default_value = 0.6
    gl.inputs["Strength"].default_value = strength
    gl.inputs["Saturation"].default_value = saturation
    gl.inputs["Size"].default_value = size
    out = tree.nodes.new("NodeGroupOutput")
    out.location = (200, 0)
    tree.links.new(rl.outputs["Image"], gl.inputs["Image"])
    tree.links.new(gl.outputs["Image"], out.inputs[0])
    scene.compositing_node_group = tree
    scene.render.use_compositing = True
    return tree


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_frames(scene: bpy.types.Scene, frames: list[int], out_dir: str | os.PathLike) -> list[Path]:
    """Render the given frames as stills; prints timing per frame."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    t_all = time.time()
    for f in frames:
        scene.frame_set(f)
        path = out / f"{f:04d}.png"
        scene.render.filepath = str(path)
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[nomiss_render] Bild {f} -> {path.name} in {time.time() - t0:.2f} s", flush=True)
        written.append(path)
    if frames:
        dt = time.time() - t_all
        print(f"[nomiss_render] {len(frames)} Bilder in {dt:.1f} s ({dt / len(frames):.2f} s/Bild)", flush=True)
    return written


def render_animation(scene: bpy.types.Scene) -> None:
    """Render frame_start..frame_end as individual stills.

    Same file names as an animation render (``scene.render.frame_path``).
    A single ``render(animation=True)`` in Blender 5.2.1 drops objects whose
    geometry is empty in the first rendered frame for the whole session
    (e.g. a band that has not started to grow yet); stills are immune.
    """
    pattern = scene.render.filepath
    t_all = time.time()
    n = 0
    try:
        for f in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(f)
            path = scene.render.frame_path(frame=f)
            scene.render.filepath = path
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            scene.render.filepath = pattern
            n += 1
            print(f"[nomiss_render] Bild {f}/{scene.frame_end} in {time.time() - t0:.2f} s", flush=True)
    finally:
        scene.render.filepath = pattern
    dt = time.time() - t_all
    print(f"[nomiss_render] Animation {n} Bilder in {dt:.1f} s ({dt / max(n, 1):.2f} s/Bild)", flush=True)


def matte_mode(scene: bpy.types.Scene, keep: list[bpy.types.Object]) -> None:
    """Only ``keep`` renders, as flat white with alpha; no DOF/MB/glare."""
    from nomiss_material import flat_emission  # local import: avoids a cycle

    white = flat_emission("NomissMatte", (1.0, 1.0, 1.0, 1.0), 1.0)
    keep_names = {o.name for o in keep}
    for ob in scene.objects:
        if ob.type in {"MESH", "CURVE", "CURVES", "VOLUME", "POINTCLOUD", "FONT"} and ob.name not in keep_names:
            ob.hide_render = True
        if ob.type == "LIGHT":
            ob.hide_render = True
    for ob in keep:
        ob.hide_render = False
        _override_material(ob, white)
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.use_motion_blur = False
    scene.compositing_node_group = None
    if scene.camera and scene.camera.data.dof:
        scene.camera.data.dof.use_dof = False
    black = bpy.data.worlds.get("NomissMatteWelt") or bpy.data.worlds.new("NomissMatteWelt")
    black.color = (0.0, 0.0, 0.0)
    scene.world = black


def _override_material(ob: bpy.types.Object, mat: bpy.types.Material) -> None:
    """Replace every material slot of ``ob`` by ``mat`` (object-linked)."""
    from nomiss_material import assign

    for i in range(max(1, len(ob.material_slots))):
        assign(ob, mat, i)


def gpu_info() -> str:
    try:
        import gpu

        return f"{gpu.platform.vendor_get()} {gpu.platform.renderer_get()} ({gpu.platform.backend_type_get()})"
    except Exception as ex:  # pragma: no cover - informational only
        return f"unbekannt ({ex})"
