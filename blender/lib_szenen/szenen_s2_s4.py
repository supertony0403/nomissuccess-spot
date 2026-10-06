"""Shared helpers for the Blender shots s2_website, s3_netz and s4_ernstfall.

Built on top of ``blender/lib`` (render setup, materials, camera, timeline),
which belongs to another task and is only imported here, never changed.

What lives here:

* :func:`parse_cli` - the shot CLI of ``nomiss_render.parse_args`` plus
  ``--proxy N`` (square proxy resolution), ``--matte-keys`` (centre-square
  check), ``--physik-json`` (rigid-body determinism check), ``--no-encode``,
* small material builders (principled, image layers with opacity, additive
  light, metal, plastic),
* :func:`bake` - fast per-frame keyframes through the 5.x layered-action API,
* :class:`Bahn`, :func:`overshoot`, :func:`rewind_time` - re-exported from
  the pure-Python ``bewegung.py`` (unit-tested without Blender),
* :class:`GN` - a tiny Geometry-Nodes graph builder,
* :func:`run` - common render/encode flow incl. the key-object mattes used by
  ``tests/test_blender_s2_s4.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "blender" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import nomiss_camera as C  # noqa: E402
import nomiss_material as M  # noqa: E402
import nomiss_render as R  # noqa: E402
import nomiss_timeline as TL  # noqa: E402

ASSETS = REPO / "blender" / "assets" / "s2_s4"
WORK = REPO / "work" / "blender" / "s2_s4"
FONT_MONO = "/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf"
FONT_SANS = str(Path.home() / ".local/share/fonts/manrope/Manrope-ExtraBold.ttf")
CENTRE = (420, 1500)  # centre square in 1920 px (both axes)

lin = M.hex_to_linear


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def parse_cli(default_samples: int = 16) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Library CLI plus the s2-s4 extras (``--proxy`` maps to ``--res``)."""
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--proxy", type=int, default=0, help="square proxy resolution, e.g. 480")
    ap.add_argument("--matte-keys", action="store_true", help="render key-object mattes at ev: frames")
    ap.add_argument("--physik-json", default="", help="write rigid-body end positions to this file")
    ap.add_argument("--no-encode", action="store_true", help="skip ProRes encode after a full render")
    ap.add_argument("--nebel", type=float, default=-1.0, help="override atmospheric fog density (0 = off)")
    own, rest = ap.parse_known_args(argv)
    if own.proxy:
        rest += ["--res", str(own.proxy)]
    args = R.parse_args(rest)
    if not args.samples:
        args.samples = max(8, default_samples // 2) if own.proxy else default_samples
    return args, own


def events_for(shot_id: str, names: Iterable[str], tl: dict) -> dict[str, int]:
    return {n: TL.event_frame(n, shot_id, tl) for n in names}


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------

def link(ob: bpy.types.Object, coll: bpy.types.Collection | None = None) -> bpy.types.Object:
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh_object(name: str, verts, faces, mat=None, *, edges=(), coll=None, smooth=False) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], list(edges), [tuple(f) for f in faces])
    me.update()
    if smooth:
        me.shade_smooth()
    if mat is not None:
        me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), coll)


def box(name, size, loc=(0, 0, 0), mat=None, *, bevel=0.0, segments=3, coll=None, rot=(0, 0, 0)) -> bpy.types.Object:
    """Axis-aligned box centred on ``loc`` (optionally with a bevel modifier)."""
    sx, sy, sz = (s * 0.5 for s in size)
    v = [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz), (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    ob = mesh_object(name, v, f, mat, coll=coll)
    ob.location = loc
    ob.rotation_euler = rot
    if bevel > 0:
        bv = ob.modifiers.new("Fase", "BEVEL")
        bv.width = bevel
        bv.segments = segments
        bv.limit_method = "ANGLE"
        bv.harden_normals = False
        ob.data.shade_smooth()
        wn = ob.modifiers.new("Normalen", "WEIGHTED_NORMAL")
        wn.keep_sharp = True
    return ob


def plane(name, w, h, loc=(0, 0, 0), mat=None, *, rot=(math.radians(90), 0, 0), coll=None, uv=True) -> bpy.types.Object:
    """Plane of ``w`` x ``h`` (in its local XY), by default standing in XZ facing -Y."""
    v = [(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)]
    ob = mesh_object(name, v, [(0, 1, 2, 3)], mat, coll=coll)
    if uv:
        uvl = ob.data.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    ob.location = loc
    ob.rotation_euler = rot
    return ob


def empty(name, loc=(0, 0, 0), coll=None) -> bpy.types.Object:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = 0.2
    ob.location = loc
    return link(ob, coll)


def text_mesh(name, body, *, font=FONT_SANS, size=0.1, extrude=0.0, align_x="CENTER", align_y="CENTER", mat=None, coll=None) -> bpy.types.Object:
    """Text converted to a mesh object (local XY, facing +Z)."""
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(font, check_existing=True)
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align_x
    cu.align_y = align_y
    cu.resolution_u = 4
    tmp = bpy.data.objects.new(name + "_txt", cu)
    link(tmp, coll)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.curves.remove(cu)
    # font meshes are not welded: make them closed (needed by EXACT booleans)
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.name = name
    if mat is not None:
        me.materials.clear()
        me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), coll)


def load_image(path: Path, *, colorspace="sRGB", name=None) -> bpy.types.Image:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} fehlt - erst 'blender/lib_szenen/assets_s2_s4.py' ausführen")
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = colorspace
    if name:
        img.name = name
    return img


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def new_mat(name: str):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if hasattr(mat, "use_nodes") and not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, nt


def value_node(nt, name: str, v: float):
    n = nt.nodes.new("ShaderNodeValue")
    n.name = n.label = name
    n.outputs[0].default_value = v
    return n.outputs[0]


def math_node(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, n.inputs[i])
        else:
            n.inputs[i].default_value = v
    return n.outputs[0]


def principled(
    name,
    color,
    *,
    rough=0.5,
    metal=0.0,
    spec=0.5,
    coat=0.0,
    coat_rough=0.05,
    sss=0.0,
    emission=None,
    estr=0.0,
    aniso=0.0,
    transmission=0.0,
    ior=1.45,
):
    mat, nt = new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.name = "BSDF"
    col = lin(color) if isinstance(color, str) else color
    b.inputs["Base Color"].default_value = col
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = coat_rough
    b.inputs["Subsurface Weight"].default_value = sss
    b.inputs["Anisotropic"].default_value = aniso
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["IOR"].default_value = ior
    if sss > 0:
        b.inputs["Subsurface Radius"].default_value = (0.6, 0.4, 0.3)
        b.inputs["Subsurface Scale"].default_value = 0.02
    if emission is not None:
        b.inputs["Emission Color"].default_value = lin(emission) if isinstance(emission, str) else emission
        b.inputs["Emission Strength"].default_value = estr
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def blended(mat, *, backface=False, shadow=False):
    """EEVEE Next sorted transparency (needed for additive light surfaces)."""
    mat.surface_render_method = "BLENDED"
    mat.use_backface_culling = not backface
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = True
    if not shadow and hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = True
    return mat


def glow_mat(name, color, strength=1.0, *, facing=0.0, alpha_noise=0.0):
    """Additive light: background stays, emission adds (``Staerke`` value).

    ``facing`` > 0 fades the light towards grazing angles (soft beams).
    """
    mat, nt = new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = lin(color) if isinstance(color, str) else color
    s = value_node(nt, "Staerke", strength)
    if facing > 0:
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = facing
        f = math_node(nt, "SUBTRACT", 1.0, lw.outputs["Facing"], clamp=True)
        s = math_node(nt, "MULTIPLY", s, math_node(nt, "POWER", f, 1.6))
    nt.links.new(s, em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    blended(mat, backface=True)
    return mat


def image_layer_mat(name, img, *, img_b=None, strength=1.0, opacity=1.0, mix_b=0.0, gloss=False):
    """Emissive image layer with alpha; values ``Deckkraft``, ``Staerke``, ``Mix``.

    With ``img_b`` the colour and alpha cross-fade from ``img`` to ``img_b``
    by ``Mix`` (used for glyphs -> bars on the typography layer).
    """
    mat, nt = new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    ta = nt.nodes.new("ShaderNodeTexImage")
    ta.image = img
    ta.interpolation = "Cubic"
    ta.extension = "CLIP"
    col, alpha = ta.outputs["Color"], ta.outputs["Alpha"]
    if img_b is not None:
        tb = nt.nodes.new("ShaderNodeTexImage")
        tb.image = img_b
        tb.interpolation = "Cubic"
        tb.extension = "CLIP"
        mv = value_node(nt, "Mix", mix_b)
        mc = nt.nodes.new("ShaderNodeMix")
        mc.data_type = "RGBA"
        nt.links.new(mv, mc.inputs["Factor"])
        nt.links.new(col, mc.inputs[6])
        nt.links.new(tb.outputs["Color"], mc.inputs[7])
        col = mc.outputs[2]
        ma = nt.nodes.new("ShaderNodeMix")
        ma.data_type = "FLOAT"
        nt.links.new(mv, ma.inputs["Factor"])
        nt.links.new(alpha, ma.inputs[2])
        nt.links.new(tb.outputs["Alpha"], ma.inputs[3])
        alpha = ma.outputs[0]
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col, em.inputs["Color"])
    nt.links.new(value_node(nt, "Staerke", strength), em.inputs["Strength"])
    surf = em.outputs[0]
    if gloss:
        gl = nt.nodes.new("ShaderNodeBsdfGlossy")
        gl.inputs["Roughness"].default_value = 0.08
        gl.inputs["Color"].default_value = (0.05, 0.05, 0.06, 1.0)
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(gl.outputs[0], add.inputs[1])
        surf = add.outputs[0]
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    fac = math_node(nt, "MULTIPLY", alpha, value_node(nt, "Deckkraft", opacity), clamp=True)
    nt.links.new(fac, mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(surf, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    blended(mat, backface=True)
    return mat


def drive_input(mat, input_name: str, value_name: str, value: float):
    """Feed a Principled input from a named Value node (keyable via node_path)."""
    nt = mat.node_tree
    b = nt.nodes["BSDF"]
    nt.links.new(value_node(nt, value_name, value), b.inputs[input_name])


def recalc_normals(ob) -> None:
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()


def node_path(name: str) -> str:
    return f'nodes["{name}"].outputs[0].default_value'


# ---------------------------------------------------------------------------
# keyframes
# ---------------------------------------------------------------------------

_INTERP = None


def _interp_enum(kind: str) -> int:
    global _INTERP
    if _INTERP is None:
        items = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items
        _INTERP = {it.identifier: it.value for it in items}
    return _INTERP[kind]


def bake(idb, path: str, frames: Sequence[float], values, *, index: int | None = None, interp: str = "LINEAR") -> None:
    """Replace the animation of ``idb.path`` by one key per frame (bulk API)."""
    ad = idb.animation_data or idb.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{idb.name}_Anim")
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
        idxs = [0 if index is None else index]
    else:
        idxs = list(range(vals.shape[1]))
    fr = np.asarray(frames, dtype=np.float64)
    n = len(fr)
    for k, i in enumerate(idxs):
        fc = ad.action.fcurve_ensure_for_datablock(idb, path, index=i)
        fc.keyframe_points.clear()
        fc.keyframe_points.add(n)
        co = np.empty(2 * n)
        co[0::2] = fr
        co[1::2] = vals[:, k]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [_interp_enum(interp)] * n)
        fc.update()


def bake_fn(idb, path: str, frames: Sequence[int], fn: Callable[[float], object], **kw) -> None:
    bake(idb, path, frames, [fn(f) for f in frames], **kw)


def bake_visible(ob, frames: Sequence[int], fn: Callable[[float], bool]) -> None:
    """Per-frame render visibility (constant keys)."""
    bake(ob, "hide_render", frames, [0.0 if fn(f) else 1.0 for f in frames], interp="CONSTANT")


# ---------------------------------------------------------------------------
# motion (pure python, see bewegung.py)
# ---------------------------------------------------------------------------

from bewegung import Bahn, overshoot, rewind_time  # noqa: E402

REEXPORTED = (Bahn, overshoot, rewind_time)  # used by the shot scripts as S.Bahn etc.


# ---------------------------------------------------------------------------
# lights and camera
# ---------------------------------------------------------------------------

def aim(ob, target) -> None:
    d = Vector(target) - Vector(ob.location)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def area(name, loc, target, size, energy, color, *, shape="SQUARE", size_y=None, coll=None, spread=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ld.energy = energy
    ld.color = color[:3]
    if spread is not None:
        ld.spread = spread
    ob = link(bpy.data.objects.new(name, ld), coll)
    ob.location = loc
    aim(ob, target)
    return ob


def point(name, loc, energy, color, *, radius=0.05, coll=None):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = energy
    ld.color = color[:3]
    ld.shadow_soft_size = radius
    ob = link(bpy.data.objects.new(name, ld), coll)
    ob.location = loc
    return ob


def spot(name, loc, target, energy, color, *, angle=40.0, blend=0.4, radius=0.05, coll=None):
    ld = bpy.data.lights.new(name, "SPOT")
    ld.energy = energy
    ld.color = color[:3]
    ld.spot_size = math.radians(angle)
    ld.spot_blend = blend
    ld.shadow_soft_size = radius
    ob = link(bpy.data.objects.new(name, ld), coll)
    ob.location = loc
    aim(ob, target)
    return ob


def camera(name, *, lens=50.0, fstop=2.8, sensor=36.0, blades=0):
    """Camera aimed by a target empty, focus on a separate empty (both baked)."""
    cam, tgt, foc = C.rig(name, (0, -5, 1), (0, 0, 1), lens=lens, fstop=fstop, sensor=sensor, blades=blades)
    return cam, tgt, foc


def bake_camera(cam, tgt, foc, frames, *, pos, target, focus=None, lens=None, fstop=None) -> None:
    """Bake camera/target/focus positions and optional lens/f-stop tracks."""
    bake_fn(cam, "location", frames, pos)
    bake_fn(tgt, "location", frames, target)
    bake_fn(foc, "location", frames, focus or target)
    if lens is not None:
        bake_fn(cam.data, "lens", frames, lens)
    if fstop is not None:
        bake_fn(cam.data, "dof.aperture_fstop", frames, fstop)


def world_gradient(scene, *, zenith, horizon, ground, strength=1.0, glow=None, clouds=0.0, name="S24Welt"):
    """Simple night sky by direction: ground / horizon / zenith (+ optional glow).

    ``glow`` = (hex colour, direction xyz, sharpness, strength) adds a soft lobe.
    """
    world = bpy.data.worlds.get(name) or bpy.data.worlds.new(name)
    if hasattr(world, "use_nodes") and not world.use_nodes:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    M.set_ramp(ramp, [(0.0, lin(ground)), (0.5, lin(horizon)), (0.62, lin(horizon)), (1.0, lin(zenith))])
    ma = nt.nodes.new("ShaderNodeMath")
    ma.operation = "MULTIPLY_ADD"
    nt.links.new(sep.outputs["Z"], ma.inputs[0])
    ma.inputs[1].default_value = 0.5
    ma.inputs[2].default_value = 0.5
    nt.links.new(ma.outputs[0], ramp.inputs["Fac"])
    col = ramp.outputs["Color"]
    if glow is not None:
        ghex, gdir, sharp, gstr = glow
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nt.links.new(tc.outputs["Generated"], dot.inputs[0])
        dot.inputs[1].default_value = Vector(gdir).normalized()
        lobe = math_node(nt, "POWER", math_node(nt, "MAXIMUM", dot.outputs["Value"], 0.0), sharp)
        add = nt.nodes.new("ShaderNodeMix")
        add.data_type = "RGBA"
        add.blend_type = "ADD"
        nt.links.new(math_node(nt, "MULTIPLY", lobe, gstr), add.inputs["Factor"])
        nt.links.new(col, add.inputs[6])
        add.inputs[7].default_value = lin(ghex)
        col = add.outputs[2]
    if clouds > 0:  # soft, wide cloud banks above the horizon
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 2.2
        noise.inputs["Detail"].default_value = 4.0
        noise.inputs["Roughness"].default_value = 0.55
        squash = nt.nodes.new("ShaderNodeMapping")
        squash.inputs["Scale"].default_value = (1.0, 1.0, 3.5)
        nt.links.new(tc.outputs["Generated"], squash.inputs["Vector"])
        nt.links.new(squash.outputs["Vector"], noise.inputs["Vector"])
        above = math_node(nt, "MULTIPLY", math_node(nt, "GREATER_THAN", sep.outputs["Z"], 0.0), clouds)
        k = math_node(nt, "ADD", 1.0, math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", noise.outputs["Fac"], 0.5), math_node(nt, "MULTIPLY", above, 2.0)))
        mul = nt.nodes.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        nt.links.new(col, mul.inputs[6])
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        for i in range(3):
            nt.links.new(k, comb.inputs[i])
        nt.links.new(comb.outputs[0], mul.inputs[7])
        col = mul.outputs[2]
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(col, bg.inputs["Color"])
    nt.links.new(value_node(nt, "Helligkeit", strength), bg.inputs["Strength"])
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    scene.world = world
    return world


def world_fog(scene, density: float, color="#8a90b8", *, anisotropy=0.35, start=0.3, end=90.0, tile="16", samples=32):
    """Thin homogeneous haze in the world volume (halos and light shafts)."""
    world = scene.world
    nt = world.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.name = "Nebel"
    vol.inputs["Color"].default_value = lin(color)
    vol.inputs["Density"].default_value = density
    vol.inputs["Anisotropy"].default_value = anisotropy
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    ee = scene.eevee
    ee.volumetric_start = start
    ee.volumetric_end = end
    ee.volumetric_tile_size = tile
    ee.volumetric_samples = samples
    ee.use_volumetric_shadows = True
    ee.volumetric_shadow_samples = 8
    return vol


# ---------------------------------------------------------------------------
# Geometry Nodes builder
# ---------------------------------------------------------------------------

class GN:
    """Tiny Geometry-Nodes builder: ``g = GN('Name'); ...; g.out(geo)``."""

    def __init__(self, name: str):
        old = bpy.data.node_groups.get(name)
        if old is not None:
            bpy.data.node_groups.remove(old)
        self.t = bpy.data.node_groups.new(name, "GeometryNodeTree")
        self.t.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        self.t.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        self.gi = self.t.nodes.new("NodeGroupInput")
        self.go = self.t.nodes.new("NodeGroupOutput")

    # inputs ---------------------------------------------------------------
    def inp(self, name: str, typ: str = "NodeSocketFloat", default=None):
        s = self.t.interface.new_socket(name=name, in_out="INPUT", socket_type=typ)
        if default is not None:
            s.default_value = default
        return self.gi.outputs[name]

    @property
    def geo(self):
        return self.gi.outputs["Geometry"]

    def out(self, sock) -> bpy.types.NodeTree:
        self.t.links.new(sock, self.go.inputs[0])
        return self.t

    # nodes ----------------------------------------------------------------
    def node(self, typ: str, **props):
        n = self.t.nodes.new(typ)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        elif v is not None:
            sock.default_value = v

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.put(n.inputs[i], v)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if scale is not None:
            self.put(n.inputs["Scale"], scale)
        return n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]

    def vec(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.put(n.inputs[i], v)
        return n.outputs[0]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.put(n.inputs[0], v)
        return n.outputs

    def attr(self, name: str, dtype: str = "FLOAT"):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=dtype)
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def store(self, geo, name: str, value, dtype: str = "FLOAT", domain: str = "POINT"):
        n = self.node("GeometryNodeStoreNamedAttribute", data_type=dtype, domain=domain)
        n.inputs["Name"].default_value = name
        self.put(n.inputs["Geometry"], geo)
        self.put(n.inputs["Value"], value)
        return n.outputs["Geometry"]

    def smoothstep(self, x, lo=0.0, hi=1.0, kind="SMOOTHERSTEP"):
        n = self.node("ShaderNodeMapRange", interpolation_type=kind)
        self.put(n.inputs["Value"], x)
        n.inputs["From Min"].default_value = lo
        n.inputs["From Max"].default_value = hi
        return n.outputs["Result"]

    def map_range(self, x, a, b, c=0.0, d=1.0, clamp=True):
        n = self.node("ShaderNodeMapRange", clamp=clamp)
        self.put(n.inputs["Value"], x)
        for name, v in (("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            self.put(n.inputs[name], v)
        return n.outputs["Result"]

    def lerp(self, a, b, t):
        return self.math("ADD", a, self.math("MULTIPLY", self.math("SUBTRACT", b, a), t))

    def vlerp(self, a, b, t):
        return self.vmath("ADD", a, self.vmath("SCALE", self.vmath("SUBTRACT", b, a), scale=t))

    def frame(self):
        return self.node("GeometryNodeInputSceneTime").outputs["Frame"]


def gn_modifier(ob, tree, name="GN") -> bpy.types.Modifier:
    mod = ob.modifiers.new(name, "NODES")
    mod.node_group = tree
    return mod


def object_material(ob, mat) -> None:
    """Link ``mat`` on the object (not the mesh data).

    Geometry-Nodes output built from primitives (instances, curve sweeps)
    carries an empty material list, so data-linked materials do not reach
    the render; an object-linked slot does.
    """
    if not ob.material_slots:
        ob.data.materials.append(None)
    ob.material_slots[0].link = "OBJECT"
    ob.material_slots[0].material = mat


def gn_ident(mod, socket_name: str) -> str:
    for item in mod.node_group.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name == socket_name:
            return item.identifier
    raise KeyError(socket_name)


def gn_set(ob, mod_name: str, socket_name: str, value) -> None:
    mod = ob.modifiers[mod_name]
    ident = gn_ident(mod, socket_name)
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        mod.properties.inputs[ident]["value"] = tuple(value) if isinstance(value, (tuple, list)) else value
    else:
        mod[ident] = value


def gn_path(ob, mod_name: str, socket_name: str) -> str:
    mod = ob.modifiers[mod_name]
    ident = gn_ident(mod, socket_name)
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        return f'modifiers["{mod_name}"].properties.inputs["{ident}"]["value"]'
    return f'modifiers["{mod_name}"]["{ident}"]'


def point_cloud(name, positions, attrs: dict[str, tuple[str, np.ndarray]], mat=None, coll=None) -> bpy.types.Object:
    """Vertex-only mesh with per-point attributes ``{name: (type, data)}``."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(positions))
    me.vertices.foreach_set("co", np.asarray(positions, dtype=np.float32).ravel())
    for aname, (kind, data) in attrs.items():
        a = me.attributes.new(aname, kind, "POINT")
        key = {"FLOAT": "value", "FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color", "INT": "value"}[kind]
        a.data.foreach_set(key, np.asarray(data, dtype=np.int32 if kind == "INT" else np.float32).ravel())
    me.update()
    if mat is not None:
        me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), coll)


# ---------------------------------------------------------------------------
# checks and render flow
# ---------------------------------------------------------------------------

def screen_bbox(scene, objs, frame) -> tuple[float, float, float, float] | None:
    """Projected bbox (px, origin top-left, 1920 space) of evaluated geometry."""
    from bpy_extras.object_utils import world_to_camera_view

    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    xs, ys = [], []
    for ob in objs:
        ev = ob.evaluated_get(dg)
        try:
            me = ev.to_mesh()
        except RuntimeError:
            continue
        if me is None:
            continue
        mw = ev.matrix_world
        for v in me.vertices:
            p = world_to_camera_view(scene, scene.camera, mw @ v.co)
            if p.z > 0:
                xs.append(p.x * 1920.0)
                ys.append((1.0 - p.y) * 1920.0)
        ev.to_mesh_clear()
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def render_key_mattes(scene, keys_by_frame: dict[int, list], out_dir: Path) -> list[Path]:
    """White alpha mattes of the event key objects (no DOF/MB/glare)."""
    all_keys = {ob.name: ob for objs in keys_by_frame.values() for ob in objs}
    R.matte_mode(scene, list(all_keys.values()))
    written = []
    for f, objs in sorted(keys_by_frame.items()):
        names = {o.name for o in objs}
        for ob in all_keys.values():
            ob.hide_render = ob.name not in names
            if ob.animation_data and ob.animation_data.action:
                for fc in C.fcurves(ob):
                    if fc.data_path == "hide_render":
                        fc.mute = True
        written += R.render_frames(scene, [f], out_dir)
    return written


def encode(shot_dir: Path, mov: Path, fps: int = 60) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[s2-s4] ffmpeg fehlt - kein .mov", flush=True)
        return
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(shot_dir / "%04d.png"),
        "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-r", str(fps), str(mov),
    ]
    subprocess.run(cmd, check=True)
    print(f"[s2-s4] {mov}", flush=True)


def run(scene, info: dict, args, own) -> None:
    """Common flow: save .blend, physics json, mattes, stills or full render."""
    sh = info["shot"]
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    if own.physik_json and "physik" in info:
        Path(own.physik_json).write_text(json.dumps(info["physik"](), indent=1) + "\n", encoding="utf-8")
        if args.frames == "none":
            return
    if args.frames == "none":
        return
    if own.matte_keys:
        names = [t.strip()[3:] for t in args.frames.split(",") if t.strip().startswith("ev:")]
        keys = {info["events"][n]: info["keys"][n] for n in names}
        render_key_mattes(scene, keys, Path(info["out"]))
        return
    frames = R.frame_list(args.frames, sh.frames, info["events"])
    if args.frames == "all":
        R.render_animation(scene)
        out = Path(info["out"])
        TL.write_shot_json(sh, out.parent / f"{sh.shot}.json")
        if not own.no_encode:
            encode(out, out.parent / f"{sh.shot}.mov", sh.fps)
    else:
        R.render_frames(scene, frames, info["out"])
