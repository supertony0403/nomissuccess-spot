"""Materials and the PRISMA night stage for the nomissuccess shots.

Colours are given as sRGB hex (brand tokens) and converted to scene-linear,
because the ``Standard`` view transform maps linear back to the exact sRGB
value for an unlit emission of strength 1.

* :func:`ribbon_material` - the logo band: colour ramp along the band (read
  from attribute ``u``), glossy lacquer, emissive edges (``kante``) and a hot
  growth head (``kopf``).
* :func:`window_material` - instanced city window lights (instance
  attributes ``farbe`` and ``hell``).
* :func:`hero_image` / :func:`world_backplate` - the website hero recipe
  (radial fields Rosa .30, Mint .20, Violett .28, Blau .92 from below) as a
  float image, projected onto a virtual screen in front of the reference
  camera so it behaves like a backplate at infinity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGO_BAND_JSON = REPO / "blender" / "assets" / "logo_band.json"

PRISMA = {
    "nacht": "#0b0c14",
    "nacht_flaeche": "#161826",
    "blau": "#145fe4",
    "violett": "#7c3aed",
    "rosa": "#e44b8d",
    "mint": "#10b981",
    "tinte": "#101223",
    "papier": "#ffffff",
}


# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------

def hex_to_srgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear(h: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    r, g, b = (srgb_to_linear(v) for v in hex_to_srgb(h))
    return (r, g, b, alpha)


def signal_gradient() -> list[tuple[float, tuple[float, float, float, float]]]:
    """PRISMA signal gradient: Blau 0 -> Violett .34 -> Rosa .66 -> Mint 1."""
    return [
        (0.00, hex_to_linear(PRISMA["blau"])),
        (0.34, hex_to_linear(PRISMA["violett"])),
        (0.66, hex_to_linear(PRISMA["rosa"])),
        (1.00, hex_to_linear(PRISMA["mint"])),
    ]


def logo_ramp(path: Path = LOGO_BAND_JSON) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Ramp sampled from logo.png along the band centreline (linear RGB)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(s[0], (s[1], s[2], s[3], 1.0)) for s in data["rampe_linear"]]


# --------------------------------------------------------------------------
# small node helpers
# --------------------------------------------------------------------------

def _new_material(name: str) -> tuple[bpy.types.Material, bpy.types.NodeTree]:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if hasattr(mat, "use_nodes") and not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, nt


def _attr(nt: bpy.types.NodeTree, name: str, kind: str = "GEOMETRY") -> bpy.types.Node:
    n = nt.nodes.new("ShaderNodeAttribute")
    n.attribute_name = name
    n.attribute_type = kind
    return n


def _math(nt, op, a, b=None, clamp=False):
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


def _value(nt, name: str, value: float) -> bpy.types.NodeSocket:
    n = nt.nodes.new("ShaderNodeValue")
    n.name = name
    n.label = name
    n.outputs[0].default_value = value
    return n.outputs[0]


def set_ramp(ramp_node: bpy.types.Node, stops) -> None:
    cr = ramp_node.color_ramp
    cr.interpolation = "LINEAR"
    cr.color_mode = "RGB"
    els = cr.elements
    while len(els) > 1:
        els.remove(els[-1])
    els[0].position = stops[0][0]
    els[0].color = stops[0][1]
    for pos, col in stops[1:]:
        e = els.new(pos)
        e.color = col


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------

def ribbon_material(
    name: str = "NomissBand",
    stops=None,
    *,
    emission: float = 0.55,
    edge: float = 2.2,
    head: float = 6.0,
    roughness: float = 0.30,
    coat_roughness: float = 0.045,
) -> bpy.types.Material:
    """Glossy gradient band with emissive edges and a hot growth head.

    Value nodes ``Leuchten`` (overall emission multiplier) and ``Kante``
    (edge glow) can be keyframed on the material node tree.
    """
    stops = stops or logo_ramp()
    mat, nt = _new_material(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    u = _attr(nt, "u")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    set_ramp(ramp, stops)
    nt.links.new(u.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # edges: smooth band along both long sides of the stadium profile
    kante = _attr(nt, "kante")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = 0.86
    mr.inputs["From Max"].default_value = 0.99
    nt.links.new(kante.outputs["Fac"], mr.inputs["Value"])
    edge_w = _math(nt, "MULTIPLY", mr.outputs["Result"], _value(nt, "Kante", edge))

    kopf = _attr(nt, "kopf")
    head_w = _math(nt, "MULTIPLY", _math(nt, "POWER", kopf.outputs["Fac"], 2.0), _value(nt, "Kopf", head))

    strength = _math(nt, "ADD", _value(nt, "Grundleuchten", emission), edge_w)
    strength = _math(nt, "ADD", strength, head_w)
    strength = _math(nt, "MULTIPLY", strength, _value(nt, "Leuchten", 1.0))

    # head glows whiter than the band
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    nt.links.new(_math(nt, "MULTIPLY", kopf.outputs["Fac"], 0.55), mix.inputs["Factor"])
    nt.links.new(ramp.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (1.0, 0.97, 0.94, 1.0)
    nt.links.new(mix.outputs[2], bsdf.inputs["Emission Color"])
    nt.links.new(strength, bsdf.inputs["Emission Strength"])

    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.65
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = coat_roughness
    bsdf.inputs["Coat IOR"].default_value = 1.55
    return mat


def assign(ob: bpy.types.Object, mat: bpy.types.Material, slot: int = 0) -> None:
    """Put ``mat`` into an object-linked slot.

    Geometry created by Geometry Nodes does not carry the input mesh's
    materials, so the material must live on the object, not on the data.
    """
    while len(ob.material_slots) <= slot:
        ob.data.materials.append(None)
    ms = ob.material_slots[slot]
    ms.link = "OBJECT"
    ms.material = mat


def flat_emission(name: str, color, strength: float = 1.0) -> bpy.types.Material:
    mat, nt = _new_material(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def window_material(name: str = "NomissFenster") -> bpy.types.Material:
    """Additive light: emission from instance attributes ``farbe``/``hell``.

    The quad itself is transparent, so a switched-off window leaves no dark
    patch in front of the dawn sky.
    """
    mat, nt = _new_material(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    farbe = _attr(nt, "farbe", "INSTANCER")
    hell = _attr(nt, "hell", "INSTANCER")
    nt.links.new(farbe.outputs["Color"], em.inputs["Color"])
    nt.links.new(_math(nt, "MULTIPLY", hell.outputs["Fac"], _value(nt, "Helligkeit", 1.0)), em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = False
    return mat


def bokeh_material(name: str = "NomissBokeh") -> bpy.types.Material:
    """Lens bokeh disc on an instanced quad (``quad_uv`` 0..1).

    Clear edge (3 % soft), slightly darker centre, faint brighter rim;
    additive (transparent + emission) so overlapping discs add up like
    light.  Colour/brightness from instance attributes ``farbe``/``hell``.
    """
    mat, nt = _new_material(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    uv = _attr(nt, "quad_uv")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    nt.links.new(uv.outputs["Vector"], sub.inputs[0])
    sub.inputs[1].default_value = (0.5, 0.5, 0.0)
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(sub.outputs[0], ln.inputs[0])
    r = _math(nt, "MULTIPLY", ln.outputs["Value"], 2.0)

    def smooth(a, b, x):
        n = nt.nodes.new("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP"
        n.inputs["From Min"].default_value = a
        n.inputs["From Max"].default_value = b
        nt.links.new(x, n.inputs["Value"])
        return n.outputs["Result"]

    edge = _math(nt, "SUBTRACT", 1.0, smooth(0.955, 1.0, r))
    shade = _math(nt, "ADD", 0.80, _math(nt, "MULTIPLY", smooth(0.30, 0.94, r), 0.20))
    ring = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", smooth(0.86, 0.95, r), smooth(0.97, 1.0, r)), 0.06)
    prof = _math(nt, "MULTIPLY", edge, _math(nt, "ADD", shade, ring))

    farbe = _attr(nt, "farbe", "INSTANCER")
    hell = _attr(nt, "hell", "INSTANCER")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(farbe.outputs["Color"], em.inputs["Color"])
    strength = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", hell.outputs["Fac"], prof), _value(nt, "Helligkeit", 1.0))
    nt.links.new(strength, em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    return mat


def facade_material(name: str = "NomissFassade", haze: str = "#161826", haze_m: float = 700.0) -> bpy.types.Material:
    """Near-black facades with aerial perspective.

    Emission = ``Dunst`` colour x (1 - exp(-distance / haze_m)) x ``Dunst
    Staerke``, so far buildings melt into the night stage instead of
    standing as black blocks.  Both values are keyable node values.
    """
    mat, nt = _new_material(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = hex_to_linear("#07080d")
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    cam = nt.nodes.new("ShaderNodeCameraData")
    fog = _math(nt, "SUBTRACT", 1.0, _math(nt, "EXPONENT", _math(nt, "DIVIDE", cam.outputs["View Distance"], -haze_m)))
    col = nt.nodes.new("ShaderNodeRGB")
    col.name = col.label = "Dunst"
    col.outputs[0].default_value = hex_to_linear(haze)
    nt.links.new(col.outputs[0], bsdf.inputs["Emission Color"])
    nt.links.new(_math(nt, "MULTIPLY", fog, _value(nt, "Dunst Staerke", 1.0)), bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# --------------------------------------------------------------------------
# PRISMA stage: the website hero recipe as a float image
# --------------------------------------------------------------------------

# (colour, cx, cy, rx, ry, [(stop, alpha), ...]) bottom -> top, CSS units (0..1)
HERO_LAYERS = [
    ("#145fe4", 0.50, 1.18, 0.80, 0.95, [(0.0, 0.92), (0.58, 0.28), (0.80, 0.0)]),
    ("#7c3aed", 0.62, 0.42, 0.60, 0.70, [(0.0, 0.28), (0.72, 0.0)]),
    ("#10b981", 0.84, 0.14, 0.50, 0.60, [(0.0, 0.20), (0.70, 0.0)]),
    ("#e44b8d", 0.16, 0.18, 0.56, 0.64, [(0.0, 0.30), (0.68, 0.0)]),
]
HERO_BASE = [(0.0, "#0b0c14"), (0.55, "#10121e"), (1.0, "#0b0c14")]


def hero_srgb(
    w: int,
    h: int,
    *,
    gains: dict[str, float] | None = None,
    blue_cy: float = 1.18,
    extra: list | None = None,
    window: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> np.ndarray:
    """Composite the hero layers like a browser does (sRGB, 'over').

    ``window`` = (x0, y0, x1, y1) of the CSS box covered by the image, so the
    image can extend beyond the box (overscan) with the same field layout.
    ``gains`` scale the layer alphas by colour hex, ``blue_cy`` moves the
    blue field (dawn rises from below), ``extra`` adds layers on top.
    """
    gains = gains or {}
    x0, y0, x1, y1 = window
    xs = np.linspace(x0, x1, w)[None, :]
    ys = np.linspace(y0, y1, h)[:, None]
    yy = np.clip(ys, 0.0, 1.0) * np.ones_like(xs)
    base_pos = [p for p, _ in HERO_BASE]
    base_cols = np.array([hex_to_srgb(c) for _, c in HERO_BASE])
    img = np.stack([np.interp(yy, base_pos, base_cols[:, k]) for k in range(3)], axis=-1)
    layers = list(HERO_LAYERS) + list(extra or [])
    for i, (col, cx, cy, rx, ry, stops) in enumerate(layers):
        if i == 0:  # the blue field from below
            cy = blue_cy
        d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
        a = np.interp(d, [s for s, _ in stops], [al for _, al in stops], right=0.0)
        a = np.clip(a * gains.get(col, 1.0), 0.0, 1.0)[..., None]
        img = img * (1 - a) + np.array(hex_to_srgb(col)) * a
    return img


def hero_image(name: str, size: int = 1024, **kw) -> bpy.types.Image:
    """Float image (scene-linear) of the hero composite."""
    srgb = hero_srgb(size, size, **kw)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    rgba = np.concatenate([lin, np.ones(lin.shape[:2] + (1,))], axis=-1)
    rgba = rgba[::-1]  # Blender images start at the bottom row
    img = bpy.data.images.get(name)
    if img is None or img.size[0] != size:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Linear Rec.709"
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    img.pack()
    return img


def world_backplate(
    scene: bpy.types.Scene,
    images: list[bpy.types.Image],
    *,
    lens_mm: float,
    sensor_mm: float = 36.0,
    overscan: float = 1.6,
    pitch: float = 0.0,
    name: str = "NomissBuehne",
) -> bpy.types.World:
    """World = hero images on a virtual screen ahead (+Y) of the camera.

    ``overscan`` is the size of the virtual screen relative to the reference
    frame; the image's CSS window must match (see ``hero_srgb(window=...)``).
    ``pitch`` (rad) tilts the screen with a camera that looks up/down.
    Value nodes: ``Mischung`` (blend image 0 -> 1), ``Anstieg`` (vertical
    offset of image 1 in screen units, + = shifted down), ``Helligkeit``.
    """
    world = bpy.data.worlds.get(name) or bpy.data.worlds.new(name)
    if hasattr(world, "use_nodes") and not world.use_nodes:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    half = (sensor_mm / lens_mm) * 0.5 * overscan  # tan(half fov) * overscan

    tc = nt.nodes.new("ShaderNodeTexCoord")
    rot = nt.nodes.new("ShaderNodeVectorRotate")
    rot.rotation_type = "X_AXIS"
    rot.inputs["Angle"].default_value = -pitch
    nt.links.new(tc.outputs["Generated"], rot.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(rot.outputs["Vector"], sep.inputs[0])
    dy = _math(nt, "MAXIMUM", sep.outputs["Y"], 0.02)
    sx = _math(nt, "ADD", _math(nt, "MULTIPLY", _math(nt, "DIVIDE", sep.outputs["X"], dy), 0.5 / half), 0.5)
    sz = _math(nt, "ADD", _math(nt, "MULTIPLY", _math(nt, "DIVIDE", sep.outputs["Z"], dy), 0.5 / half), 0.5)
    rise = _value(nt, "Anstieg", 0.0)

    texs = []
    for i, img in enumerate(images):
        v = sz if i == 0 else _math(nt, "ADD", sz, rise)
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(sx, comb.inputs["X"])
        nt.links.new(v, comb.inputs["Y"])
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.extension = "EXTEND"
        tex.interpolation = "Cubic"
        nt.links.new(comb.outputs[0], tex.inputs["Vector"])
        texs.append(tex)

    col = texs[0].outputs["Color"]
    if len(texs) > 1:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(_value(nt, "Mischung", 0.0), mix.inputs["Factor"])
        nt.links.new(texs[0].outputs["Color"], mix.inputs[6])
        nt.links.new(texs[1].outputs["Color"], mix.inputs[7])
        col = mix.outputs[2]

    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(col, bg.inputs["Color"])
    nt.links.new(_value(nt, "Helligkeit", 1.0), bg.inputs["Strength"])
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    scene.world = world
    return world


def node_value(tree_owner, name: str) -> bpy.types.NodeSocket:
    """Output socket of a named Value node (for keyframing)."""
    return tree_owner.node_tree.nodes[name].outputs[0]


def key_node_value(tree_owner, name: str, frame: float, value: float) -> None:
    sock = node_value(tree_owner, name)
    sock.default_value = value
    tree_owner.node_tree.keyframe_insert(f'nodes["{name}"].outputs[0].default_value', frame=frame)


def kelvin_rgb(k: float) -> tuple[float, float, float]:
    """Approximate linear RGB of a black body (1000-12000 K), max = 1."""
    t = k / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    rgb = [min(max(c, 0.0), 255.0) / 255.0 for c in (r, g, b)]
    lin = [srgb_to_linear(c) for c in rgb]
    m = max(lin)
    return tuple(c / m for c in lin)  # type: ignore[return-value]
