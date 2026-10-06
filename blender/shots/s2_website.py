"""Shot s2_website - the website as a lit shop window on a wet night street.

Beats (event frames are read from timeline.json at render time):

* shot start .. ``window_reveal``: the camera drifts down a dark street, then
  turns towards the facade; the shop window (the real nomissuccess.de start
  page) powers up as it swings into view,
* ``window_land``: the camera comes to rest frontally, the wet pavement
  mirrors the page,
* ``layers_explode``: the page splits into its real layers - base colour,
  colour fields, layout grid, typography (glyphs turn into bright bars) and
  the real Astro source of exactly this hero - which fly apart in depth while
  the camera orbits 25 degrees to the side (shallow depth of field),
* ``layers_snap``: the layers rush back and lock in (overshoot < 3 %),
* ``tower_fall``: a tower of soft plastic building blocks, each engraved with
  a Baukasten word, is knocked over (baked Bullet rigid bodies) and falls off
  the ledge out of frame,
* ``speed_line``: a passing headlight sweeps the facade (Fusion draws the line),
* ``key_turn``: a metal key with an N-shaped bitting slides into the lock on
  the pilaster and turns 90 degrees,
* scene end: the camera swings back and dives towards the page (cut into s3).

The page is three coincident layers (grund + bild + typo), so it looks exactly
like the screenshot at rest and its colour fields drift like on the real site.

Usage::

    blender -b --factory-startup -P blender/shots/s2_website.py -- [--proxy 480] [--frames a-b|ev:name]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib_szenen"))

import szenen_s2_s4 as S  # noqa: E402  (also puts blender/lib on sys.path)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import nomiss_camera as C  # noqa: E402
import nomiss_material as M  # noqa: E402
import nomiss_render as R  # noqa: E402
import nomiss_timeline as TL  # noqa: E402

SHOT = "s2_website"
EVENTS = (
    "window_reveal", "window_land", "layers_explode", "layers_snap", "tower_fall",
    "speed_line", "de_pin", "calendar_ticks", "month_carousel", "key_turn",
)

# ---- set geometry (metres; facade front at y = 0, facing -Y) ---------------
OX0, OX1, OZ0, OZ1 = -1.55, 2.35, 0.78, 2.78   # shop window opening
WALL_T = 0.45
WEB_W, WEB_H = 2.4, 1.5
WEB_C = Vector((-0.15, 0.30, 1.73))           # page centre
LEDGE_FRONT = -0.42
SIDEWALK_Z = 0.12
TOWER_X, TOWER_Y = 1.62, -0.25
PILASTER = (2.55, 3.15)
LOCK = Vector((0.80, 0.282, 1.28))         # lock cylinder rises out of the page

WORDS = [  # bottom -> top: (word, width, height, colour)
    ("Cookie-Banner", 0.54, 0.17, "#b9a27e"),
    ("Template", 0.46, 0.17, "#8e9db8"),
    ("Plugin", 0.40, 0.16, "#c99aa3"),
    ("Tracking", 0.44, 0.165, "#93ad98"),
    ("Theme", 0.38, 0.16, "#b4a6c4"),
    ("Widget", 0.40, 0.155, "#c7b28a"),
    ("Slider", 0.36, 0.16, "#86a3a6"),
    ("Pop-up", 0.38, 0.15, "#c4998a"),
    ("Add-on", 0.34, 0.155, "#a3a9b6"),
]
BLOCK_D = 0.25
FOG = 0.0     # world haze tested on 06.10.: froxel artefacts at affordable settings -> off

# page layers: name, image, rest y-offset, explode offset (dy, dx, dz, yaw deg).
# An exploded view along the page normal, fanned left/down so every sheet
# shows its edge behind the next one.
LAYERS = [
    ("Grund", "web_grund.png", 0.000, (0.30, 0.10, 0.05, 0.0)),
    ("Bild", "web_bild.png", -0.004, (-0.14, -0.06, -0.01, 1.0)),
    ("Raster", "web_raster.png", -0.008, (-0.62, -0.24, -0.06, 2.0)),
    ("Typo", "web_typo.png", -0.012, (-1.10, -0.42, -0.11, 3.0)),
    ("Code", "web_code.png", -0.016, (-1.58, -0.60, -0.16, 4.0)),
]


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def facade_mat():
    mat, nt = S.new_mat("S2Fassade")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    M.set_ramp(ramp, [(0.3, S.lin("#141419")), (0.7, S.lin("#1d1d24"))])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.78
    b.inputs["Specular IOR Level"].default_value = 0.35
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 60.0
    nt.links.new(fine.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def wet_ground_mat(name, base, *, pavers=False, puddle=0.55, dry_rough=0.34):
    """Wet pavement/asphalt: puddles are mirror-like, the rest damp."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.55
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.6
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    wet = nt.nodes.new("ShaderNodeMapRange")
    wet.inputs["From Min"].default_value = puddle - 0.06
    wet.inputs["From Max"].default_value = puddle + 0.04
    nt.links.new(noise.outputs["Fac"], wet.inputs["Value"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = dry_rough
    rough.inputs["To Max"].default_value = 0.025
    nt.links.new(wet.outputs["Result"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], b.inputs["Roughness"])
    col = nt.nodes.new("ShaderNodeMix")
    col.data_type = "RGBA"
    nt.links.new(wet.outputs["Result"], col.inputs["Factor"])
    col.inputs[6].default_value = S.lin(base)
    col.inputs[7].default_value = tuple(c * 0.55 for c in S.lin(base)[:3]) + (1.0,)
    base_col = col.outputs[2]
    bump = nt.nodes.new("ShaderNodeBump")
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 140.0
    nt.links.new(tc.outputs["Object"], fine.inputs["Vector"])
    if pavers:
        brick = nt.nodes.new("ShaderNodeTexBrick")
        brick.inputs["Scale"].default_value = 1.0
        brick.inputs["Mortar Size"].default_value = 0.012
        brick.inputs["Brick Width"].default_value = 0.40
        brick.inputs["Row Height"].default_value = 0.20
        brick.inputs["Color1"].default_value = (1, 1, 1, 1)
        brick.inputs["Color2"].default_value = (0.85, 0.85, 0.85, 1)
        brick.inputs["Mortar"].default_value = (0, 0, 0, 1)
        nt.links.new(tc.outputs["Object"], brick.inputs["Vector"])
        h = S.math_node(nt, "ADD", brick.outputs["Fac"], S.math_node(nt, "MULTIPLY", fine.outputs["Fac"], 0.08))
        nt.links.new(h, bump.inputs["Height"])
        dark = nt.nodes.new("ShaderNodeMix")
        dark.data_type = "RGBA"
        dark.blend_type = "MULTIPLY"
        dark.inputs["Factor"].default_value = 1.0
        nt.links.new(base_col, dark.inputs[6])
        nt.links.new(brick.outputs["Color"], dark.inputs[7])
        base_col = dark.outputs[2]
        bump.inputs["Strength"].default_value = 0.25
    else:
        nt.links.new(fine.outputs["Fac"], bump.inputs["Height"])
        bump.inputs["Strength"].default_value = 0.03
    # puddles are flat: no bump where wet
    nt.links.new(S.math_node(nt, "MULTIPLY", S.math_node(nt, "SUBTRACT", 1.0, wet.outputs["Result"]), 0.25 if pavers else 0.04), bump.inputs["Strength"])
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(base_col, b.inputs["Base Color"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Specular IOR Level"].default_value = 0.6
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def window_lights_mat():
    """Lit window: frame and mullion cross stay dark, light falls off upwards."""
    mat, nt = S.new_mat("S2Fensterlicht")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "farbe"
    st = nt.nodes.new("ShaderNodeAttribute")
    st.attribute_name = "hell"
    uv = nt.nodes.new("ShaderNodeUVMap")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    u, v = sep.outputs["X"], sep.outputs["Y"]
    mm = S.math_node

    def inside(x, lo, hi):
        return mm(nt, "MULTIPLY", mm(nt, "GREATER_THAN", x, lo), mm(nt, "LESS_THAN", x, hi))

    def gap(x, c, w):
        return mm(nt, "GREATER_THAN", mm(nt, "ABSOLUTE", mm(nt, "SUBTRACT", x, c)), w)

    mask = mm(nt, "MULTIPLY", inside(u, 0.06, 0.94), inside(v, 0.07, 0.93))
    mask = mm(nt, "MULTIPLY", mask, gap(u, 0.5, 0.018))
    mask = mm(nt, "MULTIPLY", mask, gap(v, 0.64, 0.016))
    grad = mm(nt, "ADD", 0.55, mm(nt, "MULTIPLY", mm(nt, "SUBTRACT", 1.0, v), 0.6))
    strength = mm(nt, "MULTIPLY", mm(nt, "MULTIPLY", st.outputs["Fac"], mask), grad)
    nt.links.new(at.outputs["Color"], em.inputs["Color"])
    nt.links.new(strength, em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def plastic_mat(name, color):
    return S.principled(name, color, rough=0.42, spec=0.45, sss=0.12, coat=0.25, coat_rough=0.28)


def engrave_mat(name, color):
    r, g, b, _ = S.lin(color)
    return S.principled(name, (r * 0.42, g * 0.42, b * 0.42, 1.0), rough=0.7, spec=0.3)


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

def build_street(coll):
    fac = facade_mat()
    stone = S.principled("S2Stein", "#2b2b31", rough=0.5, spec=0.5, coat=0.35, coat_rough=0.12)
    metal = S.principled("S2Mast", "#17181d", rough=0.38, metal=0.7)
    side = wet_ground_mat("S2Gehweg", "#121318", pavers=True, puddle=0.52)
    road = wet_ground_mat("S2Asphalt", "#09090c", puddle=0.47, dry_rough=0.28)
    dark_glass = S.principled("S2Fensterglas", "#050608", rough=0.06, spec=0.7)

    # facade with the shop-window opening
    S.box("Fassade_L", (OX0 + 16.0, WALL_T, 10.0), ((OX0 - 16.0) / 2, WALL_T / 2, 5.0), fac, coll=coll)
    S.box("Fassade_R", (30.0 - OX1, WALL_T, 10.0), ((OX1 + 30.0) / 2, WALL_T / 2, 5.0), fac, coll=coll)
    S.box("Fassade_U", (OX1 - OX0, WALL_T, OZ0), ((OX0 + OX1) / 2, WALL_T / 2, OZ0 / 2), fac, coll=coll)
    S.box("Fassade_O", (OX1 - OX0, WALL_T, 10.0 - OZ1), ((OX0 + OX1) / 2, WALL_T / 2, (OZ1 + 10.0) / 2), fac, coll=coll)
    # architectural detail: plinth, cornice, pilasters, window ledge
    S.box("Sockel", (46.0, 0.06, 0.55), (7.0, -0.03, 0.275), stone, coll=coll)
    S.box("Gesims", (46.0, 0.16, 0.2), (7.0, -0.08, 3.35), stone, bevel=0.01, coll=coll)
    for x0, x1 in ((-6.2, -5.6), (-11.2, -10.6), PILASTER, (7.4, 8.0), (12.4, 13.0), (17.4, 18.0)):
        S.box(f"Pilaster_{x0:+.1f}", (x1 - x0, 0.06, 2.75), ((x0 + x1) / 2, -0.03, 0.55 + 1.375), stone, coll=coll)
    ledge = S.box("Sims", (OX1 - OX0 + 0.2, WALL_T - LEDGE_FRONT, 0.08), ((OX0 + OX1) / 2, (WALL_T + LEDGE_FRONT) / 2, OZ0 - 0.04), stone, bevel=0.008, coll=coll)
    # display box behind the opening
    inner = S.principled("S2Auslage", "#0e0e14", rough=0.6)
    S.box("Auslage_Boden", (OX1 - OX0 + 0.4, 1.4, 0.05), ((OX0 + OX1) / 2, WALL_T + 0.7, OZ0 - 0.025), inner, coll=coll)
    S.box("Auslage_Rueck", (OX1 - OX0 + 0.4, 0.05, OZ1 - OZ0 + 0.4), ((OX0 + OX1) / 2, WALL_T + 1.3, (OZ0 + OZ1) / 2), inner, coll=coll)
    S.box("Auslage_Decke", (OX1 - OX0 + 0.4, 1.4, 0.05), ((OX0 + OX1) / 2, WALL_T + 0.7, OZ1 + 0.025), inner, coll=coll)
    # sidewalk, curb, road
    sw = S.box("Gehweg", (90.0, 3.6, SIDEWALK_Z), (10.0, -1.8, SIDEWALK_Z / 2), side, coll=coll)
    S.box("Bordstein", (90.0, 0.18, SIDEWALK_Z + 0.02), (10.0, -3.6, (SIDEWALK_Z + 0.02) / 2), stone, bevel=0.02, coll=coll)
    rd = S.plane("Strasse", 90.0, 9.0, (10.0, -8.0, 0.0), road, rot=(0, 0, 0))
    S.box("Gehweg_gegenueber", (90.0, 3.6, SIDEWALK_Z), (10.0, -14.2, SIDEWALK_Z / 2), side, coll=coll)
    S.box("Fassade_gegenueber", (95.0, 0.5, 13.0), (12.0, -16.25, 6.5), fac, coll=coll)
    S.box("Fassade_Ende", (0.5, 16.0, 14.0), (64.0, -8.0, 7.0), fac, coll=coll)

    # window lights for bokeh: our upper floors, the opposite side, the far end
    rng = np.random.default_rng(22)
    quads, cols, hells, mats = [], [], [], []

    def window(cx, cz, w, h, y, nrm_y, lit_p, kelvin=(2600, 3600), bright=(0.3, 1.0)):
        lit = rng.random() < lit_p
        if nrm_y < 0:
            v = [(cx - w / 2, y, cz - h / 2), (cx + w / 2, y, cz - h / 2), (cx + w / 2, y, cz + h / 2), (cx - w / 2, y, cz + h / 2)]
        else:
            v = [(cx + w / 2, y, cz - h / 2), (cx - w / 2, y, cz - h / 2), (cx - w / 2, y, cz + h / 2), (cx + w / 2, y, cz + h / 2)]
        quads.append(v)
        k = rng.uniform(*kelvin) if rng.random() < 0.8 else rng.uniform(5200, 6800)
        cols.append((*M.kelvin_rgb(k), 1.0))
        hells.append(rng.uniform(*bright) if lit else 0.0)
        mats.append(0 if lit else 1)

    for x in np.arange(-14.5, 29.0, 2.6):
        for cz in (5.0, 7.6):
            near = -5.0 < x < 6.0  # above the shop window: keep it calm and dark
            window(x, cz, 1.25, 1.55, -0.012, -1, 0.0 if near else 0.12, kelvin=(2400, 3000), bright=(0.12, 0.4))
    for x in np.arange(-20.0, 60.0, 2.4):
        for cz in (1.7, 4.6, 7.4, 10.2):
            window(x, cz, 1.2 if cz > 2 else 2.0, 1.5 if cz > 2 else 1.9, -15.99, 1, 0.42 if cz > 2 else 0.6, bright=(0.5, 1.8))
    for y in np.arange(-15.0, -1.0, 2.4):
        for cz in (1.7, 4.6, 7.4, 10.2):
            vx = 63.74
            w, h = 1.2, 1.5
            quads.append([(vx, y + w / 2, cz - h / 2), (vx, y - w / 2, cz - h / 2), (vx, y - w / 2, cz + h / 2), (vx, y + w / 2, cz + h / 2)])
            lit = rng.random() < 0.5
            cols.append((*M.kelvin_rgb(rng.uniform(2600, 4200)), 1.0))
            hells.append(rng.uniform(0.6, 2.0) if lit else 0.0)
            mats.append(0 if lit else 1)
    verts = [v for q in quads for v in q]
    faces = [(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3) for i in range(len(quads))]
    wl = S.mesh_object("Fensterlichter", verts, faces, window_lights_mat(), coll=coll)
    uvl = wl.data.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", np.tile([0, 0, 1, 0, 1, 1, 0, 1], len(quads)).astype(np.float32))
    wl.data.materials.append(dark_glass)
    wl.data.polygons.foreach_set("material_index", mats)
    ac = wl.data.attributes.new("farbe", "FLOAT_COLOR", "FACE")
    ac.data.foreach_set("color", np.asarray(cols, dtype=np.float32).ravel())
    ah = wl.data.attributes.new("hell", "FLOAT", "FACE")
    ah.data.foreach_set("value", np.asarray(hells, dtype=np.float32))

    # street lamps: posts + warm heads (bokeh) + light
    lamps = []
    for x, y in [(-7.0, -3.2), (9.0, -3.2), (25.0, -3.2), (41.0, -3.2), (1.0, -12.8), (17.0, -12.8), (33.0, -12.8), (49.0, -12.8)]:
        S.box(f"Mast_{x:.0f}_{y:.0f}", (0.11, 0.11, 4.6), (x, y, 2.3 + SIDEWALK_Z), metal, coll=coll)
        arm_y = y + (0.55 if y > -8 else -0.55)
        S.box(f"Arm_{x:.0f}_{y:.0f}", (0.09, 1.2, 0.07), (x, (y + arm_y) / 2, 4.85), metal, coll=coll)
        S.box(f"Kopf_{x:.0f}_{y:.0f}", (0.42, 0.3, 0.1), (x, arm_y, 4.8), metal, coll=coll)
        glow = S.principled(f"S2Lampe_{x:.0f}_{y:.0f}", "#000000", rough=0.5, emission=M.kelvin_rgb(2700) + (1.0,), estr=60.0)
        S.plane(f"Leuchte_{x:.0f}_{y:.0f}", 0.36, 0.24, (x, arm_y, 4.745), glow, rot=(math.pi, 0, 0), coll=coll)
        lamp = S.spot(f"Licht_{x:.0f}_{y:.0f}", (x, arm_y, 4.7), (x, arm_y, 0.0), 900.0, M.kelvin_rgb(2700), angle=118.0, blend=0.8, radius=0.15, coll=coll)
        lamp.data.use_shadow = (x, y) in ((-7.0, -3.2), (9.0, -3.2))
        lamps.append(lamp)
    return {"ledge": ledge, "sidewalk": sw, "road": rd, "lamps": lamps}


def build_page(coll):
    """The website as coincident emissive layers inside the opening."""
    layers = {}
    for name, fname, y_off, _ in LAYERS:
        img = S.load_image(S.ASSETS / fname)
        if name == "Typo":
            mat = S.image_layer_mat("S2Web_Typo", img, img_b=S.load_image(S.ASSETS / "web_balken.png"), strength=0.0)
        else:
            mat = S.image_layer_mat(f"S2Web_{name}", img, strength=0.0, opacity=1.0 if name in ("Grund", "Bild") else 0.0)
        if name in ("Grund", "Bild", "Typo"):
            mat.surface_render_method = "DITHERED"  # visible in screen-space reflections
        ob = S.plane(f"Web_{name}", WEB_W, WEB_H, (WEB_C.x, WEB_C.y + y_off, WEB_C.z), mat, coll=coll)
        layers[name] = ob
        # hairline outline, only lit while the page is exploded
        w2, h2, t = WEB_W / 2, WEB_H / 2, 0.006
        v = [(-w2, -h2, 0), (w2, -h2, 0), (w2, h2, 0), (-w2, h2, 0), (-w2 + t, -h2 + t, 0), (w2 - t, -h2 + t, 0), (w2 - t, h2 - t, 0), (-w2 + t, h2 - t, 0)]
        f = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        fr = S.mesh_object(f"Web_{name}_Kontur", v, f, S.glow_mat(f"S2Kontur_{name}", "#a9c4ff", 0.0), coll=coll)
        fr.parent = ob
        fr.location = (0.0, 0.0, 0.003)
        layers[name + "_Kontur"] = fr
    # the colour fields drift like on the real site (CSS treiben-a/-b)
    nt = layers["Bild"].active_material.node_tree
    tex = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage")
    tex.extension = "EXTEND"
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.name = "Drift"
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    return layers


def block_mesh(name, w, h, d, word, color, coll):
    """Rounded plastic block with the word engraved into its front face."""
    ob = S.box(name, (w, d, h), (0, 0, 0), plastic_mat(f"S2Stein_{word}", color), bevel=0.016, segments=4, coll=coll)
    size = min(0.085, 1.55 * (w - 0.08) / max(len(word), 4))
    txt = S.text_mesh(f"{name}_Gravur", word, size=size, extrude=0.006, mat=engrave_mat(f"S2Gravur_{word}", color), coll=coll)
    txt.rotation_euler = (math.radians(90), 0, 0)
    txt.location = (0.0, -d / 2 + 0.0035, 0.0)
    bo = ob.modifiers.new("Gravur", "BOOLEAN")
    bo.operation = "DIFFERENCE"
    bo.solver = "EXACT"
    bo.object = txt
    bo.material_mode = "TRANSFER"
    bo.use_hole_tolerant = True  # font meshes are not closed
    ob.data.materials.append(txt.data.materials[0])
    # keep the boolean result fixed: apply both modifiers
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    bpy.data.meshes.remove(old)
    bpy.data.objects.remove(txt, do_unlink=True)
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def build_tower(coll, rng):
    blocks = []
    z = OZ0
    for i, (word, w, h, col) in enumerate(WORDS):
        ob = block_mesh(f"Baustein_{i}_{word}", w, h, BLOCK_D, word, col, coll)
        jitter_x = 0.0 if i == 0 else rng.uniform(-0.025, 0.025)
        yaw = 0.0 if i == 0 else rng.uniform(-5.0, 5.0)
        ob.location = (TOWER_X + jitter_x, TOWER_Y + rng.uniform(-0.01, 0.01), z + h / 2)
        ob.rotation_euler = (0.0, 0.0, math.radians(yaw))
        z += h
        blocks.append(ob)
    return blocks


def build_key_lock(coll):
    steel = S.principled("S2Edelstahl", "#a7abb3", rough=0.3, metal=1.0, aniso=0.5)
    nickel = S.principled("S2Neusilber", "#cfd2d8", rough=0.14, metal=1.0)
    black = S.principled("S2Schlitz", "#020203", rough=0.9)
    # escutcheon + rotating core
    lock = S.empty("Schloss", tuple(LOCK), coll)
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=0.046, depth=0.012, location=(0.0, 0.0, 0.0), rotation=(math.radians(90), 0, 0))
    rosette = bpy.context.object
    rosette.name = "Schloss_Rosette"
    rosette.data.materials.append(steel)
    bv = rosette.modifiers.new("Fase", "BEVEL")
    bv.width = 0.003
    bv.segments = 4
    rosette.data.shade_smooth()
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.019, depth=0.006, location=(0.0, -0.0075, 0.0), rotation=(math.radians(90), 0, 0))
    core = bpy.context.object
    core.name = "Schloss_Kern"
    core.data.materials.append(nickel)
    bv = core.modifiers.new("Fase", "BEVEL")
    bv.width = 0.0012
    bv.segments = 3
    core.data.shade_smooth()
    slot = S.box("Schloss_Schlitz", (0.0034, 0.017, 0.004), (0, 0, 0), black, coll=coll)
    slot.parent = core
    slot.location = (0.0, 0.0, 0.0012)  # core local: axis = local Z (towards -Y world), local Y = world Z
    ring_mat = S.principled("S2Ring", "#000000", rough=0.3, emission=S.lin("#10b981"), estr=0.0)
    S.drive_input(ring_mat, "Emission Strength", "Leuchten", 0.0)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.031, minor_radius=0.0014, major_segments=96, minor_segments=12, location=(0.0, -0.0062, 0.0), rotation=(math.radians(90), 0, 0))
    ring = bpy.context.object
    ring.name = "Schloss_Ring"
    ring.data.materials.append(ring_mat)
    for ob in (rosette, core, ring):
        for c in ob.users_collection:
            c.objects.unlink(ob)
        coll.objects.link(ob)
        ob.parent = lock

    # key profile in the (y, z) plane; origin = shoulder, blade along +y
    blade = [(0.0, -0.0052), (0.0, 0.0050), (0.008, 0.0050), (0.008, 0.0102), (0.024, 0.0058), (0.024, 0.0102),
             (0.034, 0.0102), (0.042, 0.0072), (0.050, 0.0094), (0.058, 0.0066), (0.065, 0.0040), (0.065, -0.0018),
             (0.060, -0.0052)]
    key_parts = []

    def prism(name, pts, thick, mat):
        n = len(pts)
        v = [(-thick / 2, y, z) for y, z in pts] + [(thick / 2, y, z) for y, z in pts]
        f = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        f += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        ob = S.mesh_object(name, v, f, mat, coll=coll)
        return ob

    bl = prism("Schluessel_Bart", blade, 0.0026, nickel)
    # bow: a rounded ring (outer polygon with a hole bridged by two prisms)
    ang = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    outer = [(-0.034 + 0.026 * math.cos(a), 0.026 * 0.86 * math.sin(a)) for a in ang]
    inner = [(-0.040 + 0.0075 * math.cos(a), 0.0075 * math.sin(a)) for a in ang]
    n = len(ang)
    th = 0.0034
    v = [(-th / 2, y, z) for y, z in outer] + [(th / 2, y, z) for y, z in outer]
    v += [(-th / 2, y, z) for y, z in inner] + [(th / 2, y, z) for y, z in inner]
    f = []
    for i in range(n):
        j = (i + 1) % n
        f.append((i, j, n + j, n + i))                      # outer wall
        f.append((2 * n + j, 2 * n + i, 3 * n + i, 3 * n + j))  # inner wall
        f.append((j, i, 2 * n + i, 2 * n + j))              # -x face
        f.append((n + i, n + j, 3 * n + j, 3 * n + i))      # +x face
    bow = S.mesh_object("Schluessel_Reide", v, f, nickel, coll=coll)
    neck = prism("Schluessel_Hals", [(-0.012, -0.0060), (0.0, -0.0068), (0.0, 0.0068), (-0.012, 0.0060)], 0.0034, nickel)
    for ob in (bl, bow, neck):
        S.recalc_normals(ob)
        bv = ob.modifiers.new("Fase", "BEVEL")
        bv.width = 0.0005
        bv.segments = 2
        bv.limit_method = "ANGLE"
        ob.data.shade_smooth()
        wn = ob.modifiers.new("Normalen", "WEIGHTED_NORMAL")
        wn.keep_sharp = True
    key = S.empty("Schluessel", (LOCK.x, LOCK.y, LOCK.z), coll)
    for ob in (bl, bow, neck):
        ob.parent = key
        key_parts.append(ob)
    return {"key": key, "key_parts": key_parts, "core": core, "ring": ring, "ring_mat": ring_mat,
            "lock": lock, "lock_parts": [rosette, core, ring]}


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------

def setup_physics(scene, blocks, colliders, pusher, f_end):
    with bpy.context.temp_override(scene=scene):
        bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    rbw.substeps_per_frame = 12
    rbw.solver_iterations = 40
    rbw.use_split_impulse = True
    rbw.point_cache.frame_start = 1
    rbw.point_cache.frame_end = f_end
    vl = bpy.context.view_layer

    def add(ob, kind):
        for o in vl.objects:
            o.select_set(False)
        vl.objects.active = ob
        ob.select_set(True)
        with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
            bpy.ops.rigidbody.object_add(type=kind)
        rb = ob.rigid_body
        rb.collision_shape = "BOX"
        rb.collision_margin = 0.001
        return rb

    for ob in colliders:
        rb = add(ob, "PASSIVE")
        rb.friction = 0.7
        rb.restitution = 0.2
    for ob in blocks:
        rb = add(ob, "ACTIVE")
        dims = ob.dimensions
        rb.mass = 0.9 * dims.x * dims.y * dims.z * 1000.0 * 0.12
        rb.friction = 0.62
        rb.restitution = 0.22
        rb.linear_damping = 0.04
        rb.angular_damping = 0.08
        rb.use_deactivation = True
        rb.use_start_deactivated = True
    rb = add(pusher, "ACTIVE")
    rb.kinematic = True
    rb.friction = 0.4


def bake_physics(scene):
    pc = scene.rigidbody_world.point_cache
    with bpy.context.temp_override(scene=scene, point_cache=pc):
        bpy.ops.ptcache.free_bake()
        bpy.ops.ptcache.bake(bake=True)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(args, own_fog: float = -1.0):
    tl = TL.load(args.timeline or None)
    sh = TL.shot(SHOT, tl)
    ev = S.events_for(SHOT, EVENTS, tl)
    N = sh.frames
    frames = list(range(1, N + 1))
    f_rev, f_land = ev["window_reveal"], ev["window_land"]
    f_ex, f_sn, f_fall = ev["layers_explode"], ev["layers_snap"], ev["tower_fall"]
    f_speed, f_key = ev["speed_line"], ev["key_turn"]
    f_scene_end = TL.scene_frames(SHOT, tl)[1]

    scene = R.clean_scene()
    out = R.setup(scene, SHOT, N, res=args.res, samples=args.samples, motion_blur=not args.no_mb,
                  glare=not args.no_glare, out_dir=args.out or None)
    coll = scene.collection
    rng = np.random.default_rng(2)

    S.world_gradient(scene, zenith="#04050a", horizon="#14152a", ground="#08080c", strength=1.0,
                     glow=("#2a1d4a", (1.0, -0.15, 0.08), 6.0, 0.6), clouds=0.45)
    fog = FOG if own_fog < 0 else own_fog
    if fog > 0:
        S.world_fog(scene, fog, "#7d7f9e", anisotropy=0.45)
    street = build_street(coll)
    page = build_page(coll)
    blocks = build_tower(coll, rng)
    kl = build_key_lock(coll)

    # ---------------- lights --------------------------------------------------
    web_light = S.area("Weblicht", (WEB_C.x, WEB_C.y - 0.05, WEB_C.z), (WEB_C.x, -5.0, WEB_C.z), WEB_W, 0.0,
                       (0.42, 0.36, 1.0), shape="RECTANGLE", size_y=WEB_H)
    web_light.data.spread = math.radians(110)
    key_l = S.area("Schaufenster_Key", (TOWER_X - 1.9, -2.6, 2.7), (TOWER_X, TOWER_Y, 1.5), 1.2, 0.0, M.kelvin_rgb(4300))
    rim = S.area("Turm_Rim", (TOWER_X + 0.9, 0.9, 2.6), (TOWER_X, TOWER_Y, 1.4), 0.8, 0.0, S.lin("#e44b8d"))
    inside = S.area("Auslage_Rosa", (OX1 - 0.5, WALL_T + 0.4, OZ1 - 0.1), (OX1 - 0.6, WALL_T + 1.3, OZ0 + 0.5), 1.4, 0.0,
                    S.lin("#e44b8d"))
    inside.data.use_shadow = False
    rim.data.use_shadow = False
    # product light for the metal: a thin strip gives one crisp highlight line
    lock_l = S.area("Schloss_Licht", (LOCK.x + 0.30, LOCK.y - 0.42, LOCK.z + 0.30), LOCK, 0.03, 0.0, M.kelvin_rgb(6000),
                    shape="RECTANGLE", size_y=0.7)
    lock_l.rotation_euler.rotate_axis("Z", math.radians(35.0))
    lock_fill = S.area("Schloss_Fill", (LOCK.x + 0.55, LOCK.y - 0.25, LOCK.z - 0.10), LOCK, 0.5, 0.0, S.lin("#7c3aed"))
    lock_fill.data.use_shadow = False
    sweep = S.spot("Scheinwerfer", (-14.0, -6.5, 0.75), (-12.0, 0.0, 1.2), 0.0, M.kelvin_rgb(5800), angle=34.0, blend=0.9, radius=0.3)

    def power(f):
        return C.span(f, f_rev - 8, f_rev + 34, "in_out_sine")

    S.bake_fn(web_light.data, "energy", frames, lambda f: 55.0 * power(f))
    S.bake_fn(key_l.data, "energy", frames, lambda f: 45.0 * power(f))
    S.bake_fn(rim.data, "energy", frames, lambda f: 40.0 * power(f))
    S.bake_fn(inside.data, "energy", frames, lambda f: 6.0 * power(f))
    S.bake_fn(lock_l.data, "energy", frames, lambda f: 2.2 * C.span(f, f_key - 120, f_key - 80, "in_out_sine"))
    S.bake_fn(lock_fill.data, "energy", frames, lambda f: 1.0 * C.span(f, f_key - 120, f_key - 80, "in_out_sine"))

    # headlight sweep synchronised with the speed line (Fusion)
    s0, s1 = f_speed - 16, f_speed + 22
    S.bake_fn(sweep, "location", frames, lambda f: (-14.0 + 30.0 * C.span(f, s0, s1, "in_out_sine"), -6.6, 0.75))
    S.bake_fn(sweep, "rotation_euler", frames,
              lambda f: tuple((Vector((-12.0 + 30.0 * C.span(f, s0, s1, "in_out_sine") + 2.5, 0.0, 1.4))
                               - Vector((-14.0 + 30.0 * C.span(f, s0, s1, "in_out_sine"), -6.6, 0.75))).to_track_quat("-Z", "Y").to_euler()))
    S.bake_fn(sweep.data, "energy", frames,
              lambda f: 5200.0 * math.sin(math.pi * C.span(f, s0, s1, "linear")) ** 0.7 if s0 <= f <= s1 else 0.0)

    # ---------------- page: power-on, drift, explode / snap --------------------
    for name, ob in page.items():
        if name.endswith("_Kontur"):
            continue
        mat = ob.active_material
        S.bake_fn(mat.node_tree, S.node_path("Staerke"), frames, lambda f: power(f))
    S.bake_fn(page["Bild"].active_material.node_tree, 'nodes["Drift"].inputs[1].default_value', frames,
              lambda f: (0.025 * math.sin(2 * math.pi * f / (60 * 22.0)), 0.02 * math.sin(2 * math.pi * f / (60 * 26.0) + 1.0), 0.0))

    def spread(f):
        if f < f_ex:
            return 0.0
        if f < f_ex + 22:
            return C.ease((f - f_ex) / 22.0, "out_expo")
        a, b = f_sn - 12, f_sn + 5
        if f < a:
            return 1.0 + 0.06 * C.span(f, f_ex + 22, a, "out_sine")
        # the layers never pass each other: no negative spread; the "snap"
        # overshoot is a common push of the whole page (see thud)
        return max(0.0, 1.06 * (1.0 - S.overshoot((f - a) / (b - a), 0.0)))

    def thud(f):  # whole page locks in: 1.5 cm back and settle (< 1 % of the travel)
        if f < f_sn or f > f_sn + 12:
            return 0.0
        u = (f - f_sn) / 12.0
        return 0.015 * math.sin(math.pi * u) * (1.0 - u)

    for name, y_off, (dy, dx, dz, yaw) in [(n, y, e) for n, _, y, e in LAYERS]:
        ob = page[name]
        base = Vector((WEB_C.x, WEB_C.y + y_off, WEB_C.z))
        off = Vector((dx, dy, dz))
        S.bake_fn(ob, "location", frames, lambda f, base=base, off=off: tuple(base + off * spread(f) + Vector((0.0, thud(f), 0.0))))
        S.bake_fn(ob, "rotation_euler", frames, lambda f, yaw=yaw: (math.radians(90), 0.0, math.radians(yaw * spread(f))))

    def show(f):  # extra layers are only there while exploded
        return C.span(f, f_ex + 2, f_ex + 16, "in_out_sine") * (1.0 - C.span(f, f_sn - 7, f_sn, "in_sine"))

    for name in ("Raster", "Code"):
        S.bake_fn(page[name].active_material.node_tree, S.node_path("Deckkraft"), frames, show)
    for name, *_ in LAYERS:
        S.bake_fn(page[name + "_Kontur"].active_material.node_tree, S.node_path("Staerke"), frames, lambda f: 1.6 * show(f))
    S.bake_fn(page["Typo"].active_material.node_tree, S.node_path("Mix"), frames,
              lambda f: C.span(f, f_ex + 4, f_ex + 18, "in_out_sine") * (1.0 - C.span(f, f_sn - 9, f_sn - 1, "in_sine")))

    # ---------------- tower physics --------------------------------------------
    bake_end = min(N, f_fall + 260)
    # a tall invisible board shoves the whole tower forward over the ledge edge;
    # it leans back a little so the upper blocks get the bigger push (tipping)
    pusher = S.box("Schieber", (0.30, 0.12, 1.10), (TOWER_X, 0.55, 0.0), None, coll=coll)
    pusher.hide_render = True
    z_mid = OZ0 + 0.20 + 0.55
    p0, p1 = f_fall - 6, f_fall + 14
    y_back = TOWER_Y + BLOCK_D / 2 + 0.07
    S.bake_fn(pusher, "location", frames[:bake_end],
              lambda f: (TOWER_X, y_back + 0.30 - 0.52 * C.span(f, p0, p1, "in_cubic") + 0.52 * C.span(f, p1 + 6, p1 + 40, "in_out_sine"), z_mid))
    S.bake_fn(pusher, "rotation_euler", frames[:bake_end], lambda f: (math.radians(-9.0), 0.0, 0.0))
    colliders = [street["ledge"], street["sidewalk"]]
    setup_physics(scene, blocks, colliders, pusher, bake_end)
    bake_physics(scene)

    # ---------------- key and lock ---------------------------------------------
    key, core = kl["key"], kl["core"]
    S.bake_fn(kl["lock"], "location", frames,
              lambda f: tuple(LOCK + Vector((0.0, 0.07 * (1.0 - C.span(f, f_key - 120, f_key - 84, "out_quint")), 0.0))))
    seat = LOCK + Vector((0.0, -0.0105, 0.0))  # shoulder of the key on the core face
    k_in0, k_in1 = f_key - 62, f_key - 26
    t0, t1 = f_key - 16, f_key + 6

    def key_loc(f):
        u = C.span(f, k_in0, k_in1, "out_quint")
        start = seat + Vector((0.08, -0.34, 0.035))
        return tuple(start.lerp(seat, u))

    def turn(f):
        return S.overshoot((f - t0) / (t1 - t0), 0.025)

    S.bake_fn(key, "location", frames, key_loc)
    S.bake_fn(key, "rotation_euler", frames,
              lambda f: (math.radians(-4.0) * (1 - C.span(f, k_in0, k_in1, "out_quint")),
                         math.radians(-90.0) * turn(f), math.radians(9.0) * (1 - C.span(f, k_in0, k_in1, "out_quint"))))
    S.bake_fn(core, "rotation_euler", frames, lambda f: (math.radians(90), math.radians(-90.0) * turn(f), 0.0))
    S.bake_fn(kl["ring_mat"].node_tree, S.node_path("Leuchten"), frames,
              lambda f: 3.5 * C.span(f, f_key - 2, f_key + 10, "out_cubic") * (1 - 0.6 * C.span(f, f_key + 10, f_key + 60, "in_out_sine")))

    # ---------------- camera ---------------------------------------------------
    cam, tgt, foc = S.camera("Kamera2", lens=32.0, fstop=2.8)
    pos = S.Bahn([
        (1, (-12.0, -7.3, 1.78)),
        (100, (-9.0, -7.1, 1.74)),
        (f_rev, (-5.4, -6.95, 1.72)),
        (f_rev + 62, (-1.5, -6.45, 1.71)),
        (f_land, (0.37, -6.65, 1.70), True),
        (f_ex - 12, (0.42, -6.25, 1.70), True),
        (f_ex + 24, (3.12, -7.26, 2.45)),
        (f_sn - 6, (3.22, -7.12, 2.47)),
        (f_fall, (1.40, -4.70, 1.60)),
        (f_fall + 70, (1.25, -5.00, 1.48), True),
        (f_speed, (-0.6, -6.60, 1.62)),
        (f_speed + 420, (-2.2, -7.40, 1.80)),
        (f_key - 230, (-1.6, -6.70, 1.74), True),
        (f_key - 110, (1.80, -2.10, 1.58)),
        (f_key - 40, (1.30, -0.14, 1.43)),
        (f_key, (1.238, 0.030, 1.401), True),
        (f_key + 8, (1.236, 0.027, 1.401), True),
        (f_scene_end, (-0.10, -1.45, 1.73)),
        (N, (-0.15, -0.95, 1.73)),
    ])
    aimp = S.Bahn([
        (1, (12.0, -3.0, 2.15)),
        (100, (9.0, -2.0, 2.0)),
        (f_rev, (2.6, 0.0, 1.80)),
        (f_rev + 62, (0.55, 0.3, 1.70)),
        (f_land, (0.32, 0.3, 1.64), True),
        (f_ex - 12, (0.34, 0.3, 1.64), True),
        (f_ex + 24, (-0.30, -0.55, 1.65)),
        (f_sn - 6, (-0.30, -0.55, 1.65)),
        (f_fall, (1.20, 0.0, 1.50)),
        (f_fall + 70, (1.05, 0.0, 1.08), True),
        (f_speed, (0.35, 0.3, 1.62)),
        (f_speed + 420, (0.55, 0.3, 1.66)),
        (f_key - 230, (0.60, 0.3, 1.62), True),
        (f_key - 110, (0.65, 0.3, 1.38)),
        (f_key - 40, (LOCK.x - 0.01, LOCK.y - 0.02, LOCK.z)),
        (f_key, (LOCK.x + 0.006, LOCK.y - 0.02, LOCK.z), True),
        (f_key + 8, (LOCK.x + 0.006, LOCK.y - 0.02, LOCK.z), True),
        (f_scene_end, (-0.12, 0.3, 1.73)),
        (N, (-0.15, 0.3, 1.73)),
    ])
    focus = S.Bahn([
        (1, (2.0, -4.0, 1.8)),
        (f_rev, (0.0, 0.3, 1.7)),
        (f_land, tuple(WEB_C)),
        (f_ex, tuple(WEB_C), True),
        (f_ex + 20, (-0.75, -1.30, 1.57)),
        (f_sn - 4, (-0.75, -1.30, 1.57), True),
        (f_sn + 6, (TOWER_X, TOWER_Y - 0.12, 1.5)),
        (f_fall + 60, (TOWER_X, -0.5, 1.0), True),
        (f_fall + 140, tuple(WEB_C)),
        (f_key - 150, tuple(WEB_C), True),
        (f_key - 70, (LOCK.x, LOCK.y - 0.03, LOCK.z)),
        (f_key + 8, (LOCK.x, LOCK.y - 0.03, LOCK.z), True),
        (f_scene_end, tuple(WEB_C)),
    ])
    lens = S.Bahn([(1, 32.0), (f_land, 40.0, True), (f_ex, 40.0, True), (f_ex + 24, 42.0), (f_fall, 40.0, True),
                   (f_key - 230, 40.0, True), (f_key - 40, 85.0), (f_key + 8, 85.0, True), (f_scene_end, 40.0), (N, 38.0)])
    fstop = S.Bahn([(1, 2.8), (f_land, 2.8, True), (f_ex, 2.8, True), (f_ex + 20, 1.8), (f_sn, 1.8, True),
                    (f_fall, 2.4, True), (f_fall + 140, 3.2, True), (f_key - 70, 2.8), (f_key + 8, 2.8, True), (N, 2.8)])
    S.bake_camera(cam, tgt, foc, frames, pos=pos, target=aimp, focus=focus, lens=lens, fstop=fstop)

    keys = {
        "window_reveal": [page["Grund"]],
        "window_land": [page["Grund"], page["Bild"], page["Typo"]],
        "layers_explode": [page[n] for n, *_ in LAYERS],
        "layers_snap": [page[n] for n, *_ in LAYERS],
        "tower_fall": blocks,
        "key_turn": kl["key_parts"] + kl["lock_parts"],
    }

    def physik():
        def pose():
            return {ob.name: [round(c, 6) for c in list(ob.matrix_world.translation) + list(ob.matrix_world.to_quaternion())] for ob in blocks}

        scene.frame_set(1)
        start = pose()
        scene.frame_set(bake_end)
        return {"bild_ende": bake_end, "ledge_z": OZ0, "start": start, "ende": pose()}

    print(f"[s2] timeline={sh.quelle} frames={N} events={ev} bake_end={bake_end}", flush=True)
    return scene, {"shot": sh, "events": ev, "out": out, "keys": keys, "physik": physik}


def main() -> None:
    args, own = S.parse_cli(default_samples=16)
    scene, info = build(args, own.nebel)
    S.run(scene, info, args, own)


if __name__ == "__main__":
    main()
