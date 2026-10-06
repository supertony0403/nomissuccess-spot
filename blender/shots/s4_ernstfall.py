"""Shot s4_ernstfall - ransomware goes for the backups, the immutable copy holds.

Beats (event frames from timeline.json at render time):

* shot start: three backup cubes leave the bays of a server cabinet and land
  in a 3-2-1 layout (two close together, one far away),
* ``red_tendrils``: dark red smoke tendrils (swept curves with growing radius,
  animated noise, emissive veins inside a translucent smoky hull) creep over
  the floor towards the cubes,
* ``immutable_lock``: a metal seal with a padlock relief is stamped onto the
  front cube,
* ``shockwave``: the tendrils strike; a light wave runs through the glass cube
  (emission band moving outwards from the impact) and a blue shell and floor
  ring expand; the tendrils recoil,
* ``rewind``: the whole scene animation - tendrils, seal, cubes, lights and
  camera - plays backwards at up to ~8x speed with a long shutter, so the
  cubes fly back into the server cabinet,
* until the scene end it settles: the bay LEDs turn mint one by one (Fusion
  writes "Hoffnung" / "getestet").

Everything that rewinds is a function of the story time ``s(f)`` and baked
per frame; the tendril geometry keeps a constant point count (the tip
collapses instead of being trimmed) so EEVEE can motion-blur it.

Usage::

    blender -b --factory-startup -P blender/shots/s4_ernstfall.py -- [--proxy 480] [--frames a-b|ev:name]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib_szenen"))

import szenen_s2_s4 as S  # noqa: E402

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import nomiss_camera as C  # noqa: E402
import nomiss_material as M  # noqa: E402
import nomiss_render as R  # noqa: E402
import nomiss_timeline as TL  # noqa: E402

SHOT = "s4_ernstfall"
EVENTS = ("red_tendrils", "immutable_lock", "shockwave", "rewind", "hope_fade")

CUBE = 0.5
SERVER = Vector((-1.45, 2.9, 0.0))
SERVER_YAW = math.radians(38.0)
CUBES = {  # name: (position, yaw deg)
    "A": (Vector((0.10, 1.15, CUBE / 2)), 31.0),   # immutable copy, front
    "B": (Vector((0.95, 1.70, CUBE / 2)), 22.0),
    "C": (Vector((-1.25, 6.60, CUBE / 2)), 12.0),  # off-site copy, far
}
RED = "#b3121f"
BLUE = "#145fe4"
MINT = "#10b981"
REWIND_F = 72  # frames the rewind takes


# ---------------------------------------------------------------------------
# story time
# ---------------------------------------------------------------------------

def story_fn(f_rw: int):
    """Shot frame -> story frame: forward, then a fast eased rewind to 1."""
    return lambda f: S.rewind_time(f, f_rw, REWIND_F)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def glass_mat(name):
    """Dark blue glass with a light-wave band (values Welle, Radius, Zentrum*)."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.name = "BSDF"
    b.inputs["Base Color"].default_value = S.lin("#0b1838")
    b.inputs["Roughness"].default_value = 0.08
    b.inputs["Specular IOR Level"].default_value = 0.8
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.02
    b.inputs["Alpha"].default_value = 0.42
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ctr = nt.nodes.new("ShaderNodeCombineXYZ")
    for i, k in enumerate("XYZ"):
        nt.links.new(S.value_node(nt, f"Zentrum{k}", 0.0), ctr.inputs[i])
    d = nt.nodes.new("ShaderNodeVectorMath")
    d.operation = "DISTANCE"
    nt.links.new(tc.outputs["Object"], d.inputs[0])
    nt.links.new(ctr.outputs[0], d.inputs[1])
    mm = S.math_node
    rad = S.value_node(nt, "Radius", -1.0)
    band = mm(nt, "SUBTRACT", 1.0, mm(nt, "DIVIDE", mm(nt, "ABSOLUTE", mm(nt, "SUBTRACT", d.outputs["Value"], rad)), 0.06), clamp=True)
    band = mm(nt, "POWER", band, 2.0)
    s = mm(nt, "MULTIPLY", band, S.value_node(nt, "Welle", 0.0))
    s = mm(nt, "ADD", s, S.value_node(nt, "Grund", 0.0))
    b.inputs["Emission Color"].default_value = (0.45, 0.70, 1.0, 1.0)
    nt.links.new(s, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    S.blended(mat, backface=False)
    return mat


def tendril_mat(name):
    """Dark red smoke: black body, emissive veins from animated noise."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.name = "BSDF"
    b.inputs["Base Color"].default_value = (0.02, 0.0, 0.003, 1.0)
    b.inputs["Roughness"].default_value = 0.55
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = 11.0
    nz.inputs["Detail"].default_value = 7.0
    nz.inputs["Roughness"].default_value = 0.66
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    zeit = S.value_node(nt, "Zeit", 0.0)
    nt.links.new(S.math_node(nt, "MULTIPLY", zeit, 0.012), nz.inputs["W"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    M.set_ramp(ramp, [(0.46, (0.0, 0.0, 0.0, 1.0)), (0.60, S.lin("#4a050c")), (0.72, S.lin(RED)), (0.82, S.lin("#ff4a3a"))])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    tip = nt.nodes.new("ShaderNodeAttribute")
    tip.attribute_name = "spitze"
    s = S.math_node(nt, "ADD", 2.2, S.math_node(nt, "MULTIPLY", tip.outputs["Fac"], 4.0))
    s = S.math_node(nt, "MULTIPLY", s, S.value_node(nt, "Leuchten", 1.0))
    nt.links.new(ramp.outputs["Color"], b.inputs["Emission Color"])
    nt.links.new(s, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


def smoke_mat(name):
    """Translucent smoky hull around a tendril (sorted transparency)."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = 3.2
    nz.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    zeit = S.value_node(nt, "Zeit", 0.0)
    nt.links.new(S.math_node(nt, "MULTIPLY", zeit, 0.009), nz.inputs["W"])
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    facing = S.math_node(nt, "SUBTRACT", 1.0, lw.outputs["Facing"], clamp=True)
    dens = S.math_node(nt, "MULTIPLY", S.math_node(nt, "POWER", facing, 1.4), S.math_node(nt, "MULTIPLY", nz.outputs["Fac"], 0.55))
    dens = S.math_node(nt, "MULTIPLY", dens, S.value_node(nt, "Dichte", 1.0))
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = S.lin("#8a0c18")
    nt.links.new(S.math_node(nt, "MULTIPLY", dens, 3.0), em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(S.math_node(nt, "MINIMUM", dens, 1.0), mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    S.blended(mat, backface=True)
    return mat


# ---------------------------------------------------------------------------
# tendril geometry nodes (constant topology)
# ---------------------------------------------------------------------------

def tendril_group():
    g = S.GN("S4Ranke")
    wachs = g.inp("Wachstum", default=0.0)
    dicke = g.inp("Dicke", default=0.04)
    zeit = g.inp("Zeit", default=0.0)
    huelle = g.inp("Huelle", default=1.0)
    m2c = g.node("GeometryNodeMeshToCurve")
    g.put(m2c.inputs["Mesh"], g.geo)
    res = g.node("GeometryNodeResampleCurve")
    g.put(res.inputs["Curve"], m2c.outputs["Curve"])
    res.inputs["Count"].default_value = 140
    cur = res.outputs["Curve"]
    u = g.node("GeometryNodeSplineParameter").outputs["Factor"]
    pos = g.node("GeometryNodeInputPosition").outputs[0]
    # organic wriggle (grows with distance from the root)
    nz = g.node("ShaderNodeTexNoise", noise_dimensions="4D")
    nz.inputs["Scale"].default_value = 1.3
    nz.inputs["Detail"].default_value = 2.0
    g.put(nz.inputs["Vector"], pos)
    g.put(nz.inputs["W"], g.math("MULTIPLY", zeit, 0.006))
    off = g.vmath("SCALE", g.vmath("SUBTRACT", nz.outputs["Color"], (0.5, 0.5, 0.5)), scale=g.math("MULTIPLY", g.smoothstep(u, 0.0, 0.25), 0.16))
    off = g.vmath("MULTIPLY", off, (1.0, 1.0, 0.45))
    # tip: points beyond the growth front collapse onto the front
    smp = g.node("GeometryNodeSampleCurve", mode="FACTOR")
    g.put(smp.inputs["Curves"], cur)
    g.put(smp.inputs["Factor"], wachs)
    grown = g.math("SUBTRACT", 1.0, g.math("GREATER_THAN", u, wachs))
    newpos = g.vlerp(smp.outputs["Position"], g.vmath("ADD", pos, off), grown)
    sp = g.node("GeometryNodeSetPosition")
    g.put(sp.inputs["Geometry"], cur)
    g.put(sp.inputs["Position"], newpos)
    # radius: root swell, noisy body, sharp tapering tip
    nr = g.node("ShaderNodeTexNoise", noise_dimensions="4D")
    nr.inputs["Scale"].default_value = 4.0
    g.put(nr.inputs["Vector"], pos)
    g.put(nr.inputs["W"], g.math("MULTIPLY", zeit, 0.01))
    length = g.node("GeometryNodeCurveLength")
    g.put(length.inputs["Curve"], cur)
    gap = g.math("MULTIPLY", g.math("SUBTRACT", wachs, u), length.outputs["Length"])  # metres behind the tip
    taper = g.smoothstep(gap, 0.0, 0.22)
    rad = g.math("MULTIPLY", dicke, g.math("ADD", 0.6, g.math("MULTIPLY", nr.outputs["Fac"], 0.8)))
    rad = g.math("MULTIPLY", rad, g.math("MULTIPLY", taper, grown))
    rad = g.math("MULTIPLY", rad, g.math("ADD", 0.35, g.math("MULTIPLY", g.math("SUBTRACT", 1.0, u), 0.9)))
    rad = g.math("MULTIPLY", rad, huelle)
    scr = g.node("GeometryNodeSetCurveRadius")
    g.put(scr.inputs["Curve"], sp.outputs["Geometry"])
    g.put(scr.inputs["Radius"], rad)
    spitze = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.smoothstep(gap, 0.0, 0.10)), grown)
    geo = g.store(scr.outputs["Curve"], "spitze", spitze)
    circ = g.node("GeometryNodeCurvePrimitiveCircle")
    circ.inputs["Resolution"].default_value = 10
    circ.inputs["Radius"].default_value = 1.0
    c2m = g.node("GeometryNodeCurveToMesh")
    g.put(c2m.inputs["Curve"], geo)
    g.put(c2m.inputs["Profile Curve"], circ.outputs["Curve"])
    # Blender 5: the sweep is scaled by this input, not by the curve radius
    g.put(c2m.inputs["Scale"], g.node("GeometryNodeInputRadius").outputs[0])
    sm = g.node("GeometryNodeSetShadeSmooth")
    g.put(sm.inputs[0], c2m.outputs["Mesh"])
    return g.out(sm.outputs[0])


def face_point(name: str, side: str, dx: float, z: float, off: float) -> Vector:
    """Point on a face of cube ``name``: ``dx`` along the face, height ``z``,
    ``off`` metres outside the face."""
    p, yaw = CUBES[name]
    a = math.radians(yaw)
    nrm = {"vorn": Vector((math.sin(a), -math.cos(a), 0.0)), "rechts": Vector((math.cos(a), math.sin(a), 0.0)),
           "links": Vector((-math.cos(a), -math.sin(a), 0.0)), "hinten": Vector((-math.sin(a), math.cos(a), 0.0))}[side]
    tan = Vector((-nrm.y, nrm.x, 0.0))
    q = Vector((p.x, p.y, 0.0)) + nrm * (CUBE / 2 + off) + tan * dx
    q.z = z
    return q


def tendril_paths(rng):
    """(target, points): creep over the floor from far outside, then climb a
    face of a cube that the camera sees; tips stop just short of the seal."""

    def creep(root: Vector, base: Vector, bend: float, n: int = 8):
        pts = []
        d = base - root
        side = Vector((-d.y, d.x, 0.0)).normalized()
        ph = rng.uniform(0, 3)
        for i in range(n):
            t = i / (n - 1)
            q = root.lerp(base, t)
            q += side * (math.sin(math.pi * t) * bend + 0.22 * math.sin(3.3 * math.pi * t + ph) * (1 - t))
            q.z = 0.075 + 0.025 * math.sin(6 * t + ph)
            pts.append(q)
        return pts

    def climb(name, side, dx0, dx1, z1):
        a = face_point(name, side, dx0, 0.10, 0.035)
        b = face_point(name, side, 0.5 * (dx0 + dx1) + rng.uniform(-0.03, 0.03), 0.5 * (0.10 + z1), 0.03)
        c = face_point(name, side, dx1, z1, 0.03)
        return [a, b, c]

    def both(target, floor, up):
        pts = floor + up
        seg = [(pts[i + 1] - pts[i]).length for i in range(len(pts) - 1)]
        base = sum(seg[: len(floor) - 1]) / sum(seg)
        return target, pts, base

    paths = [
        both("A", creep(Vector((-4.9, 0.1, 0.05)), face_point("A", "vorn", -0.20, 0.05, 0.10), 0.6), climb("A", "vorn", -0.19, -0.13, 0.33)),
        both("A", creep(Vector((-2.8, -2.9, 0.05)), face_point("A", "vorn", 0.16, 0.05, 0.10), -0.4), climb("A", "vorn", 0.17, 0.12, 0.19)),
        both("A", creep(Vector((3.4, 5.4, 0.05)), face_point("A", "rechts", 0.12, 0.05, 0.10), 0.7), climb("A", "rechts", 0.10, -0.02, 0.36)),
        both("B", creep(Vector((5.2, 1.6, 0.05)), face_point("B", "vorn", 0.18, 0.05, 0.10), -0.5), climb("B", "vorn", 0.17, 0.08, 0.30)),
        both("B", creep(Vector((2.8, 7.8, 0.05)), face_point("B", "rechts", 0.10, 0.05, 0.10), 0.8), climb("B", "rechts", 0.08, -0.05, 0.28)),
        both("C", creep(Vector((-5.4, 8.6, 0.05)), face_point("C", "vorn", -0.15, 0.05, 0.10), 0.6), climb("C", "vorn", -0.14, -0.05, 0.30)),
        both("C", creep(Vector((1.8, 10.8, 0.05)), face_point("C", "rechts", 0.0, 0.05, 0.10), -0.7), climb("C", "rechts", 0.0, 0.05, 0.25)),
        ("-", creep(Vector((-5.2, -1.8, 0.05)), Vector((-1.2, 0.5, 0.05)), 0.5), 0.85),
        ("-", creep(Vector((5.4, 4.0, 0.05)), Vector((2.0, 2.7, 0.05)), -0.4), 0.85),
    ]
    return paths


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------

def build_server(coll):
    alu = S.principled("S4Gehaeuse", "#16181e", rough=0.34, metal=0.8, aniso=0.3)
    bay_dark = S.principled("S4Schacht", "#030305", rough=0.8)
    root = S.empty("Server", tuple(SERVER), coll)
    root.rotation_euler = (0, 0, SERVER_YAW)
    body = S.box("Server_Gehaeuse", (1.0, 0.8, 1.55), (0, 0, 0.775), alu, bevel=0.02, segments=3, coll=coll)
    body.parent = root
    bays, leds, led_mats = [], [], []
    for i, z in enumerate((1.18, 0.78, 0.38)):
        bay = S.box(f"Server_Schacht_{i}", (0.66, 0.02, 0.32), (0.0, -0.405, z), bay_dark, coll=coll)
        bay.parent = root
        lm = S.principled(f"S4LED_{i}", "#000000", emission=S.lin(BLUE), estr=0.0)
        S.drive_input(lm, "Emission Strength", "Leuchten", 0.0)
        mix = lm.node_tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.name = "Farbe"
        mix.inputs[6].default_value = S.lin(BLUE)
        mix.inputs[7].default_value = S.lin(MINT)
        lm.node_tree.links.new(S.value_node(lm.node_tree, "Gruen", 0.0), mix.inputs["Factor"])
        lm.node_tree.links.new(mix.outputs[2], lm.node_tree.nodes["BSDF"].inputs["Emission Color"])
        led = S.box(f"Server_LED_{i}", (0.58, 0.012, 0.012), (0.0, -0.412, z - 0.19), lm, coll=coll)
        led.parent = root
        bays.append(bay)
        leds.append(led)
        led_mats.append(lm)
    strip_m = S.principled("S4Streifen", "#000000", emission=S.lin(BLUE), estr=0.0)
    S.drive_input(strip_m, "Emission Strength", "Leuchten", 0.0)
    strip = S.box("Server_Streifen", (0.012, 0.012, 1.2), (0.44, -0.412, 0.78), strip_m, coll=coll)
    strip.parent = root
    return root, [body] + bays + leds + [strip], led_mats, strip_m


def bay_world(i: int) -> Vector:
    """World position just inside bay ``i``."""
    z = (1.18, 0.78, 0.38)[i]
    local = Vector((0.0, -0.25, z))
    c, s = math.cos(SERVER_YAW), math.sin(SERVER_YAW)
    return SERVER + Vector((local.x * c - local.y * s, local.x * s + local.y * c, local.z))


def build_cube(name, coll):
    mat = glass_mat(f"S4Glas_{name}")
    cube = S.box(f"Wuerfel_{name}", (CUBE, CUBE, CUBE), (0, 0, 0), mat, bevel=0.03, segments=4, coll=coll)
    edge_m = S.principled(f"S4Kante_{name}", "#000000", emission=S.lin("#5b8cff"), estr=1.6)
    edges = S.box(f"Wuerfel_{name}_Kanten", (CUBE + 0.004, CUBE + 0.004, CUBE + 0.004), (0, 0, 0), edge_m, coll=coll)
    wf = edges.modifiers.new("Draht", "WIREFRAME")
    wf.thickness = 0.0045
    wf.use_even_offset = True
    edges.parent = cube
    plat_m = S.principled(f"S4Platte_{name}", "#0a0d16", rough=0.25, metal=0.6, emission=S.lin("#3d7bff"), estr=0.0)
    S.drive_input(plat_m, "Emission Strength", "Leuchten", 1.2)
    platters = []
    for k in range(4):
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.14, depth=0.014, location=(0, 0, -0.105 + k * 0.07))
        pl = bpy.context.object
        pl.name = f"Wuerfel_{name}_Platte_{k}"
        pl.data.materials.append(plat_m)
        for c in pl.users_collection:
            c.objects.unlink(pl)
        coll.objects.link(pl)
        pl.parent = cube
        platters.append(pl)
    return cube, mat, platters, plat_m


def build_seal(coll):
    steel = S.principled("S4Siegel", "#b8bcc4", rough=0.24, metal=1.0, aniso=0.6)
    icon_m = S.principled("S4Siegel_Symbol", "#d9dde4", rough=0.18, metal=1.0, emission=S.lin(MINT), estr=0.0)
    S.drive_input(icon_m, "Emission Strength", "Leuchten", 0.0)
    root = S.empty("Siegel", (0, 0, 0), coll)
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=0.105, depth=0.014, location=(0, 0, 0), rotation=(math.radians(90), 0, 0))
    disc = bpy.context.object
    disc.name = "Siegel_Scheibe"
    disc.data.materials.append(steel)
    bv = disc.modifiers.new("Fase", "BEVEL")
    bv.width = 0.004
    bv.segments = 4
    disc.data.shade_smooth()
    for c in disc.users_collection:
        c.objects.unlink(disc)
    coll.objects.link(disc)
    disc.parent = root
    # padlock relief: body + shackle arch (in the disc plane XZ, raised along -Y)
    bm = bmesh.new()

    def prism(poly, y0, y1):
        vs0 = [bm.verts.new((x, y0, z)) for x, z in poly]
        vs1 = [bm.verts.new((x, y1, z)) for x, z in poly]
        bm.faces.new(vs0[::-1])
        bm.faces.new(vs1)
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((vs0[i], vs0[j], vs1[j], vs1[i]))

    prism([(-0.034, -0.040), (0.034, -0.040), (0.034, 0.010), (-0.034, 0.010)], -0.007, -0.011)
    ang = np.linspace(0.0, math.pi, 24)
    outer = [(0.024 * math.cos(a), 0.010 + 0.026 * math.sin(a)) for a in ang]
    inner = [(0.015 * math.cos(a), 0.010 + 0.017 * math.sin(a)) for a in ang[::-1]]
    prism(outer + inner, -0.007, -0.010)
    prism([(-0.006, -0.026), (0.006, -0.026), (0.006, -0.012), (-0.006, -0.012)], -0.011, -0.0125)
    me = bpy.data.meshes.new("Siegel_Symbol")
    bm.to_mesh(me)
    bm.free()
    icon = S.link(bpy.data.objects.new("Siegel_Symbol", me), coll)
    icon.data.materials.append(icon_m)
    S.recalc_normals(icon)
    icon.parent = root
    return root, [disc, icon], icon_m


def build_shock(coll):
    shell_m, snt = S.new_mat("S4Schale")
    so = snt.nodes.new("ShaderNodeOutputMaterial")
    lw = snt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    rim_w = S.math_node(snt, "POWER", lw.outputs["Facing"], 3.0)  # bright only at the silhouette
    sem = snt.nodes.new("ShaderNodeEmission")
    sem.inputs["Color"].default_value = S.lin("#7fb2ff")
    snt.links.new(S.math_node(snt, "MULTIPLY", rim_w, S.value_node(snt, "Staerke", 0.0)), sem.inputs["Strength"])
    str_ = snt.nodes.new("ShaderNodeBsdfTransparent")
    sadd = snt.nodes.new("ShaderNodeAddShader")
    snt.links.new(str_.outputs[0], sadd.inputs[0])
    snt.links.new(sem.outputs[0], sadd.inputs[1])
    snt.links.new(sadd.outputs[0], so.inputs["Surface"])
    S.blended(shell_m, backface=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    shell = bpy.context.object
    shell.name = "Schockwelle"
    shell.data.materials.append(shell_m)
    shell.data.shade_smooth()
    for c in shell.users_collection:
        c.objects.unlink(shell)
    coll.objects.link(shell)
    ring_m = S.new_mat("S4Ring")[0]
    nt = ring_m.node_tree
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    uv = nt.nodes.new("ShaderNodeUVMap")
    d = nt.nodes.new("ShaderNodeVectorMath")
    d.operation = "DISTANCE"
    d.inputs[1].default_value = (0.5, 0.5, 0.0)
    nt.links.new(uv.outputs["UV"], d.inputs[0])
    rr = S.math_node(nt, "SUBTRACT", 1.0, S.math_node(nt, "DIVIDE", S.math_node(nt, "ABSOLUTE", S.math_node(nt, "SUBTRACT", d.outputs["Value"], 0.46)), 0.035), clamp=True)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = S.lin("#7fb2ff")
    nt.links.new(S.math_node(nt, "MULTIPLY", S.math_node(nt, "POWER", rr, 2.0), S.value_node(nt, "Staerke", 0.0)), em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    S.blended(ring_m, backface=True)
    ring = S.plane("Bodenring", 2.0, 2.0, (0, 0, 0.004), ring_m, rot=(0, 0, 0), coll=coll)
    return shell, shell_m, ring, ring_m


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(args):
    tl = TL.load(args.timeline or None)
    sh = TL.shot(SHOT, tl)
    ev = S.events_for(SHOT, EVENTS, tl)
    N = sh.frames
    frames = list(range(1, N + 1))
    f_ten, f_lock, f_shock, f_rw = ev["red_tendrils"], ev["immutable_lock"], ev["shockwave"], ev["rewind"]
    st = story_fn(f_rw)
    f_back = f_rw + REWIND_F
    rng = np.random.default_rng(4)

    scene = R.clean_scene()
    out = R.setup(scene, SHOT, N, res=args.res, samples=args.samples, motion_blur=not args.no_mb,
                  glare=not args.no_glare, out_dir=args.out or None)
    coll = scene.collection
    ee = scene.eevee
    ee.motion_blur_max = 220
    ee.motion_blur_steps = 2
    # longer shutter while the scene rewinds
    S.bake_fn(scene, "render.motion_blur_shutter", frames,
              lambda f: 0.5 + 0.5 * C.span(f, f_rw - 4, f_rw + 10, "in_out_sine") * (1.0 - C.span(f, f_back - 14, f_back, "in_out_sine")))

    # ---------------- stage: blue night, red during the attack ----------------
    win = (-0.3, -0.3, 1.3, 1.3)
    blue = M.hero_image("S4_Blau", 1024, gains={"#145fe4": 0.75, "#7c3aed": 0.30, "#e44b8d": 0.10, "#10b981": 0.12},
                        blue_cy=1.24, window=win)
    red = M.hero_image("S4_Rot", 1024, gains={"#145fe4": 0.15, "#7c3aed": 0.25, "#e44b8d": 0.55, "#10b981": 0.0},
                       blue_cy=1.24, extra=[("#8a0b18", 0.55, 1.15, 0.85, 0.70, [(0.0, 0.70), (0.78, 0.0)])], window=win)
    world = M.world_backplate(scene, [blue, red], lens_mm=35.0, sensor_mm=36.0, overscan=1.6)

    def attack(s):  # 0 calm -> 1 under attack -> 0 after the shockwave
        return C.span(s, f_ten, f_shock - 10, "in_out_sine") * (1.0 - C.span(s, f_shock, f_shock + 60, "in_out_sine"))

    S.bake_fn(world.node_tree, S.node_path("Mischung"), frames, lambda f: 0.85 * attack(st(f)))
    S.bake_fn(world.node_tree, S.node_path("Helligkeit"), frames, lambda f: 0.42 + 0.35 * attack(st(f)))
    S.plane("Boden", 60.0, 60.0, (0.0, 5.0, 0.0), S.principled("S4Boden", "#04060c", rough=0.32, spec=0.5), rot=(0, 0, 0), coll=coll)

    # ---------------- server, cubes, seal --------------------------------------
    server, server_parts, led_mats, strip_m = build_server(coll)
    cubes = {}
    for i, name in enumerate(("A", "B", "C")):
        cube, mat, platters, plat_m = build_cube(name, coll)
        pos, yaw = CUBES[name]
        t0 = 26 + i * 20
        t1 = t0 + (96 if name != "C" else 120)
        start = bay_world(i)

        def place(s, start=start, pos=pos, t0=t0, t1=t1):
            u = C.span(s, t0, t1, "in_out_cubic")
            ctrl = start.lerp(pos, 0.5) + Vector((0.0, 0.0, 0.9))
            p = (1 - u) ** 2 * start + 2 * (1 - u) * u * ctrl + u * u * pos
            settle = 0.012 * math.sin(math.pi * C.span(s, t1, t1 + 14, "linear")) * (1 - C.span(s, t1, t1 + 14, "linear"))
            return tuple(p + Vector((0.0, 0.0, -settle)))

        S.bake_fn(cube, "location", frames, lambda f, place=place: place(st(f)))
        S.bake_fn(cube, "scale", frames, lambda f, t0=t0, t1=t1: (0.5 + 0.5 * C.span(st(f), t0, t0 + (t1 - t0) * 0.6, "out_cubic"),) * 3)
        S.bake_fn(cube, "rotation_euler", frames,
                  lambda f, yaw=yaw, t0=t0, t1=t1, i=i: (0.0, 0.0, math.radians(SERVER_YAW * 57.2958 + (yaw - SERVER_YAW * 57.2958) * C.span(st(f), t0, t1, "in_out_cubic"))))
        for k, pl in enumerate(platters):
            S.bake_fn(pl, "rotation_euler", frames, lambda f, k=k: (0.0, 0.0, 0.004 * st(f) * (1 + 0.3 * k)))
        cubes[name] = (cube, mat, platters, plat_m)

    seal, seal_parts, icon_m = build_seal(coll)
    cubeA = cubes["A"][0]
    a_pos, a_yaw = CUBES["A"]
    ya = math.radians(a_yaw)
    front = Vector((math.sin(ya), -math.cos(ya), 0.0))  # outward normal of A's front face
    seat = a_pos + front * (CUBE / 2 + 0.007) + Vector((0.0, 0.0, 0.02))

    def seal_loc(s):
        u = C.span(s, f_lock - 26, f_lock - 3, "out_quint")
        press = 0.006 * math.sin(math.pi * C.span(s, f_lock - 3, f_lock + 6, "linear"))
        p = seat + front * (0.55 * (1 - u) - press) + Vector((0.0, 0.0, 0.18 * (1 - u)))
        return tuple(p)

    S.bake_fn(seal, "location", frames, lambda f: seal_loc(st(f)))
    S.bake_fn(seal, "rotation_euler", frames, lambda f: (0.0, math.radians(-160.0) * (1 - C.span(st(f), f_lock - 26, f_lock - 3, "out_quint")), ya))
    S.bake_fn(seal, "scale", frames, lambda f: (C.span(st(f), f_lock - 30, f_lock - 24, "linear"),) * 3)
    S.bake_fn(icon_m.node_tree, S.node_path("Leuchten"), frames,
              lambda f: 2.4 * C.span(st(f), f_lock - 1, f_lock + 8, "out_cubic") * (1.0 - 0.5 * C.span(st(f), f_lock + 30, f_lock + 90, "in_out_sine")))

    # ---------------- tendrils ---------------------------------------------------
    grp = tendril_group()
    core_mat = tendril_mat("S4Ranke")
    hull_mat = smoke_mat("S4Rauch")
    tendrils = []
    for k, (target, pts, w_base) in enumerate(tendril_paths(rng)):
        verts = [tuple(p) for p in pts]
        edges = [(i, i + 1) for i in range(len(verts) - 1)]
        start = f_ten + k * 9 + int(rng.integers(0, 12))
        dick = rng.uniform(0.060, 0.085) if target != "-" else 0.040
        for hull, mat in ((1.0, core_mat), (2.0, hull_mat)):
            ob = S.mesh_object(f"Ranke_{k}_{'Kern' if hull == 1.0 else 'Rauch'}", verts, [], mat, edges=edges, coll=coll)
            S.gn_modifier(ob, grp, "Ranke")
            S.object_material(ob, mat)
            S.gn_set(ob, "Ranke", "Huelle", hull)

            def grow(s, start=start, w_base=w_base, decor=(target == "-")):
                # by arc length: over the floor until just after the seal lands,
                # then climbing the face, the strike lands on the shockwave
                if decor:
                    w = w_base * C.span(s, start, f_shock - 20, "out_sine")
                else:
                    w = w_base * C.span(s, start, f_lock + 12, "out_sine")
                    w += 0.85 * (1.0 - w_base) * C.span(s, f_lock + 12, f_shock - 8, "in_out_sine")
                    w += 0.15 * (1.0 - w_base) * C.span(s, f_shock - 8, f_shock, "in_cubic")
                return w * (1.0 - 0.74 * C.span(s, f_shock, f_shock + 58, "out_cubic"))

            def thick(s, dick=dick):
                return dick * (0.35 + 0.65 * C.span(s, f_ten, f_shock - 20, "in_out_sine")) * (1.0 - 0.45 * C.span(s, f_shock, f_shock + 50, "out_cubic"))

            S.bake(ob, S.gn_path(ob, "Ranke", "Wachstum"), frames, [grow(st(f)) for f in frames])
            S.bake(ob, S.gn_path(ob, "Ranke", "Dicke"), frames, [thick(st(f)) for f in frames])
            S.bake(ob, S.gn_path(ob, "Ranke", "Zeit"), frames, [st(f) + 37.0 * k for f in frames])
            tendrils.append(ob)
    for mat in (core_mat, hull_mat):
        S.bake_fn(mat.node_tree, S.node_path("Zeit"), frames, lambda f: st(f))
    S.bake_fn(core_mat.node_tree, S.node_path("Leuchten"), frames,
              lambda f: (1.0 + 1.2 * attack(st(f))) * (1.0 - 0.6 * C.span(st(f), f_shock, f_shock + 40, "out_cubic")))
    S.bake_fn(hull_mat.node_tree, S.node_path("Dichte"), frames, lambda f: 0.45 + 0.9 * attack(st(f)))
    red_lights = []
    for k, p in enumerate([(-0.55, 0.45, 0.65), (0.95, 0.85, 0.65), (-0.6, 5.6, 0.6)]):
        L = S.point(f"Rotlicht_{k}", p, 0.0, S.lin("#d01020"), radius=0.05)
        L.data.use_shadow = False
        L.data.specular_factor = 0.0
        S.bake_fn(L.data, "energy", frames, lambda f, k=k: 90.0 * attack(st(f)) * (0.85 + 0.15 * math.sin(st(f) * 0.11 + k)))
        red_lights.append(L)

    # ---------------- shockwave ----------------------------------------------------
    shell, shell_m, ring, ring_m = build_shock(coll)
    impact_local = Vector((-CUBE / 2, 0.0, 0.02))  # tendril A strikes the left face (object space)
    for nm in ("A",):
        gm = cubes[nm][1]
        for k, v in zip("XYZ", impact_local):
            gm.node_tree.nodes[f"Zentrum{k}"].outputs[0].default_value = v
        S.bake_fn(gm.node_tree, S.node_path("Radius"), frames, lambda f: -0.2 + 1.25 * C.span(st(f), f_shock, f_shock + 26, "out_cubic"))
        S.bake_fn(gm.node_tree, S.node_path("Welle"), frames, lambda f: 9.0 * C.span(st(f), f_shock - 1, f_shock + 2, "linear") * (1.0 - C.span(st(f), f_shock + 8, f_shock + 30, "in_sine")))
    S.bake_fn(cubes["A"][3].node_tree, S.node_path("Leuchten"), frames,
              lambda f: 1.2 + 6.0 * math.exp(-((st(f) - f_shock - 3) / 7.0) ** 2) + 0.8 * C.span(st(f), f_lock, f_lock + 20, "in_out_sine"))
    S.bake_fn(shell, "location", frames, lambda f: tuple(a_pos + Vector((-0.1, 0.0, 0.0))))
    S.bake_fn(shell, "scale", frames, lambda f: (0.2 + 0.75 * C.span(st(f), f_shock, f_shock + 22, "out_cubic"),) * 3)
    S.bake_fn(shell_m.node_tree, S.node_path("Staerke"), frames,
              lambda f: 3.5 * C.span(st(f), f_shock - 1, f_shock + 2, "linear") * (1.0 - C.span(st(f), f_shock + 3, f_shock + 22, "out_sine")))
    S.bake_fn(ring, "location", frames, lambda f: (a_pos.x, a_pos.y, 0.004))
    S.bake_fn(ring, "scale", frames, lambda f: (0.3 + 4.2 * C.span(st(f), f_shock, f_shock + 40, "out_cubic"),) * 3)
    S.bake_fn(ring_m.node_tree, S.node_path("Staerke"), frames,
              lambda f: 3.0 * C.span(st(f), f_shock - 1, f_shock + 2, "linear") * (1.0 - C.span(st(f), f_shock + 6, f_shock + 40, "out_sine")))

    # ---------------- lights ---------------------------------------------------------
    key = S.area("Key", (2.4, -1.6, 3.2), (0.3, 1.6, 0.3), 1.2, 0.0, M.kelvin_rgb(7000))
    key.data.specular_factor = 0.25
    rim = S.area("Rim", (-1.0, 4.5, 2.4), (0.3, 1.4, 0.3), 1.5, 0.0, S.lin("#3d7bff"))
    rim.data.use_shadow = False
    rim.data.specular_factor = 0.08
    top = S.area("Oberlicht", (0.4, 1.4, 3.4), (0.4, 1.4, 0.0), 1.6, 0.0, M.kelvin_rgb(8000), shape="RECTANGLE", size_y=0.2)
    top.data.specular_factor = 0.2
    S.bake_fn(top.data, "energy", frames, lambda f: 40.0 * (1.0 - 0.6 * attack(st(f))))
    S.bake_fn(key.data, "energy", frames, lambda f: 160.0 * (1.0 - 0.55 * attack(st(f))))
    S.bake_fn(rim.data, "energy", frames, lambda f: 220.0 * (1.0 - 0.6 * attack(st(f))))
    # bay LEDs: blue while the backups exist, mint one by one after the restore
    for i, lm in enumerate(led_mats):
        S.bake_fn(lm.node_tree, S.node_path("Leuchten"), frames, lambda f: 5.0)
        S.bake_fn(lm.node_tree, S.node_path("Gruen"), frames, lambda f, i=i: C.span(f, f_back + 30 + 16 * i, f_back + 40 + 16 * i, "in_out_sine"))
    S.bake_fn(strip_m.node_tree, S.node_path("Leuchten"), frames, lambda f: 3.0)
    srv_light = S.area("Server_Licht", tuple(SERVER + Vector((1.2, -1.3, 1.6))), tuple(SERVER + Vector((0, 0, 0.8))), 1.0, 0.0, M.kelvin_rgb(6500))
    srv_light.data.specular_factor = 0.3
    S.bake_fn(srv_light.data, "energy", frames, lambda f: 25.0 + 35.0 * C.span(f, f_back, f_back + 60, "in_out_sine"))

    # ---------------- camera (story time, then a calm post move) ----------------------
    cam, tgt, foc = S.camera("Kamera4", lens=35.0, fstop=2.8)
    main_pos = S.Bahn([
        (1, (2.75, -2.55, 1.45)),
        (150, (2.30, -2.05, 1.18)),
        (f_ten, (2.05, -1.80, 1.02)),
        (f_lock - 30, (1.30, -1.05, 0.80)),
        (f_lock, (1.05, -0.88, 0.72), True),
        (f_shock, (1.15, -1.10, 0.80)),
        (f_rw, (1.45, -1.45, 0.92)),
    ])
    main_tgt = S.Bahn([
        (1, (-0.45, 2.35, 0.55)),
        (150, (0.15, 2.0, 0.35)),
        (f_ten, (0.45, 1.95, 0.30)),
        (f_lock - 30, (0.20, 1.35, 0.30)),
        (f_lock, (0.12, 1.25, 0.30), True),
        (f_shock, (0.20, 1.40, 0.30)),
        (f_rw, (0.44, 1.70, 0.32)),
    ])
    main_foc = S.Bahn([(1, tuple(SERVER + Vector((0, 0, 0.8)))), (120, tuple(a_pos)), (f_lock, tuple(seat)), (f_shock, tuple(a_pos)), (f_rw, tuple(a_pos))])
    post_pos = S.Bahn([(f_back, main_pos(1), True), (N, (0.35, -0.35, 1.15))])
    post_tgt = S.Bahn([(f_back, main_tgt(1), True), (N, tuple(SERVER + Vector((0.2, -0.3, 0.75))))])
    post_foc = S.Bahn([(f_back, main_foc(1), True), (N, tuple(SERVER + Vector((0.0, -0.4, 0.8))))])

    def pick(main, post):
        return lambda f: post(f) if f > f_back else main(st(f))

    lens = S.Bahn([(1, 35.0), (f_lock, 42.0, True), (f_rw, 38.0)])
    S.bake_camera(cam, tgt, foc, frames, pos=pick(main_pos, post_pos), target=pick(main_tgt, post_tgt),
                  focus=pick(main_foc, post_foc), lens=lambda f: lens(st(f)) if f <= f_back else 35.0 + 3.0 * C.span(f, f_back, N, "in_out_sine"),
                  fstop=lambda f: 2.4 if f <= f_back else 2.8)

    keys = {
        "red_tendrils": [cubes[n][0] for n in ("A", "B", "C")],
        "immutable_lock": [cubeA] + seal_parts,
        "shockwave": [cubeA] + seal_parts,
        "rewind": [cubes[n][0] for n in ("A", "B", "C")],
    }
    print(f"[s4] timeline={sh.quelle} frames={N} events={ev} rewind_end={f_back} tendrils={len(tendrils)}", flush=True)
    return scene, {"shot": sh, "events": ev, "out": out, "keys": keys}


def main() -> None:
    args, own = S.parse_cli(default_samples=16)
    scene, info = build(args)
    S.run(scene, info, args, own)


if __name__ == "__main__":
    main()
