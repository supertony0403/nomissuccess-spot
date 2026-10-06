"""Shot s3_netz - behind the website the network begins.

Beats (event frames from timeline.json at render time):

* shot start: the real page fills the frame (continues the push of s2),
* ``dive_into_pixels``: the page turns into a mosaic of 7840 pixel instances
  that peel away radially from the centre; the camera flies through the hole
  and the pixels become the nodes of a network lattice (the rest glows out as
  dust); fine connection lines with data pulses fade in,
* ``attacker``: red packets appear at the outer edge and creep closer,
* ``firewall_wall``: a wall of vertical blue light slats (#145fe4) rises,
* ``packets_bounce``: red packets hit the wall and bounce with sparks, green
  ones pass through to a node,
* ``radar_sweep``: a scanning beam (light cone + spot) sweeps the field,
* ``net_vanish``: the company network inside the wall dissolves into the dark,
  the beam finds nothing,
* ``mtls_lock``: a plain aluminium laptop draws a fine band to exactly one
  service cube, a padlock snaps shut on it.

All motion that matters is computed in Python from the event frames
(packets, sparks, camera), or analytically in Geometry Nodes from scene time
(pixels, nodes, sparks), so a re-timed voice only needs a re-render.

Usage::

    blender -b --factory-startup -P blender/shots/s3_netz.py -- [--proxy 480] [--frames a-b|ev:name]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib_szenen"))

import szenen_s2_s4 as S  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import nomiss_camera as C  # noqa: E402
import nomiss_material as M  # noqa: E402
import nomiss_render as R  # noqa: E402
import nomiss_ribbon as RB  # noqa: E402
import nomiss_timeline as TL  # noqa: E402

SHOT = "s3_netz"
EVENTS = ("dive_into_pixels", "attacker", "firewall_wall", "packets_bounce", "radar_sweep", "net_vanish",
          "zero_ports", "mtls_lock")

WEB_W, WEB_H = 2.4, 1.5
GRID = (160, 100)
PIX = WEB_W / GRID[0]
WALL_P0 = Vector((-2.6, 9.4, 0.0))
WALL_P1 = Vector((3.0, 11.2, 0.0))
WALL_H = (-1.35, 1.25)
SLAT_W, SLAT_STEP = 0.19, 0.25
FLOOR_Z = -1.35
RED = "#ff3048"
GREEN = "#10b981"
BLUE = "#145fe4"
NODE_COL = (0.16, 0.38, 1.0, 1.0)
LAPTOP = Vector((-0.42, 5.05, -0.72))
SERVICE = Vector((0.48, 6.25, -0.25))
RADAR = Vector((-9.0, 15.5, 2.6))


def wall_dir() -> Vector:
    return (WALL_P1 - WALL_P0).normalized()


def wall_normal() -> Vector:
    """Unit normal of the wall pointing outwards (away from the company network)."""
    d = wall_dir()
    n = Vector((d.y, -d.x, 0.0))
    return n if n.y > 0 else -n


def side_of_wall(p) -> float:
    """> 0 outside (internet), < 0 inside (company network)."""
    return (Vector((p[0], p[1], 0.0)) - WALL_P0).dot(wall_normal())


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def attr_emission_mat(name, *, dithered=True):
    """Emission from point attributes ``farbe`` x ``hell``; values Staerke/Deckkraft."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    col = nt.nodes.new("ShaderNodeAttribute")
    col.attribute_name = "farbe"
    hell = nt.nodes.new("ShaderNodeAttribute")
    hell.attribute_name = "hell"
    nt.links.new(col.outputs["Color"], em.inputs["Color"])
    nt.links.new(S.math_node(nt, "MULTIPLY", hell.outputs["Fac"], S.value_node(nt, "Staerke", 1.0)), em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(S.value_node(nt, "Deckkraft", 1.0), mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    mat.surface_render_method = "DITHERED" if dithered else "BLENDED"
    return mat


def tile_mat(name):
    """Page tiles: each tile shows its own piece of the real page (uvc + lokal),
    fading to the node colour by the flight progress ``e``."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    a_uvc = nt.nodes.new("ShaderNodeAttribute")
    a_uvc.attribute_name = "uvc"
    a_lok = nt.nodes.new("ShaderNodeAttribute")
    a_lok.attribute_name = "lokal"
    a_e = nt.nodes.new("ShaderNodeAttribute")
    a_e.attribute_name = "e"
    a_h = nt.nodes.new("ShaderNodeAttribute")
    a_h.attribute_name = "hell"
    sl = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(a_lok.outputs["Vector"], sl.inputs[0])
    su = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(a_uvc.outputs["Vector"], su.inputs[0])
    mm = S.math_node
    u = mm(nt, "ADD", su.outputs["X"], mm(nt, "MULTIPLY", sl.outputs["X"], 1.0 / GRID[0]))
    v = mm(nt, "ADD", su.outputs["Y"], mm(nt, "MULTIPLY", sl.outputs["Z"], 1.0 / GRID[1]))
    uv = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(u, uv.inputs[0])
    nt.links.new(v, uv.inputs[1])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = S.load_image(S.ASSETS / "web_komposit.png")
    tex.extension = "EXTEND"
    tex.interpolation = "Linear"
    nt.links.new(uv.outputs[0], tex.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(mm(nt, "POWER", a_e.outputs["Fac"], 1.6), mix.inputs["Factor"])
    nt.links.new(tex.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = NODE_COL
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(mix.outputs[2], em.inputs["Color"])
    nt.links.new(mm(nt, "MULTIPLY", a_h.outputs["Fac"], S.value_node(nt, "Staerke", 1.0)), em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    ms = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(S.value_node(nt, "Deckkraft", 1.0), ms.inputs[0])
    nt.links.new(tr.outputs[0], ms.inputs[1])
    nt.links.new(em.outputs[0], ms.inputs[2])
    nt.links.new(ms.outputs[0], out.inputs["Surface"])
    mat.surface_render_method = "DITHERED"
    return mat


def objcolor_glow_mat(name, color, strength):
    """Additive light whose intensity is the object's colour alpha (per-object keys)."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = S.lin(color) if isinstance(color, str) else color
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    nt.links.new(S.math_node(nt, "MULTIPLY", oi.outputs["Alpha"], strength), em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    S.blended(mat, backface=True)
    return mat


def radial_glow_mat(name, color, strength):
    """Additive soft disc (UV centre bright -> rim dark), intensity = object alpha."""
    mat, nt = S.new_mat(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = S.lin(color)
    uv = nt.nodes.new("ShaderNodeUVMap")
    vm = nt.nodes.new("ShaderNodeVectorMath")
    vm.operation = "DISTANCE"
    vm.inputs[1].default_value = (0.5, 0.5, 0.0)
    nt.links.new(uv.outputs["UV"], vm.inputs[0])
    ring = S.math_node(nt, "SUBTRACT", 1.0, S.math_node(nt, "MULTIPLY", vm.outputs["Value"], 2.0), clamp=True)
    ring = S.math_node(nt, "POWER", ring, 2.2)
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    s = S.math_node(nt, "MULTIPLY", S.math_node(nt, "MULTIPLY", ring, oi.outputs["Alpha"]), strength)
    nt.links.new(s, em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    S.blended(mat, backface=True)
    return mat


def slat_mat():
    """Firewall slat: bright base and top lines, soft vertical light in between."""
    mat, nt = S.new_mat("S3Lamelle")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    uv = nt.nodes.new("ShaderNodeUVMap")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    u, v = sep.outputs["X"], sep.outputs["Y"]
    mm = S.math_node
    body = mm(nt, "ADD", 0.05, mm(nt, "MULTIPLY", mm(nt, "POWER", mm(nt, "SUBTRACT", 1.0, v), 2.2), 0.55))
    side = mm(nt, "SUBTRACT", 1.0, mm(nt, "POWER", mm(nt, "ABSOLUTE", mm(nt, "MULTIPLY", mm(nt, "SUBTRACT", u, 0.5), 2.0)), 6.0))
    body = mm(nt, "MULTIPLY", body, side)
    lo = mm(nt, "LESS_THAN", v, 0.012)
    hi = mm(nt, "GREATER_THAN", v, 0.992)
    edges = mm(nt, "MULTIPLY", mm(nt, "ADD", lo, hi), 2.6)
    em = nt.nodes.new("ShaderNodeEmission")
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(mm(nt, "MINIMUM", edges, 1.0), mix.inputs["Factor"])
    mix.inputs[6].default_value = S.lin(BLUE)
    mix.inputs[7].default_value = S.lin("#9ec0ff")
    nt.links.new(mix.outputs[2], em.inputs["Color"])
    nt.links.new(mm(nt, "MULTIPLY", mm(nt, "ADD", body, edges), S.value_node(nt, "Staerke", 0.0)), em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    S.blended(mat, backface=True)
    return mat


def lines_mat():
    """Connection lines: faint blue with data pulses travelling along ``u``."""
    mat, nt = S.new_mat("S3Linien")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    u = nt.nodes.new("ShaderNodeAttribute")
    u.attribute_name = "u"
    ph = nt.nodes.new("ShaderNodeAttribute")
    ph.attribute_name = "phase"
    mm = S.math_node
    t = S.value_node(nt, "Zeit", 0.0)
    wave = mm(nt, "SINE", mm(nt, "ADD", mm(nt, "MULTIPLY", u.outputs["Fac"], 6.283), mm(nt, "ADD", mm(nt, "MULTIPLY", t, -0.09), mm(nt, "MULTIPLY", ph.outputs["Fac"], 40.0))))
    pulse = mm(nt, "POWER", mm(nt, "MAXIMUM", wave, 0.0), 24.0)
    s = mm(nt, "MULTIPLY", mm(nt, "ADD", 0.12, mm(nt, "MULTIPLY", pulse, 1.6)), S.value_node(nt, "Staerke", 0.0))
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.20, 0.42, 1.0, 1.0)
    nt.links.new(s, em.inputs["Strength"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    S.blended(mat, backface=True)
    return mat


def floor_mat():
    """Dark glossy floor with a faint dot grid fading out with distance."""
    mat, nt = S.new_mat("S3Boden")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.name = "BSDF"
    b.inputs["Base Color"].default_value = S.lin("#05060c")
    b.inputs["Roughness"].default_value = 0.22
    b.inputs["Specular IOR Level"].default_value = 0.55
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    mm = S.math_node

    def near_line(c):
        f = mm(nt, "FRACT", mm(nt, "MULTIPLY", c, 1.0))
        return mm(nt, "LESS_THAN", mm(nt, "ABSOLUTE", mm(nt, "SUBTRACT", f, 0.5)), 0.022)

    dots = mm(nt, "MULTIPLY", near_line(sep.outputs["X"]), near_line(sep.outputs["Y"]))
    lines = mm(nt, "MAXIMUM", near_line(sep.outputs["X"]), near_line(sep.outputs["Y"]))
    g = mm(nt, "ADD", mm(nt, "MULTIPLY", dots, 0.55), mm(nt, "MULTIPLY", lines, 0.0))
    d = nt.nodes.new("ShaderNodeVectorMath")
    d.operation = "DISTANCE"
    d.inputs[1].default_value = (0.0, 7.5, 0.0)
    nt.links.new(tc.outputs["Object"], d.inputs[0])
    fall = mm(nt, "SUBTRACT", 1.0, mm(nt, "DIVIDE", d.outputs["Value"], 9.0), clamp=True)
    g = mm(nt, "MULTIPLY", mm(nt, "MULTIPLY", g, mm(nt, "POWER", fall, 1.5)), S.value_node(nt, "Gitter", 0.0))
    b.inputs["Emission Color"].default_value = S.lin("#3d6fe0")
    nt.links.new(g, b.inputs["Emission Strength"])
    nt.links.new(b.outputs[0], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------
# geometry-node groups
# ---------------------------------------------------------------------------

def pixel_group():
    """Pixels -> nodes: analytic flight driven by scene time and point attributes."""
    g = S.GN("S3Pixel")
    start = g.inp("Start", default=144.0)
    dauer = g.inp("Dauer", default=90.0)
    staffel = g.inp("Staffel", default=50.0)
    vanish = g.inp("Vanish", default=1.0e6)
    frame = g.frame()
    verz = g.attr("verz")
    knoten = g.attr("knoten")
    t = g.math("DIVIDE", g.math("SUBTRACT", g.math("SUBTRACT", frame, start), g.math("MULTIPLY", verz, staffel)), dauer, clamp=True)
    e = g.smoothstep(t)
    # nodes take the long way (ease), dust leaves fast (out-cubic-ish via sqrt)
    e_dust = g.math("POWER", t, 0.55)
    e_mix = g.lerp(e_dust, e, knoten)
    p0 = g.attr("start", "FLOAT_VECTOR")
    p1 = g.attr("ziel", "FLOAT_VECTOR")
    bend = g.attr("bogen", "FLOAT_VECTOR")
    arc = g.math("SINE", g.math("MULTIPLY", e_mix, math.pi))
    pos = g.vmath("ADD", g.vlerp(p0, p1, e_mix), g.vmath("SCALE", bend, scale=arc))
    sp = g.node("GeometryNodeSetPosition")
    g.put(sp.inputs["Geometry"], g.geo)
    g.put(sp.inputs["Position"], pos)
    # vanish of the company network (nodes only), staggered
    av = g.attr("aus_verz")
    v0 = g.math("ADD", vanish, g.math("MULTIPLY", av, 50.0))
    fade = g.map_range(frame, v0, g.math("ADD", v0, 34.0))
    fade = g.math("MULTIPLY", fade, knoten)
    # scale: flat pixel tile -> node cube (dust shrinks to nothing)
    node_s = g.math("MULTIPLY", g.math("MULTIPLY", knoten, 0.044), g.math("SUBTRACT", 1.0, g.math("MULTIPLY", fade, 0.7)))
    sc = g.vlerp((PIX, 0.004, PIX), g.vec(node_s, node_s, node_s), e_mix)
    spin = g.attr("spin", "FLOAT_VECTOR")
    rot = g.vmath("SCALE", spin, scale=g.math("MULTIPLY", e_mix, g.math("SUBTRACT", 1.0, knoten)))
    e2r = g.node("FunctionNodeEulerToRotation")
    g.put(e2r.inputs[0], rot)
    # emission: page piece -> node blue (shader), dust glows out
    h_end = g.lerp(0.7, 1.7, knoten)
    hell = g.lerp(1.0, h_end, e_mix)
    hell = g.math("MULTIPLY", hell, g.math("SUBTRACT", 1.0, fade))
    pulse = g.math("ADD", 1.0, g.math("MULTIPLY", g.math("MULTIPLY", knoten, 0.22), g.math("SINE", g.math("ADD", g.math("MULTIPLY", frame, 0.06), g.math("MULTIPLY", verz, 37.0)))))
    hell = g.math("MULTIPLY", hell, pulse)
    geo = g.store(sp.outputs["Geometry"], "e", e_mix)
    geo = g.store(geo, "hell", hell)
    cube = g.node("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (1.0, 1.0, 1.0)
    lok = g.store(cube.outputs["Mesh"], "lokal", g.node("GeometryNodeInputPosition").outputs[0], "FLOAT_VECTOR")
    iop = g.node("GeometryNodeInstanceOnPoints")
    g.put(iop.inputs["Points"], geo)
    g.put(iop.inputs["Instance"], lok)
    g.put(iop.inputs["Rotation"], e2r.outputs[0])
    g.put(iop.inputs["Scale"], sc)
    real = g.node("GeometryNodeRealizeInstances")
    g.put(real.inputs[0], iop.outputs["Instances"])
    # cull: dust after the whole dive (one topology change while nothing moves
    # fast), nodes after the network has vanished
    done = g.math("ADD", g.math("ADD", start, staffel), g.math("ADD", dauer, 2.0))
    dust_gone = g.math("MULTIPLY", g.math("GREATER_THAN", frame, done), g.math("LESS_THAN", knoten, 0.5))
    nodes_gone = g.math("GREATER_THAN", frame, g.math("ADD", vanish, 90.0))
    sel = g.math("MAXIMUM", dust_gone, nodes_gone)
    dl = g.node("GeometryNodeDeleteGeometry", domain="POINT")
    g.put(dl.inputs["Geometry"], geo)
    g.put(dl.inputs["Selection"], g.math("GREATER_THAN", sel, 0.5))
    g.t.links.new(dl.outputs[0], iop.inputs["Points"])
    return g.out(real.outputs[0])


def lines_group():
    g = S.GN("S3Linien")
    radius = g.inp("Radius", default=0.0035)
    m2c = g.node("GeometryNodeMeshToCurve")
    g.put(m2c.inputs["Mesh"], g.geo)
    par = g.node("GeometryNodeSplineParameter")
    cur = g.store(m2c.outputs["Curve"], "u", par.outputs["Factor"])
    circ = g.node("GeometryNodeCurvePrimitiveCircle")
    circ.inputs["Resolution"].default_value = 6
    g.put(circ.inputs["Radius"], radius)
    c2m = g.node("GeometryNodeCurveToMesh")
    g.put(c2m.inputs["Curve"], cur)
    g.put(c2m.inputs["Profile Curve"], circ.outputs["Curve"])
    return g.out(c2m.outputs["Mesh"])


def sparks_group():
    """Ballistic sparks: p0 + v0 t + g t^2/2, aligned to velocity, fading."""
    g = S.GN("S3Funken")
    frame = g.frame()
    t0 = g.attr("t0")
    life = g.attr("life")
    tf = g.math("SUBTRACT", frame, t0)
    alive = g.math("MULTIPLY", g.math("GREATER_THAN", tf, -0.001), g.math("LESS_THAN", tf, life))
    ts = g.math("DIVIDE", g.math("MAXIMUM", tf, 0.0), 60.0)
    p0 = g.attr("p0", "FLOAT_VECTOR")
    v0 = g.attr("v0", "FLOAT_VECTOR")
    grav = (0.0, 0.0, -5.5)
    pos = g.vmath("ADD", g.vmath("ADD", p0, g.vmath("SCALE", v0, scale=ts)), g.vmath("SCALE", grav, scale=g.math("MULTIPLY", g.math("MULTIPLY", ts, ts), 0.5)))
    vel = g.vmath("ADD", v0, g.vmath("SCALE", grav, scale=ts))
    sp = g.node("GeometryNodeSetPosition")
    g.put(sp.inputs["Geometry"], g.geo)
    g.put(sp.inputs["Position"], pos)
    u = g.math("DIVIDE", g.math("MAXIMUM", tf, 0.0), life, clamp=True)
    k = g.math("MULTIPLY", alive, g.math("SUBTRACT", 1.0, u))
    hell = g.math("MULTIPLY", g.math("POWER", k, 1.5), 9.0)
    col = g.vlerp((1.0, 0.92, 0.85), S.lin(RED)[:3], u)
    geo = g.store(sp.outputs["Geometry"], "farbe", col, "FLOAT_COLOR")
    geo = g.store(geo, "hell", hell)
    al = g.node("FunctionNodeAlignRotationToVector")
    al.axis = "X"
    g.put(al.inputs["Vector"], vel)
    cube = g.node("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (1.0, 1.0, 1.0)
    iop = g.node("GeometryNodeInstanceOnPoints")
    g.put(iop.inputs["Points"], geo)
    g.put(iop.inputs["Instance"], cube.outputs["Mesh"])
    g.put(iop.inputs["Rotation"], al.outputs[0])
    g.put(iop.inputs["Scale"], g.vec(g.math("MULTIPLY", k, 0.045), g.math("MULTIPLY", k, 0.0045), g.math("MULTIPLY", k, 0.0045)))
    real = g.node("GeometryNodeRealizeInstances")
    g.put(real.inputs[0], iop.outputs["Instances"])
    return g.out(real.outputs[0])


# ---------------------------------------------------------------------------
# network layout
# ---------------------------------------------------------------------------

def node_lattice(rng) -> np.ndarray:
    """The company network lies below the eye line, the sides rise a little;
    the corridor in front of the camera stays clear so the wall reads."""
    pts = []
    for x in np.arange(-4.4, 4.6, 0.92):
        for y in np.arange(2.2, 11.0, 0.98):
            for z in (-0.98, -0.32, 0.42):
                p = np.array([x, y, z]) + rng.normal(0.0, 0.13, 3)
                if side_of_wall(p) > -0.38:
                    continue
                if z > 0.0 and abs(x) < 2.3:
                    continue  # upper layer only at the sides
                if y < 5.6 and abs(p[0] - 0.2) < 1.3 and p[2] > -0.55:
                    continue  # keep the view corridor free
                pts.append(p)
    return np.array(pts)


def lattice_edges(nodes: np.ndarray, rng) -> list[tuple[int, int]]:
    edges = set()
    for i, p in enumerate(nodes):
        d = np.linalg.norm(nodes - p, axis=1)
        order = np.argsort(d)[1:5]
        for j in order:
            if d[j] < 1.45 and rng.random() < 0.72:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def build_pixels(rng, nodes, coll):
    w, h = GRID
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    xs = (gx + 0.5) / w * WEB_W - WEB_W / 2
    zs = WEB_H / 2 - (gy + 0.5) / h * WEB_H
    start = np.stack([xs, np.zeros_like(xs), zs], axis=-1).reshape(-1, 3)
    uvc = np.stack([(gx + 0.5) / w, 1.0 - (gy + 0.5) / h, np.zeros_like(xs, dtype=np.float64)], axis=-1).reshape(-1, 3)
    n = len(start)
    r = np.hypot(start[:, 0] / (WEB_W / 2), start[:, 2] / (WEB_H / 2)) / math.sqrt(2.0)
    verz = np.clip(r + rng.normal(0.0, 0.05, n), 0.0, 1.0)
    knoten = np.zeros(n)
    idx = rng.choice(n, size=len(nodes), replace=False)
    knoten[idx] = 1.0
    ziel = np.empty((n, 3))
    # dust: outwards and into the depth, a little up or down
    radial = np.stack([start[:, 0], np.zeros(n), start[:, 2]], axis=1)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-3)
    ziel[:] = start + radial * rng.uniform(0.6, 2.6, (n, 1)) + np.array([0.0, 1.0, 0.0]) * rng.uniform(0.8, 5.0, (n, 1))
    ziel[:, 2] += rng.normal(0.0, 0.35, n)
    ziel[idx] = nodes
    bogen = np.zeros((n, 3))
    bogen[idx] = rng.normal(0.0, 0.35, (len(idx), 3)) + np.array([0.0, 0.0, 0.25])
    spin = rng.normal(0.0, 5.0, (n, 3))
    aus = rng.uniform(0.0, 1.0, n)
    # vanish: nodes closer to the wall go first (the scan finds them last)
    for k, i in enumerate(idx):
        aus[i] = float(np.clip(0.15 + 0.85 * (-side_of_wall(nodes[k]) / 5.0) + rng.normal(0, 0.08), 0.0, 1.0))
    ob = S.point_cloud("Pixelnetz", start, {
        "start": ("FLOAT_VECTOR", start), "verz": ("FLOAT", verz), "knoten": ("FLOAT", knoten), "ziel": ("FLOAT_VECTOR", ziel),
        "bogen": ("FLOAT_VECTOR", bogen), "spin": ("FLOAT_VECTOR", spin), "uvc": ("FLOAT_VECTOR", uvc),
        "aus_verz": ("FLOAT", aus),
    }, tile_mat("S3Pixel"), coll)
    S.gn_modifier(ob, pixel_group(), "Pixel")
    S.object_material(ob, ob.data.materials[0])
    return ob


def build_lines(nodes, edges, rng, coll):
    verts, eds, phase = [], [], []
    for a, b in edges:
        i = len(verts)
        verts += [tuple(nodes[a]), tuple(nodes[b])]
        eds.append((i, i + 1))
        ph = rng.random()
        phase += [ph, ph]
    ob = S.mesh_object("Verbindungen", verts, [], lines_mat(), edges=eds, coll=coll)
    a = ob.data.attributes.new("phase", "FLOAT", "POINT")
    a.data.foreach_set("value", np.asarray(phase, dtype=np.float32))
    S.gn_modifier(ob, lines_group(), "Linien")
    S.object_material(ob, ob.data.materials[0])
    return ob


def build_wall(coll):
    mat = slat_mat()
    d = wall_dir()
    length = (WALL_P1 - WALL_P0).length
    n = int(length / SLAT_STEP) + 1
    yaw = math.atan2(d.y, d.x)
    slats = []
    h = WALL_H[1] - WALL_H[0]
    for i in range(n):
        p = WALL_P0 + d * (i * SLAT_STEP)
        # origin on the floor: the slat grows upwards by scaling local Y
        v = [(-SLAT_W / 2, 0.0, 0.0), (SLAT_W / 2, 0.0, 0.0), (SLAT_W / 2, h, 0.0), (-SLAT_W / 2, h, 0.0)]
        ob = S.mesh_object(f"Lamelle_{i:02d}", v, [(0, 1, 2, 3)], mat, coll=coll)
        uvl = ob.data.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
        ob.location = (p.x, p.y, WALL_H[0])
        ob.rotation_euler = (math.radians(90), 0.0, yaw)
        slats.append(ob)
    return slats


def packet_mesh(name, color, coll):
    mat = S.principled(f"S3Paket_{name}", "#000000", rough=0.4, emission=S.lin(color), estr=0.0)
    nt = mat.node_tree
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    nt.links.new(S.math_node(nt, "MULTIPLY", oi.outputs["Alpha"], 9.0), nt.nodes["BSDF"].inputs["Emission Strength"])
    ob = S.box(name, (0.26, 0.095, 0.095), (0, 0, 0), mat, bevel=0.034, segments=3, coll=coll)
    return ob


# ---------------------------------------------------------------------------
# packet choreography (pure python, deterministic)
# ---------------------------------------------------------------------------

def schedule_packets(ev, rng):
    """Red: creep in, hit the wall, bounce.  Green: pass through to a node."""
    f_att, f_wall, f_bounce, f_radar = ev["attacker"], ev["firewall_wall"], ev["packets_bounce"], ev["radar_sweep"]
    d = wall_dir()
    length = (WALL_P1 - WALL_P0).length
    nrm = wall_normal()
    reds, greens = [], []

    def wall_point(s, z):
        p = WALL_P0 + d * (s * length)
        return Vector((p.x, p.y, z))

    # first wave: appears at the edge with ``attacker`` and waits for the wall
    for k in range(6):
        hit = wall_point(rng.uniform(0.15, 0.85), rng.uniform(-0.6, 0.8))
        spawn = hit + nrm * rng.uniform(3.0, 4.0) + d * rng.uniform(-1.0, 1.0) + Vector((0, 0, rng.uniform(-0.3, 0.3)))
        wait = hit + nrm * rng.uniform(0.9, 1.5) + Vector((0, 0, rng.uniform(-0.2, 0.2)))
        reds.append({"t_spawn": f_att + k * 5 - 6, "spawn": spawn, "wait": wait, "hit": hit,
                     "t_hit": f_wall + 26 + k * 9 + int(rng.integers(0, 6))})
    # main wave around packets_bounce, one hit exactly on the event
    hits = sorted([f_bounce] + [int(f_bounce + rng.uniform(-110, 330)) for _ in range(15)])
    for t_hit in hits:
        hit = wall_point(0.5, 0.05) if t_hit == f_bounce else wall_point(rng.uniform(0.18, 0.82), rng.uniform(-0.8, 0.8))
        spawn = hit + nrm * rng.uniform(3.2, 4.5) + d * rng.uniform(-1.5, 1.5) + Vector((0, 0, rng.uniform(-0.4, 0.4)))
        reds.append({"t_spawn": t_hit - int(rng.uniform(55, 80)), "spawn": spawn, "wait": None, "hit": hit, "t_hit": t_hit})
    for k in range(10):
        t_pass = int(f_bounce - 30 + k * (f_radar - f_bounce - 20) / 10 + rng.uniform(-8, 8))
        through = wall_point(rng.uniform(0.2, 0.8), rng.uniform(-0.5, 0.7))
        spawn = through + nrm * rng.uniform(4.5, 6.0) + d * rng.uniform(-1.2, 1.2)
        greens.append({"t_spawn": t_pass - int(rng.uniform(50, 70)), "spawn": spawn, "through": through, "t_pass": t_pass})
    return reds, greens


def red_track(r, rng_seed):
    rng = np.random.default_rng(rng_seed)
    nrm = wall_normal()
    v_in = (r["hit"] - (r["wait"] if r["wait"] is not None else r["spawn"]))
    refl = (v_in - 2 * v_in.dot(-nrm) * (-nrm)) * 0.32 if v_in.length > 0 else nrm
    refl += Vector((0, 0, 0.35)) + Vector(rng.normal(0, 0.25, 3))
    spin = Vector(rng.normal(0, 9.0, 3))
    t_sp, t_hit = r["t_spawn"], r["t_hit"]
    t_fade = t_hit + 46

    def at(f):
        if f < t_sp or f > t_fade:
            return None
        if r["wait"] is not None:
            t_go = t_hit - 22
            if f < t_go:
                u = C.span(f, t_sp, t_sp + 70, "out_cubic")
                wob = Vector((0.0, 0.06 * math.sin(f * 0.07 + rng_seed), 0.05 * math.sin(f * 0.05 + 2 * rng_seed)))
                return r["spawn"].lerp(r["wait"], u) + wob * u, 1.0
            u = C.span(f, t_go, t_hit, "in_cubic")
            start = r["wait"] + Vector((0.0, 0.06 * math.sin(t_go * 0.07 + rng_seed), 0.05 * math.sin(t_go * 0.05 + 2 * rng_seed)))
            return start.lerp(r["hit"], u), 1.0
        if f <= t_hit:
            u = C.span(f, t_sp, t_hit, "in_sine")
            return r["spawn"].lerp(r["hit"], u), 1.0
        tb = (f - t_hit) / 60.0
        return r["hit"] + refl * (tb * 2.2 - tb * tb * 0.9) + Vector((0, 0, -1.6 * tb * tb)), 1.0 - C.span(f, t_hit + 8, t_fade, "in_sine")

    return at, spin


def bake_packet(ob, at, frames, spin=None, fade_in=8):
    locs, rots, scs, alphas = [], [], [], []
    prev = None
    first = None
    for f in frames:
        r = at(f)
        if r is None:
            locs.append(tuple(prev) if prev is not None else (0.0, 0.0, -50.0))
            rots.append((0.0, 0.0, 0.0))
            scs.append((0.0, 0.0, 0.0))
            alphas.append(0.0)
            continue
        p, a = r
        if first is None:
            first = f
        nxt = at(f + 1)
        v = (nxt[0] - p) if nxt is not None else (p - prev if prev is not None else Vector((1, 0, 0)))
        q = v.to_track_quat("X", "Z") if v.length > 1e-6 else Vector((1, 0, 0)).to_track_quat("X", "Z")
        e = q.to_euler()
        if spin is not None and a < 0.999:
            e = (e.x + spin.x * (1 - a), e.y + spin.y * (1 - a), e.z + spin.z * (1 - a))
        k = C.span(f, first, first + fade_in, "out_cubic")
        locs.append(tuple(p))
        rots.append(tuple(e))
        scs.append((k, k, k))
        alphas.append(a * k)
        prev = p
    S.bake(ob, "location", frames, locs)
    S.bake(ob, "rotation_euler", frames, rots)
    S.bake(ob, "scale", frames, scs)
    S.bake(ob, "color", frames, [(1.0, 1.0, 1.0, a) for a in alphas])


# ---------------------------------------------------------------------------
# laptop, service, band, padlock
# ---------------------------------------------------------------------------

def build_laptop(coll):
    alu = S.principled("S3Alu", "#c6c9cf", rough=0.27, metal=1.0, aniso=0.4)
    deck = S.principled("S3Tasten", "#141519", rough=0.6)
    root = S.empty("Laptop", tuple(LAPTOP), coll)
    base = S.box("Laptop_Unterteil", (0.36, 0.25, 0.016), (0, 0, 0.008), alu, bevel=0.006, segments=4, coll=coll)
    keys = S.box("Laptop_Tastatur", (0.30, 0.11, 0.002), (0, 0.035, 0.0162), deck, coll=coll)
    pad = S.box("Laptop_Trackpad", (0.11, 0.07, 0.0015), (0, -0.075, 0.0162), S.principled("S3Pad", "#a9adb4", rough=0.35, metal=0.9), coll=coll)
    hinge = S.empty("Laptop_Scharnier", (0.0, 0.125, 0.016), coll)
    lid = S.box("Laptop_Deckel", (0.36, 0.006, 0.235), (0, 0.003, 0.1175), alu, bevel=0.004, segments=3, coll=coll)
    screen_img = S.load_image(S.ASSETS / "web_code.png")
    scr_mat = S.image_layer_mat("S3Bildschirm", screen_img, strength=0.0)
    scr_mat.surface_render_method = "DITHERED"
    scr = S.plane("Laptop_Bildschirm", 0.33, 0.206, (0, -0.0005, 0.1175), scr_mat, rot=(math.radians(90), 0, 0), coll=coll)
    for ob in (base, keys, pad, hinge):
        ob.parent = root
    for ob in (lid, scr):
        ob.parent = hinge
    hinge.rotation_euler = (math.radians(-12.0), 0, 0)  # lid open ~102 deg
    root.rotation_euler = (0, 0, math.radians(-24.0))
    return root, [base, keys, pad, lid, scr], scr_mat


def build_service(coll):
    glass = S.principled("S3Dienst", "#0a0d1a", rough=0.12, spec=0.6, coat=0.6, coat_rough=0.05)
    cube = S.box("Dienst", (0.24, 0.24, 0.24), tuple(SERVICE), glass, bevel=0.012, segments=3, coll=coll)
    edge_mat = S.principled("S3Dienst_Kante", "#000000", emission=S.lin("#6fd8c0"), estr=0.0)
    S.drive_input(edge_mat, "Emission Strength", "Leuchten", 0.0)
    wire = S.box("Dienst_Kanten", (0.246, 0.246, 0.246), tuple(SERVICE), edge_mat, coll=coll)
    wf = wire.modifiers.new("Draht", "WIREFRAME")
    wf.thickness = 0.005
    wf.use_even_offset = True
    core = S.box("Dienst_Kern", (0.09, 0.09, 0.09), tuple(SERVICE), S.principled("S3Dienst_Kern", "#000000", emission=S.lin(BLUE), estr=0.0), coll=coll)
    S.drive_input(core.active_material, "Emission Strength", "Leuchten", 0.0)
    return cube, wire, core, edge_mat


def build_padlock(coll):
    steel = S.principled("S3Schloss", "#b9bdc5", rough=0.22, metal=1.0)
    root = S.empty("Vorhaengeschloss", (0, 0, 0), coll)
    body = S.box("Schloss_Koerper", (0.058, 0.024, 0.048), (0, 0, 0), steel, bevel=0.007, segments=4, coll=coll)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.019, minor_radius=0.0042, major_segments=48, minor_segments=12,
                                     location=(0, 0, 0), rotation=(math.radians(90), 0, 0))
    torus = bpy.context.object
    torus.name = "Schloss_Buegel"
    # keep the upper half of the torus, extend the legs downwards
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(torus.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.y < -0.002], context="VERTS")  # torus local y = world z
    bm.to_mesh(torus.data)
    bm.free()
    torus.data.materials.append(steel)
    torus.data.shade_smooth()
    for c in torus.users_collection:
        c.objects.unlink(torus)
    coll.objects.link(torus)
    legs = []
    for sx in (-0.019, 0.019):
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.0042, depth=0.026, location=(sx, 0, -0.013))
        leg = bpy.context.object
        leg.data.materials.append(steel)
        leg.data.shade_smooth()
        for c in leg.users_collection:
            c.objects.unlink(leg)
        coll.objects.link(leg)
        legs.append(leg)
    shackle = S.empty("Schloss_Buegel_Achse", (0, 0, 0.022), coll)
    for ob in [torus] + legs:
        ob.parent = shackle
    body.parent = root
    shackle.parent = root
    return root, shackle, [body, torus] + legs


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(args):
    tl = TL.load(args.timeline or None)
    sh = TL.shot(SHOT, tl)
    ev = S.events_for(SHOT, EVENTS, tl)
    N = sh.frames
    frames = list(range(1, N + 1))
    f_dive, f_att, f_wall = ev["dive_into_pixels"], ev["attacker"], ev["firewall_wall"]
    f_bounce, f_radar, f_van, f_zero, f_lock = ev["packets_bounce"], ev["radar_sweep"], ev["net_vanish"], ev["zero_ports"], ev["mtls_lock"]
    rng = np.random.default_rng(3)

    scene = R.clean_scene()
    out = R.setup(scene, SHOT, N, res=args.res, samples=args.samples, motion_blur=not args.no_mb,
                  glare=not args.no_glare, out_dir=args.out or None)
    coll = scene.collection

    # ---------------- stage ---------------------------------------------------
    win = (-0.3, -0.3, 1.3, 1.3)
    night = M.hero_image("S3_Buehne", 1024, gains={"#145fe4": 0.55, "#7c3aed": 0.30, "#e44b8d": 0.12, "#10b981": 0.18},
                         blue_cy=1.30, window=win)
    world = M.world_backplate(scene, [night], lens_mm=35.0, sensor_mm=36.0, overscan=1.6)
    S.bake_fn(world.node_tree, S.node_path("Helligkeit"), frames,
              lambda f: 0.10 + 0.55 * C.span(f, f_dive + 30, f_dive + 130, "in_out_sine") * (1.0 - 0.55 * C.span(f, f_van, f_van + 90, "in_out_sine")))
    floor = S.plane("Boden", 60.0, 60.0, (0.0, 8.0, FLOOR_Z), floor_mat(), rot=(0, 0, 0), coll=coll)
    S.bake_fn(floor.active_material.node_tree, S.node_path("Gitter"), frames,
              lambda f: 0.9 * C.span(f, f_dive + 60, f_dive + 160, "in_out_sine") * (1.0 - 0.8 * C.span(f, f_van, f_van + 80, "in_out_sine")))

    # the page as it ends in s2 (real image), then the pixel mosaic
    page_mat = S.image_layer_mat("S3Seite", S.load_image(S.ASSETS / "web_komposit.png"), strength=1.0)
    page_mat.surface_render_method = "DITHERED"
    page = S.plane("Seite", WEB_W, WEB_H, (0.0, 0.0, 0.0), page_mat, coll=coll)
    S.bake_fn(page_mat.node_tree, S.node_path("Deckkraft"), frames, lambda f: 1.0 - C.span(f, f_dive - 4, f_dive + 6, "in_out_sine"))

    nodes = node_lattice(rng)
    edges = lattice_edges(nodes, rng)
    pixels = build_pixels(rng, nodes, coll)
    S.gn_set(pixels, "Pixel", "Start", float(f_dive))
    S.gn_set(pixels, "Pixel", "Dauer", 84.0)
    S.gn_set(pixels, "Pixel", "Staffel", 54.0)
    S.gn_set(pixels, "Pixel", "Vanish", float(f_van - 20))
    S.bake_fn(pixels.active_material.node_tree, S.node_path("Deckkraft"), frames, lambda f: C.span(f, f_dive - 8, f_dive + 2, "in_out_sine"))
    lines = build_lines(nodes, edges, rng, coll)
    S.bake_fn(lines.active_material.node_tree, S.node_path("Zeit"), frames, lambda f: float(f))
    S.bake_fn(lines.active_material.node_tree, S.node_path("Staerke"), frames,
              lambda f: C.span(f, f_dive + 70, f_dive + 150, "in_out_sine") * (1.0 - C.span(f, f_van - 10, f_van + 50, "in_out_sine")))

    # ---------------- firewall ------------------------------------------------
    slats = build_wall(coll)
    n_sl = len(slats)
    for i, ob in enumerate(slats):
        t0 = f_wall - 14 + i * 1.2
        S.bake_fn(ob, "scale", frames, lambda f, t0=t0: (1.0, max(1e-3, C.span(f, t0, t0 + 26, "out_cubic")), 1.0))
    S.bake_fn(slats[0].active_material.node_tree, S.node_path("Staerke"), frames,
              lambda f: 1.0 * C.span(f, f_wall - 14, f_wall + 10, "out_cubic") * (1.0 - 0.75 * C.span(f, f_van, f_van + 70, "in_out_sine") - 0.25 * C.span(f, f_zero - 40, f_zero + 10, "in_out_sine"))
              + 0.35 * math.exp(-((f - f_wall - 12) / 9.0) ** 2))
    mid = (WALL_P0 + WALL_P1) * 0.5
    wl1 = S.area("Wandlicht_1", tuple(mid + Vector((0.0, -0.4, 1.6))), tuple(mid + Vector((0.0, -2.5, FLOOR_Z))), 5.0, 0.0, S.lin(BLUE),
                 shape="RECTANGLE", size_y=0.4)
    wl1.data.use_shadow = False
    S.bake_fn(wl1.data, "energy", frames, lambda f: 35.0 * C.span(f, f_wall - 10, f_wall + 16, "out_cubic") * (1.0 - 0.75 * C.span(f, f_van, f_van + 70, "in_out_sine")))

    # ---------------- packets, impacts, sparks ----------------------------------
    reds, greens = schedule_packets(ev, rng)
    impacts = []
    for k, r in enumerate(reds):
        ob = packet_mesh(f"Rot_{k:02d}", RED, coll)
        at, spin = red_track(r, 100 + k)
        bake_packet(ob, at, frames, spin)
        impacts.append((r["t_hit"], r["hit"], (r["hit"] - (r["wait"] if r["wait"] is not None else r["spawn"])).normalized()))
    for k, gpk in enumerate(greens):
        ob = packet_mesh(f"Gruen_{k:02d}", GREEN, coll)
        target = Vector(nodes[int(rng.integers(0, len(nodes)))])
        t_sp, t_pass = gpk["t_spawn"], gpk["t_pass"]
        t_arr = t_pass + int(rng.uniform(40, 60))

        def at(f, g=gpk, target=target, t_sp=t_sp, t_pass=t_pass, t_arr=t_arr):
            if f < t_sp or f > t_arr + 10:
                return None
            if f <= t_pass:
                return g["spawn"].lerp(g["through"], C.span(f, t_sp, t_pass, "linear")), 1.0
            u = C.span(f, t_pass, t_arr, "out_sine")
            ctrl = g["through"] + (g["through"] - g["spawn"]).normalized() * 0.8
            p = (1 - u) ** 2 * g["through"] + 2 * (1 - u) * u * ctrl + u * u * target
            return p, 1.0 - C.span(f, t_arr - 4, t_arr + 10, "in_sine")

        bake_packet(ob, at, frames)
    flash_mat = radial_glow_mat("S3Treffer", "#5f8dff", 2.6)
    nrm = wall_normal()
    yaw = math.atan2(wall_dir().y, wall_dir().x)
    sp_pos, sp_vel, sp_t0, sp_life = [], [], [], []
    for k, (t_hit, hit, vin) in enumerate(impacts):
        fl = S.plane(f"Treffer_{k:02d}", 0.55, 0.55, tuple(hit + nrm * 0.02), flash_mat, rot=(math.radians(90), 0, yaw), coll=coll)
        S.bake_fn(fl, "scale", frames, lambda f, t=t_hit: (0.25 + 0.95 * C.span(f, t, t + 14, "out_cubic"),) * 3)
        S.bake_fn(fl, "color", frames, lambda f, t=t_hit: (1, 1, 1, (1.0 - C.span(f, t, t + 16, "out_sine")) if f >= t else 0.0))
        out_dir = (nrm * 0.8 + Vector((0, 0, 0.25))).normalized()  # sparks spray back outwards
        for _ in range(18):
            d = (out_dir + Vector(rng.normal(0, 0.55, 3))).normalized()
            sp_pos.append(tuple(hit + nrm * 0.03))
            sp_vel.append(tuple(d * rng.uniform(1.4, 4.2)))
            sp_t0.append(float(t_hit))
            sp_life.append(float(rng.uniform(12, 30)))
    sparks = S.point_cloud("Funken", sp_pos, {"p0": ("FLOAT_VECTOR", np.array(sp_pos)), "v0": ("FLOAT_VECTOR", np.array(sp_vel)),
                                              "t0": ("FLOAT", np.array(sp_t0)), "life": ("FLOAT", np.array(sp_life))},
                           attr_emission_mat("S3Funken"), coll)
    S.gn_modifier(sparks, sparks_group(), "Funken")
    S.object_material(sparks, sparks.data.materials[0])

    # ---------------- radar ----------------------------------------------------
    cone_len, cone_ang = 19.0, math.radians(5.5)
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=0.0, radius2=cone_len * math.tan(cone_ang), depth=cone_len,
                                    location=(0, 0, 0), end_fill_type="NOTHING")
    cone = bpy.context.object
    cone.name = "Radarkegel"
    # apex at the origin, opening along -Z of the object
    for v in cone.data.vertices:
        v.co.z = -(v.co.z + cone_len / 2)
    S.recalc_normals(cone)
    beam = S.glow_mat("S3Radar", "#ff5a6e", 0.0, facing=0.08)
    nt = beam.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    along = S.math_node(nt, "SUBTRACT", 1.0, S.math_node(nt, "DIVIDE", S.math_node(nt, "ABSOLUTE", sep.outputs["Z"]), cone_len), clamp=True)
    em = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeEmission")
    old = em.inputs["Strength"].links[0].from_socket
    nt.links.new(S.math_node(nt, "MULTIPLY", old, S.math_node(nt, "POWER", along, 1.8)), em.inputs["Strength"])
    cone.data.materials.append(beam)
    cone.location = tuple(RADAR)
    for c in cone.users_collection:
        c.objects.unlink(cone)
    coll.objects.link(cone)
    radar_spot = S.spot("Radarlicht", tuple(RADAR), (0, 6, FLOOR_Z), 0.0, S.lin("#ff4a60"), angle=13.0, blend=0.35, radius=0.05)
    centre = Vector((0.6, 6.4, FLOOR_Z + 0.2))
    base_dir = (centre - RADAR).normalized()

    def radar_dir(f):
        a = math.radians(24.0 - 52.0 * C.span(f, f_radar - 10, f_van + 150, "in_out_sine"))
        dx, dy = base_dir.x * math.cos(a) - base_dir.y * math.sin(a), base_dir.x * math.sin(a) + base_dir.y * math.cos(a)
        return Vector((dx, dy, base_dir.z))

    S.bake_fn(cone, "rotation_euler", frames, lambda f: tuple(radar_dir(f).to_track_quat("-Z", "Y").to_euler()))
    S.bake_fn(radar_spot, "rotation_euler", frames, lambda f: tuple(radar_dir(f).to_track_quat("-Z", "Y").to_euler()))
    radar_on = lambda f: C.span(f, f_radar - 16, f_radar + 10, "out_cubic") * (1.0 - C.span(f, f_lock - 120, f_lock - 60, "in_out_sine"))  # noqa: E731
    S.bake_fn(beam.node_tree, S.node_path("Staerke"), frames, lambda f: 0.03 * radar_on(f))
    core = cone.copy()
    core.data = cone.data.copy()
    core.name = "Radarkern"
    for v in core.data.vertices:
        v.co.x *= 0.22
        v.co.y *= 0.22
    core_mat = S.glow_mat("S3Radarkern", "#ff8a96", 0.0, facing=0.12)
    nt2 = core_mat.node_tree
    tc2 = nt2.nodes.new("ShaderNodeTexCoord")
    sep2 = nt2.nodes.new("ShaderNodeSeparateXYZ")
    nt2.links.new(tc2.outputs["Object"], sep2.inputs[0])
    along2 = S.math_node(nt2, "SUBTRACT", 1.0, S.math_node(nt2, "DIVIDE", S.math_node(nt2, "ABSOLUTE", sep2.outputs["Z"]), cone_len), clamp=True)
    em2 = next(n for n in nt2.nodes if n.bl_idname == "ShaderNodeEmission")
    old2 = em2.inputs["Strength"].links[0].from_socket
    nt2.links.new(S.math_node(nt2, "MULTIPLY", old2, S.math_node(nt2, "POWER", along2, 1.2)), em2.inputs["Strength"])
    core.data.materials.clear()
    core.data.materials.append(core_mat)
    coll.objects.link(core)
    S.bake_fn(core, "rotation_euler", frames, lambda f: tuple(radar_dir(f).to_track_quat("-Z", "Y").to_euler()))
    S.bake_fn(core_mat.node_tree, S.node_path("Staerke"), frames, lambda f: 0.35 * radar_on(f))
    S.bake_fn(radar_spot.data, "energy", frames, lambda f: 9000.0 * radar_on(f))

    # ---------------- mTLS: laptop -> one service, padlock ------------------------
    laptop, laptop_parts, scr_mat = build_laptop(coll)
    cube, wire, core, edge_mat = build_service(coll)
    lock_root, shackle, lock_parts = build_padlock(coll)
    f_lap = f_zero - 40
    S.bake_fn(laptop, "location", frames, lambda f: tuple(LAPTOP + Vector((0.0, 0.0, -0.25 * (1 - C.span(f, f_lap - 30, f_lap + 20, "out_cubic"))))))
    S.bake_fn(laptop, "scale", frames, lambda f: (C.span(f, f_lap - 30, f_lap - 22, "linear"),) * 3)
    S.bake_fn(scr_mat.node_tree, S.node_path("Staerke"), frames, lambda f: 0.55 * C.span(f, f_lap + 10, f_lap + 40, "in_out_sine"))
    for ob in (cube, wire, core):
        S.bake_fn(ob, "scale", frames, lambda f: (C.span(f, f_lap - 6, f_lap + 24, "out_cubic"),) * 3)
    S.bake_fn(edge_mat.node_tree, S.node_path("Leuchten"), frames,
              lambda f: 1.2 * C.span(f, f_lap, f_lap + 30, "in_out_sine") + 5.0 * math.exp(-((f - f_lock - 3) / 6.0) ** 2))
    S.bake_fn(core.active_material.node_tree, S.node_path("Leuchten"), frames,
              lambda f: 2.0 * C.span(f, f_lap, f_lap + 30, "in_out_sine") + 4.0 * C.span(f, f_lock - 2, f_lock + 8, "out_cubic"))
    lap_key = S.area("Laptop_Licht", tuple(LAPTOP + Vector((-0.6, -0.7, 0.9))), tuple(LAPTOP), 1.2, 0.0, M.kelvin_rgb(6200))
    S.bake_fn(lap_key.data, "energy", frames, lambda f: 18.0 * C.span(f, f_lap - 20, f_lap + 30, "in_out_sine"))
    lock_light = S.area("Schloss_Licht", tuple(SERVICE + Vector((-0.35, -0.75, 0.45))), tuple(SERVICE + Vector((0, -0.16, -0.03))),
                        0.25, 0.0, M.kelvin_rgb(6500), shape="RECTANGLE", size_y=0.6)
    S.bake_fn(lock_light.data, "energy", frames, lambda f: 6.0 * C.span(f, f_lock - 30, f_lock - 10, "in_out_sine"))

    # band: from the laptop screen edge in an arc to the service cube face
    a = LAPTOP + Vector((0.10, 0.06, 0.20))
    b = SERVICE + Vector((-0.045, -0.126, 0.045))
    mid1 = a.lerp(b, 0.35) + Vector((0.0, -0.1, 0.42))
    mid2 = a.lerp(b, 0.72) + Vector((0.0, -0.15, 0.28))
    from nomiss_logo import catmull_rom
    path = catmull_rom([tuple(a), tuple(mid1), tuple(mid2), tuple(b)], samples_per_seg=40)
    band_mat = M.ribbon_material("S3Band", [(0.0, S.lin(BLUE)), (0.55, S.lin("#7c3aed")), (1.0, S.lin(GREEN))], emission=1.4, edge=1.0, head=7.0)
    band = RB.build(path, 0.012, 0.0024, name="mTLS_Band", material=band_mat, collection=coll)
    RB.set_input(band, "Kopf Laenge", 0.12)
    f_b0, f_b1 = f_lock - 70, f_lock - 14
    RB.bake_input(band, "Ende", frames, lambda f: C.span(f, f_b0, f_b1, "in_out_cubic"))
    RB.bake_input(band, "Kopf Laenge", frames, lambda f: 0.12 * (1.0 - C.span(f, f_b1, f_b1 + 20, "in_out_sine")))

    lock_pos = SERVICE + Vector((0.01, -0.165, -0.035))
    S.bake_fn(lock_root, "location", frames,
              lambda f: tuple(lock_pos + Vector((0.0, -0.06 * (1 - C.span(f, f_lock - 24, f_lock - 8, "out_quint")), 0.20 * (1 - C.span(f, f_lock - 24, f_lock - 8, "out_quint"))))))
    S.bake_fn(lock_root, "scale", frames, lambda f: (2.2 * C.span(f, f_lock - 26, f_lock - 20, "linear"),) * 3)
    S.bake_fn(lock_root, "rotation_euler", frames, lambda f: (0.0, 0.0, math.radians(-10.0)))
    S.bake_fn(shackle, "location", frames, lambda f: (0.0, 0.0, 0.034 - 0.012 * S.overshoot((f - (f_lock - 7)) / 9.0, 0.03)))
    ring = S.plane("Schloss_Puls", 0.6, 0.6, tuple(SERVICE + Vector((0, -0.13, 0))), radial_glow_mat("S3Puls", "#9ff0d6", 2.2),
                   rot=(math.radians(90), 0, math.radians(-10.0)), coll=coll)
    S.bake_fn(ring, "scale", frames, lambda f: (0.05 + 0.9 * C.span(f, f_lock, f_lock + 22, "out_cubic"),) * 3)
    S.bake_fn(ring, "color", frames, lambda f: (1, 1, 1, (1.0 - C.span(f, f_lock, f_lock + 24, "out_sine")) if f >= f_lock else 0.0))

    # ---------------- camera -----------------------------------------------------
    cam, tgt, foc = S.camera("Kamera3", lens=40.0, fstop=2.8)
    mid_net = Vector((0.3, 6.2, 0.0))
    pos = S.Bahn([
        (1, (0.0, -1.95, 0.0)),
        (31, (0.0, -1.45, 0.0)),
        (f_dive, (0.0, -1.02, 0.0)),
        (f_dive + 34, (0.04, 0.20, 0.03)),
        (f_dive + 110, (0.25, 2.3, 0.45)),
        (f_att, (0.35, 2.4, 0.50)),
        (f_wall, (0.10, 0.2, 0.66), True),
        (f_bounce, (0.55, 4.7, 0.30)),
        (f_bounce + 230, (-0.15, 5.1, 0.36)),
        (f_radar, (0.6, 0.4, 3.7), True),
        (f_van + 60, (0.4, 0.9, 3.4)),
        (f_zero, (-0.25, 2.55, 0.42)),
        (f_lock, (-0.18, 2.90, 0.30), True),
        (N, (-0.17, 2.93, 0.29)),
    ])
    aimp = S.Bahn([
        (1, (0.0, 0.0, 0.0)),
        (f_dive, (0.0, 0.0, 0.0)),
        (f_dive + 34, (0.10, 4.0, 0.0)),
        (f_dive + 110, (0.30, 10.0, 0.0)),
        (f_att, (0.40, 11.0, 0.0)),
        (f_wall, (-0.12, 10.3, 0.05), True),
        (f_bounce, (0.40, 10.4, 0.05)),
        (f_bounce + 230, (0.20, 10.5, 0.10)),
        (f_radar, (0.3, 8.6, -0.7), True),
        (f_van + 60, (0.3, 8.4, -0.7)),
        (f_zero, (-0.03, 5.7, -0.46)),
        (f_lock, (-0.03, 5.7, -0.46), True),
        (N, (-0.03, 5.7, -0.46)),
    ])
    focus = S.Bahn([
        (1, (0.0, 0.0, 0.0)),
        (f_dive, (0.0, 0.0, 0.0)),
        (f_dive + 60, (0.3, 6.0, 0.0)),
        (f_att, (0.4, 12.0, 0.0)),
        (f_wall, tuple(mid + Vector((0, 0, 0.1)))),
        (f_bounce + 230, tuple(mid + Vector((0, 0, 0.1))), True),
        (f_radar, tuple(mid_net)),
        (f_zero, tuple(LAPTOP.lerp(SERVICE, 0.5))),
        (f_lock, tuple(LAPTOP.lerp(SERVICE, 0.55)), True),
    ])
    lens = S.Bahn([(1, 40.0), (f_dive, 40.0, True), (f_dive + 60, 32.0), (f_att, 34.0), (f_wall, 34.0, True), (f_radar, 30.0, True),
                   (f_zero, 38.0), (f_lock, 38.0, True)])
    fstop = S.Bahn([(1, 2.8), (f_dive + 40, 2.0), (f_wall, 2.8), (f_radar, 3.5, True), (f_zero, 2.4), (f_lock, 2.4, True)])
    S.bake_camera(cam, tgt, foc, frames, pos=pos, target=aimp, focus=focus, lens=lens, fstop=fstop)

    k_ev = next(k for k, r in enumerate(reds) if r["t_hit"] == f_bounce)
    keys = {
        "dive_into_pixels": [page],
        "firewall_wall": slats,
        "packets_bounce": [bpy.data.objects[f"Rot_{k_ev:02d}"], bpy.data.objects[f"Treffer_{k_ev:02d}"]],
        "net_vanish": slats,
        "mtls_lock": laptop_parts + [cube, band] + lock_parts,
    }
    print(f"[s3] timeline={sh.quelle} frames={N} events={ev} nodes={len(nodes)} edges={len(edges)} "
          f"slats={n_sl} red={len(reds)} green={len(greens)} sparks={len(sp_pos)}", flush=True)
    return scene, {"shot": sh, "events": ev, "out": out, "keys": keys}


def main() -> None:
    args, own = S.parse_cli(default_samples=16)
    scene, info = build(args)
    S.run(scene, info, args, own)


if __name__ == "__main__":
    main()
