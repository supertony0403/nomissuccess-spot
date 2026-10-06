"""Night city as a bokeh field: dark facades plus instanced window lights.

Windows are points on the visible facades; a Geometry-Nodes group instances
a small quad on every point (aligned to the facade) and computes the
instance attribute ``hell`` per frame:

    hell = hell0 * (1 - smoothstep(aus_f, aus_f + fade, frame)) * Helligkeit

``aus_f`` is the shot frame at which a window switches off (1e9 = never),
so "lights off" is a property of the data, not of keyframes.  The global
``Helligkeit`` input fades the whole city (dawn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import bpy
import numpy as np

from nomiss_material import assign, bokeh_material, facade_material, kelvin_rgb, window_material

GROUP = "NomissFenster"
MOD = "NomissFenster"
NEVER = 1.0e9


def window_group() -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.get(GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(GROUP, "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    s = ng.interface.new_socket(name="Helligkeit", in_out="INPUT", socket_type="NodeSocketFloat")
    s.default_value = 1.0
    s = ng.interface.new_socket(name="Ausblenden Bilder", in_out="INPUT", socket_type="NodeSocketFloat")
    s.default_value = 5.0
    nodes, links = ng.nodes, ng.links
    gi = nodes.new("NodeGroupInput")
    go = nodes.new("NodeGroupOutput")

    def attr(name, dtype="FLOAT"):
        n = nodes.new("GeometryNodeInputNamedAttribute")
        n.data_type = dtype
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    grid = nodes.new("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value = 1.0
    grid.inputs["Size Y"].default_value = 1.0
    grid.inputs["Vertices X"].default_value = 2
    grid.inputs["Vertices Y"].default_value = 2
    uv = nodes.new("GeometryNodeStoreNamedAttribute")  # quad UV for the bokeh disc shader
    uv.data_type = "FLOAT_VECTOR"
    uv.domain = "CORNER"
    uv.inputs["Name"].default_value = "quad_uv"
    links.new(grid.outputs["Mesh"], uv.inputs["Geometry"])
    links.new(grid.outputs["UV Map"], uv.inputs["Value"])
    align = nodes.new("FunctionNodeAlignRotationToVector")
    align.axis = "Z"
    links.new(attr("nrm", "FLOAT_VECTOR"), align.inputs["Vector"])
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(gi.outputs["Geometry"], inst.inputs["Points"])
    links.new(uv.outputs["Geometry"], inst.inputs["Instance"])
    links.new(align.outputs["Rotation"], inst.inputs["Rotation"])
    links.new(attr("groesse", "FLOAT_VECTOR"), inst.inputs["Scale"])

    t = nodes.new("GeometryNodeInputSceneTime")
    aus = attr("aus_f")
    aus_end = nodes.new("ShaderNodeMath")
    aus_end.operation = "ADD"
    links.new(aus, aus_end.inputs[0])
    links.new(gi.outputs["Ausblenden Bilder"], aus_end.inputs[1])
    mr = nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    links.new(t.outputs["Frame"], mr.inputs["Value"])
    links.new(aus, mr.inputs["From Min"])
    links.new(aus_end.outputs[0], mr.inputs["From Max"])
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    m1 = nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    links.new(attr("hell0"), m1.inputs[0])
    links.new(mr.outputs["Result"], m1.inputs[1])
    m2 = nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    links.new(m1.outputs[0], m2.inputs[0])
    links.new(gi.outputs["Helligkeit"], m2.inputs[1])
    st = nodes.new("GeometryNodeStoreNamedAttribute")
    st.data_type = "FLOAT"
    st.domain = "INSTANCE"
    st.inputs["Name"].default_value = "hell"
    links.new(inst.outputs["Instances"], st.inputs["Geometry"])
    links.new(m2.outputs[0], st.inputs["Value"])
    links.new(st.outputs["Geometry"], go.inputs[0])
    return ng


@dataclass
class Stadt:
    fassaden: bpy.types.Object | None
    fenster: bpy.types.Object
    pos: np.ndarray       # (N, 3) window centres
    hell0: np.ndarray     # (N,) base brightness

    def set_off_frames(self, aus_f: np.ndarray) -> None:
        a = self.fenster.data.attributes["aus_f"]
        a.data.foreach_set("value", np.asarray(aus_f, dtype=np.float32))
        self.fenster.data.update()

    def input_path(self, name: str) -> str:
        mod = self.fenster.modifiers[MOD]
        ident = next(i.identifier for i in mod.node_group.interface.items_tree if getattr(i, "in_out", "") == "INPUT" and i.name == name)
        return f'modifiers["{MOD}"].properties.inputs["{ident}"]["value"]'

    def key_brightness(self, frame: float, value: float) -> None:
        mod = self.fenster.modifiers[MOD]
        ident = next(i.identifier for i in mod.node_group.interface.items_tree if getattr(i, "in_out", "") == "INPUT" and i.name == "Helligkeit")
        mod.properties.inputs[ident]["value"] = value
        self.fenster.keyframe_insert(self.input_path("Helligkeit"), frame=frame)


def _box(verts, faces, x0, x1, y0, y1, z0, z1):
    b = len(verts)
    verts += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces += [(b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5), (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7), (b + 4, b + 5, b + 6, b + 7)]


def build(
    name: str,
    *,
    seed: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_buildings: int,
    height: tuple[float, float],
    eye: tuple[float, float, float],
    lit: float = 0.32,
    brightness: float = 6.0,
    x_weight=None,
    warm: float = 0.6,
    facades: bool = True,
    log_depth: bool = False,
    height_cap=None,
    collection: bpy.types.Collection | None = None,
) -> Stadt:
    """Random city block field seen from ``eye``.

    ``x_weight(x) -> 0..1`` thins out buildings (e.g. a calm left half).
    Only the two facades facing ``eye`` get windows.  ``warm`` is the share
    of warm (2500-3300 K) windows, the rest is neutral or cool white.
    ``facades=False`` keeps only the lights (no dark silhouettes).
    ``log_depth`` samples depth log-uniformly (as many blocks per octave of
    distance, so near low blocks fill the lower frame); ``height_cap(x, y)``
    limits building height, e.g. to keep near blocks below eye level.
    """
    rng = np.random.default_rng(seed)
    coll = collection or bpy.context.scene.collection
    verts: list = []
    faces: list = []
    wpos, wnrm, wsize, wcol, whell = [], [], [], [], []
    ex, ey, _ = eye
    placed = 0
    tries = 0
    while placed < n_buildings and tries < n_buildings * 20:
        tries += 1
        x = rng.uniform(*x_range)
        y = (
            float(np.exp(rng.uniform(math.log(y_range[0]), math.log(y_range[1]))))
            if log_depth
            else rng.uniform(*y_range)
        )
        if x_weight is not None and rng.random() > x_weight(x):
            continue
        placed += 1
        w = rng.uniform(14.0, 42.0)
        d = rng.uniform(14.0, 36.0)
        h = float(np.clip(rng.lognormal(math.log(np.mean(height)), 0.45), height[0], height[1]))
        if height_cap is not None:
            h = min(h, max(height[0], float(height_cap(x, y))))
        x0, x1, y0, y1 = x - w / 2, x + w / 2, y - d / 2, y + d / 2
        _box(verts, faces, x0, x1, y0, y1, 0.0, h)
        floor_h = rng.uniform(3.1, 3.6)
        col_w = rng.uniform(2.6, 3.4)
        b_lit = float(np.clip(lit * rng.uniform(0.5, 1.6), 0.05, 0.75))
        # facades facing the eye: front (-Y side) and the side towards eye x
        sides = [((x0, y0), (x1, y0), (0.0, -1.0, 0.0))]
        if ex < x0:
            sides.append(((x0, y1), (x0, y0), (-1.0, 0.0, 0.0)))
        elif ex > x1:
            sides.append(((x1, y0), (x1, y1), (1.0, 0.0, 0.0)))
        for (ax, ay), (bx, by), n in sides:
            length = math.hypot(bx - ax, by - ay)
            ncol = max(1, int(length / col_w) - 1)
            nfl = max(1, int((h - 4.0) / floor_h))
            for i in range(ncol):
                f = (i + 1) / (ncol + 1)
                px, py = ax + (bx - ax) * f + n[0] * 0.08, ay + (by - ay) * f + n[1] * 0.08
                for j in range(nfl):
                    if rng.random() > b_lit:
                        continue
                    pz = 4.0 + j * floor_h + floor_h * 0.45
                    k = rng.random()
                    kelvin = (
                        rng.uniform(2500, 3300)
                        if k < warm
                        else rng.uniform(3600, 4600)
                        if k < warm + (1 - warm) * 0.6
                        else rng.uniform(5600, 7200)
                    )
                    wpos.append((px, py, pz))
                    wnrm.append(n)
                    wsize.append((rng.uniform(1.2, 1.8), rng.uniform(1.2, 1.7), 1.0))
                    wcol.append((*kelvin_rgb(kelvin), 1.0))
                    whell.append(brightness * float(np.clip(rng.lognormal(0.0, 0.55), 0.25, 4.0)))

    me = bpy.data.meshes.new(f"{name}_Fassaden")
    me.from_pydata(verts, [], faces)
    fass = bpy.data.objects.new(f"{name}_Fassaden", me)
    coll.objects.link(fass)
    assign(fass, facade_material())
    fass.hide_render = not facades

    pos = np.asarray(wpos, dtype=np.float64)
    pm = bpy.data.meshes.new(f"{name}_Fenster")
    pm.from_pydata(pos.tolist(), [], [])
    n = len(pos)
    for aname, kind, data, key in (
        ("nrm", "FLOAT_VECTOR", np.asarray(wnrm), "vector"),
        ("groesse", "FLOAT_VECTOR", np.asarray(wsize), "vector"),
        ("farbe", "FLOAT_COLOR", np.asarray(wcol), "color"),
        ("hell0", "FLOAT", np.asarray(whell), "value"),
        ("aus_f", "FLOAT", np.full(n, NEVER), "value"),
    ):
        a = pm.attributes.new(aname, kind, "POINT")
        a.data.foreach_set(key, data.astype(np.float32).ravel())
    fen = bpy.data.objects.new(f"{name}_Fenster", pm)
    coll.objects.link(fen)
    assign(fen, window_material())
    mod = fen.modifiers.new(MOD, "NODES")
    mod.node_group = window_group()
    print(f"[nomiss_stadt] {name}: {placed} Gebäude, {n} Fensterlichter", flush=True)
    return Stadt(fass, fen, pos, np.asarray(whell))


def _points_object(name: str, pos, nrm, size, col, hell, coll, material) -> bpy.types.Object:
    pm = bpy.data.meshes.new(name)
    pm.from_pydata(np.asarray(pos).tolist(), [], [])
    n = len(pos)
    for aname, kind, data, key in (
        ("nrm", "FLOAT_VECTOR", np.asarray(nrm), "vector"),
        ("groesse", "FLOAT_VECTOR", np.asarray(size), "vector"),
        ("farbe", "FLOAT_COLOR", np.asarray(col), "color"),
        ("hell0", "FLOAT", np.asarray(hell), "value"),
        ("aus_f", "FLOAT", np.full(n, NEVER), "value"),
    ):
        a = pm.attributes.new(aname, kind, "POINT")
        a.data.foreach_set(key, np.asarray(data, dtype=np.float32).ravel())
    ob = bpy.data.objects.new(name, pm)
    coll.objects.link(ob)
    assign(ob, material)
    mod = ob.modifiers.new(MOD, "NODES")
    mod.node_group = window_group()
    return ob


@dataclass
class Ebene:
    """One depth layer of bokeh discs."""

    distance: float        # metres in front of the reference camera
    count: int
    size_px: tuple[float, float]
    brightness: tuple[float, float]


def bokeh_layers(
    name: str,
    *,
    seed: int,
    cam_loc,
    cam_rot,
    lens_mm: float,
    sensor_mm: float,
    layers: list[Ebene],
    region: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    x_weight=None,
    cool: float = 0.15,
    kelvin: tuple[float, float] = (2900.0, 4500.0),
    collection: bpy.types.Collection | None = None,
) -> Stadt:
    """Designed bokeh discs in depth layers (instead of noisy DOF blur).

    Positions are drawn in screen space of a reference camera (``cam_loc``,
    ``cam_rot`` as Euler) inside ``region`` = (x0, x1, y0, y1) in -1..1
    screen units (y up), placed at each layer's distance and sized so that
    they appear ``size_px`` wide at 1920 px from the reference pose.  The
    discs face the camera's viewing direction; their look (clear edge,
    slightly darker centre) comes from :func:`nomiss_material.bokeh_material`.
    """
    from mathutils import Euler, Vector

    rng = np.random.default_rng(seed)
    coll = collection or bpy.context.scene.collection
    rot = Euler(cam_rot).to_matrix()
    right, up, back = rot.col[0], rot.col[1], rot.col[2]
    fwd = -back
    tan_h = sensor_mm / lens_mm * 0.5
    eye = Vector(cam_loc)
    pos, nrm, size, col, hell = [], [], [], [], []
    for layer in layers:
        placed: list[tuple[float, float, float]] = []
        tries = 0
        while len(placed) < layer.count and tries < layer.count * 400:
            tries += 1
            sx = rng.uniform(region[0], region[1])
            sy = rng.uniform(region[2], region[3])
            if x_weight is not None and rng.random() > x_weight(sx, sy):
                continue
            px = rng.uniform(*layer.size_px)
            r_s = px / 1920.0  # radius in screen units (-1..1 spans 1920 px)
            if any((sx - a) ** 2 + (sy - b) ** 2 < (0.75 * (r_s + c)) ** 2 for a, b, c in placed):
                continue  # keep discs mostly apart: calm, not a cloud
            placed.append((sx, sy, r_s))
            d = layer.distance * rng.uniform(0.85, 1.15)
            p = eye + fwd * d + right * (sx * d * tan_h) + up * (sy * d * tan_h)
            w = px / 1920.0 * 2.0 * d * tan_h
            pos.append(tuple(p))
            nrm.append(tuple(back))
            size.append((w, w, 1.0))
            k = rng.uniform(7000, 10000) if rng.random() < cool else rng.uniform(*kelvin)
            col.append((*kelvin_rgb(k), 1.0))
            hell.append(rng.uniform(*layer.brightness))
    ob = _points_object(f"{name}_Bokeh", pos, nrm, size, col, hell, coll, bokeh_material())
    print(f"[nomiss_stadt] {name}: {len(pos)} Bokeh-Scheiben in {len(layers)} Ebenen", flush=True)
    return Stadt(None, ob, np.asarray(pos), np.asarray(hell))
