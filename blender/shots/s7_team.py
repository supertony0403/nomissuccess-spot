"""Shot s7_team - one team instead of a call centre, documentation, no black box.

Set pieces stand along +X on a dark glossy floor; the camera travels from one
to the next (frames from timeline.json at render time):

* ``converge``: four stand-ins of the night - shop window with the real
  website, a firewall wall of blue light panels, a backup cube and a
  dashboard card - spiral into one orbit; the logo band wraps them and
  cinches them into one bundle; the camera circles it,
* ``tickets_fall``: a house of cards made of 26 ticket slips (#48213 ...)
  collapses - Bullet rigid bodies, simulated once, baked to keyframes and
  frozen in ``blender/assets/s5_s7/tickets_bake.json`` (deterministic),
* ``direct_line``: the band shoots straight across the frame and leads the
  camera on,
* ``docs_fly``: documentation pages (runbook, access, configuration) fly
  into an open folder, which closes,
* ``blackbox_unfold``: a glossy black cube folds open into a net on the
  floor; what stays is an open wireframe cube with rosa (#e44b8d) edges.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib_szenen"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Quaternion, Vector  # noqa: E402

import nomiss_camera as C  # noqa: E402
import nomiss_material as M  # noqa: E402
import nomiss_ribbon as RB  # noqa: E402
import szenen_s5_s7 as S  # noqa: E402

SHOT = "s7_team"
EVENTS = ("converge", "tickets_fall", "direct_line", "docs_fly", "blackbox_unfold")
LENS = 50.0
K0 = Vector((0.0, 0.0, 0.78))        # bundle centre
T0 = Vector((2.6, 0.35, 0.0))        # ticket tower base
F0 = Vector((5.4, 0.25, 0.0))        # folder
B0 = Vector((8.0, 0.35, 0.0))        # black box (on the floor)
CARD = (0.10, 0.15, 0.0022)          # ticket slip: width (y), height, thickness
LEAN = math.radians(17.0)
SIM_RELEASE = 10                     # sim frame where the tower is released
SIM_FRAMES = 240


# ==========================================================================
# ticket tower (house of cards) - pure spec for the rigid-body bake
# ==========================================================================

def tower_spec() -> list[dict]:
    """House of cards: 4 levels of A-frames plus flat slips (26 slips)."""
    w, h, t = CARD
    rng = np.random.default_rng(48213)
    bodies = []
    z = 0.0
    rise = h * math.cos(LEAN)
    spacing = 2 * h * math.sin(LEAN) + 0.012
    for level in range(4):
        n = 4 - level
        xs = [(j - (n - 1) / 2) * spacing for j in range(n)]
        for j, xc in enumerate(xs):
            for side in (-1, 1):
                ang = side * LEAN  # top leans towards the A-frame centre
                cx = xc + side * (0.5 * h * math.sin(LEAN) + t * 0.6)
                cz = z + 0.5 * rise + 0.0004
                jitter = rng.normal(0, 0.0004, 3)
                Mx = (Matrix.Translation(T0 + Vector((cx + jitter[0], jitter[1], cz)))
                      @ Matrix.Rotation(-ang, 4, "Y") @ Matrix.Rotation(math.radians(-90.0), 4, "Y"))  # text top (+x) up
                bodies.append({"name": f"L{level}F{j}{'l' if side < 0 else 'r'}", "size": (h, w, t), "matrix": [list(r) for r in Mx]})
        z += rise + t + 0.001
        for j in range(n - 1):
            xm = (xs[j] + xs[j + 1]) / 2
            Mx = Matrix.Translation(T0 + Vector((xm, 0.0, z - t / 2 - 0.0002))) @ Matrix.Rotation(rng.normal(0, 0.01), 4, "Z")
            bodies.append({"name": f"L{level}D{j}", "size": (h, w, t), "matrix": [list(r) for r in Mx]})
        if n - 1 > 0:
            z += t + 0.0006
    # shove: the bottom-left slip is pulled out just before the release
    first = bodies[0]
    M0 = Matrix(first["matrix"])
    M1 = Matrix.Translation(Vector((-0.035, -0.012, 0.0))) @ M0
    first["kinematic"] = [(1, [list(r) for r in M0]), (SIM_RELEASE - 6, [list(r) for r in M0]), (SIM_RELEASE, [list(r) for r in M1])]
    first["release"] = SIM_RELEASE + 1
    for b in bodies:
        b["mass"] = 0.004
        b["friction"] = 0.6
        b["bounce"] = 0.05
        if "kinematic" not in b:
            b["kinematic"] = [(1, b["matrix"])]
            b["release"] = SIM_RELEASE
    return bodies


def tower_bake(*, write: bool = False, fresh: bool = False) -> dict:
    return S.rigid_bake(tower_spec(), frames=SIM_FRAMES, cache=S.TICKET_BAKE, write=write, fresh=fresh, seed_text="tickets-v1")


# ==========================================================================
# props
# ==========================================================================

def ticket_mesh(name: str, size, cell: int, mats, atlas=(6, 5)):
    """Thin slip box; both faces show atlas cell ``cell`` (x=height axis)."""
    hgt, wid, thk = size
    cols, rows = atlas
    cx, cy = cell % cols, cell // cols
    u0, u1 = cx / cols, (cx + 1) / cols
    v1, v0 = 1 - cy / rows, 1 - (cy + 1) / rows
    x, y, z = hgt / 2, wid / 2, thk / 2
    verts = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    # slip "up" (text top) points to +x; the image is portrait (v = height)
    faces = [(4, 5, 6, 7), (3, 2, 1, 0), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

    def uv(px, py):  # local (x along height, y along width) -> atlas
        return (u0 + (py / wid + 0.5) * (u1 - u0), v0 + (px / hgt + 0.5) * (v1 - v0))

    uvs = []
    for fi, f in enumerate(faces):
        for vi in f:
            vx, vy, _ = verts[vi]
            if fi == 0:
                uvs.append(uv(vx, vy))
            elif fi == 1:  # the underside is seen from the other side: mirror it back
                uvs.append(uv(vx, -vy))
            else:
                uvs.append((u0, v0))
    return S.mesh_object(name, verts, faces, uvs=uvs, mats=mats, mat_index=[0, 0, 1, 1, 1, 1])


def paper_flat(name: str, img, *, back=(0.93, 0.93, 0.92, 1.0)):
    """Matte paper with a printed front and a plain back."""
    mat, b = S.new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    uvn = b.n("ShaderNodeUVMap", uv_map="UVMap")
    tex = b.n("ShaderNodeTexImage", image=img, interpolation="Cubic")
    b.put(tex.inputs["Vector"], uvn.outputs["UV"])
    geo = b.n("ShaderNodeNewGeometry")
    col = b.mix(geo.outputs["Backfacing"], tex.outputs["Color"], back)
    b.put(bs.inputs["Base Color"], col)
    bs.inputs["Roughness"].default_value = 0.6
    bs.inputs["Specular IOR Level"].default_value = 0.3
    bs.inputs["Sheen Weight"].default_value = 0.2
    b.put(bs.inputs["Emission Color"], col)
    bs.inputs["Emission Strength"].default_value = 0.05
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    return mat


def firewall_wall(name: str, cols=3, rows=4, tw=0.095, th=0.068, gap=0.011):
    """Grid of blue light panels with a per-panel brightness attribute."""
    rng = np.random.default_rng(3)
    verts, faces, hell = [], [], []
    W = cols * tw + (cols - 1) * gap
    H = rows * th + (rows - 1) * gap
    for r in range(rows):
        for c in range(cols):
            x0 = -W / 2 + c * (tw + gap)
            y0 = -H / 2 + r * (th + gap)
            base = len(verts)
            verts += [(x0, y0, 0), (x0 + tw, y0, 0), (x0 + tw, y0 + th, 0), (x0, y0 + th, 0)]
            faces.append([base, base + 1, base + 2, base + 3])
            v = float(rng.uniform(0.55, 1.0))
            hell += [v] * 4
    mat, b = S.new_mat(f"{name}_Mat")
    out = b.n("ShaderNodeOutputMaterial")
    attr = b.n("ShaderNodeAttribute", attribute_name="hell", attribute_type="GEOMETRY")
    em = b.n("ShaderNodeEmission")
    blau = S.hex_lin(S.PRISMA["blau"])
    em.inputs["Color"].default_value = (blau[0] * 1.2, blau[1] * 1.25, blau[2] * 1.1, 1.0)
    b.put(em.inputs["Strength"], b.m("MULTIPLY", attr.outputs["Fac"], b.value("Glut", 3.2)))
    b.t.links.new(em.outputs[0], out.inputs["Surface"])
    ob = S.mesh_object(name, verts, faces, mats=[mat])
    a = ob.data.attributes.new("hell", "FLOAT", "POINT")
    a.data.foreach_set("value", np.asarray(hell, dtype=np.float32))
    frame = S.slab(name + "_Rahmen", W + 0.03, H + 0.03, 0.018, 0.008,
                   mats=[S.principled("WandRahmen", S.hex_lin("#0c0d16"), rough=0.35, metal=0.6)] * 3 + [S.edge_material()])
    frame.parent = ob
    frame.location = (0.0, 0.0, -0.011)
    return ob, (W, H)


def backup_cube(name: str, size=0.24):
    dark = S.principled("BackupKoerper", S.hex_lin("#12141f"), rough=0.22, metal=0.3, coat=0.8, coat_rough=0.04)
    ob = S.slab(name, size, size, size, 0.03, mats=[dark, S.edge_material(), dark, dark])
    mint = S.hex_lin(S.PRISMA["mint"])
    led = S.emission_mat("BackupStreifen", (mint[0] * 1.1, mint[1] * 1.1, mint[2] * 1.1, 1.0), 4.0)
    for i in range(3):
        z = (i - 1) * size * 0.24
        verts = [(-size * 0.32, z - 0.004, 0), (size * 0.32, z - 0.004, 0), (size * 0.32, z + 0.004, 0), (-size * 0.32, z + 0.004, 0)]
        st = S.mesh_object(f"{name}_Streifen{i}", verts, [[0, 1, 2, 3]], mats=[led])
        st.parent = ob
        st.location = (0.0, 0.0, size / 2 + 0.0008)
        st.visible_shadow = False
    return ob


def tube(name, p0, p1, radius, mat, ring=10):
    """Capped cylinder between two points."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    L = np.linalg.norm(d)
    t = d / L
    a = np.cross(t, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(t, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    bv = np.cross(t, a)
    verts, faces = [], []
    for end in (p0, p1):
        for k in range(ring):
            ang = 2 * math.pi * k / ring
            verts.append(tuple(end + radius * (math.cos(ang) * a + math.sin(ang) * bv)))
    for k in range(ring):
        faces.append([k, (k + 1) % ring, ring + (k + 1) % ring, ring + k])
    faces.append(list(reversed(range(ring))))
    faces.append([ring + k for k in range(ring)])
    return S.mesh_object(name, verts, faces, mats=[mat], smooth=True)


# ==========================================================================
# build
# ==========================================================================

def build(args) -> S.ShotCtx:
    ctx = S.start_shot(SHOT, args, EVENTS)
    scene, ev = ctx.scene, ctx.events
    n_frames = ctx.sh.frames
    f_conv, f_tick, f_line, f_docs, f_box = (ev[e] for e in EVENTS)

    S.stage_world(
        scene,
        [
            ((0.6, 1.0, 0.25), S.PRISMA["rosa"], 0.07, 30.0),
            ((-1.0, 0.7, 0.4), S.PRISMA["violett"], 0.09, 30.0),
            ((1.0, 0.3, -0.2), S.PRISMA["blau"], 0.08, 28.0),
            ((0.0, -1.0, 0.3), S.PRISMA["rosa"], 0.05, 34.0),
        ],
        zenith="#07070d",
        horizon="#131322",
    )
    S.floor("Boden", (4.0, 0.2, 0.0), (16.0, 9.0), fade_inner=1.6, fade_outer=3.3, rough=0.32)

    # ---- converge: four stand-ins + the band --------------------------------
    win = S.screenshot_card("Schaufenster", "webseite-eigen", 0.30, shadow=False)
    frame = S.slab("Schaufenster_Rahmen", 0.30 + 0.026, win.h + 0.026, 0.022, 0.010,
                   mats=[S.principled("RahmenMetall", S.hex_lin("#1a1b27"), rough=0.25, metal=0.85)] * 3 + [S.edge_material()])
    frame.parent = win.ob
    frame.location = (0.0, 0.0, -0.010)
    wall, _ = firewall_wall("FirewallWand", tw=0.080, th=0.058, gap=0.009)
    cube = backup_cube("BackupWuerfel", 0.18)
    W, H = S.SIZES["grafana-monitoring"]
    dash = S.screenshot_card("DashboardKarte", "grafana-monitoring", 0.30, shadow=False,
                             uv_rect=(24 / W, 1 - 600 / H, 1584 / W, 1 - 38 / H))
    members = [win.ob, wall, cube, dash.ob]
    # bundle frame: local -Y faces the camera at the converge event
    slot_pos = [Vector((-0.125, -0.07, 0.07)), Vector((0.0, 0.10, 0.12)), Vector((0.01, -0.15, -0.11)), Vector((0.13, -0.05, 0.015))]
    slot_yaw = [math.radians(14.0), 0.0, math.radians(32.0), math.radians(-14.0)]
    starts = [(-140.0, 2.3, 0.45), (-70.0, 2.5, 1.0), (60.0, 2.6, -0.35), (130.0, 2.4, 0.7)]  # angle offset, radius, height
    psi0 = math.radians(-6.0)

    def bundle_yaw(f):
        return psi0 + math.radians(24.0) * S.span(f, f_conv, f_conv + 240, "in_out_sine")

    frames_c = list(range(1, f_conv + 260))
    for ob, pl, yl, (a_off, r_far, h_far), i in zip(members, slot_pos, slot_yaw, starts, range(4)):
        arrive = f_conv - 6 + 3 * i
        mats = []
        for f in frames_c:
            s = S.span(f, 1, arrive, "out_cubic")
            Rb = Matrix.Rotation(bundle_yaw(f), 4, "Z")
            final = Matrix.Translation(K0) @ Rb @ Matrix.Translation(pl) @ Matrix.Rotation(yl, 4, "Z") @ Matrix.Rotation(math.radians(90.0), 4, "X")
            v = final.translation - K0
            rh = Vector((v.x, v.y, 0.0))
            r_slot = rh.length
            ang = math.atan2(rh.y, rh.x) + math.radians(a_off) * (1 - s)
            r = r_slot + (r_far - r_slot) * (1 - s)
            hh = v.z + (h_far - v.z) * (1 - S.span(f, 1, arrive, "out_quint"))
            pos = K0 + Vector((r * math.cos(ang), r * math.sin(ang), hh))
            tumble = Quaternion(Vector((0.3, 1.0, 0.2)).normalized(), 2.4 * (1 - s) ** 2)
            q = tumble @ final.to_quaternion()
            mats.append(Matrix.Translation(pos) @ q.to_matrix().to_4x4())
        S.bake_transform(ob, frames_c, mats)

    def orbit_angle(f):
        return bundle_yaw(f) - psi0

    bmat = M.ribbon_material("BandS7", emission=0.75, edge=1.8, head=5.0)
    pts = np.stack([np.zeros(400), np.linspace(0, 1, 400), np.zeros(400)], axis=1)
    band = RB.build(pts, 0.052, 0.004, name="TeamBand", material=bmat)
    RB.set_input(band, "Morph", 0.0)
    RB.set_input(band, "Spirale Drehung", (math.radians(90.0), 0.0, 0.0))
    RB.set_input(band, "Spirale Windungen", 1.55)
    RB.set_input(band, "Spirale Laenge", 0.24)
    RB.set_input(band, "Spirale Konus", 0.15)
    RB.set_input(band, "Drall Phase", math.pi / 2)
    RB.set_input(band, "Spirale Mitte", tuple(K0 + Vector((0, 0, 0.01))))
    RB.set_input(band, "Kopf Laenge", 0.12)
    bf = list(range(60, f_conv + 260, 2))
    RB.bake_input(band, "Ende", bf, lambda f: S.span(f, 70, f_conv - 4, "out_cubic"))
    RB.bake_input(band, "Spirale Radius", bf, lambda f: 1.1 + (0.33 - 1.1) * S.span(f, 70, f_conv, "out_quint"))
    RB.bake_input(band, "Spirale Phase", bf, lambda f: -2.6 * (1 - S.span(f, 70, f_conv, "out_cubic")) + orbit_angle(f))
    RB.bake_input(band, "Kopf Laenge", bf, lambda f: 0.12 * (1 - S.span(f, f_conv - 10, f_conv + 10)))
    band.hide_render = True
    band.keyframe_insert("hide_render", frame=70)
    band.hide_render = False
    band.keyframe_insert("hide_render", frame=71)
    M.key_node_value(bmat, "Leuchten", f_conv - 2, 1.0)
    M.key_node_value(bmat, "Leuchten", f_conv + 4, 2.2)
    M.key_node_value(bmat, "Leuchten", f_conv + 40, 1.0)

    # ---- tickets: baked rigid-body collapse ----------------------------------
    data = tower_bake(write=args.bake)
    spec = tower_spec()
    atlas = S.load_image(S.ASSETS / "tickets.png")
    tmat = paper_flat("Ticket_Mat", atlas)
    edge_p = S.principled("TicketKante", (0.85, 0.84, 0.8, 1.0), rough=0.7)
    off = f_tick - SIM_RELEASE - 3            # shot frame = sim frame + off
    mats_all = np.asarray(data["matrizen"])
    tickets = []
    for i, bd in enumerate(spec):
        ob = ticket_mesh(f"Ticket_{bd['name']}", bd["size"], i % 30, [tmat, edge_p])
        sim_frames = list(range(1, SIM_FRAMES + 1))
        frames_t = [1] + [s + off for s in sim_frames if s + off > 1]
        mm = [mats_all[i, 0]] + [mats_all[i, s - 1] for s in sim_frames if s + off > 1]
        S.bake_transform(ob, frames_t, mm)
        tickets.append(ob)

    # ---- the direct line -------------------------------------------------------
    lmat = M.ribbon_material("BandLinie", emission=1.2, edge=2.2, head=8.0)
    p_a = T0 + Vector((-1.9, -0.45, 0.24))
    p_b = F0 + Vector((-0.2, -0.20, 0.22))
    lpts = S.bezier(tuple(p_a), tuple(p_a + Vector((1.4, 0.02, 0.06))), tuple(p_b + Vector((-1.6, 0.0, 0.10))), tuple(p_b), 240)
    line = RB.build(lpts, 0.014, 0.0025, name="DirekteLinie", material=lmat)
    RB.set_input(line, "Kopf Laenge", 0.06)
    lf = list(range(f_line - 26, f_line + 1))
    RB.bake_input(line, "Ende", lf, lambda f: S.span(f, f_line - 24, f_line, "in_out_sine"))
    RB.key_input(line, "Start", f_docs - 120, 0.0, interp="SINE", easing="EASE_IN_OUT")
    RB.key_input(line, "Start", f_docs - 40, 1.0)
    line.hide_render = True
    line.keyframe_insert("hide_render", frame=f_line - 25)
    line.hide_render = False
    line.keyframe_insert("hide_render", frame=f_line - 24)

    # ---- documentation folder -----------------------------------------------
    folder_yaw = math.radians(10.0)
    navy = S.principled("OrdnerDeckel", S.hex_lin("#1a1c2c"), rough=0.4, metal=0.1, coat=0.5, coat_rough=0.1)
    rosa = S.hex_lin(S.PRISMA["rosa"])
    rosa_m = S.principled("OrdnerRosa", rosa, rough=0.35, emit=rosa, emit_strength=0.6)
    fw, fh, ft = 0.25, 0.33, 0.008
    base_ob = S.slab("Ordner_Boden", fw, fh, ft, 0.008, mats=[navy, rosa_m, navy, rosa_m])
    base_ob.location = F0 + Vector((0, 0, ft / 2))
    base_ob.rotation_euler = (0, 0, folder_yaw)
    spine = S.empty("Ordner_Ruecken", (0, 0, 0))
    spine.parent = base_ob
    hinge_up = 0.0045  # the hinge sits half a page stack above the base, so the
    spine.location = (-fw / 2, 0.0, ft / 2 + hinge_up)  # closed cover clears the pages
    cover = S.slab("Ordner_Deckel", fw, fh, ft, 0.008, mats=[navy, rosa_m, navy, rosa_m])
    cover.parent = spine
    cover.location = (-fw / 2 - 0.004, 0.0, -ft / 2 - hinge_up)
    C.key(spine, "rotation_euler", 1, (0.0, 0.0, 0.0), interp="SINE", easing="EASE_IN_OUT")
    C.key(spine, "rotation_euler", f_docs + 4, (0.0, 0.0, 0.0), interp="SINE", easing="EASE_IN_OUT")
    C.key(spine, "rotation_euler", f_docs + 40, (0.0, math.radians(178.0), 0.0))
    tab = S.slab("Ordner_Reiter", 0.05, 0.022, 0.003, 0.004, mats=[rosa_m] * 4)
    tab.parent = cover
    tab.location = (-fw / 2 + 0.04, fh / 2 + 0.009, 0.0)

    pages = []
    docs = [("runbook", -0.9), ("zugaenge", -0.6), ("konfiguration", -0.3), ("runbook", 0.1), ("konfiguration", 0.4)]
    pw, ph = 0.21, 0.297
    gx, gy = np.meshgrid(np.linspace(-pw / 2, pw / 2, 7), np.linspace(-ph / 2, ph / 2, 9))
    pgrid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    rng = np.random.default_rng(12)
    folder_M = Matrix.Translation(base_ob.location) @ Matrix.Rotation(folder_yaw, 4, "Z")
    for k, (key_k, lat) in enumerate(docs):
        img = S.load_image(S.ASSETS / f"doku_{key_k}.png")
        pm = paper_flat(f"Doku_{k}", img)
        pg = S.grid_mesh(f"Seite_{k}", pgrid, 7, 9, mats=[pm])
        pg.shape_key_add(name="Basis", from_mix=False)
        curl = pg.shape_key_add(name="Wellung", from_mix=False)
        cz = 0.02 * (pgrid[:, 0] / (pw / 2)) ** 2 + 0.012 * (pgrid[:, 1] / (ph / 2))
        curl.data.foreach_set("co", np.stack([pgrid[:, 0], pgrid[:, 1], cz], axis=1).astype(np.float32).ravel())
        land = f_docs - (len(docs) - 1 - k) * 6
        f0 = land - 34
        curl.value = 1.0
        curl.keyframe_insert("value", frame=f0)
        curl.value = 0.0
        curl.keyframe_insert("value", frame=land)
        home = folder_M @ Matrix.Translation(Vector((0.0, 0.0, ft + 0.0015 * (k + 1))))
        start = folder_M @ Matrix.Translation(Vector((lat - 0.6, -0.9 + 0.1 * k, 0.9 + 0.08 * k)))
        q0 = Quaternion(Vector(rng.normal(size=3)).normalized(), rng.uniform(0.8, 1.6)) @ start.to_quaternion()
        q1 = home.to_quaternion()
        ctrl = (start.translation + home.translation) * 0.5 + Vector((0, 0, 0.45))
        fr_k = [1] + list(range(f0, land + 1))
        mm = []
        for f in fr_k:
            s = S.span(f, f0, land, "out_cubic")
            p = (1 - s) ** 2 * start.translation + 2 * (1 - s) * s * ctrl + s * s * home.translation
            q = q0.slerp(q1, S.span(f, f0, land, "smoother"))
            mm.append(Matrix.Translation(p) @ q.to_matrix().to_4x4())
        S.bake_transform(pg, fr_k, mm)
        pg.hide_render = True
        pg.keyframe_insert("hide_render", frame=f0 - 1)
        pg.hide_render = False
        pg.keyframe_insert("hide_render", frame=f0)
        pages.append(pg)

    # ---- the black box ----------------------------------------------------------
    size = 0.42
    black = S.principled("PianoSchwarz", S.hex_lin("#050508"), rough=0.08, metal=0.0, spec=0.75, coat=1.0, coat_rough=0.02)
    hinge_defs = [  # (name, hinge location, panel offset from hinge, open rotation axis/angle, start frame)
        ("vorne", Vector((0, -size / 2, 0)), Vector((0, 0, size / 2)), ("X", math.radians(90.0)), f_box + 4),
        ("hinten", Vector((0, size / 2, 0)), Vector((0, 0, size / 2)), ("X", -math.radians(90.0)), f_box + 10),
        ("links", Vector((-size / 2, 0, 0)), Vector((0, 0, size / 2)), ("Y", -math.radians(90.0)), f_box + 7),
        ("rechts", Vector((size / 2, 0, 0)), Vector((0, 0, size / 2)), ("Y", math.radians(90.0)), f_box + 6),
        ("oben", Vector((0, size / 2, size)), Vector((0, -size / 2, 0)), ("X", -math.radians(125.0)), f_box),
    ]
    panels = []
    for name, hinge, offset, (axis, ang), f0 in hinge_defs:
        piv = S.empty(f"Box_Gelenk_{name}", tuple(B0 + hinge))
        if name in ("vorne", "hinten"):
            pnl = S.slab(f"Box_{name}", size, size, 0.012, 0.01, mats=[black] * 4)
            pnl.rotation_euler = (math.radians(90.0), 0, 0)
        elif name in ("links", "rechts"):
            pnl = S.slab(f"Box_{name}", size, size, 0.012, 0.01, mats=[black] * 4)
            pnl.rotation_euler = (0, math.radians(90.0), 0)
        else:
            pnl = S.slab(f"Box_{name}", size, size, 0.012, 0.01, mats=[black] * 4)
        pnl.parent = piv
        pnl.location = offset
        rot0 = (0.0, 0.0, 0.0)
        rot1 = (ang, 0.0, 0.0) if axis == "X" else (0.0, ang, 0.0)
        C.key(piv, "rotation_euler", f0, rot0, interp="QUINT", easing="EASE_IN_OUT")
        C.key(piv, "rotation_euler", f0 + 32, rot1)
        panels.append(pnl)
    base_p = S.slab("Box_unten", size, size, 0.012, 0.01, mats=[black] * 4)
    base_p.location = B0 + Vector((0, 0, 0.006))
    wire_m = S.emission_mat("DrahtRosa", (rosa[0] * 1.15, rosa[1] * 1.15, rosa[2] * 1.15, 1.0), 0.0, value_name="Glut")
    S.key_value(wire_m, "Glut", f_box - 4, 1.8, "SINE", "EASE_IN_OUT")
    S.key_value(wire_m, "Glut", f_box + 26, 7.0, "SINE", "EASE_IN_OUT")
    S.key_value(wire_m, "Glut", f_box + 70, 4.5)
    corners = [Vector((x, y, z)) for x in (-1, 1) for y in (-1, 1) for z in (0, 1)]
    wire = []
    r_w = 0.0055
    for i, a in enumerate(corners):
        for j in range(i + 1, len(corners)):
            b2 = corners[j]
            if sum(1 for k in range(3) if a[k] != b2[k]) != 1:
                continue  # only the 12 cube edges
            pa = B0 + Vector((a.x * size / 2, a.y * size / 2, a.z * size))
            pb = B0 + Vector((b2.x * size / 2, b2.y * size / 2, b2.z * size))
            wire.append(tube(f"Draht_{i}{j}", tuple(pa), tuple(pb), r_w, wire_m))
    glow = S.point_light("BoxInnen", tuple(B0 + Vector((0, 0, size * 0.55))), 0.0, (1.0, 0.45, 0.65), radius=0.08)
    glow.data.specular_factor = 0.0
    C.key(glow.data, "energy", f_box, 0.0, interp="SINE", easing="EASE_IN_OUT")
    C.key(glow.data, "energy", f_box + 30, 6.0)

    # ---- lights -------------------------------------------------------------------
    key = S.area_light("Key", (-1.5, -2.6, 2.6), (2.0, 0.3, 0.4), 3.0, 45.0, M.kelvin_rgb(5600))
    rim = S.area_light("Rim", (5.0, 3.5, 2.0), (4.0, 0.3, 0.4), 3.0, 60.0, M.kelvin_rgb(8200))
    for big in (key, rim):  # no light boxes mirrored in the glossy floor
        big.data.specular_factor = 0.08
    for nm, loc, aim, sz, en in (
        ("BuendelLicht", K0 + Vector((-0.9, -1.2, 0.9)), K0, 0.8, 8.0),
        ("TicketLicht", T0 + Vector((-0.6, -1.0, 1.1)), T0 + Vector((0, 0, 0.25)), 0.9, 9.0),
        ("OrdnerLicht", F0 + Vector((-0.5, -0.9, 1.2)), F0, 0.9, 8.0),
    ):
        lt = S.area_light(nm, tuple(loc), tuple(aim), sz, en, M.kelvin_rgb(5400))
        lt.data.specular_factor = 0.25
    box_key = S.area_light("BoxLicht", tuple(B0 + Vector((-1.1, -1.0, 1.3))), tuple(B0 + Vector((0, 0, 0.2))), 1.6, 16.0, M.kelvin_rgb(6500))
    box_key.data.shape = "RECTANGLE"
    box_key.data.size_y = 0.25
    box_key.data.specular_factor = 0.3
    strip = S.area_light("BoxGlanz", tuple(B0 + Vector((0.25, 1.7, 1.35))), tuple(B0 + Vector((0, 0, 0.42))), 1.4, 80.0, M.kelvin_rgb(7000), shape="RECTANGLE", size_y=0.07)
    side = S.area_light("BoxSeite", tuple(B0 + Vector((-1.3, 1.2, 0.25))), tuple(B0 + Vector((0, 0, 0.21))), 0.08, 40.0, M.kelvin_rgb(7500), shape="RECTANGLE", size_y=1.2)
    side.data.diffuse_factor = 0.0
    strip.data.diffuse_factor = 0.0  # only the glossy streak on the piano-black box
    # light linking: the box lights reach only the box, not the glossy floor
    recv = bpy.data.collections.new("BoxEmpfaenger")
    for ob in panels + [base_p]:
        recv.objects.link(ob)
    for lt in (strip, box_key, side):
        lt.light_linking.receiver_collection = recv
    S.dust("Staub", 40, (-1.5, -1.6, 0.1), (9.5, 1.8, 1.8), seed=77, size=(0.003, 0.007), strength=0.8, color=(1.0, 0.8, 0.9, 1.0))

    # ---- camera ---------------------------------------------------------------------
    def orbit(deg, dist, h):
        a = math.radians(deg)
        return K0 + Vector((dist * math.cos(a), dist * math.sin(a), h))

    cam_keys = [
        (1, orbit(-118.0, 2.35, 0.32)),
        (f_conv, orbit(-96.0, 2.05, 0.22)),
        (f_conv + 150, orbit(-62.0, 2.10, 0.30)),
        (f_tick - 22, T0 + Vector((-0.40, -1.90, 0.52))),
        (f_tick + 6, T0 + Vector((-0.38, -1.86, 0.52))),    # hold the tower through the event
        (f_tick + 70, T0 + Vector((-0.28, -1.30, 1.00))),   # then look down on the slip pile
        (f_line - 30, T0 + Vector((-0.10, -1.55, 0.52))),
        (f_line + 40, T0 + Vector((0.95, -1.75, 0.62))),
        (f_docs - 110, F0 + Vector((-0.26, -1.08, 0.96))),
        (f_docs + 30, F0 + Vector((-0.16, -0.98, 0.90))),
        (f_box - 30, B0 + Vector((-0.55, -1.75, 0.80))),
        (f_box + 40, B0 + Vector((-0.30, -1.95, 0.95))),
        (n_frames, B0 + Vector((0.30, -1.95, 1.00))),
    ]
    tgt_keys = [
        (1, K0 + Vector((0.0, 0.0, -0.02))),
        (f_conv, K0 + Vector((0.0, 0.0, -0.01))),
        (f_conv + 150, K0 + Vector((0.02, 0.0, 0.0))),
        (f_tick - 22, T0 + Vector((0.02, 0.0, 0.27))),
        (f_tick + 6, T0 + Vector((0.02, 0.0, 0.26))),
        (f_tick + 70, T0 + Vector((0.05, 0.05, 0.02))),
        (f_line - 30, T0 + Vector((0.08, 0.0, 0.16))),
        (f_line + 40, T0 + Vector((1.25, 0.0, 0.25))),
        (f_docs - 110, F0 + Vector((0.0, 0.05, 0.05))),
        (f_docs + 30, F0 + Vector((0.0, 0.05, 0.04))),
        (f_box - 30, B0 + Vector((0.0, 0.0, 0.24))),
        (f_box + 40, B0 + Vector((0.0, 0.0, 0.20))),
        (n_frames, B0 + Vector((0.0, 0.0, 0.22))),
    ]
    foc_keys = [
        (1, K0),
        (f_conv + 150, K0),
        (f_tick - 22, T0 + Vector((0, 0, 0.2))),
        (f_line - 30, T0 + Vector((0, 0, 0.1))),
        (f_line + 40, T0 + Vector((1.3, -0.3, 0.3))),
        (f_docs - 110, F0),
        (f_docs + 30, F0),
        (f_box - 30, B0 + Vector((0, 0, 0.2))),
        (n_frames, B0 + Vector((0, 0, 0.2))),
    ]
    cam, tgt, foc = C.rig("Kamera7", tuple(cam_keys[0][1]), tuple(tgt_keys[0][1]), lens=LENS, fstop=2.0)
    frames = list(range(1, n_frames + 1))
    S.bake_fast(cam, "location", frames, [tuple(S.smooth_path(f, cam_keys)) for f in frames])
    S.bake_fast(tgt, "location", frames, [tuple(S.smooth_path(f, tgt_keys)) for f in frames])
    S.bake_fast(foc, "location", frames, [tuple(S.smooth_path(f, foc_keys)) for f in frames])
    C.key(cam.data.dof, "aperture_fstop", 1, 2.0)
    C.key(cam.data.dof, "aperture_fstop", f_tick - 22, 2.4)
    C.key(cam.data.dof, "aperture_fstop", f_box - 30, 2.8)

    ctx.key("converge", *members)
    ctx.key("tickets_fall", *tickets)
    p_mid, _ = S.polyline_at(lpts, 0.55 * S.arc_lengths(lpts)[-1])
    ctx.key("direct_line", S.empty("Linie_Punkt", tuple(p_mid)))
    ctx.key("docs_fly", base_ob)
    ctx.key("blackbox_unfold", base_p, *panels)
    ctx.extra["band"] = band

    def probe_extra(c):
        """Box walls open outwards; standing slips are not upside down."""
        c.scene.frame_set(1)
        upright = []
        for ob, bd in zip(tickets, spec):
            if "F" in bd["name"]:  # A-frame slips stand; local +x is the text top
                upright.append(ob.matrix_world.to_3x3().col[0].z > 0.5)
        c.scene.frame_set(f_box + 60)
        outside = {}
        for pnl in panels:
            ctr = pnl.matrix_world.translation - B0
            outside[pnl.name] = bool(max(abs(ctr.x), abs(ctr.y)) > size / 2 + 0.02 or ctr.z > size + 0.02)
        return {"zettel_aufrecht": float(np.mean(upright)), "box_aussen": outside}

    ctx.extra["probe"] = probe_extra
    print(f"[s7] frames={n_frames} conv={f_conv} tick={f_tick} line={f_line} docs={f_docs} box={f_box} bake={data['hash']}", flush=True)
    return ctx


def main() -> None:
    args = S.parse_args()
    if args.encode:
        S.encode(SHOT)
        return
    if args.bake_check:
        import nomiss_render as R

        R.clean_scene()
        data = tower_bake(fresh=True)
        Path(args.bake_check).write_text(json.dumps(data) + "\n", encoding="utf-8")
        print(f"[s7] Bake-Pruefung -> {args.bake_check} ({data['hash']})", flush=True)
        return
    ctx = build(args)
    S.run(ctx, args)


if __name__ == "__main__":
    main()
