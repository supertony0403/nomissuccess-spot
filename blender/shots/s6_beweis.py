"""Shot s6_beweis - "Klingt nach Prospekt? Ist es nicht." -> real monitoring.

Beats (frames from timeline.json at render time):

* ``prospect_paper``: a glossy tri-fold brochure hovers; a gust catches it
  and it flutters away (hinged panels flap, the sheet bends in a travelling
  wave, it tumbles out of frame),
* ``grafana_flyover``: the real Grafana screenshot (3360 px) lies in space
  like a landscape; the camera flies low over it (view ~60 deg from the
  normal, f/1.4) while the curves draw themselves: an animated mask reveals
  the coloured series left to right behind a glowing leading edge, the
  panel chrome is already there,
* ``stamp``: +-1 s around the event the camera settles into a calm, readable
  high view (the stamp itself is a Fusion layer),
* ``fan_panels``: the real PBS and Proxmox screenshots slide out from under
  the landscape and fan up beside it.
"""

from __future__ import annotations

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
import szenen_s5_s7 as S  # noqa: E402

SHOT = "s6_beweis"
EVENTS = ("prospect_paper", "grafana_flyover", "stamp", "fan_panels")
LENS = 35.0
GW = 6.0                                  # landscape width (m)
BRO = Vector((-1.55, -0.95, 0.95))        # brochure hover position


def look(normal: Vector, up: Vector = Vector((0, 0, 1))) -> Matrix:
    """Rotation whose local +Z points along ``normal`` and +Y towards ``up``."""
    n = normal.normalized()
    u = (up - n * up.dot(n)).normalized()
    r = u.cross(n)
    return Matrix((tuple(r), tuple(u), tuple(n))).transposed().to_4x4()


def reveal_material(name: str, image, uv_rect):
    """Screenshot face whose coloured series appear behind a moving edge.

    Value nodes: ``Front`` (0..1 across the image), ``Kante`` (edge glow),
    ``Hell`` (overall brightness).  Unrevealed areas keep the dim panel
    chrome but lose every saturated pixel (the curves).
    """
    mat, b = S.new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    uvn = b.n("ShaderNodeUVMap", uv_map="UVMap")
    tex = b.n("ShaderNodeTexImage", image=image, interpolation="Cubic", extension="EXTEND")
    b.put(tex.inputs["Vector"], uvn.outputs["UV"])
    col = tex.outputs["Color"]
    hsv = b.n("ShaderNodeSeparateColor", mode="HSV")
    b.put(hsv.inputs[0], col)
    sat, val = hsv.outputs[1], hsv.outputs[2]
    # thresholds in scene-linear values (the sRGB texture is linearised)
    curve = b.m("MULTIPLY", b.smooth(sat, 0.30, 0.55), b.smooth(val, 0.012, 0.045))
    dark = (0.012, 0.013, 0.017, 1.0)
    hidden = b.scale_color(b.mix(curve, col, dark), 0.62)
    u = b.sep(uvn.outputs["UV"])[0]
    uu = b.m("DIVIDE", b.m("SUBTRACT", u, uv_rect[0]), uv_rect[2] - uv_rect[0])
    front = b.value("Front", 0.0)
    shown = b.m("SUBTRACT", 1.0, b.smooth(uu, b.m("SUBTRACT", front, 0.004), b.m("ADD", front, 0.002)))
    colr = b.mix(shown, hidden, col)
    q = b.m("DIVIDE", b.m("SUBTRACT", uu, front), 0.0028)
    edge = b.m("EXPONENT", b.m("MULTIPLY", b.m("MULTIPLY", q, q), -1.0))
    edge = b.m("MULTIPLY", edge, b.m("ADD", 0.06, b.m("MULTIPLY", curve, 1.1)))
    edge = b.m("MULTIPLY", edge, b.value("Kante", 0.0))
    mint = S.hex_lin(S.PRISMA["mint"])
    glow = b.scale_color((0.35 + mint[0], 0.55 + mint[1], 0.45 + mint[2]), edge)
    hell = b.value("Hell", 1.0)
    emis = b.scale_color(b.add_color(colr, glow), hell)
    b.put(bs.inputs["Base Color"], b.scale_color(colr, 0.12))
    bs.inputs["Roughness"].default_value = 0.3
    bs.inputs["Specular IOR Level"].default_value = 0.4
    bs.inputs["Coat Weight"].default_value = 0.12
    bs.inputs["Coat Roughness"].default_value = 0.05
    b.put(bs.inputs["Emission Color"], emis)
    bs.inputs["Emission Strength"].default_value = 1.0
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    return mat


def brochure_positions(grid: np.ndarray, a_l: float, a_r: float, wave: float, phase: float, bend: float) -> np.ndarray:
    """Tri-fold brochure: side panels hinged at x = +-0.05, plus a bending wave."""
    x, y = grid[:, 0], grid[:, 1]
    P = np.zeros((len(grid), 3))
    P[:, 0], P[:, 1] = x, y
    left, right = x < -0.05, x > 0.05
    d = -0.05 - x[left]
    P[left, 0] = -0.05 - d * math.cos(a_l)
    P[left, 2] = d * math.sin(a_l)
    d = x[right] - 0.05
    P[right, 0] = 0.05 + d * math.cos(a_r)
    P[right, 2] = d * math.sin(a_r)
    P[:, 2] += wave * np.sin(9.0 * y + 6.0 * x + phase) * (0.3 + np.abs(x) / 0.15) + bend * (y / 0.105) ** 2
    return P


def build(args) -> S.ShotCtx:
    ctx = S.start_shot(SHOT, args, EVENTS)
    scene, ev = ctx.scene, ctx.events
    n_frames = ctx.sh.frames
    f_pro, f_fly, f_stamp, f_fan = (ev[e] for e in EVENTS)
    f_rev0, f_rev1 = f_fly - 40, f_fly + 160   # edge already running at the event, done before the stamp window

    S.stage_world(
        scene,
        [
            ((0.2, 1.0, 0.05), S.PRISMA["mint"], 0.11, 32.0),
            ((-1.0, 0.6, 0.30), S.PRISMA["blau"], 0.14, 36.0),
            ((1.0, 0.4, 0.45), S.PRISMA["violett"], 0.10, 30.0),
        ],
        zenith="#06070c",
        horizon="#10121f",
    )

    # ---- the landscape: real Grafana screenshot -----------------------------
    key = "grafana-dashboard"
    uv = S.screenshot_uv(key)
    x0, y0, x1, y1 = S.SCREENSHOTS[key]["crop"]
    GH = GW * (y1 - y0) / (x1 - x0)
    img = S.load_image(S.SCREENSHOTS[key]["datei"])
    rmat = reveal_material("Grafana_Landschaft", img, uv)
    land = S.slab("Grafana", GW, GH, 0.02, GW * 16 / 1920, uv_rect=uv, rim=GW * 0.0012,
                  mats=[rmat, S.edge_material(), S.back_material(), S.rim_material("LandRand", strength=0.05)])
    land.location = (0.0, 0.0, -0.01)
    S.key_value(rmat, "Hell", 1, 0.55, "SINE", "EASE_IN_OUT")
    S.key_value(rmat, "Hell", f_rev0, 0.95)
    S.key_value(rmat, "Front", f_rev0, 0.0, "SINE", "EASE_IN_OUT")
    S.key_value(rmat, "Front", f_rev1, 1.0)
    S.key_value(rmat, "Kante", f_rev0 - 10, 0.0, "LINEAR")
    S.key_value(rmat, "Kante", f_rev0 + 6, 1.0, "LINEAR")
    S.key_value(rmat, "Kante", f_rev1 - 6, 1.0, "LINEAR")
    S.key_value(rmat, "Kante", f_rev1 + 14, 0.0, "LINEAR")

    # ---- the brochure ----------------------------------------------------------
    gx, gy = np.meshgrid(np.linspace(-0.15, 0.15, 61), np.linspace(-0.105, 0.105, 43))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    out_img = S.load_image(S.ASSETS / "prospekt_aussen.png")
    in_img = S.load_image(S.ASSETS / "prospekt_innen.png")
    pmat = brochure_material("Prospekt_Mat", out_img, in_img)
    bro = S.grid_mesh("Prospekt", grid, 61, 43, mats=[pmat])
    hover_frames = list(range(1, f_pro + 90))

    def shape(f):
        gust = S.span(f, f_pro - 4, f_pro + 10, "out_cubic")
        t = f / 60.0
        flap = gust * math.exp(-max(0.0, f - f_pro) / 70.0)
        a_l = math.radians(32 + 4 * math.sin(2 * math.pi * t * 0.35) + 70 * flap * math.sin(2 * math.pi * t * 4.2))
        a_r = math.radians(28 + 4 * math.sin(2 * math.pi * t * 0.35 + 1.3) + 65 * flap * math.sin(2 * math.pi * t * 4.2 + 2.1))
        wave = 0.004 + 0.022 * gust
        return brochure_positions(grid, a_l, a_r, wave, 2 * math.pi * t * (0.4 + 3.0 * gust), 0.006)

    S.bake_shape_frames(bro, hover_frames, [shape(f) for f in hover_frames], lead=0)

    cam_hover = BRO + Vector((0.30, -0.98, 0.22))
    face = (cam_hover - BRO).normalized()
    base_rot = look(face)
    fly_dir = Vector((-1.0, 0.25, 0.75)).normalized()
    mats = []
    for f in hover_frames:
        t = f / 60.0
        g = S.span(f, f_pro, f_pro + 70, "in_quad")
        drift = Vector((0.004 * math.sin(t * 1.1), 0.0, 0.006 * math.sin(t * 1.6)))
        pos = BRO + drift + fly_dir * (2.6 * g) + Vector((0.0, -0.25, 0.0)) * S.span(f, f_pro, f_pro + 30, "out_cubic")
        spin = Quaternion(Vector((0.35, 1.0, 0.25)).normalized(), 5.5 * g + 0.06 * math.sin(t * 0.9))
        tilt = Quaternion(Vector((1.0, 0.0, 0.0)), math.radians(-6.0 + 3.0 * math.sin(t * 0.7)))
        q = spin @ base_rot.to_quaternion() @ tilt
        mats.append(Matrix.Translation(pos) @ q.to_matrix().to_4x4())
    S.bake_transform(bro, hover_frames, mats)
    bro.hide_render = False
    bro.keyframe_insert("hide_render", frame=f_pro + 88)
    bro.hide_render = True
    bro.keyframe_insert("hide_render", frame=f_pro + 89)

    # ---- PBS + Proxmox cards fanning out --------------------------------------
    cam_end = Vector((1.85, -4.70, 3.20))
    tgt_end = Vector((2.72, 0.25, 0.25))
    cards = []
    finals = [
        ("proxmox-ui", Vector((2.32, -0.15, 0.80)), math.radians(-7.0), f_fan - 6),
        ("pbs-dashboard", Vector((3.08, 0.30, 1.10)), math.radians(6.0), f_fan),
    ]
    for k, (key_k, centre, roll, land_f) in enumerate(finals):
        card = S.screenshot_card(f"Faecher_{k}", key_k, 1.85)
        n = (cam_end - centre).normalized()
        R_end = look(n) @ Matrix.Rotation(roll, 4, "Z")
        start = Vector((GW / 2 - 0.85, 0.30 + 0.25 * k, -0.06))
        R_start = Matrix.Rotation(math.radians(-4.0 + 3 * k), 4, "Z")
        mid = Vector((GW / 2 + 0.55, 0.45 + 0.3 * k, 0.05))
        frames_k = [1] + list(range(land_f - 44, land_f + 1))
        mk = []
        for f in frames_k:
            s = S.span(f, land_f - 44, land_f, "out_quint")
            s_rot = S.span(f, land_f - 30, land_f, "smoother")
            p = Vector(_bez(start, mid, centre + Vector((0.25, -0.1, -0.35)), centre, s))
            q = R_start.to_quaternion().slerp(R_end.to_quaternion(), s_rot)
            mk.append(Matrix.Translation(p) @ q.to_matrix().to_4x4())
        S.bake_transform(card.ob, frames_k, mk)
        card.ob.hide_render = True
        card.ob.keyframe_insert("hide_render", frame=land_f - 45)
        card.ob.hide_render = False
        card.ob.keyframe_insert("hide_render", frame=land_f - 44)
        cards.append(card.ob)

    # ---- lights ----------------------------------------------------------------
    S.area_light("Key", (-3.0, -3.5, 3.0), (0.0, 0.0, 0.0), 3.0, 70.0, M.kelvin_rgb(5600))
    rim = S.area_light("Rim", (4.5, 3.5, 2.2), (1.5, 0.5, 0.3), 2.0, 60.0, M.kelvin_rgb(8000))
    rim.data.specular_factor = 0.0  # no light box mirrored in the landscape
    S.area_light("Prospektlicht", tuple(BRO + Vector((-0.5, -0.8, 0.6))), tuple(BRO), 0.6, 3.0, M.kelvin_rgb(5400))
    kl = S.area_light("Kartenlicht", (2.0, -2.0, 2.4), (3.0, 0.7, 0.6), 1.6, 26.0, M.kelvin_rgb(6200))
    kl.data.specular_factor = 0.15
    S.dust("Staub", 30, (-3.0, -3.0, 0.2), (3.5, 1.5, 1.6), seed=66, size=(0.003, 0.007), strength=0.8, color=(0.75, 0.95, 0.88, 1.0))

    # ---- camera ------------------------------------------------------------------
    yaw = math.radians(24.0)
    fwd = Vector((math.sin(yaw), math.cos(yaw), 0.0))

    def low(x, y, h):
        c = Vector((x, y, h))
        return c, c + fwd * (h / math.tan(math.radians(30.0))) - Vector((0, 0, h))

    c200, t200 = low(-2.70, -1.60, 0.50)
    c300, t300 = low(-1.30, -1.50, 0.55)
    c390, t390 = low(-0.20, -1.40, 0.62)
    cam_keys = [
        (1, cam_hover + Vector((0.04, 0.05, 0.0))),
        (f_pro, cam_hover),
        (f_pro + 55, cam_hover + Vector((0.2, -0.5, -0.05))),
        (f_fly, c200),
        (f_fly + 75, c300),
        (f_fly + 125, c390),
        (f_stamp - 62, Vector((0.25, -3.05, 2.15))),        # settled before the stamp window
        (f_stamp + 60, Vector((0.32, -3.12, 2.22))),
        (f_fan - 8, cam_end + Vector((-0.08, 0.10, -0.06))),
        (n_frames, cam_end + Vector((0.32, 0.30, -0.10))),
    ]
    tgt_keys = [
        (1, BRO + Vector((-0.05, 0.0, -0.035))),
        (f_pro, BRO + Vector((-0.05, 0.0, -0.035))),
        (f_pro + 55, Vector((-1.9, -0.2, 0.25))),
        (f_fly, t200),
        (f_fly + 75, t300),
        (f_fly + 125, t390),
        (f_stamp - 62, Vector((0.05, -0.15, 0.0))),
        (f_stamp + 60, Vector((0.08, -0.12, 0.0))),
        (f_fan - 8, tgt_end + Vector((-0.05, 0.05, 0.0))),
        (n_frames, tgt_end),
    ]
    foc_keys = [
        (1, BRO),
        (f_pro + 30, BRO + fly_dir * 0.3),
        (f_fly - 20, t200),
        (f_fly + 75, t300),
        (f_fly + 125, t390),
        (f_stamp - 62, Vector((0.05, -0.15, 0.0))),
        (f_stamp + 60, Vector((0.08, -0.12, 0.0))),
        (f_fan - 8, Vector((2.9, 0.7, 0.6))),
        (n_frames, Vector((2.95, 0.72, 0.65))),
    ]
    cam, tgt, foc = C.rig("Kamera6", tuple(cam_keys[0][1]), tuple(tgt_keys[0][1]), lens=LENS, fstop=2.0)
    frames = list(range(1, n_frames + 1))
    S.bake_fast(cam, "location", frames, [tuple(S.smooth_path(f, cam_keys)) for f in frames])
    S.bake_fast(tgt, "location", frames, [tuple(S.smooth_path(f, tgt_keys)) for f in frames])
    S.bake_fast(foc, "location", frames, [tuple(S.smooth_path(f, foc_keys)) for f in frames])
    C.key(cam.data.dof, "aperture_fstop", 1, 2.2)
    C.key(cam.data.dof, "aperture_fstop", f_fly - 30, 1.4)
    C.key(cam.data.dof, "aperture_fstop", f_fly + 125, 1.4)
    C.key(cam.data.dof, "aperture_fstop", f_stamp - 62, 4.0)
    C.key(cam.data.dof, "aperture_fstop", f_fan - 8, 3.2)

    # probe points live on the landscape, independent of the camera:
    # the reveal edge across the lower panel row, and the dashboard centre
    front_fc = [fc for fc in C.fcurves(rmat.node_tree) if fc.data_path == 'nodes["Front"].outputs[0].default_value'][0]
    p_fly = S.empty("Enthuellungskante", (0.0, 0.0, 0.0))
    S.bake_fast(p_fly, "location", frames, [(-GW / 2 + GW * front_fc.evaluate(f), -0.55, 0.0) for f in frames])
    p_stamp = S.empty("Dashboard_Mitte", (0.0, 0.25, 0.0))

    def probe_extra(c):
        """Largest screen motion (px/frame at 1920) of landscape points
        within +-1 s of the stamp."""
        pts = np.array([(x, y, 0.0) for x in (-1.5, 0.0, 1.5) for y in (-1.0, 0.0, 1.0)])
        prev, worst = None, 0.0
        for f in range(f_stamp - 60, f_stamp + 61):
            c.scene.frame_set(f)
            px = S.project_points(c.scene, pts)
            if prev is not None:
                worst = max(worst, float(np.nanmax(np.linalg.norm(px - prev, axis=1))))
            prev = px
        return {"ruhe_px_pro_bild": round(worst, 3)}

    ctx.extra["probe"] = probe_extra
    ctx.key("prospect_paper", bro)
    ctx.key("grafana_flyover", p_fly)
    ctx.key("stamp", p_stamp)
    ctx.key("fan_panels", *cards)
    print(f"[s6] frames={n_frames} prospekt={f_pro} flug={f_fly} stempel={f_stamp} faecher={f_fan} GH={GH:.2f}", flush=True)
    return ctx


def _bez(p0, p1, p2, p3, t):
    a, b, c, d = (np.asarray(v, dtype=np.float64) for v in (p0, p1, p2, p3))
    return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d


def brochure_material(name, out_img, in_img):
    """Glossy coated print; outside print on the front, inside on the back."""
    mat, b = S.new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    uvn = b.n("ShaderNodeUVMap", uv_map="UVMap")
    t1 = b.n("ShaderNodeTexImage", image=out_img, interpolation="Cubic")
    t2 = b.n("ShaderNodeTexImage", image=in_img, interpolation="Cubic")
    b.put(t1.inputs["Vector"], uvn.outputs["UV"])
    uvs = b.sep(uvn.outputs["UV"])
    b.put(t2.inputs["Vector"], b.comb(b.m("SUBTRACT", 1.0, uvs[0]), uvs[1], 0.0))  # inside seen from behind
    geo = b.n("ShaderNodeNewGeometry")
    col = b.mix(geo.outputs["Backfacing"], t1.outputs["Color"], t2.outputs["Color"])
    b.put(bs.inputs["Base Color"], col)
    bs.inputs["Roughness"].default_value = 0.22
    bs.inputs["Specular IOR Level"].default_value = 0.6
    bs.inputs["Coat Weight"].default_value = 1.0
    bs.inputs["Coat Roughness"].default_value = 0.03
    b.put(bs.inputs["Emission Color"], col)
    bs.inputs["Emission Strength"].default_value = 0.06
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    return mat


def main() -> None:
    args = S.parse_args()
    if args.encode:
        S.encode(SHOT)
        return
    ctx = build(args)
    S.run(ctx, args)


if __name__ == "__main__":
    main()
