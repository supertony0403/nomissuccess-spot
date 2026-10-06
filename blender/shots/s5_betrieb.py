"""Shot s5_betrieb - licence invoice -> paper plane -> Proxmox, rollback,
rolling update, dashboards.  One continuous camera move through one space.

Beats (all frames from timeline.json at render time):

* ``invoice_spin``: a slightly curled A4 invoice hovers; the amount's digit
  drums spin faster and faster into an unreadable vertical smear (shader
  integrated blur, never a real number),
* fold: the sheet folds itself into a dart plane (rigid fold kinematics,
  one shape key per frame) and turns towards its target,
* ``paper_plane``: it flies an arc (the logo band grows behind it as the
  migration path) and dives into the real Proxmox screenshot card; a mint
  ring ripples over the card,
* ``rollback_path``: a dashed, glowing return path draws itself back from
  the card to the launch point, where a ring lights up,
* ``rolling_update``: five container blocks on a host base are replaced one
  after another (old sinks into the base, new drops in), the status line
  below stays green the whole time,
* ``dashboards``: six panel tiles cut from the real Grafana screenshot fly
  in and lock into the dashboard; the Proxmox card dims behind it.
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

import json  # noqa: E402

import nomiss_camera as C  # noqa: E402
import nomiss_material as M  # noqa: E402
import nomiss_ribbon as RB  # noqa: E402
import szenen_s5_s7 as S  # noqa: E402

SHOT = "s5_betrieb"
EVENTS = ("invoice_spin", "paper_plane", "rollback_path", "rolling_update", "dashboards")
LENS = 50.0
SHEET_W, SHEET_H = 0.21, 0.297
GRID = (101, 143)

A = Vector((0.0, 0.0, 0.0))                 # launch point (invoice)
B = Vector((1.15, 1.30, 0.10))              # Proxmox card centre
YAW = math.radians(-25.0)                   # card faces front-left (towards A)
E_N = Vector((math.sin(YAW), -math.cos(YAW), 0.0))   # card normal (towards viewer)
E_X = Vector((math.cos(YAW), math.sin(YAW), 0.0))    # card right
E_Z = Vector((0.0, 0.0, 1.0))
CARD_W = 0.95
CONTACT_LOCAL = (-0.13, 0.05)

TILES = [  # grafana-monitoring.png panels (px): CPU, load, memory, gauge, disk io, disk space
    ("CPU", (24, 38, 808, 296)),
    ("Last", (816, 38, 1584, 296)),
    ("Speicher", (24, 342, 1196, 600)),
    ("Anzeige", (1204, 342, 1584, 600)),
    ("DiskIO", (24, 646, 808, 903)),
    ("Platten", (816, 646, 1584, 903)),
]
BOARD_W = 0.70
C_ROW = B + E_X * 1.55 + E_N * 0.30 + E_Z * -0.42      # host base with containers
D_CENTRE = C_ROW + E_Z * 0.56 + E_N * -0.06              # dashboard above the row


def frame_basis(yaw: float) -> Matrix:
    """Rotation of a card standing upright, front (+Z local) along E_N."""
    return (Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(math.radians(90.0), 4, "X"))


def build(args) -> S.ShotCtx:
    ctx = S.start_shot(SHOT, args, EVENTS)
    scene, ev = ctx.scene, ctx.events
    fps = ctx.sh.fps
    n_frames = ctx.sh.frames
    f_inv, f_plane, f_back, f_roll, f_dash = (ev[e] for e in EVENTS)
    f_fold0 = f_inv + 17
    f_fold1 = f_fold0 + 70
    f_launch = f_fold1 + 4

    # ---- stage (green register) -------------------------------------------
    S.stage_world(
        scene,
        [
            ((0.95, 0.80, -0.25), S.PRISMA["mint"], 0.13, 33.0),
            ((-0.95, 0.60, 0.50), S.PRISMA["violett"], 0.14, 38.0),
            ((0.15, 1.00, -1.05), S.PRISMA["blau"], 0.22, 36.0),
            ((0.70, 0.80, 0.95), S.PRISMA["rosa"], 0.07, 30.0),
            ((-0.6, -1.0, 0.2), S.PRISMA["mint"], 0.10, 45.0),
        ],
    )

    # ---- the invoice / paper plane ----------------------------------------
    plane = S.DartPlane(SHEET_W, SHEET_H)
    grid, (nx, ny) = plane.sheet_grid(*GRID)
    lay = json.loads((S.ASSETS / "rechnung_layout.json").read_text(encoding="utf-8"))
    front = S.load_image(S.ASSETS / "rechnung.png")
    strip = S.load_image(S.ASSETS / "ziffern.png")
    strip.colorspace_settings.name = "Non-Color"
    pmat = S.paper_material("Rechnung_Papier", front, digits=(strip, lay))
    paper = S.grid_mesh("Rechnung", grid, nx, ny, mats=[pmat])

    # curl (gentle bend while hovering), as its own shape key
    paper.shape_key_add(name="Basis", from_mix=False)
    curl = paper.shape_key_add(name="Wellung", from_mix=False)
    xn, yn = grid[:, 0] / (SHEET_W / 2), grid[:, 1] / (SHEET_H / 2)
    cz = 0.016 * xn ** 2 + 0.008 * yn * xn + 0.010 * np.clip(-yn - 0.3, 0, None) ** 2
    curl.data.foreach_set("co", np.stack([grid[:, 0], grid[:, 1], cz], axis=1).astype(np.float32).ravel())
    # the curl is gone before the first fold, otherwise the flap pokes through
    for f in list(range(1, f_fold0 - 24, 6)) + [f_fold0 - 24, f_fold0]:
        v = (0.85 + 0.15 * math.sin(2 * math.pi * f / 170.0)) * (1.0 - S.span(f, f_fold0 - 24, f_fold0))
        curl.value = v
        curl.keyframe_insert("value", frame=f)

    fold_frames = list(range(f_fold0 + 1, f_fold1 + 1))
    fold_pos = [plane.positions(grid, plane.angles_at((f - f_fold0) / (f_fold1 - f_fold0))) for f in fold_frames]
    S.bake_shape_frames(paper, fold_frames, fold_pos, lead=f_fold0)
    final = fold_pos[-1]
    fr = plane.flight_frame(final, grid)
    c_local = Vector(final.mean(axis=0))
    loc_basis = Matrix((tuple(fr["rechts"]), tuple(fr["vor"]), tuple(fr["oben"]))).transposed()  # columns

    # rolling digits: positions + shader blur per column
    ratios = [0.18, 0.27, 0.39, 0.53, 0.68, 0.86, 1.0]
    rng = np.random.default_rng(5)
    pos0 = rng.uniform(0, 10, len(ratios))
    digit_frames = list(range(1, f_plane + 30, 2))

    def speed(f):
        return 0.6 + 2.0 * S.span(f, 25, f_inv, "in_cubic")

    acc = np.cumsum([speed(f) * 2 for f in digit_frames])  # 2-frame steps
    for k, r in enumerate(ratios):
        S.bake_value(pmat, f"Rolle{k}", digit_frames, lambda f, k=k, r=r: float(pos0[k] + r * acc[digit_frames.index(f)]))
        S.bake_value(pmat, f"Schliere{k}", digit_frames, lambda f, r=r: float(min(1.6, 0.06 + 0.55 * r * speed(f))))

    # ---- Proxmox card ------------------------------------------------------
    card = S.screenshot_card("ProxmoxKarte", "proxmox-ui", CARD_W, ripple=True)
    card.ob.matrix_world = Matrix.Translation(B) @ frame_basis(YAW)
    contact = card.ob.matrix_world @ Vector((CONTACT_LOCAL[0], CONTACT_LOCAL[1], CARD_W * 0.006))
    S.key_value(card.mat, "WelleX", 1, CONTACT_LOCAL[0])
    S.key_value(card.mat, "WelleY", 1, CONTACT_LOCAL[1])
    S.key_value(card.mat, "WelleR", f_plane, 0.0, "CUBIC", "EASE_OUT")
    S.key_value(card.mat, "WelleR", f_plane + 60, 0.62)
    S.key_value(card.mat, "WelleA", f_plane - 1, 0.0, "LINEAR")
    S.key_value(card.mat, "WelleA", f_plane + 2, 2.6, "SINE", "EASE_IN")
    S.key_value(card.mat, "WelleA", f_plane + 60, 0.0)
    S.key_value(card.mat, "Hell", f_dash - 45, 1.0, "SINE", "EASE_IN_OUT")
    S.key_value(card.mat, "Hell", f_dash, 0.32)

    # ---- flight path --------------------------------------------------------
    p_launch = A + Vector((0.0, 0.0, 0.03))
    path = S.bezier(
        tuple(p_launch),
        tuple(p_launch + Vector((0.10, 0.22, 0.55))),
        tuple(contact - E_X * 0.62 + E_N * 0.34 + E_Z * 0.24),   # sweep in from the left, dive in at ~30 deg
        tuple(contact),
        260,
    )
    L = S.arc_lengths(path)

    def flight_s(f):
        u = S.clamp01((f - f_launch) / (f_plane - f_launch))
        return L[-1] * (0.55 * u ** 2.2 + 0.45 * u)

    # trail: the logo band grows behind the plane
    bmat = M.ribbon_material("BandS5", emission=1.1, edge=2.0, head=7.0)
    trail = RB.build(S.polyline_resample(path, 220), 0.011, 0.0022, name="Umzugsbahn", material=bmat)
    RB.set_input(trail, "Kopf Laenge", 0.10)
    trail.hide_render = True
    trail.keyframe_insert("hide_render", frame=f_launch)
    trail.hide_render = False
    trail.keyframe_insert("hide_render", frame=f_launch + 1)
    tframes = list(range(f_launch - 1, f_plane + 2))
    lag = 0.12  # the band leaves the plane at its tail, not inside it
    RB.bake_input(trail, "Ende", tframes, lambda f: float(min(1.0, max(0.0, flight_s(f) - lag) / (L[-1] - lag))) if f >= f_launch else 0.0)
    RB.key_input(trail, "Kopf Laenge", f_plane, 0.10)
    RB.key_input(trail, "Kopf Laenge", f_plane + 20, 0.0)
    M.key_node_value(bmat, "Leuchten", f_plane + 10, 1.0)
    M.key_node_value(bmat, "Leuchten", f_plane + 80, 0.55)

    # ---- paper transform: hover -> turn while folding -> flight -> into card
    def r_sheet(f):
        yaw = math.radians(-24.0 + 9.0 * S.span(f, 1, f_fold0, "in_out_sine"))
        tilt = math.radians(-8.0 + 2.5 * math.sin(2 * math.pi * f / 200.0))
        roll = math.radians(4.0 * math.sin(2 * math.pi * f / 260.0 + 1.0))
        return (Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(math.radians(90.0) + tilt, 4, "X") @ Matrix.Rotation(roll, 4, "Z")).to_quaternion()

    def r_flight(s):
        p, t = S.polyline_at(path, s, L)
        t = Vector(t)
        up = E_Z - t * t.dot(E_Z)
        up.normalize()
        right = t.cross(up)
        # bank from curvature
        p2, t2 = S.polyline_at(path, min(s + 0.05, L[-1]), L)
        bank = max(-0.6, min(0.6, -2.0 * Vector(t2 - np.asarray(t)).dot(right)))
        q_bank = Quaternion(t, bank)
        world = Matrix((tuple(right), tuple(t), tuple(up))).transposed()
        R = (q_bank.to_matrix() @ world @ loc_basis.transposed()).to_quaternion()
        return R

    q_launch = r_flight(0.0)
    hover_anchor = A + Vector((0.0, 0.0, 0.0))
    paper_frames = list(range(1, f_plane + 14))
    mats = []
    for f in paper_frames:
        bob = Vector((0.0, 0.0, 0.006 * math.sin(2 * math.pi * f / 150.0)))
        if f < f_launch:
            s_turn = S.span(f, f_fold0 + 18, f_launch, "smoother")
            q = r_sheet(min(f, f_fold0)).slerp(q_launch, s_turn)
            anchor_local = c_local * s_turn
            anchor_world = (hover_anchor + bob).lerp(p_launch, S.span(f, f_fold0 + 18, f_launch))
        else:
            s = flight_s(f)
            if f <= f_plane:
                p, _ = S.polyline_at(path, s, L)
                q = r_flight(s)
            else:  # keep flying straight into the card
                p_end, t_end = S.polyline_at(path, L[-1], L)
                v = (L[-1] - flight_s(f_plane - 1))
                p = p_end + t_end * v * (f - f_plane)
                q = r_flight(L[-1])
            anchor_local = c_local
            anchor_world = Vector(p)
        Rm = q.to_matrix().to_4x4()
        loc = anchor_world - (q.to_matrix() @ anchor_local)
        mats.append(Matrix.Translation(loc) @ Rm)
    S.bake_transform(paper, paper_frames, mats)
    paper.hide_render = False
    paper.keyframe_insert("hide_render", frame=f_plane + 12)
    paper.hide_render = True
    paper.keyframe_insert("hide_render", frame=f_plane + 13)

    # ---- dashed return path + launch ring -----------------------------------
    p_back0 = card.ob.matrix_world @ Vector((-CARD_W * 0.42, -CARD_W * 0.24, 0.0))
    back_pts = S.bezier(
        tuple(p_back0),
        tuple(p_back0 + E_N * 0.30 + E_Z * -0.40),
        tuple(A + Vector((0.40, -0.10, -0.42))),
        tuple(A + Vector((0.0, 0.0, -0.075))),
        240,
    )
    mint = S.hex_lin(S.PRISMA["mint"])
    dashes, dmat = S.dashed_path("Rueckweg", back_pts, dash=0.034, gap=0.022, radius=0.0052, color=mint, strength=4.5)
    S.key_value(dmat, "Front", f_back, 0.0, "CUBIC", "EASE_OUT")
    S.key_value(dmat, "Front", f_back + 62, 1.0)
    ring = ring_mesh("Startring", A + Vector((0.0, 0.0, -0.08)), 0.045, 0.0032)
    rmat = S.emission_mat("Startring_Mat", mint, 0.0, value_name="Glut")
    ring.data.materials.append(rmat)
    S.key_value(rmat, "Glut", f_back + 52, 0.0, "CUBIC", "EASE_OUT")
    S.key_value(rmat, "Glut", f_back + 64, 9.0, "SINE", "EASE_IN_OUT")
    S.key_value(rmat, "Glut", f_back + 110, 3.5)
    C.key(ring, "scale", f_back + 52, (0.2, 0.2, 0.2), interp="QUINT", easing="EASE_OUT")
    C.key(ring, "scale", f_back + 68, (1.0, 1.0, 1.0))

    # ---- host base with rolling update ---------------------------------------
    c_row = C_ROW
    base_w, base_d, base_h = 1.00, 0.22, 0.24
    dark = S.principled("HostDunkel", S.hex_lin("#161826"), rough=0.32, metal=0.35, spec=0.5, coat=0.4, coat_rough=0.08)
    edge = S.principled("HostKante", (0.35, 0.37, 0.44, 1.0), rough=0.25, metal=0.9)
    base = S.slab("Host", base_w, base_d, base_h, 0.018, mats=[dark, edge, dark, edge])
    base.matrix_world = Matrix.Translation(c_row + E_Z * (-base_h / 2)) @ Matrix.Rotation(YAW, 4, "Z")
    line, lmat = status_line("Statuslinie", base_w * 0.94, mint)
    line.parent = base
    line.matrix_parent_inverse = Matrix()
    line.location = (0.0, -base_d / 2 - 0.0015, base_h * 0.5 - 0.035)
    for i in range(5):
        t_i = f_roll + 10 * i
        S.key_value(lmat, "Puls", t_i - 4, (i + 0.5) / 5.0 - 0.05, "LINEAR")
        S.key_value(lmat, "Puls", t_i + 6, (i + 0.5) / 5.0 + 0.05, "LINEAR")
    S.key_value(lmat, "PulsA", f_roll - 8, 0.0, "LINEAR")
    S.key_value(lmat, "PulsA", f_roll, 1.0, "LINEAR")
    S.key_value(lmat, "PulsA", f_roll + 48, 1.0, "LINEAR")
    S.key_value(lmat, "PulsA", f_roll + 62, 0.0, "LINEAR")

    blk = (0.15, 0.15, 0.20)
    old_led = S.emission_mat("LED_alt", S.hex_lin("#7b88a8"), 2.2)
    new_led = S.emission_mat("LED_neu", mint, 6.0)
    body = S.principled("Container", S.hex_lin("#1d2030"), rough=0.3, metal=0.25, spec=0.5, coat=0.6, coat_rough=0.05)
    body_new = S.principled("ContainerNeu", S.hex_lin("#1f2a2c"), rough=0.28, metal=0.25, spec=0.5, coat=0.6, coat_rough=0.05)
    rimc = S.principled("ContainerRand", (0.45, 0.47, 0.55, 1.0), rough=0.22, metal=0.9)
    blocks = []
    for i in range(5):
        x = (i - 2) * 0.19
        home = c_row + E_X * x + E_Z * (blk[2] / 2)
        t_i = f_roll + 10 * i
        old = container(f"Container_alt{i}", blk, [body, rimc, body, rimc], old_led)
        new = container(f"Container_neu{i}", blk, [body_new, rimc, body_new, rimc], new_led)
        rot = Matrix.Rotation(YAW, 4, "Z")
        frames_o = list(range(t_i - 4, t_i + 16))
        S.bake_transform(old, [1] + frames_o, [Matrix.Translation(home) @ rot] + [
            Matrix.Translation(home + E_Z * (-0.235 * S.span(f, t_i - 4, t_i + 14, "in_cubic"))) @ rot for f in frames_o])
        frames_n = list(range(t_i + 2, t_i + 28))
        S.bake_transform(new, [1] + frames_n, [Matrix.Translation(home + E_Z * 1.4) @ rot] + [
            Matrix.Translation(home + E_Z * (1.4 * (1.0 - S.span(f, t_i + 2, t_i + 26, "out_quint")))) @ rot for f in frames_n])
        for ob_, f_on in ((new, t_i + 2),):
            ob_.hide_render = True
            ob_.keyframe_insert("hide_render", frame=f_on - 1)
            ob_.hide_render = False
            ob_.keyframe_insert("hide_render", frame=f_on)
        old.hide_render = False
        old.keyframe_insert("hide_render", frame=t_i + 15)
        old.hide_render = True
        old.keyframe_insert("hide_render", frame=t_i + 16)
        blocks += [old, new]

    # ---- dashboard tiles -----------------------------------------------------
    d_centre = D_CENTRE
    board = frame_basis(YAW)
    px_w = 1584 - 24
    scale = BOARD_W / px_w
    cx_px, cy_px = (24 + 1584) / 2, (38 + 903) / 2
    W, H = S.SIZES["grafana-monitoring"]
    rng = np.random.default_rng(76)
    tiles = []
    for k, (name, (x0, y0, x1, y1)) in enumerate(TILES):
        uv = (x0 / W, 1 - y1 / H, x1 / W, 1 - y0 / H)
        tw = (x1 - x0) * scale
        tile = S.screenshot_card(f"Kachel_{name}", "grafana-monitoring", tw, uv_rect=uv, shadow=False, depth=0.006, glow=0.95)
        off = Vector((((x0 + x1) / 2 - cx_px) * scale, -((y0 + y1) / 2 - cy_px) * scale, 0.0))
        home = Matrix.Translation(d_centre) @ board @ Matrix.Translation(off)
        land = f_dash - (len(TILES) - 1 - k) * 3
        start_off = Vector((rng.uniform(-0.55, 0.55), rng.uniform(-0.35, 0.45), rng.uniform(0.25, 0.65)))
        start_rot = Quaternion(Vector(rng.normal(size=3)).normalized(), rng.uniform(0.6, 1.2))
        hq = home.to_quaternion()
        frames_t = [1] + list(range(land - 32, land + 1))
        mats_t = []
        for f in frames_t:
            s = S.span(f, land - 32, land, "out_quint")
            loc = home.translation + (board.to_3x3() @ start_off) * (1 - s)
            q = (hq @ start_rot).slerp(hq, S.span(f, land - 32, land, "out_cubic"))
            mats_t.append(Matrix.Translation(loc) @ q.to_matrix().to_4x4())
        S.bake_transform(tile.ob, frames_t, mats_t)
        tile.ob.hide_render = True
        tile.ob.keyframe_insert("hide_render", frame=land - 33)
        tile.ob.hide_render = False
        tile.ob.keyframe_insert("hide_render", frame=land - 32)
        tiles.append(tile.ob)

    # ---- dust for depth --------------------------------------------------------
    S.dust("Staub", 34, (-1.2, -1.6, -1.0), (3.2, 2.6, 1.2), seed=55, size=(0.0025, 0.006), strength=0.9, color=(0.75, 0.95, 0.88, 1.0))

    # ---- lights -----------------------------------------------------------------
    S.area_light("Key", (-1.3, -1.7, 1.5), (0.5, 0.6, 0.0), 1.8, 22.0, M.kelvin_rgb(5600))
    S.area_light("Rim", (2.9, 2.7, 1.3), (0.8, 0.9, 0.0), 1.2, 30.0, M.kelvin_rgb(8200))
    S.area_light("Fill", (0.4, -1.2, -1.3), (0.6, 0.6, 0.0), 2.0, 6.0, (0.55, 1.0, 0.82))
    S.area_light("Papierlicht", (-0.55, -0.85, 0.55), (0.0, 0.0, 0.0), 0.6, 2.6, M.kelvin_rgb(5200))
    S.area_light("Hostlicht", tuple(c_row + E_N * 0.9 + E_Z * 0.9 + E_X * -0.6), tuple(c_row), 1.0, 7.0, M.kelvin_rgb(6000))
    S.area_light("Hostkante", tuple(c_row + E_N * -0.8 + E_Z * 0.7 + E_X * 0.9), tuple(c_row + E_Z * 0.3), 0.8, 9.0, M.kelvin_rgb(9000))

    # ---- camera -----------------------------------------------------------------
    M_ = (A + B) * 0.5
    cam_keys = [
        (1, A + Vector((0.13, -1.28, 0.19))),
        (f_inv, A + Vector((0.06, -1.05, 0.13))),
        (f_fold1, A + Vector((-0.06, -1.25, 0.26))),
        (f_plane - 22, B + E_N * 2.95 + E_X * -0.85 + E_Z * 0.38),
        (f_plane, B + E_N * 2.80 + E_X * -0.55 + E_Z * 0.28),
        (f_back - 75, B + E_N * 2.70 + E_X * -0.70 + E_Z * 0.22),
        (f_back - 6, M_ + Vector((-0.32, -3.95, 0.38))),
        (f_back + 95, M_ + Vector((-0.16, -3.75, 0.30))),
        (f_roll - 8, c_row + E_N * 2.75 + E_X * -0.36 + E_Z * 0.42),
        (f_roll + 40, c_row + E_N * 2.60 + E_X * -0.30 + E_Z * 0.48),
        (f_dash - 6, c_row + E_N * 2.35 + E_X * -0.34 + E_Z * 0.66),
        (n_frames, c_row + E_N * 2.02 + E_X * 0.30 + E_Z * 0.60),
    ]
    tgt_keys = [
        (1, A + Vector((-0.045, 0.0, -0.01))),
        (f_inv, A + Vector((-0.04, 0.0, -0.005))),
        (f_fold1, A + Vector((0.04, 0.08, 0.03))),
        (f_plane - 22, B + E_N * 0.30 + E_X * -0.30),
        (f_plane, B + E_X * -0.06 + E_Z * -0.02),
        (f_back - 75, B + E_X * -0.10 + E_Z * -0.04),
        (f_back - 6, M_ + Vector((0.10, 0.0, -0.20))),
        (f_back + 95, M_ + Vector((0.13, 0.0, -0.20))),
        (f_roll - 8, c_row + E_Z * 0.08 + E_X * 0.03),
        (f_roll + 40, c_row + E_Z * 0.12 + E_X * 0.03),
        (f_dash - 6, d_centre + E_Z * -0.20 + E_X * 0.02),
        (n_frames, d_centre + E_Z * -0.19 + E_X * 0.02),
    ]
    foc_keys = [
        (1, A),
        (f_fold1, A + Vector((0.02, 0.03, 0.03))),
        (f_plane - 22, B + E_N * 0.5),
        (f_plane, contact),
        (f_back - 6, M_),
        (f_roll - 8, c_row),
        (f_roll + 40, c_row),
        (f_dash - 12, d_centre),
        (n_frames, d_centre),
    ]
    cam, tgt, foc = C.rig("Kamera5", tuple(cam_keys[0][1]), tuple(tgt_keys[0][1]), lens=LENS, fstop=2.0)
    frames = list(range(1, n_frames + 1))
    S.bake_fast(cam, "location", frames, [tuple(S.smooth_path(f, cam_keys)) for f in frames])
    S.bake_fast(tgt, "location", frames, [tuple(S.smooth_path(f, tgt_keys)) for f in frames])
    S.bake_fast(foc, "location", frames, [tuple(S.smooth_path(f, foc_keys)) for f in frames])
    C.key(cam.data.dof, "aperture_fstop", 1, 2.0)
    C.key(cam.data.dof, "aperture_fstop", f_back, 2.4)
    C.key(cam.data.dof, "aperture_fstop", f_roll - 8, 1.8)

    # ---- key objects per event (centre-square probe) ----------------------------
    ctx.key("invoice_spin", paper)
    ctx.key("paper_plane", paper, card.ob)
    ctx.key("rollback_path", card.ob, dashes, ring)
    ctx.key("rolling_update", base)
    ctx.key("dashboards", *tiles)
    print(f"[s5] frames={n_frames} inv={f_inv} fold={f_fold0}-{f_fold1} plane={f_plane} back={f_back} roll={f_roll} dash={f_dash}", flush=True)
    return ctx


def ring_mesh(name, centre, radius, tube, seg=64, ring=10):
    verts, faces = [], []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        cx, cy = math.cos(a), math.sin(a)
        for j in range(ring):
            b = 2 * math.pi * j / ring
            r = radius + tube * math.cos(b)
            verts.append((centre.x + r * cx, centre.y + r * cy, centre.z + tube * math.sin(b)))
    for i in range(seg):
        for j in range(ring):
            a0 = i * ring + j
            a1 = i * ring + (j + 1) % ring
            b0 = ((i + 1) % seg) * ring + j
            b1 = ((i + 1) % seg) * ring + (j + 1) % ring
            faces.append([a0, b0, b1, a1])
    ob = S.mesh_object(name, verts, faces, smooth=True)
    # scale about the ring centre
    me = ob.data
    me.transform(Matrix.Translation(-centre))
    ob.location = centre
    return ob


def status_line(name, length, color):
    """Thin emissive strip; ``Puls`` (0..1 along x) adds a travelling glow."""
    mat, b = S.new_mat(f"{name}_Mat")
    out = b.n("ShaderNodeOutputMaterial")
    tc = b.n("ShaderNodeTexCoord")
    x = b.m("ADD", b.m("DIVIDE", b.sep(tc.outputs["Object"])[0], length), 0.5)
    q = b.m("DIVIDE", b.m("SUBTRACT", x, b.value("Puls", 0.0)), 0.035)
    puls = b.m("MULTIPLY", b.m("EXPONENT", b.m("MULTIPLY", b.m("MULTIPLY", q, q), -1.0)), b.value("PulsA", 0.0))
    em = b.n("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    b.put(em.inputs["Strength"], b.m("ADD", 5.0, b.m("MULTIPLY", puls, 14.0)))
    b.t.links.new(em.outputs[0], out.inputs["Surface"])
    h = 0.007
    verts = [(-length / 2, 0, -h / 2), (length / 2, 0, -h / 2), (length / 2, 0, h / 2), (-length / 2, 0, h / 2)]
    ob = S.mesh_object(name, verts, [[0, 1, 2, 3]], mats=[mat])
    ob.visible_shadow = False
    return ob, mat


def container(name, size, mats, led_mat):
    sx, sy, sz = size
    ob = S.slab(name, sx, sy, sz, 0.016, mats=mats)
    w, h = sx * 0.34, 0.011
    verts = [(-w / 2, -sy / 2 - 0.0012, sz * 0.22 - h / 2), (w / 2, -sy / 2 - 0.0012, sz * 0.22 - h / 2),
             (w / 2, -sy / 2 - 0.0012, sz * 0.22 + h / 2), (-w / 2, -sy / 2 - 0.0012, sz * 0.22 + h / 2)]
    led = S.mesh_object(name + "_LED", verts, [[0, 1, 2, 3]], mats=[led_mat])
    led.parent = ob
    led.visible_shadow = False
    return ob


def main() -> None:
    args = S.parse_args()
    if args.encode:
        S.encode(SHOT)
        return
    ctx = build(args)
    S.run(ctx, args)


if __name__ == "__main__":
    main()
