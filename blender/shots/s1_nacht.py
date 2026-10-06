"""Shot s1_nacht - 03:12, the city sleeps, the band cuts in as a line of light.

Beats (all read from timeline.json at render time):

* ``ribbon_enter``: the logo band cuts in from the right as a thin line of
  light with a hot head, turns towards the camera and grows into a broad,
  twisting, lacquered band with the logo gradient,
* the camera follows in one long push-in with a lateral drift and a slight
  roll, so the three bokeh depth layers slide against each other,
* the night city is a field of designed bokeh discs (clear edge, slightly
  darker centre), warm white with a few cool ones; the left half of the frame
  stays calm for the typography,
* ``lights_off``: about 30 % of the discs switch off, staggered outwards,
* scene end (+ handle): the band's head races away into the depth while its
  body stays in frame up to the last frame - the hand-over to s2.

Usage (see blender/render.sh)::

    blender -b --factory-startup -P blender/shots/s1_nacht.py -- [--res 480] [--frames 1,mid,last]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import nomiss_camera as C  # noqa: E402
import nomiss_material as M  # noqa: E402
import nomiss_render as R  # noqa: E402
import nomiss_ribbon as RB  # noqa: E402
import nomiss_stadt as ST  # noqa: E402
import nomiss_timeline as TL  # noqa: E402

SHOT = "s1_nacht"
LENS = 45.0
SENSOR = 36.0
Z0 = 30.0
BAND_W = 0.34
BAND_T = 0.065
OFF_SHARE = 0.30
COLOUR_SPAN_M = 48.0  # the full logo gradient spans the first 48 m of band
EVENTS = ("ribbon_enter", "clock_roll", "sleep_letters", "lights_off", "led_1", "led_2", "led_3", "led_4", "bell_morph")

# Band path (world, metres).  It cuts in from the right close to the camera
# and then runs ahead of it; the camera chases it from (-1, -4) to about
# (1.6, 26), so the band's near part slides out of frame on the right.
WAYPOINTS = [
    (16.0, 12.0, Z0 - 1.2),
    (8.5, 13.5, Z0 - 1.6),
    (5.0, 17.0, Z0 - 1.95),
    (3.4, 23.0, Z0 - 1.80),
    (3.6, 30.0, Z0 - 1.35),
    (4.2, 38.0, Z0 - 1.05),
    (3.5, 48.0, Z0 - 1.30),
    (2.5, 60.0, Z0 - 1.55),
    (1.9, 78.0, Z0 - 1.25),
    (1.3, 110.0, Z0 - 0.6),
    (0.8, 170.0, Z0 + 0.4),
    (0.4, 300.0, Z0 + 1.8),
    (0.2, 520.0, Z0 + 3.5),
]


def path_points() -> tuple[np.ndarray, np.ndarray]:
    from nomiss_logo import catmull_rom, resample  # pure numpy helpers

    p = resample(np.asarray(catmull_rom(WAYPOINTS, 60)), 0.06)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return p, s


def build(args):
    tl = TL.load(args.timeline or None)
    sh = TL.shot(SHOT, tl)
    ev = {name: TL.event_frame(name, SHOT, tl) for name in EVENTS}
    fps = sh.fps
    f_end = sh.frames
    f_enter = ev["ribbon_enter"]
    f_off = ev["lights_off"]
    _, f_scene_end = TL.scene_frames(SHOT, tl)
    f_pull = f_scene_end - int(1.6 * fps)  # the head races into the depth

    scene = R.clean_scene()
    out = R.setup(
        scene,
        SHOT,
        sh.frames,
        res=args.res,
        samples=args.samples or 64,
        motion_blur=not args.no_mb,
        glare=not args.no_glare,
        out_dir=args.out or None,
    )
    if not args.no_glare:
        R.compositor_glow(scene, threshold=0.8, strength=0.3, size=0.7)

    # ---- camera path -------------------------------------------------------
    def k_of(f):
        return C.ease((f - 1) / (f_end - 1), "in_out_sine")

    def cam_loc(f):
        k = k_of(f)
        push = 6.0 * C.ease((f - f_pull) / (f_end - f_pull), "in_quad") if f > f_pull else 0.0
        return (-1.0 + 2.4 * k, -4.0 + 35.0 * k + push, Z0 + 0.1 + 0.5 * k)

    # ---- stage: the night hero, subtle -------------------------------------
    pitch = math.radians(-3.0)
    win = (-0.3, -0.3, 1.3, 1.3)
    night = M.hero_image(
        "Buehne_Nacht1",
        1024,
        gains={"#145fe4": 0.48, "#7c3aed": 0.6, "#e44b8d": 0.45, "#10b981": 0.32},
        blue_cy=1.28,
        window=win,
    )
    M.world_backplate(scene, [night], lens_mm=LENS, sensor_mm=SENSOR, overscan=1.6, pitch=pitch)

    # ---- city as bokeh in three depth layers -------------------------------
    ref_loc = Vector(cam_loc((f_end + 1) // 2))
    ref_rot = (math.radians(90.0) + pitch, 0.0, 0.0)
    stadt = ST.bokeh_layers(
        "Stadt1",
        seed=11,
        cam_loc=ref_loc,
        cam_rot=ref_rot,
        lens_mm=LENS,
        sensor_mm=SENSOR,
        layers=[
            ST.Ebene(distance=70.0, count=7, size_px=(165.0, 215.0), brightness=(0.40, 0.55)),
            ST.Ebene(distance=170.0, count=18, size_px=(100.0, 130.0), brightness=(0.55, 0.85)),
            ST.Ebene(distance=450.0, count=42, size_px=(52.0, 70.0), brightness=(0.75, 1.15)),
        ],
        region=(-1.15, 1.15, -1.05, 0.06),
        x_weight=lambda sx, sy: 0.10 if sx < -0.08 else (0.5 if sx < 0.15 else 1.0),
        cool=0.16,
        kelvin=(3300.0, 4800.0),
    )
    rng = np.random.default_rng(31)
    pos = stadt.pos
    n = len(pos)
    pick = rng.choice(n, size=int(n * OFF_SHARE), replace=False)
    centre = np.array([ref_loc.x + 40.0, ref_loc.y + 150.0])
    d = np.linalg.norm(pos[pick, :2] - centre, axis=1)
    rank = np.argsort(np.argsort(d)) / max(len(pick) - 1, 1)
    aus = np.full(n, ST.NEVER)
    aus[pick] = f_off + (0.05 + 1.1 * rank + rng.normal(0.0, 0.1, len(pick))).clip(0.0, 1.4) * fps
    stadt.set_off_frames(aus)

    # ---- the band ----------------------------------------------------------
    pts, s = path_points()
    total = s[-1]
    u = np.clip(s / COLOUR_SPAN_M, 0.0, 1.0)
    band_mat = M.ribbon_material("Band1", emission=0.95, edge=2.4, head=4.5, roughness=0.34)
    band = RB.build(pts, BAND_W, BAND_T, name="Band", material=band_mat, u=u)
    RB.set_input(band, "Drall", total / 16.0)       # one turn every 16 m
    RB.set_input(band, "Kopf Laenge", 2.5 / total)  # 2.5 m glowing head
    RB.set_input(band, "Schweif Laenge", 9.0 / total)  # tail tapers over 9 m

    def head_m(f: float) -> float:
        # never 0: a 0.3 m stub off-frame keeps the geometry non-empty (an
        # object that is empty in a render session's first frame is dropped)
        if f <= f_enter:
            return 0.3
        t = (f - f_enter) / fps
        s_h = 0.3 + 20.0 * C.ease(t / 1.3, "out_cubic") + 2.0 * max(0.0, t - 0.8)
        if f > f_pull:
            s_h += 330.0 * C.ease((f - f_pull) / (f_end - f_pull), "in_quad")
        return min(s_h, total)

    def tail_m(f: float) -> float:
        """The tail follows the head ~20 m behind (a moving ribbon of light);
        while the head races away it lags, so the band stretches into depth."""
        if f <= f_pull:
            return max(0.0, head_m(f) - 22.0)
        return max(0.0, head_m(f_pull) - 22.0) + 9.0 * C.ease((f - f_pull) / (f_end - f_pull), "in_out_sine")

    frames = range(1, f_end + 1)
    RB.bake_input(band, "Ende", frames, lambda f: head_m(f) / total)
    RB.bake_input(band, "Start", frames, lambda f: tail_m(f) / total)
    RB.bake_input(band, "Drall Phase", frames, lambda f: 1.1 * f / fps)

    def point_at(arc: float) -> Vector:
        i = int(np.clip(np.searchsorted(s, arc), 0, len(s) - 1))
        return Vector(pts[i])

    # ---- camera: push-in with lateral drift and slight roll ---------------
    fstop = C.fstop_for_bokeh(lens_mm=LENS, sensor_mm=SENSOR, subject_m=16.0, background_m=1e9, bokeh_px=9.0, res_px=1920)
    cam, tgt, foc = C.rig("Kamera1", cam_loc(1), (0.0, 60.0, Z0 - 3.0), lens=LENS, fstop=fstop, sensor=SENSOR)

    def look(f):
        k = k_of(f)
        h = point_at(max(min(head_m(f), 70.0) - 6.0, 0.0))
        w = 0.62 * C.ease((f - f_enter) / (2.5 * fps), "in_out_sine")
        base = Vector((0.8 + 1.8 * k, 60.0 + 35.0 * k, Z0 - 4.2 + 0.6 * k))
        return tuple(base.lerp(h + Vector((-1.2, 0.0, -0.3)), w))

    def roll(f):
        return math.radians(-1.6 + 3.4 * C.ease((f - f_enter) / (f_end - f_enter), "in_out_sine"))

    C.bake_look(cam, frames, cam_loc, look, roll)

    def focus(f):
        h = point_at(max(head_m(f) - 7.0, 0.0))
        rest = Vector((6.5, 16.0, Z0 - 1.6))
        p = rest.lerp(h, C.ease((f - f_enter) / (0.9 * fps), "in_out_sine"))
        if f > f_pull:  # stay on the near body while the head races away
            p = p.lerp(point_at(max(head_m(f_pull) - 10.0, 0.0)), C.ease((f - f_pull) / (0.6 * fps), "in_out_sine"))
        return tuple(p)

    C.bake(foc, "location", frames, focus)

    # ---- light: cool rim from the blue stage + a travelling key glint -----
    def area(name, loc, target, size, energy, color):
        ld = bpy.data.lights.new(name, "AREA")
        ld.size = size
        ld.energy = energy
        ld.color = color
        ob = bpy.data.objects.new(name, ld)
        ob.location = loc
        ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(ob)
        return ob

    area("Kante", (-14.0, 70.0, Z0 + 16.0), (5.0, 30.0, Z0 - 1.5), 14.0, 12000.0, M.kelvin_rgb(9500))
    area("Unten", (9.0, 45.0, Z0 - 12.0), (5.5, 30.0, Z0 - 1.5), 12.0, 5000.0, (0.25, 0.45, 1.0))
    glanz = area("Glanz", (-8.0, 0.0, Z0 + 6.0), (5.0, 30.0, Z0 - 1.5), 3.0, 1800.0, M.kelvin_rgb(6000))
    glanz.data.diffuse_factor = 0.15
    C.key(glanz, "location", 1, (-9.0, -4.0, Z0 + 7.0), interp="SINE", easing="EASE_IN_OUT")
    C.key(glanz, "location", f_end, (10.0, 22.0, Z0 + 5.0))

    print(
        f"[s1] timeline={sh.quelle} provisorisch={sh.provisorisch} frames={sh.frames} enter={f_enter} "
        f"lights_off={f_off} pull={f_pull} fstop={fstop:.2f} band={total:.1f} m discs_off={len(pick)}/{n}",
        flush=True,
    )
    return scene, {"shot": sh, "events": ev, "out": out, "band": band}


def main() -> None:
    args = R.parse_args()
    scene, info = build(args)
    sh = info["shot"]
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    if args.matte:
        R.matte_mode(scene, [bpy.data.objects[args.matte]])
    frames = R.frame_list(args.frames, sh.frames, info["events"])
    if args.frames == "all" and not args.matte:
        R.render_animation(scene)
        TL.write_shot_json(sh, Path(info["out"]).parent / f"{SHOT}.json")
    else:
        R.render_frames(scene, frames, info["out"])


if __name__ == "__main__":
    main()
