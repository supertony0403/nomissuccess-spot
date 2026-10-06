"""Shot s8_morgen - morning: the band spirals in and folds into the 3D "N".

Beats (all read from timeline.json at render time):

* shot start .. ``logo_fold``: the band arrives as a slowly turning spiral out
  of the depth (soft, out of focus) while the camera pushes in on an arc,
* ``dawn``: the PRISMA hero glow rises from below (night blue -> blue/mint),
  the city lights fade, a blue/mint light from below catches the band,
* ``logo_fold``: the band has folded into the N - staggered smootherstep per
  point, every point lands with zero velocity (no overshoot at all),
* a single specular glint sweeps across the N, then it rests with a minimal
  breathing motion (scale +-0.4 %) until the end of the scene + handle.

Framing: the N is centred horizontally, its centre at y = 875 px of 1920
(about y 725-1025), 300 px tall, so the delivery crops (16:9 band y 420-1500,
9:16 column x 420-1500) both keep it and leave calm room below for type.

Usage (see blender/render.sh)::

    blender -b --factory-startup -P blender/shots/s8_morgen.py -- [--res 480] [--frames 1,mid,last]
"""

from __future__ import annotations

import math
import shutil
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

SHOT = "s8_morgen"
LOGO_H = 1.0          # world height of the N (m)
LOGO_PX = 300.0       # on-screen height at 1920 px
LOGO_Y_PX = 875.0     # on-screen centre (from top) at 1920 px
HORIZON_Y_PX = 1480.0 # where the (distant) horizon sits
LENS = 85.0
SENSOR = 36.0
EYE_Z = 30.0
FOLD_S = 2.0          # duration of the fold (morph) before logo_fold
STAGGER = 0.5
EVENTS = ("dawn", "logo_fold", "clock_roll_7", "bell_sleep", "claim", "cta", "url")


def geometry() -> dict:
    tan_half = SENSOR / LENS * 0.5
    dist = LOGO_H * 1920.0 / LOGO_PX * LENS / SENSOR
    pitch = math.atan((HORIZON_Y_PX - LOGO_Y_PX) / 960.0 * tan_half)
    logo = Vector((0.0, 0.0, EYE_Z + dist * math.tan(pitch)))
    cam = Vector((0.0, -dist, EYE_Z))
    return {"dist": dist, "pitch": pitch, "logo": logo, "cam": cam, "shift_y": -(960.0 - LOGO_Y_PX) / 1920.0}


def build(args):
    tl = TL.load(args.timeline or None)
    sh = TL.shot(SHOT, tl)
    ev = {name: TL.event_frame(name, SHOT, tl) for name in EVENTS}
    fps = sh.fps
    f_fold = ev["logo_fold"]
    f_dawn = ev["dawn"]
    f_end = sh.frames
    f_m0 = f_fold - int(round(FOLD_S * fps))

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
    g = geometry()
    L = g["logo"]

    # ---- stage: night hero -> dawn hero rising from below -----------------
    win = (-0.3, -0.3, 1.3, 1.3)  # CSS window of the 1.6x overscanned screen
    night = M.hero_image(
        "Buehne_Nacht",
        1024,
        gains={"#145fe4": 0.42, "#7c3aed": 0.55, "#e44b8d": 0.45, "#10b981": 0.35},
        blue_cy=1.34,
        window=win,
    )
    dawn = M.hero_image(
        "Buehne_Morgen",
        1024,
        gains={"#145fe4": 1.0, "#7c3aed": 0.85, "#e44b8d": 0.7, "#10b981": 0.9},
        blue_cy=1.10,
        extra=[("#10b981", 0.70, 1.08, 0.55, 0.50, [(0.0, 0.42), (0.75, 0.0)])],
        window=win,
    )
    world = M.world_backplate(scene, [night, dawn], lens_mm=LENS, sensor_mm=SENSOR, overscan=1.6, pitch=g["pitch"])
    f_dawn_full = f_dawn + int(2.6 * fps)
    M.key_node_value(world, "Mischung", 1, 0.0)
    M.key_node_value(world, "Mischung", f_dawn, 0.0)
    M.key_node_value(world, "Mischung", f_dawn_full, 1.0)
    M.key_node_value(world, "Anstieg", f_dawn, 0.32)
    M.key_node_value(world, "Anstieg", f_dawn_full + int(0.6 * fps), 0.0)

    # ---- city bokeh low in the frame, fading with dawn ---------------------
    stadt = ST.bokeh_layers(
        "Stadt8",
        seed=8,
        cam_loc=g["cam"],
        cam_rot=(math.radians(90.0) + g["pitch"], 0.0, 0.0),
        lens_mm=LENS,
        sensor_mm=SENSOR,
        layers=[
            ST.Ebene(distance=70.0, count=5, size_px=(150.0, 200.0), brightness=(0.35, 0.5)),
            ST.Ebene(distance=170.0, count=14, size_px=(95.0, 125.0), brightness=(0.5, 0.8)),
            ST.Ebene(distance=450.0, count=34, size_px=(50.0, 68.0), brightness=(0.7, 1.1)),
        ],
        region=(-1.15, 1.15, -1.1, -0.30),
        cool=0.2,
        kelvin=(3300.0, 4800.0),
    )
    stadt.key_brightness(1, 1.0)
    stadt.key_brightness(f_dawn, 1.0)
    stadt.key_brightness(f_dawn_full + int(0.8 * fps), 0.0)

    # ---- the band ----------------------------------------------------------
    band_mat = M.ribbon_material("Band8", emission=0.7, edge=1.6, head=0.0, roughness=0.42)
    logo = RB.build_logo(LOGO_H, name="LogoN", material=band_mat, location=L)
    RB.set_input(logo, "Staffelung", STAGGER)
    RB.set_input(logo, "Spirale Windungen", 1.3)
    RB.set_input(logo, "Spirale Konus", 0.35)
    RB.set_input(logo, "Spirale Drall", 1.0)
    RB.set_input(logo, "Spirale Skala", 1.0)

    frames = range(1, f_fold + 1)

    def arrive(f):  # 0 at shot start -> 1 at logo_fold, decelerating
        return C.span(f, 1, f_fold, "out_cubic")

    RB.bake_input(logo, "Morph", frames, lambda f: C.span(f, f_m0, f_fold, "linear"))
    RB.bake_input(logo, "Spirale Mitte", frames, lambda f: tuple(C.lerp(Vector((2.2, 11.0, -2.0)), Vector((0.0, 0.0, 0.0)), arrive(f))))
    RB.bake_input(logo, "Spirale Radius", frames, lambda f: C.lerp(1.35, 0.6, arrive(f)))
    RB.bake_input(logo, "Spirale Laenge", frames, lambda f: C.lerp(5.0, 1.0, arrive(f)))
    RB.bake_input(logo, "Spirale Phase", frames, lambda f: 5.2 * C.span(f, 1, f_fold, "out_cubic") - 2.0)
    RB.bake_input(
        logo,
        "Spirale Drehung",
        frames,
        lambda f: tuple(C.lerp(Vector((0.55, 0.0, 0.65)), Vector((0.15, 0.0, 0.2)), arrive(f))),
    )

    # breathing after the glint: scale +-0.4 %, rotation +-0.25 deg
    f_breath = f_fold + int(1.6 * fps)

    def breath(f):
        a = C.span(f, f_breath, f_breath + fps, "in_out_sine")
        t = (f - f_breath) / fps
        return a * math.sin(2 * math.pi * t / 4.8)

    after = range(f_fold, f_end + 1)
    C.bake(logo, "scale", after, lambda f: (1.0 + 0.004 * breath(f),) * 3)
    C.bake(logo, "rotation_euler", after, lambda f: (0.0, 0.0, math.radians(0.25) * breath(f + 0.7 * fps)))

    # ---- lights ------------------------------------------------------------
    def area(name, loc, target, size, energy, color, shape="SQUARE", size_y=None):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = shape
        ld.size = size
        if size_y is not None:
            ld.size_y = size_y
        ld.energy = energy
        ld.color = color
        ob = bpy.data.objects.new(name, ld)
        ob.location = loc
        scene.collection.objects.link(ob)
        direction = Vector(target) - Vector(loc)
        ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return ob

    area("Key", L + Vector((-3.6, -6.5, 3.2)), L, 3.0, 220.0, M.kelvin_rgb(5600))
    area("Rim", L + Vector((3.0, 3.5, 2.6)), L, 2.0, 260.0, M.kelvin_rgb(8000))
    up = area("Morgenlicht", L + Vector((0.0, -2.2, -3.6)), L, 6.0, 0.0, (0.25, 0.55, 1.0), "RECTANGLE", 2.0)
    C.key(up.data, "energy", f_dawn, 0.0, interp="SINE", easing="EASE_IN_OUT")
    C.key(up.data, "energy", f_dawn_full, 140.0)
    C.key(up.data, "color", f_dawn, (0.10, 0.35, 1.0), interp="SINE", easing="EASE_IN_OUT")
    C.key(up.data, "color", f_dawn_full + fps, (0.12, 0.85, 0.6))

    # glint: a long thin strip sweeping across in front of the N
    f_g0 = f_fold + int(0.25 * fps)
    f_g1 = f_g0 + int(1.15 * fps)
    glint = area("Glanz", L + Vector((-1.6, -6.0, 0.0)), L, 0.10, 0.0, (1.0, 0.98, 0.95), "RECTANGLE", 5.0)
    glint.rotation_euler.rotate_axis("Z", math.radians(28.0))
    glint.data.diffuse_factor = 0.0  # pure specular streak, no wash
    glint.data.volume_factor = 0.0
    C.key(glint, "location", f_g0, tuple(L + Vector((-1.7, -6.0, 0.25))), interp="SINE", easing="EASE_IN_OUT")
    C.key(glint, "location", f_g1, tuple(L + Vector((1.7, -6.0, -0.25))))
    C.key(glint.data, "energy", f_g0 - 1, 0.0, interp="LINEAR")
    C.key(glint.data, "energy", f_g0 + int(0.15 * fps), 420.0, interp="LINEAR")
    C.key(glint.data, "energy", f_g1 - int(0.15 * fps), 420.0, interp="LINEAR")
    C.key(glint.data, "energy", f_g1, 0.0, interp="LINEAR")

    # ---- camera: arc push-in that lands on logo_fold ----------------------
    fstop = C.fstop_for_bokeh(lens_mm=LENS, sensor_mm=SENSOR, subject_m=g["dist"], background_m=1e9, bokeh_px=10.0, res_px=1920)
    cam, tgt, foc = C.rig("Kamera8", g["cam"], L, lens=LENS, fstop=fstop, sensor=SENSOR, blades=0)
    cam.data.shift_y = g["shift_y"]
    start = g["cam"] + Vector((1.15, -3.2, -0.55))
    C.key(cam, "location", 1, tuple(start), interp="QUINT", easing="EASE_OUT")
    C.key(cam, "location", f_fold, tuple(g["cam"]), interp="CONSTANT")

    print(
        f"[s8] timeline={sh.quelle} provisorisch={sh.provisorisch} frames={sh.frames} "
        f"dawn={f_dawn} fold={f_fold} morph_start={f_m0} fstop={fstop:.2f} dist={g['dist']:.2f}",
        flush=True,
    )
    return scene, {"shot": sh, "events": ev, "out": out, "logo": logo, "geo": g}


def logo_mask(scene, info, path: str, res: int) -> None:
    """Frontal orthographic alpha of the final N (for the IoU test)."""
    logo = info["logo"]
    if logo.animation_data:
        logo.animation_data_clear()
    logo.scale = (1.0, 1.0, 1.0)
    logo.rotation_euler = (0.0, 0.0, 0.0)
    RB.set_input(logo, "Morph", 1.0)
    cd = bpy.data.cameras.new("Ortho")
    cd.type = "ORTHO"
    cd.ortho_scale = LOGO_H * 340.0 / 333.0
    cam = bpy.data.objects.new("Ortho", cd)
    cam.location = info["geo"]["logo"] + Vector((0.0, -20.0, 0.0))
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.resolution_x = scene.render.resolution_y = res
    R.matte_mode(scene, [logo])
    tmp = Path(path).with_suffix("")
    written = R.render_frames(scene, [1], tmp)
    shutil.move(str(written[0]), path)
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    args = R.parse_args()
    scene, info = build(args)
    sh = info["shot"]
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    if args.logo_mask:
        logo_mask(scene, info, args.logo_mask, args.res)
        return
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
