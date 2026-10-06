"""Camera rig, easing and keyframe helpers for the nomissuccess shots.

* :func:`rig` - camera tracking a target empty (also the DOF focus), plus an
  optional separate focus empty for focus pulls.
* :func:`ease` - easing curves (all exactly 0 at t=0 and 1 at t=1).
* :func:`key` / :func:`set_key_interp` - keyframes with Blender's Penner
  easings (``QUINT`` + ``EASE_OUT`` lands without overshoot), working with
  the layered actions of Blender 4.4+/5.x.
* :func:`bake` - per-frame keys from a Python function for motion that is
  easier to write as code than as curves.
* :func:`fstop_for_bokeh` - aperture that gives a background blur disc of a
  chosen size in pixels.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import bpy
from mathutils import Vector


# --------------------------------------------------------------------------
# easing
# --------------------------------------------------------------------------

def _clamp01(t: float) -> float:
    return 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else t


def ease(t: float, kind: str = "in_out_cubic") -> float:
    t = _clamp01(t)
    if kind == "linear":
        return t
    if kind == "in_out_sine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if kind == "in_sine":
        return 1.0 - math.cos(0.5 * math.pi * t)
    if kind == "out_sine":
        return math.sin(0.5 * math.pi * t)
    if kind == "in_quad":
        return t * t
    if kind == "in_cubic":
        return t ** 3
    if kind == "out_cubic":
        return 1.0 - (1.0 - t) ** 3
    if kind == "in_out_cubic":
        return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    if kind == "out_quint":
        return 1.0 - (1.0 - t) ** 5
    if kind == "in_out_quint":
        return 16 * t ** 5 if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2
    if kind == "out_expo":
        return 1.0 if t >= 1.0 else 1.0 - 2.0 ** (-10.0 * t)
    if kind == "smoothstep":
        return t * t * (3 - 2 * t)
    if kind == "smootherstep":
        return t * t * t * (t * (6 * t - 15) + 10)
    raise ValueError(f"unbekannte Kurve: {kind}")


def span(frame: float, f0: float, f1: float, kind: str = "in_out_cubic") -> float:
    """Eased progress of ``frame`` inside [f0, f1] (0 before, 1 after)."""
    if f1 <= f0:
        return 1.0 if frame >= f1 else 0.0
    return ease((frame - f0) / (f1 - f0), kind)


def lerp(a, b, t):
    if isinstance(a, (tuple, list, Vector)):
        return type(a)(x + (y - x) * t for x, y in zip(a, b)) if not isinstance(a, Vector) else a.lerp(b, t)
    return a + (b - a) * t


# --------------------------------------------------------------------------
# keyframes (layered actions aware)
# --------------------------------------------------------------------------

def fcurves(id_data) -> list:
    ad = getattr(id_data, "animation_data", None)
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "layers") and len(act.layers):
        try:
            from bpy_extras.anim_utils import action_get_channelbag_for_slot

            cb = action_get_channelbag_for_slot(act, ad.action_slot)
            return list(cb.fcurves) if cb is not None else []
        except ImportError:  # pragma: no cover
            pass
    return list(getattr(act, "fcurves", []))


def set_key_interp(id_data, data_path: str, frame: float, interp: str = "BEZIER", easing: str = "AUTO", index: int = -1) -> None:
    for fc in fcurves(id_data):
        if fc.data_path != data_path or (index >= 0 and fc.array_index != index):
            continue
        for kp in fc.keyframe_points:
            if abs(kp.co.x - frame) < 1e-3:
                kp.interpolation = interp
                kp.easing = easing


def key(id_data, prop: str, frame: float, value, *, index: int = -1, interp: str = "BEZIER", easing: str = "AUTO") -> None:
    """Set ``id_data.prop = value`` and key it at ``frame``.

    ``interp``/``easing`` shape the segment that *starts* at this key.
    """
    if index >= 0:
        getattr(id_data, prop)[index] = value
    else:
        setattr(id_data, prop, value)
    id_data.keyframe_insert(prop, frame=frame, index=index)
    set_key_interp(id_data, prop, frame, interp, easing, index)


def bake(id_data, prop: str, frames: Iterable[int], fn: Callable[[float], object]) -> None:
    """Key ``prop`` on every frame from ``fn(frame)``; linear in between."""
    frames = list(frames)
    for f in frames:
        setattr(id_data, prop, fn(f))
        id_data.keyframe_insert(prop, frame=f)
    for fc in fcurves(id_data):
        if fc.data_path == prop:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


# --------------------------------------------------------------------------
# rig
# --------------------------------------------------------------------------

def _empty(name: str, location, collection) -> bpy.types.Object:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "PLAIN_AXES"
    ob.empty_display_size = 0.2
    ob.location = location
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def rig(
    name: str,
    location,
    target,
    *,
    lens: float = 50.0,
    fstop: float = 2.8,
    sensor: float = 36.0,
    blades: int = 0,
    focus=None,
    clip=(0.05, 5000.0),
    collection: bpy.types.Collection | None = None,
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object]:
    """Camera + aim target (+ focus empty, defaults to the target)."""
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = sensor
    cd.clip_start, cd.clip_end = clip
    cam = bpy.data.objects.new(name, cd)
    cam.location = location
    (collection or bpy.context.scene.collection).objects.link(cam)
    tgt = _empty(f"{name}_Ziel", target, collection)
    con = cam.constraints.new("TRACK_TO")
    con.target = tgt
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    foc = _empty(f"{name}_Fokus", focus if focus is not None else target, collection)
    cd.dof.use_dof = True
    cd.dof.focus_object = foc
    cd.dof.aperture_fstop = fstop
    cd.dof.aperture_blades = blades
    bpy.context.scene.camera = cam
    return cam, tgt, foc


def look_at_euler(loc, target, roll: float = 0.0, prev=None):
    """Euler rotation that points a camera from ``loc`` at ``target``.

    ``roll`` (rad) turns it about its viewing axis; ``prev`` keeps the
    Euler angles continuous between frames (no flips for motion blur).
    """
    from mathutils import Matrix

    q = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y")
    q = q @ Matrix.Rotation(roll, 4, "Z").to_quaternion()
    e = q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")
    return e


def bake_look(cam, frames, loc_fn, target_fn, roll_fn=lambda f: 0.0) -> None:
    """Bake location + rotation of ``cam`` per frame (no constraint)."""
    for con in list(cam.constraints):
        cam.constraints.remove(con)
    prev = None
    for f in frames:
        loc = loc_fn(f)
        e = look_at_euler(loc, target_fn(f), roll_fn(f), prev)
        prev = e
        cam.location = loc
        cam.rotation_euler = e
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    for fc in fcurves(cam):
        if fc.data_path in ("location", "rotation_euler"):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def fstop_for_bokeh(
    *, lens_mm: float, sensor_mm: float, subject_m: float, background_m: float, bokeh_px: float, res_px: int
) -> float:
    """Aperture number that blurs a background point into ``bokeh_px``."""
    frame_w = subject_m * sensor_mm / lens_mm  # frame width at the subject (m)
    rel = 1.0 - subject_m / background_m if math.isfinite(background_m) else 1.0
    aperture = (bokeh_px / res_px) * frame_w / max(rel, 1e-6)  # entrance pupil (m)
    return (lens_mm / 1000.0) / aperture


def frame_height_at(distance: float, lens_mm: float, sensor_mm: float = 36.0) -> float:
    """Visible frame height (= width, square render) at ``distance``."""
    return distance * sensor_mm / lens_mm
