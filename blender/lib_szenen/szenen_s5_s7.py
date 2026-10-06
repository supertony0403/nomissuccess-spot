"""Shared helpers for the nomissuccess shots s5_betrieb, s6_beweis, s7_team.

Pure-Python parts (importable without Blender, unit-tested):

* :class:`DartPlane` - rigid fold kinematics of a classic dart paper plane.
  Every fold is a rotation of a set of material points about a hinge whose
  axis is carried by material points itself, so partial folds compose
  correctly and every rigid panel stays isometric.
* :func:`dash_layout`, :func:`polyline_resample`, :func:`bezier` - paths.
* :data:`SCREENSHOTS` - the real screenshots with crops that remove vendor
  logos (Proxmox header bar, Grafana sidebar, "Powered by Grafana").

Blender parts (``bpy``): CLI, stage world, cards with rounded corners, thin
edge and soft shadow, paper material with rolling digits, per-frame shape-key
and transform baking, rigid-body simulation baked to keyframes, centre-square
probe, and the render/encode runner.  The library in ``blender/lib`` is only
imported, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # Blender-only parts
    import bpy
    from mathutils import Matrix, Quaternion, Vector
except ImportError:  # pragma: no cover - pure-python tests
    bpy = None

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "blender" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

ASSETS = REPO / "blender" / "assets" / "s5_s7"
SITE = REPO.parent / "nomissuccess-website" / "src" / "assets"
RENDERS = REPO / "renders"
WORK = REPO / "work" / "blender" / "s5_s7"
TICKET_BAKE = ASSETS / "tickets_bake.json"
CENTRE = (420.0, 1500.0)
DEFAULT_SAMPLES = 16

# real screenshots; crop = (x0, y0, x1, y1) in source pixels
SCREENSHOTS: dict[str, dict] = {
    "proxmox-ui": {"datei": SITE / "proxmox-ui.png", "crop": (0, 36, 1920, 1080)},
    "pbs-dashboard": {"datei": SITE / "pbs-dashboard.png", "crop": (0, 42, 1920, 1080)},
    "grafana-dashboard": {"datei": SITE / "grafana-dashboard.png", "crop": (100, 0, 3360, 2100)},
    "grafana-monitoring": {"datei": SITE / "grafana-monitoring.png", "crop": (0, 0, 1600, 900)},
    "webseite-eigen": {"datei": SITE / "webseite-eigen.png", "crop": (0, 0, 1120, 700)},
}
SIZES = {
    "proxmox-ui": (1920, 1080),
    "pbs-dashboard": (1920, 1080),
    "grafana-dashboard": (3360, 2100),
    "grafana-monitoring": (1600, 926),
    "webseite-eigen": (1120, 760),
}

PRISMA = {
    "nacht": "#0b0c14",
    "nacht_flaeche": "#161826",
    "blau": "#145fe4",
    "violett": "#7c3aed",
    "rosa": "#e44b8d",
    "mint": "#10b981",
    "tinte": "#101223",
    "papier": "#ffffff",
}


def hex_lin(h: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (out[0], out[1], out[2], a)


# ==========================================================================
# easing (pure)
# ==========================================================================

def clamp01(t: float) -> float:
    return 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else t


def smootherstep(t: float) -> float:
    t = clamp01(t)
    return t * t * t * (t * (6 * t - 15) + 10)


def span(f: float, f0: float, f1: float, kind: str = "smoother") -> float:
    """Eased progress of ``f`` in [f0, f1]."""
    if f1 <= f0:
        return 1.0 if f >= f1 else 0.0
    t = clamp01((f - f0) / (f1 - f0))
    if kind == "linear":
        return t
    if kind == "smoother":
        return smootherstep(t)
    if kind == "smooth":
        return t * t * (3 - 2 * t)
    if kind == "out_cubic":
        return 1 - (1 - t) ** 3
    if kind == "out_quint":
        return 1 - (1 - t) ** 5
    if kind == "in_cubic":
        return t ** 3
    if kind == "in_quad":
        return t * t
    if kind == "out_quad":
        return 1 - (1 - t) ** 2
    if kind == "in_out_sine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if kind == "out_expo":
        return 1.0 if t >= 1.0 else 1.0 - 2.0 ** (-10.0 * t)
    raise ValueError(kind)


def lerp(a, b, t):
    if isinstance(a, (tuple, list)):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    return a + (b - a) * t


def keyed_path(f: float, keys: list[tuple[float, tuple]], kind: str = "smoother") -> tuple:
    """Piecewise eased interpolation through ``[(frame, value), ...]``."""
    if f <= keys[0][0]:
        return tuple(keys[0][1])
    for (f0, a), (f1, b) in zip(keys, keys[1:]):
        if f <= f1:
            return lerp(tuple(a), tuple(b), span(f, f0, f1, kind))
    return tuple(keys[-1][1])


def smooth_path(f: float, keys: list[tuple[float, tuple]]) -> np.ndarray:
    """Monotone cubic Hermite through ``[(frame, xyz), ...]`` (per axis).

    Fritsch-Butland tangents never overshoot a key, so a camera flowing
    through set pieces stops exactly where a beat is framed; a key whose
    neighbours lie on both sides in an axis is passed with zero velocity in
    that axis.  First and last key are at rest.
    """
    fs = np.asarray([float(k[0]) for k in keys])
    ps = np.asarray([np.asarray(k[1], dtype=np.float64) for k in keys])
    if np.any(np.diff(fs) <= 0):
        raise ValueError(f"Kamera-Keys nicht aufsteigend: {fs.tolist()}")
    if f <= fs[0]:
        return ps[0].copy()
    if f >= fs[-1]:
        return ps[-1].copy()
    h = np.diff(fs)
    delta = np.diff(ps, axis=0) / h[:, None]
    m = np.zeros_like(ps)
    for i in range(1, len(ps) - 1):
        d0, d1 = delta[i - 1], delta[i]
        same = d0 * d1 > 0
        w0, w1 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = (w0 + w1) / (w0 / d0 + w1 / d1)
        m[i] = np.where(same, hm, 0.0)
    i = int(min(max(np.searchsorted(fs, f, side="right") - 1, 0), len(fs) - 2))
    t = (f - fs[i]) / h[i]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ps[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
            + (-2 * t3 + 3 * t2) * ps[i + 1] + (t3 - t2) * h[i] * m[i + 1])


def catmull(f: float, keys: list[tuple[float, tuple]], rest: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """C1-smooth Hermite spline through keyed points ``[(frame, xyz), ...]``.

    Inner tangents follow Catmull-Rom scaled to the (non-uniform) key
    spacing, so a camera flows through inner keys without stopping; the
    first/last key are at rest when ``rest`` says so.
    """
    fs = [float(k[0]) for k in keys]
    ps = [np.asarray(k[1], dtype=np.float64) for k in keys]
    if f <= fs[0]:
        return ps[0]
    if f >= fs[-1]:
        return ps[-1]
    i = max(0, min(len(fs) - 2, int(np.searchsorted(fs, f, side="right")) - 1))
    d1 = fs[i + 1] - fs[i]
    t = (f - fs[i]) / d1

    def tangent(j: int) -> np.ndarray:
        if j == 0:
            return np.zeros(3) if rest[0] else (ps[1] - ps[0]) / (fs[1] - fs[0])
        if j == len(ps) - 1:
            return np.zeros(3) if rest[1] else (ps[-1] - ps[-2]) / (fs[-1] - fs[-2])
        return (ps[j + 1] - ps[j - 1]) / (fs[j + 1] - fs[j - 1])

    m1, m2 = tangent(i) * d1, tangent(i + 1) * d1
    p1, p2 = ps[i], ps[i + 1]
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * p1 + (t3 - 2 * t2 + t) * m1 + (-2 * t3 + 3 * t2) * p2 + (t3 - t2) * m2


# ==========================================================================
# paths (pure)
# ==========================================================================

def bezier(p0, p1, p2, p3, n: int = 200) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def arc_lengths(pts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def polyline_at(pts: np.ndarray, s: float, lengths: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit tangent at arc length ``s``."""
    L = arc_lengths(pts) if lengths is None else lengths
    s = min(max(s, 0.0), L[-1])
    i = int(min(max(np.searchsorted(L, s) - 1, 0), len(pts) - 2))
    seg = L[i + 1] - L[i]
    t = 0.0 if seg <= 0 else (s - L[i]) / seg
    p = pts[i] + (pts[i + 1] - pts[i]) * t
    d = pts[i + 1] - pts[i]
    n = np.linalg.norm(d)
    return p, (d / n if n > 0 else np.array([0.0, 1.0, 0.0]))


def polyline_resample(pts: np.ndarray, n: int) -> np.ndarray:
    L = arc_lengths(pts)
    return np.array([polyline_at(pts, s, L)[0] for s in np.linspace(0, L[-1], n)])


def dash_layout(pts: np.ndarray, dash: float, gap: float) -> list[tuple[float, float]]:
    """Arc-length intervals ``(s0, s1)`` of the dashes along a polyline."""
    L = arc_lengths(np.asarray(pts, dtype=np.float64))[-1]
    out = []
    s = 0.0
    while s < L - 1e-9:
        out.append((s, min(s + dash, L)))
        s += dash + gap
    return out


# ==========================================================================
# paper plane fold kinematics (pure)
# ==========================================================================

@dataclass
class Fold:
    name: str
    p0: tuple[float, float]          # fold line in the reference state before this fold
    p1: tuple[float, float]
    sample: tuple[float, float]      # a point of the moving side (reference state)
    hinge: tuple[tuple[float, float], tuple[float, float]]  # material points (sheet coords)
    static: tuple[float, float]      # material point on the static side
    direction: tuple[float, float, float]  # initial motion of the flap (object frame)
    offset: float                    # layer offset of the hinge (along direction)
    only: tuple[int, bool] | None = None  # (fold index, member) material filter


def _reflect(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (b - a) / np.linalg.norm(b - a)
    v = p - a
    proj = np.outer(v @ d, d)
    return a + 2 * proj - v


def _rotate(p: np.ndarray, a: np.ndarray, u: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation of points ``p`` about the axis through ``a`` along ``u``."""
    v = p - a
    c, s = math.cos(ang), math.sin(ang)
    cross = np.cross(u, v)
    dot = v @ u
    return a + v * c + cross * s + np.outer(dot, u) * (1 - c)


class DartPlane:
    """Classic dart paper plane folded from a ``width`` x ``height`` sheet.

    Sheet coordinates: x right, y up (nose at ``+height/2``), printed side
    facing +Z.  Folds in order: top corners to the centre line (1L, 1R), the
    new edges to the centre line again (2L, 2R), the sheet in half (3), and
    both wings down (4R, 4L).  The final pose: nose +Y, wings above the
    keel (+X), right wing towards +Z.
    """

    WING = 0.12  # wing hinge distance from the keel (fraction of width)

    def __init__(self, width: float = 0.21, height: float = 0.297, eps: float = 0.00015):
        self.W, self.H = float(width), float(height)
        hw, hh = self.W / 2, self.H / 2
        t = math.tan(math.radians(22.5))
        d = self.WING * self.W
        nose = (0.0, hh)
        y2 = hh - hw / t
        e = eps
        self.folds = [
            Fold("1L", nose, (-hw, hh - hw), (-hw + 1e-3, hh - 1e-3), (nose, (-hw, hh - hw)), (0.0, 0.0), (0, 0, -1), e),
            Fold("1R", nose, (hw, hh - hw), (hw - 1e-3, hh - 1e-3), (nose, (hw, hh - hw)), (0.0, 0.0), (0, 0, -1), e),
            Fold("2L", nose, (-hw, y2), (-hw + 1e-3, 0.0), (nose, (-hw, y2)), (0.0, -0.05), (0, 0, -1), 3 * e),
            Fold("2R", nose, (hw, y2), (hw - 1e-3, 0.0), (nose, (hw, y2)), (0.0, -0.05), (0, 0, -1), 3 * e),
            Fold("3", nose, (0.0, -hh), (-0.05, 0.0), (nose, (0.0, -hh)), (0.05, -0.05), (0, 0, -1), 7 * e),
            Fold("4R", (d, hh), (d, -hh), (d + 0.01, -0.05), ((d, 0.0), (d, -hh)), (d * 0.5, -0.05), (0, 0, 1), 0.0, (4, False)),
            Fold("4L", (d, hh), (d, -hh), (d + 0.01, -0.05), ((-d, 0.0), (-d, -hh)), (-d * 0.5, -0.05), (0, 0, -1), 0.0, (4, True)),
        ]
        self.final = {"1L": math.pi, "1R": math.pi, "2L": math.pi, "2R": math.pi, "3": math.pi - 0.22, "4R": math.radians(86.0), "4L": math.radians(86.0)}
        self._calibrate()

    # ---- grids -----------------------------------------------------------
    def sheet_grid(self, nx: int = 101, ny: int = 143) -> tuple[np.ndarray, tuple[int, int]]:
        """Regular grid over the sheet, rows from bottom to top."""
        xs = np.linspace(-self.W / 2, self.W / 2, nx)
        ys = np.linspace(-self.H / 2, self.H / 2, ny)
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1), (nx, ny)

    # ---- masks -----------------------------------------------------------
    def _masks(self, pts2: np.ndarray) -> np.ndarray:
        """Membership of every material point in every fold (n, K)."""
        R = np.asarray(pts2, dtype=np.float64).copy()
        masks = np.zeros((len(R), len(self.folds)), dtype=bool)
        for k, fd in enumerate(self.folds):
            a, b = np.asarray(fd.p0), np.asarray(fd.p1)
            dvec = b - a
            side = np.sign(dvec[0] * (fd.sample[1] - a[1]) - dvec[1] * (fd.sample[0] - a[0]))
            c = dvec[0] * (R[:, 1] - a[1]) - dvec[1] * (R[:, 0] - a[0])
            m = side * c > 1e-12 * max(self.W, self.H)
            if fd.only is not None:
                idx, member = fd.only
                m &= masks[:, idx] == member
            masks[:, k] = m
            if m.any():
                R[m] = _reflect(R[m], a, b)
        return masks

    def signatures(self, pts2: np.ndarray) -> np.ndarray:
        return self._masks(pts2)

    def _extra(self) -> np.ndarray:
        pts = []
        for fd in self.folds:
            pts.extend([fd.hinge[0], fd.hinge[1], fd.static])
        return np.asarray(pts, dtype=np.float64)

    # ---- kinematics --------------------------------------------------------
    def _apply(self, pts2: np.ndarray, angles: dict[str, float], signs=None) -> np.ndarray:
        n = len(pts2)
        allp = np.concatenate([np.asarray(pts2, dtype=np.float64), self._extra()])
        masks = self._masks(allp)
        P = np.concatenate([allp, np.zeros((len(allp), 1))], axis=1)
        for k, fd in enumerate(self.folds):
            ang = angles.get(fd.name, 0.0)
            j = n + 3 * k
            h0, h1, st = P[j], P[j + 1], P[j + 2]
            u = (h1 - h0) / np.linalg.norm(h1 - h0)
            nrm = np.cross(h1 - h0, st - h0)
            nrm /= np.linalg.norm(nrm)
            sn, sr = signs[k] if signs is not None else (1.0, 1.0)
            dirv = sn * nrm
            if ang == 0.0:
                continue
            a = h0 + fd.offset * dirv
            m = masks[:, k]
            P[m] = _rotate(P[m], a, u, sr * ang)
        return P

    def _calibrate(self) -> None:
        """Choose normal and rotation signs from the reference state."""
        grid, _ = self.sheet_grid(41, 57)
        n = len(grid)
        signs: list[tuple[float, float]] = []
        for k, fd in enumerate(self.folds):
            ref = {f.name: math.pi for f in self.folds[:k]}
            # positions before fold k with the signs found so far
            P = self._apply(grid, ref, signs + [(1.0, 1.0)] * (len(self.folds) - k))
            allp = np.concatenate([grid, self._extra()])
            masks = self._masks(allp)
            j = n + 3 * k
            h0, h1, st = P[j], P[j + 1], P[j + 2]
            u = (h1 - h0) / np.linalg.norm(h1 - h0)
            nrm = np.cross(h1 - h0, st - h0)
            nrm /= np.linalg.norm(nrm)
            want = np.asarray(fd.direction, dtype=np.float64)
            sn = 1.0 if nrm @ want >= 0 else -1.0
            mov = P[:n][masks[:n, k]]
            c = mov.mean(axis=0)
            r = (c - h0) - ((c - h0) @ u) * u
            sr = 1.0 if np.cross(u, r) @ want >= 0 else -1.0
            signs.append((sn, sr))
        self.signs = signs

    def positions(self, pts2: np.ndarray, angles: dict[str, float]) -> np.ndarray:
        """3D positions (n, 3) of sheet points for the given fold angles."""
        return self._apply(pts2, angles, self.signs)[: len(pts2)]

    def angles_at(self, t: float) -> dict[str, float]:
        """Fold angles for the global fold progress ``t`` in [0, 1]."""
        sched = {
            "1L": (0.00, 0.17),
            "1R": (0.05, 0.22),
            "2L": (0.24, 0.41),
            "2R": (0.29, 0.46),
            "3": (0.50, 0.70),
            "4R": (0.70, 0.94),
            "4L": (0.72, 0.96),
        }
        out = {}
        for name, (a, b) in sched.items():
            full = math.pi if name in ("3",) else self.final[name]
            out[name] = full * smootherstep((t - a) / (b - a))
        # the half fold opens slightly into the keel V at the end
        out["3"] -= (math.pi - self.final["3"]) * smootherstep((t - 0.80) / 0.20)
        return out

    def flight_frame(self, pos: np.ndarray, pts2: np.ndarray | None = None) -> dict:
        """Keel point, forward/up/right axes, span and mirror error."""
        if pts2 is None:
            pts2, (nx, ny) = self.sheet_grid(41, 57)
        else:
            xs = np.unique(np.round(pts2[:, 0], 12))
            nx = len(xs)
            ny = len(pts2) // nx
        keel = np.abs(pts2[:, 0]) < 1e-9
        kp = pos[keel]
        ky = pts2[keel, 1]
        fwd = kp[np.argmax(ky)] - kp[np.argmin(ky)]
        fwd /= np.linalg.norm(fwd)
        K = kp.mean(axis=0)
        d = self.WING * self.W
        col = np.abs(np.abs(pts2[:, 0]) - d)
        roots = pos[col <= col.min() + 1e-9]
        up = roots.mean(axis=0) - K
        up -= (up @ fwd) * fwd
        up /= np.linalg.norm(up)
        right = np.cross(fwd, up)
        span_w = float((pos @ right).max() - (pos @ right).min())
        idx = np.arange(nx * ny).reshape(ny, nx)
        mirror = idx[:, ::-1].ravel()
        refl = pos[mirror] - 2 * np.outer((pos[mirror] - K) @ right, right)
        sym = float(np.max(np.linalg.norm(refl - pos, axis=1)))
        return {"kiel": K, "vor": fwd, "oben": up, "rechts": right, "spannweite": span_w, "symmetrie_fehler": sym}


# ==========================================================================
# Blender: command line + shot context
# ==========================================================================

def parse_args(argv: list[str] | None = None):
    """Shot CLI: ``--proxy N`` and the probe/bake/resume flags on top of
    ``nomiss_render.parse_args``."""
    import nomiss_render as R

    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--proxy", type=int, default=0)
    ap.add_argument("--probe", default="")
    ap.add_argument("--bake-check", default="")
    ap.add_argument("--bake", action="store_true", help="re-simulate and freeze the rigid-body bake")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--encode", action="store_true", help="only encode renders/<shot>/ to .mov + .json")
    own, rest = ap.parse_known_args(argv)
    args = R.parse_args(rest)
    if own.proxy:
        args.res = own.proxy
    for k, v in vars(own).items():
        setattr(args, k.replace("-", "_"), v)
    if not args.samples:
        args.samples = DEFAULT_SAMPLES
    return args


@dataclass
class ShotCtx:
    shot_id: str
    scene: object
    sh: object                      # nomiss_timeline.Shot
    events: dict[str, int]
    out: Path
    keys: dict[str, list] = field(default_factory=dict)   # event -> key objects
    extra: dict = field(default_factory=dict)

    def key(self, event: str, *objs) -> None:
        self.keys.setdefault(event, []).extend(objs)


def start_shot(shot_id: str, args, event_ids: tuple[str, ...]) -> ShotCtx:
    import nomiss_render as R
    import nomiss_timeline as TL

    tl = TL.load(args.timeline or None)
    sh = TL.shot(shot_id, tl)
    events = {e: TL.event_frame(e, shot_id, tl) for e in event_ids}
    scene = R.clean_scene()
    out = R.setup(
        scene,
        shot_id,
        sh.frames,
        res=args.res,
        samples=args.samples,
        motion_blur=not args.no_mb,
        glare=not args.no_glare,
        out_dir=args.out or None,
    )
    scene.eevee.use_shadows = True
    return ShotCtx(shot_id, scene, sh, events, Path(out))


# ==========================================================================
# Blender: generic builders
# ==========================================================================

def link(ob, collection=None):
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh_object(name: str, verts, faces, *, uvs=None, mats=None, mat_index=None, smooth=False, collection=None):
    """Mesh from numpy verts/faces; ``uvs`` per face-corner (loop order)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, dtype=np.float64).tolist(), [], [list(map(int, f)) for f in faces])
    me.validate()
    if uvs is not None:
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32).ravel())
    for m in mats or []:
        me.materials.append(m)
    if mat_index is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_index, dtype=np.int32))
    me.polygons.foreach_set("use_smooth", np.full(len(me.polygons), bool(smooth)))
    me.update()
    ob = bpy.data.objects.new(name, me)
    return link(ob, collection)


def empty(name: str, loc=(0, 0, 0), collection=None):
    ob = bpy.data.objects.new(name, None)
    ob.location = loc
    ob.empty_display_size = 0.1
    return link(ob, collection)


def rounded_rect(w: float, h: float, r: float, seg: int = 8) -> np.ndarray:
    """Counter-clockwise outline (n, 2) of a rounded rectangle centred at 0."""
    r = min(r, w / 2, h / 2)
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0.0), (-w / 2 + r, h / 2 - r, 90.0), (-w / 2 + r, -h / 2 + r, 180.0), (w / 2 - r, -h / 2 + r, 270.0)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90.0 * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.asarray(pts)


def slab(name: str, w: float, h: float, depth: float, r: float, *, uv_rect=(0.0, 0.0, 1.0, 1.0), rim: float = 0.0, mats=(), seg: int = 8, collection=None):
    """Rounded slab in local XY, front face +Z.

    Material slots: 0 front (UV-mapped to ``uv_rect`` = u0, v0, u1, v1),
    1 side edge, 2 back, 3 hairline rim on the front (if ``rim`` > 0).
    """
    outer = rounded_rect(w, h, r, seg)
    n = len(outer)
    z1, z0 = depth / 2, -depth / 2
    verts, faces, idx, uvs = [], [], [], []
    u0, v0, u1, v1 = uv_rect

    def uv_of(p):
        return (u0 + (p[0] / w + 0.5) * (u1 - u0), v0 + (p[1] / h + 0.5) * (v1 - v0))

    def add(ps, z):
        base = len(verts)
        verts.extend([(p[0], p[1], z) for p in ps])
        return base

    fo = add(outer, z1)
    bo = add(outer, z0)
    if rim > 0:
        inner = rounded_rect(w - 2 * rim, h - 2 * rim, max(r - rim, 1e-4), seg)
        fi = add(inner, z1)
        faces.append([fi + i for i in range(n)])
        idx.append(0)
        uvs.extend([uv_of(p) for p in inner])
        for i in range(n):
            j = (i + 1) % n
            faces.append([fo + i, fo + j, fi + j, fi + i])
            idx.append(3)
            uvs.extend([uv_of(outer[i]), uv_of(outer[j]), uv_of(inner[j]), uv_of(inner[i])])
    else:
        faces.append([fo + i for i in range(n)])
        idx.append(0)
        uvs.extend([uv_of(p) for p in outer])
    faces.append([bo + i for i in reversed(range(n))])
    idx.append(2)
    uvs.extend([(0.0, 0.0)] * n)
    for i in range(n):
        j = (i + 1) % n
        faces.append([bo + i, bo + j, fo + j, fo + i])
        idx.append(1)
        uvs.extend([(0.0, 0.0)] * 4)
    mats = list(mats)
    if not mats:
        raise ValueError("slab braucht mindestens ein Material")
    while len(mats) < 4:
        mats.append(mats[-1])
    return mesh_object(name, verts, faces, uvs=uvs, mats=mats, mat_index=idx, collection=collection)


def rounded_box(name: str, size, r: float, mats=(), collection=None):
    """Box with rounded vertical edges (slab standing on Z)."""
    sx, sy, sz = size
    ob = slab(name, sx, sy, sz, r, mats=mats, seg=6, collection=collection)
    return ob


def area_light(name, loc, target, size, energy, color, shape="SQUARE", size_y=None, shadow=True, collection=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ld.energy = energy
    ld.color = color
    if hasattr(ld, "use_shadow"):
        ld.use_shadow = shadow
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return link(ob, collection)


def point_light(name, loc, energy, color, radius=0.05, collection=None):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = energy
    ld.color = color
    ld.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    return link(ob, collection)


# ---- node helpers ----------------------------------------------------------

class NT:
    """Tiny shader-node builder."""

    def __init__(self, tree):
        self.t = tree

    def n(self, typ, **props):
        node = self.t.nodes.new(typ)
        for k, v in props.items():
            setattr(node, k, v)
        return node

    def put(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        else:
            sock.default_value = v

    def m(self, op, a, b=None, c=None, clamp=False):
        node = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.put(node.inputs[i], v)
        return node.outputs[0]

    def value(self, name, v):
        node = self.n("ShaderNodeValue", name=name, label=name)
        node.outputs[0].default_value = v
        return node.outputs[0]

    def mix(self, fac, a, b):
        node = self.n("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
        self.put(node.inputs[0], fac)
        self.put(node.inputs[6], a)
        self.put(node.inputs[7], b)
        return node.outputs[2]

    def mixf(self, fac, a, b):
        node = self.n("ShaderNodeMix", data_type="FLOAT")
        self.put(node.inputs[0], fac)
        self.put(node.inputs[2], a)
        self.put(node.inputs[3], b)
        return node.outputs[0]

    def smooth(self, x, lo, hi, kind="SMOOTHSTEP"):
        node = self.n("ShaderNodeMapRange", interpolation_type=kind)
        self.put(node.inputs["Value"], x)
        self.put(node.inputs["From Min"], lo)
        self.put(node.inputs["From Max"], hi)
        return node.outputs["Result"]

    def sep(self, v):
        node = self.n("ShaderNodeSeparateXYZ")
        self.put(node.inputs[0], v)
        return node.outputs

    def comb(self, x, y, z=0.0):
        node = self.n("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.put(node.inputs[i], v)
        return node.outputs[0]

    def scale_color(self, col, s):
        node = self.n("ShaderNodeVectorMath", operation="SCALE")
        self.put(node.inputs[0], col)
        self.put(node.inputs["Scale"], s)
        return node.outputs["Vector"]

    def add_color(self, a, b):
        node = self.n("ShaderNodeVectorMath", operation="ADD")
        self.put(node.inputs[0], a)
        self.put(node.inputs[1], b)
        return node.outputs["Vector"]


def new_mat(name: str):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if hasattr(mat, "use_nodes") and not mat.use_nodes:
        mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat, NT(mat.node_tree)


def set_alpha_mode(mat, mode: str = "DITHERED", shadow: bool = True) -> None:
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = mode
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = shadow


def principled(name, color, *, rough=0.4, metal=0.0, spec=0.5, coat=0.0, coat_rough=0.05, emit=None, emit_strength=0.0, alpha=1.0):
    mat, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    bs.inputs["Base Color"].default_value = color
    bs.inputs["Roughness"].default_value = rough
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Specular IOR Level"].default_value = spec
    bs.inputs["Coat Weight"].default_value = coat
    bs.inputs["Coat Roughness"].default_value = coat_rough
    if emit is not None:
        bs.inputs["Emission Color"].default_value = emit
        bs.inputs["Emission Strength"].default_value = emit_strength
    bs.inputs["Alpha"].default_value = alpha
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    return mat


def emission_mat(name, color, strength=1.0, *, value_name=None):
    mat, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    em = b.n("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    s = b.value(value_name, strength) if value_name else strength
    b.put(em.inputs["Strength"], s)
    b.t.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def load_image(path: Path, name: str | None = None):
    img = bpy.data.images.load(str(path), check_existing=True)
    if name:
        img.name = name
    img.colorspace_settings.name = "sRGB"
    return img


def screenshot_uv(key: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = SCREENSHOTS[key]["crop"]
    W, H = SIZES[key]
    return (x0 / W, 1.0 - y1 / H, x1 / W, 1.0 - y0 / H)


def screen_material(name: str, image, *, glow: float = 0.92, value_prefix: str = "", ripple: bool = False, aspect: float = 1.6):
    """Screenshot face: exact colours (emission) plus a glossy coat sweep.

    Value nodes: ``<prefix>Hell`` (brightness 0..1), and with ``ripple`` a
    mint ring ``<prefix>WelleR`` (radius in card heights) / ``WelleA``.
    """
    mat, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    uv = b.n("ShaderNodeUVMap", uv_map="UVMap")
    tex = b.n("ShaderNodeTexImage", image=image, interpolation="Cubic", extension="EXTEND")
    b.put(tex.inputs["Vector"], uv.outputs["UV"])
    hell = b.value(value_prefix + "Hell", 1.0)
    col = b.scale_color(tex.outputs["Color"], hell)
    b.put(bs.inputs["Base Color"], b.scale_color(col, 0.22))
    bs.inputs["Roughness"].default_value = 0.42
    bs.inputs["Specular IOR Level"].default_value = 0.35
    bs.inputs["Coat Weight"].default_value = 0.35
    bs.inputs["Coat Roughness"].default_value = 0.06
    emis = col
    strength = glow
    if ripple:
        # ring around the contact point (uv 0..1 of the face), mint
        geo = b.n("ShaderNodeTexCoord")
        s = b.sep(geo.outputs["Object"])
        cx, cy = b.value(value_prefix + "WelleX", 0.0), b.value(value_prefix + "WelleY", 0.0)
        dx = b.m("SUBTRACT", s[0], cx)
        dy = b.m("SUBTRACT", s[1], cy)
        dist = b.m("SQRT", b.m("ADD", b.m("MULTIPLY", dx, dx), b.m("MULTIPLY", dy, dy)))
        rr = b.value(value_prefix + "WelleR", 0.0)
        width = b.m("ADD", 0.012, b.m("MULTIPLY", rr, 0.08))
        q = b.m("DIVIDE", b.m("SUBTRACT", dist, rr), width)
        ring = b.m("EXPONENT", b.m("MULTIPLY", b.m("MULTIPLY", q, q), -1.0))
        inner = b.m("MULTIPLY", b.m("LESS_THAN", dist, rr), 0.05)
        amp = b.value(value_prefix + "WelleA", 0.0)
        w = b.m("MULTIPLY", b.m("ADD", ring, inner), amp)
        mint = hex_lin(PRISMA["mint"])
        glow_col = b.scale_color((mint[0] * 1.4, mint[1] * 1.4, mint[2] * 1.4), w)
        emis = b.add_color(col, glow_col)
    b.put(bs.inputs["Emission Color"], emis)
    bs.inputs["Emission Strength"].default_value = strength
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    return mat


def edge_material(name="KarteKante"):
    return principled(name, (0.58, 0.6, 0.66, 1.0), rough=0.28, metal=0.85, spec=0.6)


def back_material(name="KarteRueck"):
    return principled(name, hex_lin("#12131c"), rough=0.45, metal=0.2, spec=0.4)


def rim_material(name="KarteRand", strength=0.5):
    return principled(name, (0.8, 0.82, 0.88, 1.0), rough=0.3, metal=0.6, emit=(0.85, 0.88, 0.95, 1.0), emit_strength=strength)


def shadow_material(name: str, w: float, h: float, r: float, blur: float, alpha: float = 0.6):
    """Black with a soft rounded-rect alpha falloff (object space)."""
    mat, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    bs.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bs.inputs["Roughness"].default_value = 1.0
    bs.inputs["Specular IOR Level"].default_value = 0.0
    tc = b.n("ShaderNodeTexCoord")
    s = b.sep(tc.outputs["Object"])
    qx = b.m("MAXIMUM", b.m("SUBTRACT", b.m("ABSOLUTE", s[0]), w / 2 - r), 0.0)
    qy = b.m("MAXIMUM", b.m("SUBTRACT", b.m("ABSOLUTE", s[1]), h / 2 - r), 0.0)
    sdf = b.m("SUBTRACT", b.m("SQRT", b.m("ADD", b.m("MULTIPLY", qx, qx), b.m("MULTIPLY", qy, qy))), r)
    a = b.m("MULTIPLY", b.m("SUBTRACT", 1.0, b.smooth(sdf, -blur * 0.35, blur)), b.value("Schatten", alpha))
    b.put(bs.inputs["Alpha"], a)
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    set_alpha_mode(mat, "BLENDED", shadow=False)
    return mat


@dataclass
class Card:
    ob: object
    w: float
    h: float
    mat: object
    shadow: object | None


def screenshot_card(name: str, key: str, width: float, *, depth: float | None = None, shadow: bool = True, ripple: bool = False, glow: float = 0.92, uv_rect=None, collection=None) -> Card:
    """A real screenshot as a card: radius 16 px (at 1920 px width), thin
    metallic edge with a hairline rim, and a soft shadow behind."""
    x0, y0, x1, y1 = SCREENSHOTS[key]["crop"]
    if uv_rect is None:
        uv_rect = screenshot_uv(key)
        px_w, px_h = x1 - x0, y1 - y0
    else:
        W, H = SIZES[key]
        px_w, px_h = (uv_rect[2] - uv_rect[0]) * W, (uv_rect[3] - uv_rect[1]) * H
    h = width * px_h / px_w
    r = width * 16.0 / 1920.0
    depth = depth if depth is not None else width * 0.012
    img = load_image(SCREENSHOTS[key]["datei"])
    mat = screen_material(f"{name}_Bild", img, glow=glow, ripple=ripple, aspect=width / h)
    ob = slab(name, width, h, depth, r, uv_rect=uv_rect, rim=width * 0.0016, mats=[mat, edge_material(), back_material(), rim_material()], collection=collection)
    sh = None
    if shadow:
        blur = width * 0.07
        sw, shh = width + 2.2 * blur, h + 2.2 * blur
        sm = shadow_material(f"{name}_Schatten", width * 0.98, h * 0.98, r * 3, blur)
        verts = [(-sw / 2, -shh / 2, 0), (sw / 2, -shh / 2, 0), (sw / 2, shh / 2, 0), (-sw / 2, shh / 2, 0)]
        sh = mesh_object(f"{name}_Schatten", verts, [[0, 1, 2, 3]], mats=[sm], collection=collection)
        sh.parent = ob
        sh.location = (width * 0.01, -h * 0.035, -depth / 2 - width * 0.03)
        sh.visible_shadow = False
    return Card(ob, width, h, mat, sh)


def mat_value(mat, name: str):
    return mat.node_tree.nodes[name].outputs[0]


def key_value(mat, name: str, frame: float, value: float, interp: str = "BEZIER", easing: str = "AUTO") -> None:
    import nomiss_camera as C

    sock = mat_value(mat, name)
    sock.default_value = value
    path = f'nodes["{name}"].outputs[0].default_value'
    mat.node_tree.keyframe_insert(path, frame=frame)
    C.set_key_interp(mat.node_tree, path, frame, interp, easing)


def bake_value(mat, name: str, frames, fn) -> None:
    """Per-frame keys of a value node (linear in between)."""
    bake_fast(mat.node_tree, f'nodes["{name}"].outputs[0].default_value', list(frames), [[fn(f)] for f in frames])


def bake_fast(id_data, path: str, frames: list, values: list) -> None:
    """Write per-frame keys directly into the f-curves (linear)."""
    import nomiss_camera as C

    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    dims = vals.shape[1]
    # create the curves with one key each
    for i in range(dims):
        if dims == 1 and not _is_array(id_data, path):
            id_data.keyframe_insert(path, frame=frames[0])
        else:
            id_data.keyframe_insert(path, frame=frames[0], index=i)
    curves = [fc for fc in C.fcurves(id_data) if fc.data_path == path]
    for fc in curves:
        i = fc.array_index if dims > 1 else 0
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(frames))
        co = np.empty(2 * len(frames))
        co[0::2] = frames
        co[1::2] = vals[:, i]
        fc.keyframe_points.foreach_set("co", co)
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.update()


def _is_array(id_data, path: str) -> bool:
    try:
        v = id_data.path_resolve(path)
    except ValueError:
        return False
    return hasattr(v, "__len__") and not isinstance(v, str)


def bake_transform(ob, frames: list, mats: list) -> None:
    """Bake location + quaternion from 4x4 world matrices (no parent)."""
    ob.rotation_mode = "QUATERNION"
    locs, quats = [], []
    prev = None
    for M in mats:
        M = Matrix(np.asarray(M).tolist()) if not isinstance(M, Matrix) else M
        loc, rot, sca = M.decompose()
        q = rot
        if prev is not None and q.dot(prev) < 0:
            q = Quaternion((-q.w, -q.x, -q.y, -q.z))
        prev = q
        locs.append(tuple(loc))
        quats.append((q.w, q.x, q.y, q.z))
    bake_fast(ob, "location", frames, locs)
    bake_fast(ob, "rotation_quaternion", frames, quats)


def bake_shape_frames(ob, frames: list, positions: list, lead: float | None = None) -> None:
    """One shape key per frame, blended linearly between neighbours.

    ``lead`` is the frame where the first key is still 0 (the basis shape);
    after the last frame the last key stays fully on.
    """
    import nomiss_camera as C

    if ob.data.shape_keys is None:
        ob.shape_key_add(name="Basis", from_mix=False)
    keys = []
    for f, P in zip(frames, positions):
        k = ob.shape_key_add(name=f"F{int(f)}", from_mix=False)
        k.data.foreach_set("co", np.asarray(P, dtype=np.float32).ravel())
        k.slider_min = 0.0
        keys.append(k)
    sk = ob.data.shape_keys
    for i, k in enumerate(keys):
        pts = []
        if i > 0:
            pts.append((frames[i - 1], 0.0))
        elif lead is not None:
            pts.append((lead, 0.0))
        pts.append((frames[i], 1.0))
        if i + 1 < len(keys):
            pts.append((frames[i + 1], 0.0))
        for f, v in pts:
            k.value = v
            k.keyframe_insert("value", frame=f)
    for fc in C.fcurves(sk):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


# ---- stage -----------------------------------------------------------------

def stage_world(scene, fields: list[tuple], *, zenith="#07080e", horizon="#121423", name="NomissStage", strength=1.0):
    """360-degree PRISMA stage: dark base with wide soft colour fields.

    ``fields`` = [(direction xyz, hex colour, alpha, half-angle deg), ...];
    fields sit off-axis so they glow at the frame edges, never as a centred
    blob.  Value node ``Helligkeit`` scales the whole stage.
    """
    world = bpy.data.worlds.get(name) or bpy.data.worlds.new(name)
    if hasattr(world, "use_nodes") and not world.use_nodes:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    b = NT(nt)
    tc = b.n("ShaderNodeTexCoord")
    nv = b.n("ShaderNodeVectorMath", operation="NORMALIZE")
    b.put(nv.inputs[0], tc.outputs["Generated"])
    d = nv.outputs["Vector"]
    z = b.sep(d)[2]
    col = b.mix(b.smooth(z, -0.25, 0.85), hex_lin(horizon), hex_lin(zenith))
    for dirv, hx, alpha, ang in fields:
        dv = np.asarray(dirv, dtype=np.float64)
        dv /= np.linalg.norm(dv)
        dot = b.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
        b.put(dot.inputs[0], d)
        dot.inputs[1].default_value = tuple(dv)
        f = b.smooth(dot.outputs["Value"], math.cos(math.radians(ang)), 1.0, "SMOOTHERSTEP")
        f = b.m("MULTIPLY", b.m("POWER", f, 1.6), alpha)
        col = b.mix(f, col, hex_lin(hx))
    bg = b.n("ShaderNodeBackground")
    b.put(bg.inputs["Color"], col)
    b.put(bg.inputs["Strength"], b.value("Helligkeit", strength))
    out = b.n("ShaderNodeOutputWorld")
    b.put(out.inputs["Surface"], bg.outputs[0])
    scene.world = world
    return world


def floor(name: str, center, size, *, fade_inner: float, fade_outer: float, color="#0d0e17", rough=0.22, collection=None):
    """Dark glossy floor with reflections that fades out radially."""
    mat, b = new_mat(f"{name}_Mat")
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    bs.inputs["Base Color"].default_value = hex_lin(color)
    bs.inputs["Roughness"].default_value = rough
    bs.inputs["Specular IOR Level"].default_value = 0.55
    tc = b.n("ShaderNodeTexCoord")
    s = b.sep(tc.outputs["Object"])
    # elliptic fade: x stretched by 0.45 so the floor runs long in X
    rx = b.m("MULTIPLY", s[0], 0.45)
    dist = b.m("SQRT", b.m("ADD", b.m("MULTIPLY", rx, rx), b.m("MULTIPLY", s[1], s[1])))
    alpha = b.m("SUBTRACT", 1.0, b.smooth(dist, fade_inner, fade_outer))
    b.put(bs.inputs["Alpha"], alpha)
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    set_alpha_mode(mat, "DITHERED")
    sx, sy = size
    verts = [(-sx / 2, -sy / 2, 0), (sx / 2, -sy / 2, 0), (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, 0)]
    ob = mesh_object(name, verts, [[0, 1, 2, 3]], mats=[mat], collection=collection)
    ob.location = center
    return ob


def dust(name: str, n: int, box_min, box_max, *, seed: int, size=(0.004, 0.012), strength=3.0, color=(0.85, 0.9, 1.0, 1.0), collection=None):
    """Sparse tiny emissive specks for depth parallax (they bokeh out)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(box_min), np.asarray(box_max)
    verts, faces = [], []
    for i in range(n):
        c = lo + rng.random(3) * (hi - lo)
        r = rng.uniform(*size)
        base = len(verts)
        # octahedron
        for d in ((r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)):
            verts.append(tuple(c + np.asarray(d)))
        for a, b2, c2 in ((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)):
            faces.append([base + a, base + b2, base + c2])
    mat = emission_mat(f"{name}_Mat", color, strength)
    ob = mesh_object(name, verts, faces, mats=[mat], collection=collection)
    ob.visible_shadow = False
    return ob


# ---- paper with rolling digits ----------------------------------------------

def paper_material(name: str, front_img, *, back=(0.93, 0.93, 0.92, 1.0), digits=None, ink="#101223"):
    """Matte paper: printed front, plain back, fibre bump.

    ``digits`` = (strip_image, layout dict) adds rolling counter columns
    in the UV window ``layout['fenster_uv']`` with per-column value nodes
    ``Rolle<k>`` (position in digits) and ``Schliere<k>`` (blur in digits).
    The blur is integrated in the shader (9 taps) because a UV scroll is not
    seen by the renderer's motion blur.
    """
    mat, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    bs = b.n("ShaderNodeBsdfPrincipled")
    uvn = b.n("ShaderNodeUVMap", uv_map="UVMap")
    uv = uvn.outputs["UV"]
    tex = b.n("ShaderNodeTexImage", image=front_img, interpolation="Cubic", extension="EXTEND")
    b.put(tex.inputs["Vector"], uv)
    col = tex.outputs["Color"]
    if digits is not None:
        strip, lay = digits
        su = b.sep(uv)
        u, v = su[0], su[1]
        u0, v0, u1, v1 = lay["fenster_uv"]
        lv = b.m("DIVIDE", b.m("SUBTRACT", v, v0), v1 - v0)
        inside_v = b.m("MULTIPLY", b.m("GREATER_THAN", v, v0), b.m("LESS_THAN", v, v1))
        lu = None
        pos = None
        blur = None
        any_col = None
        for k, (a, c) in enumerate(lay["spalten"]):
            mk = b.m("MULTIPLY", b.m("GREATER_THAN", u, a), b.m("LESS_THAN", u, c))
            lu_k = b.m("MULTIPLY", b.m("DIVIDE", b.m("SUBTRACT", u, a), c - a), mk)
            p_k = b.m("MULTIPLY", b.value(f"Rolle{k}", 0.0), mk)
            bl_k = b.m("MULTIPLY", b.value(f"Schliere{k}", 0.0), mk)
            lu = lu_k if lu is None else b.m("ADD", lu, lu_k)
            pos = p_k if pos is None else b.m("ADD", pos, p_k)
            blur = bl_k if blur is None else b.m("ADD", blur, bl_k)
            any_col = mk if any_col is None else b.m("ADD", any_col, mk)
        base_v = b.m("DIVIDE", b.m("ADD", lv, pos), 10.0)
        taps = 9
        acc = None
        for i in range(taps):
            off = b.m("MULTIPLY", blur, (i / (taps - 1) - 0.5) / 10.0)
            tv = b.m("ADD", base_v, off)
            t = b.n("ShaderNodeTexImage", image=strip, interpolation="Linear", extension="REPEAT")
            b.put(t.inputs["Vector"], b.comb(lu, tv, 0.0))
            r = b.sep(t.outputs["Color"])[0]
            acc = r if acc is None else b.m("ADD", acc, r)
        inkv = b.m("MULTIPLY", b.m("MULTIPLY", acc, 1.0 / taps), b.m("MULTIPLY", inside_v, any_col))
        # the roll window shades darker towards its top and bottom edge
        edge = b.m("SUBTRACT", 1.0, b.m("POWER", b.m("ABSOLUTE", b.m("SUBTRACT", b.m("MULTIPLY", lv, 2.0), 1.0)), 6.0))
        inkv = b.m("MULTIPLY", inkv, b.m("MAXIMUM", edge, 0.0))
        col = b.mix(inkv, col, hex_lin(ink))
    geo = b.n("ShaderNodeNewGeometry")
    col = b.mix(geo.outputs["Backfacing"], col, back)
    b.put(bs.inputs["Base Color"], col)
    bs.inputs["Roughness"].default_value = 0.62
    bs.inputs["Specular IOR Level"].default_value = 0.32
    bs.inputs["Sheen Weight"].default_value = 0.25
    bs.inputs["Sheen Roughness"].default_value = 0.4
    noise = b.n("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 420.0
    noise.inputs["Detail"].default_value = 6.0
    bump = b.n("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06
    bump.inputs["Distance"].default_value = 0.0004
    b.put(bump.inputs["Height"], noise.outputs["Fac"])
    b.put(bs.inputs["Normal"], bump.outputs["Normal"])
    b.put(bs.inputs["Emission Color"], col)
    bs.inputs["Emission Strength"].default_value = 0.05
    b.t.links.new(bs.outputs[0], out.inputs["Surface"])
    return mat


def grid_mesh(name: str, pts2: np.ndarray, nx: int, ny: int, uv_rect=(0.0, 0.0, 1.0, 1.0), mats=(), collection=None):
    """Quad grid over sheet points (rows bottom->top), UV from the bounds."""
    verts = np.concatenate([pts2, np.zeros((len(pts2), 1))], axis=1)
    idx = np.arange(nx * ny).reshape(ny, nx)
    q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)
    lo, hi = pts2.min(axis=0), pts2.max(axis=0)
    uvp = (pts2 - lo) / (hi - lo)
    u0, v0, u1, v1 = uv_rect
    uvp = np.stack([u0 + uvp[:, 0] * (u1 - u0), v0 + uvp[:, 1] * (v1 - v0)], axis=1)
    uvs = uvp[q.ravel()]
    return mesh_object(name, verts, q, uvs=uvs, mats=list(mats), smooth=True, collection=collection)


# ---- dashed glowing path ------------------------------------------------------

def dashed_path(name: str, pts: np.ndarray, *, dash: float, gap: float, radius: float, color, strength=4.0, reverse=False, collection=None):
    """Capsule dashes along a polyline; one mesh, attribute ``s`` (0..1 along
    the drawing direction).  Value node ``Front`` (0..1) reveals the dashes;
    the leading dash glows hotter.  ``reverse`` draws from the end."""
    pts = np.asarray(pts, dtype=np.float64)
    L = arc_lengths(pts)
    dashes = dash_layout(pts, dash, gap)
    verts, faces, svals = [], [], []
    ring = 8
    for s0, s1 in dashes:
        sl = np.linspace(s0, s1, 6)
        base = len(verts)
        for j, s in enumerate(sl):
            p, t = polyline_at(pts, s, L)
            a = np.cross(t, [0, 0, 1.0])
            if np.linalg.norm(a) < 1e-6:
                a = np.cross(t, [1.0, 0, 0])
            a /= np.linalg.norm(a)
            bvec = np.cross(t, a)
            taper = math.sin(math.pi * (0.08 + 0.84 * j / (len(sl) - 1)))
            rr = radius * (0.35 + 0.65 * taper)
            for k in range(ring):
                ang = 2 * math.pi * k / ring
                verts.append(tuple(p + rr * (math.cos(ang) * a + math.sin(ang) * bvec)))
                sv = s / L[-1]
                svals.append(1.0 - sv if reverse else sv)
        for j in range(len(sl) - 1):
            for k in range(ring):
                a0 = base + j * ring + k
                a1 = base + j * ring + (k + 1) % ring
                faces.append([a0, a1, a1 + ring, a0 + ring])
        faces.append([base + k for k in reversed(range(ring))])
        last = base + (len(sl) - 1) * ring
        faces.append([last + k for k in range(ring)])
    mat, b = new_mat(f"{name}_Mat")
    out = b.n("ShaderNodeOutputMaterial")
    attr = b.n("ShaderNodeAttribute", attribute_name="s", attribute_type="GEOMETRY")
    front = b.value("Front", 0.0)
    diff = b.m("SUBTRACT", front, attr.outputs["Fac"])
    shown = b.m("GREATER_THAN", diff, 0.0)
    hot = b.m("MULTIPLY", b.m("SUBTRACT", 1.0, b.smooth(diff, 0.0, 0.06)), 5.0)
    em = b.n("ShaderNodeEmission")
    em.inputs["Color"].default_value = color
    b.put(em.inputs["Strength"], b.m("MULTIPLY", b.m("ADD", b.value("Staerke", strength), hot), shown))
    tr = b.n("ShaderNodeBsdfTransparent")
    mix = b.n("ShaderNodeMixShader")
    b.put(mix.inputs[0], shown)
    b.t.links.new(tr.outputs[0], mix.inputs[1])
    b.t.links.new(em.outputs[0], mix.inputs[2])
    b.t.links.new(mix.outputs[0], out.inputs["Surface"])
    set_alpha_mode(mat, "DITHERED", shadow=False)
    ob = mesh_object(name, verts, faces, mats=[mat], smooth=True, collection=collection)
    a = ob.data.attributes.new("s", "FLOAT", "POINT")
    a.data.foreach_set("value", np.asarray(svals, dtype=np.float32))
    ob.visible_shadow = False
    return ob, mat


# ==========================================================================
# rigid body: simulate once, bake to keyframes, cache to JSON
# ==========================================================================

SIM_VERSION = 2
# every constant that shapes the simulation; all of it goes into the hash
SIM_PARAMS = {
    "fps": 60,
    "gravity": -9.81,
    "substeps": 30,
    "iterations": 40,
    "split_impulse": True,
    "shape": "BOX",
    "linear_damping": 0.08,
    "angular_damping": 0.12,
    "margin": 0.0004,
    "ground_size": 40.0,
    "ground_friction": 0.7,
    "fallprobe_z": 3.0,
}


def bake_spec(bodies: list[dict], *, frames: int, ground_z: float = 0.0, seed_text: str = "") -> dict:
    """Everything that determines the simulation result (hash input)."""
    return {"version": SIM_VERSION, "params": SIM_PARAMS, "bodies": bodies, "frames": frames,
            "ground_z": ground_z, "seed": seed_text, "blender": bpy.app.version_string}


def bake_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def simulate_rigid(bodies: list[dict], *, frames: int, ground_z: float = 0.0, seed_text: str = "") -> dict:
    """Run a Bullet simulation at 60 fps in a scratch scene.

    ``bodies`` = [{"name", "size": (x, y, z), "matrix": 4x4 list, "mass",
    "friction", "bounce", "kinematic": [(frame, 4x4), ...], "release"}].
    A lone box falls freely far away from everything (``fallprobe``) so the
    time step can be verified from the result.
    Returns {"hash", "fps", "frames", "matrizen": (n, frames, 4, 4), "fallprobe"}.
    """
    P = SIM_PARAMS
    digest = bake_hash(bake_spec(bodies, frames=frames, ground_z=ground_z, seed_text=seed_text))
    main = bpy.context.window.scene if bpy.context.window else bpy.context.scene
    sim = bpy.data.scenes.new("RigidSim")
    sim.render.fps = P["fps"]          # Bullet steps 1/fps per frame (default scene: 24)
    sim.render.fps_base = 1.0
    if bpy.context.window:
        bpy.context.window.scene = sim
    sim.frame_start, sim.frame_end = 1, frames
    with bpy.context.temp_override(scene=sim):
        bpy.ops.rigidbody.world_add()
    rbw = sim.rigidbody_world
    rbw.substeps_per_frame = P["substeps"]
    rbw.solver_iterations = P["iterations"]
    rbw.use_split_impulse = P["split_impulse"]
    rbw.point_cache.frame_start = 1
    rbw.point_cache.frame_end = frames
    sim.gravity = (0.0, 0.0, P["gravity"])
    coll = bpy.data.collections.new("RigidSimColl")
    sim.collection.children.link(coll)
    rbw.collection = coll

    def box(name, size, M):
        sx, sy, sz = size
        v = [(-sx / 2, -sy / 2, -sz / 2), (sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, -sz / 2), (-sx / 2, sy / 2, -sz / 2),
             (-sx / 2, -sy / 2, sz / 2), (sx / 2, -sy / 2, sz / 2), (sx / 2, sy / 2, sz / 2), (-sx / 2, sy / 2, sz / 2)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(v, [], f)
        ob = bpy.data.objects.new(name, me)
        sim.collection.objects.link(ob)
        ob.matrix_world = Matrix(M)
        with bpy.context.temp_override(scene=sim, object=ob, active_object=ob, selected_objects=[ob]):
            bpy.ops.rigidbody.object_add()
        rb = ob.rigid_body
        rb.collision_shape = P["shape"]
        rb.use_margin = True
        rb.collision_margin = P["margin"]
        rb.linear_damping = P["linear_damping"]
        rb.angular_damping = P["angular_damping"]
        rb.use_deactivation = False
        return ob

    objs = []
    for bd in bodies:
        ob = box("rb_" + bd["name"], bd["size"], bd["matrix"])
        rb = ob.rigid_body
        rb.type = "ACTIVE"
        rb.mass = bd.get("mass", 0.01)
        rb.friction = bd.get("friction", 0.5)
        rb.restitution = bd.get("bounce", 0.1)
        kin = bd.get("kinematic")
        if kin:
            rb.kinematic = True
            ob.rotation_mode = "QUATERNION"
            for f, M in kin:
                loc, rot, _ = Matrix(M).decompose()
                ob.location = loc
                ob.rotation_quaternion = rot
                ob.keyframe_insert("location", frame=f)
                ob.keyframe_insert("rotation_quaternion", frame=f)
            release = bd.get("release")
            if release:
                rb.keyframe_insert("kinematic", frame=release - 1)
                rb.kinematic = False
                rb.keyframe_insert("kinematic", frame=release)
        objs.append(ob)
    probe = box("rb_fallprobe", (0.1, 0.1, 0.1), Matrix.Translation((P["ground_size"] * 0.4, 0.0, P["fallprobe_z"])))
    probe.rigid_body.type = "ACTIVE"
    probe.rigid_body.linear_damping = 0.0
    probe.rigid_body.angular_damping = 0.0
    gs = P["ground_size"]
    g = box("rb_ground", (gs, gs, 0.2), Matrix.Translation((0.0, 0.0, ground_z - 0.1)))
    g.rigid_body.type = "PASSIVE"
    g.rigid_body.friction = P["ground_friction"]

    mats = np.zeros((len(objs), frames, 4, 4))
    fall = np.zeros(frames)
    dg = sim.view_layers[0].depsgraph
    with bpy.context.temp_override(scene=sim):
        for f in range(1, frames + 1):
            sim.frame_set(f)
            for i, ob in enumerate(objs):
                mats[i, f - 1] = np.asarray(ob.evaluated_get(dg).matrix_world)
            fall[f - 1] = probe.evaluated_get(dg).matrix_world.translation.z
    # clean up the scratch scene
    if bpy.context.window:
        bpy.context.window.scene = main
    for ob in list(sim.objects):
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    bpy.data.collections.remove(coll)
    bpy.data.scenes.remove(sim)
    return {"hash": digest, "fps": P["fps"], "frames": frames, "matrizen": mats.round(9).tolist(),
            "fallprobe": fall.round(9).tolist()}


class StaleBake(RuntimeError):
    pass


def rigid_bake(bodies: list[dict], *, frames: int, cache: Path, write: bool = False, fresh: bool = False, **kw) -> dict:
    """The frozen bake from ``cache`` (render path) or a new simulation.

    Rendering never writes the committed file: a hash mismatch raises
    :class:`StaleBake` unless ``write`` (CLI ``--bake``) re-simulates and
    replaces the file atomically.  ``fresh`` simulates without touching it.
    """
    digest = bake_hash(bake_spec(bodies, frames=frames, **kw))
    if not fresh and not write:
        if not cache.exists():
            raise StaleBake(f"{cache.name} fehlt - mit --bake erzeugen")
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("hash") != digest:
            raise StaleBake(f"{cache.name} ist veraltet ({data.get('hash')} != {digest}) - mit --bake neu erzeugen")
        print(f"[s5_s7] Starrkoerper-Bake aus {cache.name} ({digest})", flush=True)
        return data
    t0 = time.time()
    data = simulate_rigid(bodies, frames=frames, **kw)
    print(f"[s5_s7] Starrkoerper simuliert in {time.time() - t0:.1f} s ({data['hash']})", flush=True)
    if write:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data) + "\n", encoding="utf-8")
        os.replace(tmp, cache)
        print(f"[s5_s7] Bake eingefroren: {cache}", flush=True)
    return data


# ==========================================================================
# probe, render, encode
# ==========================================================================

def project_points(scene, pts: np.ndarray, res: int = 1920) -> np.ndarray:
    """World points (n, 3) -> pixel coords (n, 2) of the 1920 frame (y down);
    rows of points behind the camera are NaN."""
    cam = scene.camera
    dg = bpy.context.evaluated_depsgraph_get()
    P = np.asarray(cam.calc_matrix_camera(dg, x=res, y=res, scale_x=1.0, scale_y=1.0))
    V = np.asarray(cam.matrix_world.inverted())
    h = np.concatenate([np.asarray(pts, dtype=np.float64), np.ones((len(pts), 1))], axis=1)
    clip = h @ (P @ V).T
    w = clip[:, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :2] / w
    px = np.stack([(ndc[:, 0] * 0.5 + 0.5) * res, (0.5 - ndc[:, 1] * 0.5) * res], axis=1)
    px[w[:, 0] <= 0] = np.nan
    return px


def project_bbox(scene, ob, res: int = 1920):
    """Bounding box (x0, y0, x1, y1) in px of all evaluated vertices."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    if ob.type == "EMPTY":
        pts = np.asarray([ev.matrix_world.translation])
    else:
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        co = co.reshape(-1, 3)
        if len(co) == 0:
            return None
        M = np.asarray(ev.matrix_world)
        pts = co @ M[:3, :3].T + M[:3, 3]
    px = project_points(bpy.context.scene, pts, res)
    px = px[~np.isnan(px[:, 0])]
    if len(px) == 0:
        return None
    return [float(px[:, 0].min()), float(px[:, 1].min()), float(px[:, 0].max()), float(px[:, 1].max())]


def write_probe(ctx: ShotCtx, path: str) -> None:
    scene = ctx.scene
    data = {"shot": ctx.shot_id, "shot_frames": ctx.sh.frames, "frames": dict(ctx.events), "bbox": {}, "kamera": {}, "typ": {}}
    for ev, f in ctx.events.items():
        scene.frame_set(f)
        cm = scene.camera.matrix_world
        data["kamera"][ev] = {"ort": [round(v, 3) for v in cm.translation], "blick": [round(v, 3) for v in -cm.col[2].xyz]}
        boxes = {}
        for ob in ctx.keys.get(ev, []):
            bb = project_bbox(scene, ob)
            if bb is not None:
                boxes[ob.name] = [round(v, 1) for v in bb]
                data["typ"][ob.name] = ob.type
        data["bbox"][ev] = boxes
    extra_fn = ctx.extra.get("probe")
    data["extra"] = extra_fn(ctx) if extra_fn else {}
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"[s5_s7] Probe -> {path}", flush=True)


def png_complete(path: Path) -> bool:
    """True if ``path`` is a PNG that was written to the end (IEND chunk)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            fh.seek(-12, os.SEEK_END)
            tail = fh.read(12)
    except OSError:
        return False
    return head == b"\x89PNG\r\n\x1a\n" and tail[4:8] == b"IEND"


def encode(shot_id: str, out_dir: Path | None = None) -> Path:
    """Complete PNG sequence -> ProRes 422 HQ .mov (60 fps) + shot .json."""
    import nomiss_timeline as TL

    sh = TL.shot(shot_id)
    seq = out_dir or (RENDERS / shot_id)
    missing = [f for f in range(1, sh.frames + 1) if not png_complete(seq / f"{f:04d}.png")]
    if missing:
        raise RuntimeError(f"{shot_id}: {len(missing)} Bilder fehlen oder sind abgeschnitten (erstes: {missing[0]})")
    mov = RENDERS / f"{shot_id}.mov"
    tmp = mov.with_name(mov.stem + ".tmp.mov")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "60", "-start_number", "1", "-i", str(seq / "%04d.png"),
           "-frames:v", str(sh.frames), "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-r", "60", str(tmp)]
    subprocess.run(cmd, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                            "-of", "default=nw=1:nk=1", str(tmp)], check=True, capture_output=True, text=True)
    n = int(probe.stdout.strip() or 0)
    if n != sh.frames:
        raise RuntimeError(f"{shot_id}: .mov hat {n} statt {sh.frames} Bilder")
    os.replace(tmp, mov)
    TL.write_shot_json(sh, RENDERS / f"{shot_id}.json")
    print(f"[s5_s7] {mov.name} ({n} Bilder) + {shot_id}.json geschrieben", flush=True)
    return mov


def run(ctx: ShotCtx, args) -> None:
    """Probe / render frames / full render + encode, per the CLI flags."""
    import nomiss_render as R
    import nomiss_camera as C

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
    if args.probe:
        write_probe(ctx, args.probe)
    if args.matte:
        # keyed visibility would undo the matte on every frame change
        for ob in ctx.scene.objects:
            for fc in C.fcurves(ob):
                if fc.data_path == "hide_render":
                    fc.mute = True
        R.matte_mode(ctx.scene, [bpy.data.objects[args.matte]])
    frames = R.frame_list(args.frames, ctx.sh.frames, ctx.events)
    if args.resume:
        frames = [f for f in frames if not png_complete(ctx.out / f"{f:04d}.png")]
    print(f"[s5_s7] {ctx.shot_id}: {len(frames)} Bilder bei {args.res}px, {args.samples} Samples", flush=True)
    R.render_frames(ctx.scene, frames, ctx.out)
    full = args.frames == "all" and not args.matte and args.res == 1920 and not args.out
    if full:
        encode(ctx.shot_id)
