"""Derive the 3D ribbon centreline of the nomissuccess "N" from ``logo.png``.

The logo is one gradient band drawn as two point-symmetric hooks:

* hook 1: left stem (up) -> arch over the top -> right leg (down, tapering
  one-sidedly along the diagonal slit) -> rounded tip,
* hook 2: rounded tip -> left leg (down, widening) -> U around the bottom ->
  right stem (up) to the top-right end.

A single continuous ribbon must connect the two hooks.  In front view the
hooks are separated by a dark channel (the diagonal slit), so the connection
runs *behind* both legs and crosses the slit perpendicularly.  In the final
pose its profile scale is 0 (``THREAD_RADIUS``), so the slit stays clean;
during the morph from the spiral the same stretch is a full-width band that
tucks itself away while it folds.

The tapering legs are produced like a real ribbon would produce them: the
band twists about its centreline (tilt) while its profile scale (radius)
shrinks, so the projected width follows the measured chord of the logo.

Run with the project venv (needs Pillow)::

    .venv/bin/python blender/lib/nomiss_logo.py

It writes ``blender/assets/logo_band.json`` (read inside Blender by
``nomiss_ribbon.logo_band()``) and debug overlays to ``work/blender/``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGO_PNG = REPO / "blender" / "assets" / "logo.png"
OUT_JSON = REPO / "blender" / "assets" / "logo_band.json"
DEBUG_DIR = REPO / "work" / "blender"

# Ribbon cross-section in logo pixels.  The stems measure 75-76 px.
BAND_WIDTH_PX = 75.5
BAND_THICKNESS_PX = 20.0
# Maximum twist of the tapering legs (degrees).  The remaining taper is
# taken up by the profile scale.
TWIST_MAX_DEG = 62.0
# Profile scale of the hidden connector thread (fraction of the band).
THREAD_RADIUS = 0.0
# Depth of the hidden connector behind the front plane (logo px, + = away).
THREAD_DEPTH_PX = 9.0
# Point spacing along the centreline (logo px).
SPACING_PX = 2.0
# Number of colour-ramp stops (Blender's Color Ramp holds at most 32).
RAMP_STOPS = 32


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------

def load_logo(path: Path = LOGO_PNG) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha HxW in 0..1, rgb HxWx3 sRGB in 0..1)."""
    from PIL import Image  # only needed when (re)generating the asset

    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float64) / 255.0
    return im[:, :, 3], im[:, :, :3]


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample ``img`` at pixel-centre coordinates (x, y); outside = 0."""
    h, w = img.shape[:2]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0

    def px(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        out = np.zeros(xx.shape + img.shape[2:], dtype=np.float64)
        out[inside] = img[yy[inside], xx[inside]]
        return out

    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    return (
        px(x0, y0) * (1 - fx) * (1 - fy)
        + px(x0 + 1, y0) * fx * (1 - fy)
        + px(x0, y0 + 1) * (1 - fx) * fy
        + px(x0 + 1, y0 + 1) * fx * fy
    )


# --------------------------------------------------------------------------
# polyline helpers
# --------------------------------------------------------------------------

def arc_length(pts: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Resample an (N, k) polyline uniformly by 2D arc length."""
    s = arc_length(pts)
    n = max(2, int(round(s[-1] / spacing)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, i]) for i in range(pts.shape[1])], axis=1)


def smooth(pts: np.ndarray, sigma: float, keep_ends: bool = True) -> np.ndarray:
    """Gaussian smoothing along the polyline index (reflect at the ends)."""
    if sigma <= 0:
        return pts.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([2 * pts[:1] - pts[r:0:-1], pts, 2 * pts[-1:] - pts[-2 : -r - 2 : -1]])
    out = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(pts.shape[1])], axis=1)
    if keep_ends:
        out[0] = pts[0]
        out[-1] = pts[-1]
    return out


def tangents(pts: np.ndarray) -> np.ndarray:
    d = np.gradient(pts[:, :2], axis=0)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def catmull_rom(waypoints: list[tuple[float, ...]], samples_per_seg: int = 24) -> np.ndarray:
    """Centripetal-free uniform Catmull-Rom through ``waypoints`` (any dim)."""
    p = np.asarray(waypoints, dtype=np.float64)
    p = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
    out.append(p[-2])
    return np.asarray(out)


# --------------------------------------------------------------------------
# centreline refinement
# --------------------------------------------------------------------------

def chord(alpha: np.ndarray, p: np.ndarray, n: np.ndarray, max_d: float = 80.0, step: float = 0.2):
    """Distances from ``p`` to the alpha=0.5 boundary along -n and +n."""
    d = np.arange(0.0, max_d, step)
    res = []
    for sgn in (-1.0, 1.0):
        xs = p[0] + sgn * n[0] * d
        ys = p[1] + sgn * n[1] * d
        a = bilinear(alpha, xs, ys)
        below = np.nonzero(a < 0.5)[0]
        if len(below) == 0:
            res.append(max_d)
            continue
        i = below[0]
        if i == 0:
            res.append(0.0)
            continue
        a0, a1 = a[i - 1], a[i]
        f = (a0 - 0.5) / max(a0 - a1, 1e-9)
        res.append(d[i - 1] + f * step)
    return res[0], res[1]


def refine(alpha: np.ndarray, pts: np.ndarray, iters: int = 6, sigma: float = 2.5) -> np.ndarray:
    """Move each point to the middle of its chord along the normal."""
    pts = resample(pts, SPACING_PX)
    for _ in range(iters):
        t = tangents(pts)
        nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
        new = pts.copy()
        for i in range(1, len(pts) - 1):
            dm, dp = chord(alpha, pts[i], nrm[i])
            new[i, :2] = pts[i, :2] + nrm[i] * (dp - dm) * 0.5
        pts = resample(smooth(new, sigma), SPACING_PX)
    return pts


def widths(alpha: np.ndarray, pts: np.ndarray) -> np.ndarray:
    t = tangents(pts)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return np.array([sum(chord(alpha, pts[i], nrm[i])) for i in range(len(pts))])


def extend_to_boundary(alpha: np.ndarray, pts: np.ndarray, at_end: bool, max_d: float = 30.0) -> np.ndarray:
    """Extend the polyline straight along its end tangent up to alpha=0.5."""
    if at_end:
        p, q = pts[-1, :2], pts[-4, :2]
    else:
        p, q = pts[0, :2], pts[3, :2]
    t = (p - q) / np.linalg.norm(p - q)
    d = np.arange(0.0, max_d, 0.1)
    a = bilinear(alpha, p[0] + t[0] * d, p[1] + t[1] * d)
    below = np.nonzero(a < 0.5)[0]
    reach = d[below[0]] if len(below) else 0.0
    if reach <= SPACING_PX * 0.5:
        return pts
    extra = np.array([p + t * s for s in np.arange(SPACING_PX, reach, SPACING_PX)] + [p + t * reach])
    extra = np.concatenate([extra, np.zeros((len(extra), pts.shape[1] - 2))], axis=1)
    return np.concatenate([pts, extra]) if at_end else np.concatenate([extra[::-1], pts])


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

@dataclass
class Band:
    pts: np.ndarray      # (N, 3) x, y (logo px, y down), z (depth px, + = away from viewer)
    w_proj: np.ndarray   # projected width in front view (logo px)
    tilt: np.ndarray     # twist about the tangent (rad)
    radius: np.ndarray   # profile scale (1 = full band)
    rgb: np.ndarray      # sampled sRGB colour (0..1)
    u: np.ndarray        # normalised arc length 0..1
    segments: dict[str, list[int]]


def hook1_guess() -> np.ndarray:
    pts = [(37.0, 329.0)]
    pts += [(37.0, y) for y in np.arange(320.0, 114.0, -6.0)]
    cx, cy, r = 114.75, 114.0, 77.4
    pts += [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in np.linspace(math.pi, 0.0, 60)]
    pts += [(192.2 + (218.5 - 192.2) * f, 114.0 + (209.0 - 114.0) * f) for f in np.linspace(0.05, 1.0, 30)]
    return np.asarray(pts)


def hook2_guess() -> np.ndarray:
    pts = [(116.5, 123.0)]
    pts += [(116.5 + (141.8 - 116.5) * f, 123.0 + (219.0 - 123.0) * f) for f in np.linspace(0.05, 1.0, 30)]
    cx, cy, r = 219.0, 219.0, 77.4
    pts += [(cx - r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(0.0, math.pi, 60)][1:]
    pts += [(296.5, y) for y in np.arange(212.0, 3.0, -6.0)]
    pts += [(296.5, 3.0)]
    return np.asarray(pts)


def taper_profile(w_proj: np.ndarray, taper: np.ndarray, twist_sign: float):
    """Tilt + radius so that the stadium profile projects to ``w_proj``."""
    W, T = BAND_WIDTH_PX, BAND_THICKNESS_PX
    phi_max = math.radians(TWIST_MAX_DEG)
    # taper progress 0 (full band) .. 1 (narrowest)
    t = np.clip((W - w_proj) / (W - T * 1.6), 0.0, 1.0) * taper
    s = t * t * (3 - 2 * t)
    phi = phi_max * s
    proj_unit = (W - T) * np.cos(phi) + T
    radius = np.where(taper > 0, w_proj / proj_unit, 1.0)
    return twist_sign * phi * (taper > 0), np.clip(radius, THREAD_RADIUS, 1.0)


def sample_colours(alpha: np.ndarray, rgb: np.ndarray, pts: np.ndarray, w_proj: np.ndarray) -> np.ndarray:
    """Average interior logo colour in a small disc around each point."""
    out = np.zeros((len(pts), 3))
    offs = [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if dx * dx + dy * dy <= 36]
    for i, (p, w) in enumerate(zip(pts, w_proj)):
        rad = max(1.0, min(6.0, 0.5 * w - 2.0))
        xs = np.array([p[0] + dx * rad / 6.0 for dx, _ in offs])
        ys = np.array([p[1] + dy * rad / 6.0 for _, dy in offs])
        a = bilinear(alpha, xs, ys)
        c = bilinear(rgb, xs, ys)
        wgt = np.clip((a - 0.9) * 10.0, 0.0, 1.0)
        if wgt.sum() < 1e-6:
            wgt = a
        out[i] = (c * wgt[:, None]).sum(0) / max(wgt.sum(), 1e-9)
    return out


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def ramp_stops(u: np.ndarray, rgb: np.ndarray, n: int = RAMP_STOPS) -> list[tuple[float, float, float, float]]:
    """Greedy pick of ``n`` stops minimising linear-interpolation error."""
    chosen = [0, len(u) - 1]
    for _ in range(n - 2):
        chosen.sort()
        approx = np.stack([np.interp(u, u[chosen], rgb[chosen, k]) for k in range(3)], axis=1)
        err = np.linalg.norm(approx - rgb, axis=1)
        err[chosen] = -1.0
        chosen.append(int(np.argmax(err)))
    chosen.sort()
    lin = srgb_to_linear(rgb)
    return [(float(u[i]), *map(float, lin[i])) for i in chosen]


def build_band(alpha: np.ndarray, rgb: np.ndarray) -> Band:
    # ---- hook 1: left stem, arch, tapering right leg ----------------------
    h1 = refine(alpha, hook1_guess())
    h1 = extend_to_boundary(alpha, h1, at_end=True)
    h1[0, 1] = 329.4  # flat foot exactly on the logo's bottom edge
    w1 = widths(alpha, h1)
    leg1 = (h1[:, 0] > 150.0) & (h1[:, 1] > 114.5)

    # ---- hook 2: tapering left leg, U, right stem --------------------------
    h2 = refine(alpha, hook2_guess())
    h2 = extend_to_boundary(alpha, h2, at_end=False)
    h2[-1, 1] = 1.6  # flat top exactly on the logo's top edge
    w2 = widths(alpha, h2)
    leg2 = (h2[:, 0] < 186.0) & (h2[:, 1] < 218.5)

    def finish(w: np.ndarray, leg: np.ndarray) -> np.ndarray:
        w = np.where(leg, np.minimum(w, BAND_WIDTH_PX), BAND_WIDTH_PX)
        ws = smooth(w[:, None], 1.5)[:, 0]
        return np.where(leg, np.clip(ws, 0.0, BAND_WIDTH_PX), BAND_WIDTH_PX)

    w1 = finish(w1, leg1)
    w2 = finish(w2, leg2)
    w1[-1] = 0.0
    w2[0] = 0.0

    # ---- connector: pinched thread behind both legs ------------------------
    tip1 = h1[-1, :2]
    tip2 = h2[0, :2]
    D = THREAD_DEPTH_PX
    way = [
        (tip1[0], tip1[1], 0.0),
        (tip1[0] + 2.5, tip1[1] - 1.5, D * 0.4),
        (tip1[0] + 1.0, tip1[1] - 12.0, D),
        (210.0, 176.0, D),
        (203.0, 158.0, D),
        (168.0, 172.0, D),          # crosses the slit perpendicularly
        (133.0, 188.0, D),
        (124.0, 160.0, D),
        (117.0, 133.0, D),
        (tip2[0] - 1.5, tip2[1] + 3.0, D * 0.4),
        (tip2[0], tip2[1], 0.0),
    ]
    con = resample(catmull_rom(way, 30), SPACING_PX)

    # ---- assemble -----------------------------------------------------------
    z1 = np.zeros((len(h1), 1))
    z2 = np.zeros((len(h2), 1))
    pts = np.concatenate([np.hstack([h1[:, :2], z1]), con[1:-1], np.hstack([h2[:, :2], z2])])
    w_proj = np.concatenate([w1, np.zeros(len(con) - 2), w2])
    n1, nc = len(h1), len(con) - 2
    segments = {"haken1": [0, n1 - 1], "verbinder": [n1, n1 + nc - 1], "haken2": [n1 + nc, len(pts) - 1]}

    taper = np.zeros(len(pts))
    taper[:n1] = leg1.astype(float)
    taper[n1 + nc :] = leg2.astype(float)
    tilt, radius = taper_profile(w_proj, taper, twist_sign=1.0)
    # hook 2 twists the other way so both legs turn "into" the slit
    tilt[n1 + nc :] *= -1.0
    radius[n1 : n1 + nc] = THREAD_RADIUS
    tilt[n1 : n1 + nc] = math.radians(90.0)

    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    u = s / s[-1]

    cols = np.zeros((len(pts), 3))
    cols[:n1] = sample_colours(alpha, rgb, h1, w1)
    cols[n1 + nc :] = sample_colours(alpha, rgb, h2, w2)
    c_a, c_b = cols[n1 - 1], cols[n1 + nc]
    f = np.linspace(0, 1, nc + 2)[1:-1, None]
    cols[n1 : n1 + nc] = c_a * (1 - f) + c_b * f
    cols = smooth(cols, 2.0)

    return Band(pts, w_proj, tilt, radius, cols, u, segments)


# --------------------------------------------------------------------------
# front-view silhouette (pure 2D, used for a quick IoU estimate)
# --------------------------------------------------------------------------

def rasterise_silhouette(band: Band, scale: int = 2) -> np.ndarray:
    """Union of the swept projected-width quads, in logo pixels x ``scale``."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (336 * scale, 333 * scale), 0)
    d = ImageDraw.Draw(img)
    t = tangents(band.pts)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    half = 0.5 * np.maximum(band.w_proj, 0.0)
    # thread: projected width = radius * thickness (edge-on)
    i0, i1 = band.segments["verbinder"]
    half[i0 : i1 + 1] = 0.5 * band.radius[i0 : i1 + 1] * BAND_THICKNESS_PX
    left = band.pts[:, :2] + nrm * half[:, None]
    right = band.pts[:, :2] - nrm * half[:, None]
    for i in range(len(band.pts) - 1):
        poly = [tuple(left[i] * scale), tuple(left[i + 1] * scale), tuple(right[i + 1] * scale), tuple(right[i] * scale)]
        d.polygon(poly, fill=255)
    return np.asarray(img) > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    from PIL import Image, ImageDraw

    alpha, rgb = load_logo()
    band = build_band(alpha, rgb)

    sil = rasterise_silhouette(band, scale=2)
    ref = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((672, 666), Image.BILINEAR)) > 127
    est = iou(sil, ref)

    stops = ramp_stops(band.u, band.rgb)
    data = {
        "_hinweis": "Erzeugt von blender/lib/nomiss_logo.py aus blender/assets/logo.png. Nicht von Hand bearbeiten.",
        "logo_px": [336, 333],
        "zentrum_px": [168.0, 166.5],
        "breite_px": BAND_WIDTH_PX,
        "dicke_px": BAND_THICKNESS_PX,
        "drall_max_grad": TWIST_MAX_DEG,
        "iou_2d_schaetzung": round(est, 4),
        "segmente": band.segments,
        "felder": ["x", "y", "z", "w_proj", "tilt", "radius", "r", "g", "b", "u"],
        "punkte": [
            [round(float(v), 4) for v in (*p, w, tl, r, *c, uu)]
            for p, w, tl, r, c, uu in zip(band.pts, band.w_proj, band.tilt, band.radius, band.rgb, band.u)
        ],
        "rampe_linear": [[round(v, 5) for v in s] for s in stops],
    }
    OUT_JSON.write_text(json.dumps(data, indent=0))
    print(f"punkte={len(band.pts)} segmente={band.segments} iou_2d={est:.4f} -> {OUT_JSON}")

    # debug overlay
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    S = 3
    base = Image.open(LOGO_PNG).convert("RGBA")
    bg = Image.new("RGBA", base.size, (16, 16, 24, 255))
    bg.alpha_composite(base)
    big = bg.convert("RGB").resize((336 * S, 333 * S), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    sil3 = np.asarray(Image.fromarray(sil.astype(np.uint8) * 255).resize((336 * S, 333 * S), Image.NEAREST)) > 127
    edge = sil3 ^ np.roll(sil3, 1, axis=0) | sil3 ^ np.roll(sil3, 1, axis=1)
    arr = np.asarray(big).copy()
    arr[edge] = (255, 255, 255)
    big = Image.fromarray(arr)
    d = ImageDraw.Draw(big)
    col = {"haken1": (255, 230, 0), "verbinder": (255, 40, 40), "haken2": (0, 255, 255)}
    for name, (i0, i1) in band.segments.items():
        p = band.pts[i0 : i1 + 1, :2] * S
        d.line([tuple(q) for q in p], fill=col[name], width=2)
    big.save(DEBUG_DIR / "logo_band_overlay.png")
    strip = np.zeros((40, 1000, 3), dtype=np.uint8)
    for x in range(1000):
        uu = x / 999
        strip[:, x] = [int(255 * np.interp(uu, band.u, band.rgb[:, k])) for k in range(3)]
    Image.fromarray(strip).resize((1000, 40)).save(DEBUG_DIR / "logo_band_rampe.png")


if __name__ == "__main__":
    main()
