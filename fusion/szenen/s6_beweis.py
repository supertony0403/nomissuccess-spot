"""Scene 6 „Kein Mockup“ — typography, 16:9 and 9:16 (accent: Mint).

* ``grafana_flyover`` „Wir betreiben selbst, was wir empfehlen.“ — line by line, each
                      line rises out of its own mask edge (6 frames apart)
* ``stamp``           stamp KEIN MOCKUP · ECHTES MONITORING: mint mono capitals in a
                      3 px frame, turned −3°, lands with a short press (scale 1.06 → 1.0,
                      edge blur 3 → 0 px) and lifts off at the end
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "s6_beweis"
STEMPEL_FONT = ("JetBrains Mono", "Bold")
STEMPEL_WINKEL = -3.0


def _satz(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("grafana_flyover")
    full = gt.satz(c, sz.bildtexte["t1"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    lines = full.lines
    h = (len(lines) - 1) * full.pitch + full.cap
    (top,) = sz.spalte([h], [])
    g = gt.Stack(c, "SatzG")
    for i, line in enumerate(lines):
        s = gt.satz(c, line, gt.HEAD, r.head_cap, tracking=full.tracking)
        y = top + i * full.pitch
        start = t + 6 * i
        tx = gt.text(sz, f"Satz{i}", s, r.left, y, motion_blur=False)
        xf = c.transform(f"Satz{i}Xf", tx, motion_blur=True, quality=12, shutter=220.0)
        c.keyframes(xf, "Center", {start: (0.5, 0.5 - c.px_h(full.cap * 1.2)), start + 22: (0.5, 0.5)},
                    ease="out_expo")
        box = (r.left - 6, y - full.cap * 0.4, s.width + 30, full.cap * 1.62)
        g.add(gt.wisch(sz, f"Satz{i}", xf, box, start, 1, richtung="hoch"))
    gt.eintrag(sz, "t1", g.top, t, sz.halten("grafana_flyover", "stamp"), slot="kopf", teile=lines,
               event="grafana_flyover", text=sz.bildtexte["t1"]["text"], abgang="hoch")


def _stempel(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("stamp")
    label = sz.bildtexte["stempel"]["text"]
    cap = 22 if sz.vertical else 24
    s = gt.satz(c, label, STEMPEL_FONT, cap, track_em=0.1)
    pad_x, pad_y, stroke = cap * 1.0, cap * 0.8, 3.0
    w, h = s.width + 2 * pad_x, cap + 2 * pad_y
    (top,) = sz.spalte([h], [])
    left = r.left + 6
    tx = gt.text(sz, "Stempel", s, left + pad_x, top + pad_y, color=gt.FARBEN["mint"], motion_blur=False)
    frame_mask = gt.ring(sz, "StempelRahmen", (left, top, w, h), stroke)
    frame = c.background("StempelRahmen", color=gt.FARBEN["mint"], alpha=1.0)
    c.apply_mask(frame, frame_mask)
    g = gt.gruppe(sz, "StempelG", [frame, tx])
    pivot = sz.px(left + w / 2, top + h / 2)
    press = c.transform("StempelXf", g, angle=STEMPEL_WINKEL, pivot=pivot, motion_blur=True, quality=8)
    c.keyframes(press, "Size", {t: 1.06, t + 6: 1.0}, ease="out_cubic")
    blur = c.tool("StempelKante", "Blur", {"Input": cw.Link(press), "XBlurSize": 3.0})
    c.keyframes(blur, "XBlurSize", {t: 3.0, t + 8: 0.0}, ease="out_cubic")
    sz.layout["stempel"] = (left, top, w, h)
    gt.eintrag(sz, "stempel", blur, t, sz.halten("stamp"), slot="stempel", teile=[label], event="stamp",
               abgang="klein", dur=12, pivot_px=(left + w / 2, top + h / 2))


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    _satz(sz)
    _stempel(sz)
    sz.fertig()
    return sz


def build(fmt: str, tl: dict) -> cw.Comp:
    return bau(fmt, tl).c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=["16x9", "9x16", "beide"], default="beide")
    ap.add_argument("--out", type=Path, default=cw.REPO / "work" / "fusion")
    args = ap.parse_args(argv)
    tl = cw.load_timeline()
    for fmt in (["16x9", "9x16"] if args.format == "beide" else [args.format]):
        print(build(fmt, tl).save(args.out / fmt / f"{SZENE}.comp"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
