"""End card (2 s after scene 8) — 16:9 and 9:16, opaque.

Night background with a very faint brand glow: three soft colour fields (Rosa, Violett,
Blau, 7–11 %) drifting slowly, like the hero of nomissuccess.de. On ``endcard_in`` the
mono label „made by“ types on, and the small word mark „nomissuccess“ (nomis 600 /
success 400 at 55 %) builds itself from staggered, blurred glyphs. Both drift up and fade
out in the last 16 frames while the last chord rings out. Centred like the lockup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "abspann"


def _schein(sz: gt.Szene) -> str:
    """Opaque night plate with three drifting, very soft colour fields."""
    c = sz.c
    w, h = c.width, c.height
    plate = gt.Stack(c, "Grund")
    night = c.background("Nacht", color=gt.FARBEN["nacht"], alpha=1.0)
    plate.add(night)
    felder = [(gt.FARBEN["rosa"], 0.10, (0.30, 0.62), (0.36, 0.58)),
              (gt.FARBEN["violett"], 0.11, (0.70, 0.40), (0.64, 0.44)),
              (gt.FARBEN["blau"], 0.08, (0.52, 0.86), (0.48, 0.80))]
    d = max(w, h) * 0.75
    for i, (col, alpha, a, b) in enumerate(felder):
        layer, mask = gt.flaeche(sz, f"Schein{i}", (a[0] * w - d / 2, a[1] * h - d / 2, d, d), col, alpha,
                                 soft=d * 0.35, ellipse=True)
        c.keyframes(mask, "Center", {0: c.px(a[0] * w, a[1] * h), sz.len - 1: c.px(b[0] * w, b[1] * h)},
                    ease="linear")
        plate.add(layer)
    return plate.top


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    c = sz.c
    sz.stack.add(_schein(sz))
    t = cw.event_frame(tl, "endcard_in") - sz.s0
    texte = {b["id"]: b for b in gt.SCRIPT["bildtexte"]["abspann"]}
    cy = c.height / 2
    cap_m, cap_l = (40, 15) if not sz.vertical else (46, 17)
    lab = gt.satz(c, texte["a"]["text"], gt.MONO, cap_l, track_em=0.22)
    a = gt.text(sz, "MadeBy", lab, c.width / 2, cy - cap_m * 0.5 - 30 - cap_l, alpha=0.6, justify="center",
                follower=gt.tippen(t + 4, delay=1.6, alpha=0.6), motion_blur=False)
    dur = 16
    end = sz.len - 1
    gt.eintrag(sz, "a", a, t, end, slot="a", teile=[lab.text], event="endcard_in", abgang="hoch", dur=dur,
               text=texte["a"]["text"])
    marke, mw, teile = gt.wortmarke(sz, "Marke", c.width / 2, cy - cap_m * 0.5, cap_m, t + 10, delay=1.6, rise=18)
    sz.layout["marke"] = (c.width / 2 - mw / 2, cy - cap_m * 0.5, mw, cap_m)
    gt.eintrag(sz, "b", marke, t, end, slot="b", teile=teile, event="endcard_in", abgang="hoch", dur=dur,
               text=texte["b"]["text"])
    sz.fertig(lesefeld=False)   # opaque card
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
