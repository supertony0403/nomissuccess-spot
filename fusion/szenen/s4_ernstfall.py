"""Scene 4 „Ernstfall“ — typography, 16:9 and 9:16 (accent: Blau).

* ``red_tendrils``   kicker BACKUP & WIEDERHERSTELLUNG
* ``immutable_lock`` „Unveränderbare Kopien.“ — glyphs snap in from 125 % (they lock),
                     a blue rule draws under the line and ends in a small square
* ``rewind``         „Regelmäßig zurückgespielt.“ — glyphs arrive right to left, sliding
                     back in like a rewind, with horizontal motion blur
* ``hope_fade``      „Hoffnung“ stands thin and pale (Manrope 200, 38 %), dissolves into
                     blur and is replaced on the spot by „getestet“ plus a mint check
                     mark that draws itself
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "s4_ernstfall"


def _zeilen(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t1, t2 = sz.ev("immutable_lock"), sz.ev("rewind")
    s1 = gt.satz(c, sz.bildtexte["t1"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    s2 = gt.satz(c, sz.bildtexte["t2"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    h1 = (len(s1.lines) - 1) * s1.pitch + s1.cap
    h2 = (len(s2.lines) - 1) * s2.pitch + s2.cap
    top1, top2 = sz.spalte([h1, h2], [s1.pitch - s1.cap + 14])

    a = gt.text(sz, "T1", s1, r.left, top1, follower=gt.zeichen(
        t1, dy=0, px_w=c.px_w(1), dur=10, delay=0.8, ease="out_cubic",
        extra={"CharacterSizeX": {t1: 1.25, t1 + 10: 1.0}, "CharacterSizeY": {t1: 1.25, t1 + 10: 1.0}}))
    yr = top1 + h1 + 12
    w = min(s1.width, 420)
    rule_box = (r.left, yr, w, 2)
    rule, m = gt.flaeche(sz, "T1Regel", rule_box, sz.farbe)
    gt.zeichne_linie(sz, m, rule_box, t1 + 6, 16)
    lock, _ = gt.flaeche(sz, "T1Schloss", (r.left + w, yr - 3, 8, 8), sz.farbe)
    g1 = gt.Stack(c, "T1G")
    g1.add(a)
    g1.add(rule)
    sq = g1.add(lock)
    c.keyframes(sq, "Blend", {t1 + 20: 0.0, t1 + 22: 1.0}, ease="linear")
    gt.eintrag(sz, "t1", g1.top, t1, sz.halten("immutable_lock", "hope_fade"), slot="kopf/1",
               teile=[s1.text.replace("\n", " ")], event="immutable_lock", abgang="links")

    b = gt.text(sz, "T2", s2, r.left, top2, follower=gt.zeichen(
        t2, dx=90, dy=0, px_w=c.px_w(1), dur=16, delay=0.9, order="right_to_left", ease="out_quart"))
    gt.eintrag(sz, "t2", b, t2, sz.halten("rewind", "hope_fade"), slot="kopf/2",
               teile=[s2.text.replace("\n", " ")], event="rewind", abgang="links")


def _hoffnung(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    d = sz.bildtexte["hoffnung"]
    t = sz.ev("hope_fade")
    cap = round(r.head_cap * 1.2)
    sh = gt.satz(c, d["text"], gt.HEAD_DUENN, cap, track_em=0.0)
    ersatz_text = d["ersatz"].replace("✓", "").strip()
    se = gt.satz(c, ersatz_text, gt.TEXT, cap, track_em=gt.HEAD_TRACK_EM)
    (top,) = sz.spalte([cap], [])
    swap = t + 12
    pale = 0.38
    h = gt.text(sz, "Hoffnung", sh, r.left, top, follower=cw.FollowerSpec(
        delay=0.7, ease="out_cubic",
        keys={"Opacity1": {t: 0.0, t + 6: pale, swap: pale, swap + 10: 0.0},
              "SoftnessX1": {swap: 0.0, swap + 10: 12.0},
              "CharacterOffset": {swap: (0.0, 0.0), swap + 10: (0.0, c.px_w(14))}}))
    e = gt.text(sz, "Getestet", se, r.left, top, follower=gt.zeichen(swap + 2, dy=-30, px_w=c.px_w(1), dur=12,
                                                                     delay=0.8))
    hk_h = cap * 0.92
    hk = gt.haken(sz, "Haken", r.left + se.width + cap * 0.42, top + cap - hk_h, hk_h, gt.FARBEN["mint"],
                  swap + 12, dur=9)
    g = gt.gruppe(sz, "HoffG", [h, e, hk])
    gt.eintrag(sz, "hoffnung", g, t, sz.halten("hope_fade"), slot="kopf", teile=[d["text"], ersatz_text],
               event="hope_fade", abgang="hoch", dur=10, symbole={"✓": "HakenS0"})


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    r = sz.r
    top = r.oben if sz.vertical else r.unten - r.kicker_cap  # type: ignore[operator]
    gt.kicker(sz, "kicker", top, sz.ev("red_tendrils"), sz.halten("red_tendrils"))
    _zeilen(sz)
    _hoffnung(sz)
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
