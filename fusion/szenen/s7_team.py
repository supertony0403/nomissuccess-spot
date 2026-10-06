"""Scene 7 „Ein Team“ — typography, 16:9 and 9:16 (accent: Rosa).

* ``converge``        „Ein Team.“ — big; the letters close ranks (tracking 1.9 → −0.015 em)
                      while they fade in
* ``converge_sub``    „Von der Website bis zur Firewall.“ — a rosa rule spans from the
                      first to the last word, the line wipes in along it
* ``tickets_fall``    „Kein Callcenter. Kein Ticketstapel.“ — glyphs drop in from above
                      with a small settle
* ``workday``         „Antwort innerhalb eines Werktags.“ — soft sweep left → right
* ``blackbox_unfold`` „Keine Black Box.“ — each glyph unfolds from a flat line (height
                      0 → 100 %), one after another
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "s7_team"


def _team(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t, t2 = sz.ev("converge"), sz.ev("converge_sub")
    cap = round(r.head_cap * 1.45)
    s1 = gt.satz(c, sz.bildtexte["t1"]["text"], gt.HEAD, cap, track_em=gt.HEAD_TRACK_EM)
    s2 = gt.satz(c, sz.bildtexte["t2"]["text"], gt.TEXT, r.sub_cap + 2, track_em=0.0, max_w=r.max_w)
    h2 = (len(s2.lines) - 1) * s2.pitch + s2.cap
    top1, top_rule, top2 = sz.spalte([cap, 2, h2], [34, 22])
    met = gt.metrics(gt.HEAD)
    weit = min(1.9, gt.spacing_for_width(met, s1.text, s1.size, c.width, r.max_w - 20))
    a = gt.text(sz, "Team", s1, r.left, top1, follower=cw.FollowerSpec(
        delay=1.0, ease="out_cubic", keys={"Opacity1": {t: 0.0, t + 14: 1.0}}))
    c.keyframes(a, "CharacterSpacing", {t: weit, t + 40: s1.tracking}, ease="out_expo")
    sz.layout["team_weit"] = weit
    gt.eintrag(sz, "t1", a, t, sz.halten("converge"), slot="kopf/1", teile=[s1.text], event="converge",
               abgang="hoch")

    box = (r.left, top_rule, s2.width, 2)
    rule, m = gt.flaeche(sz, "VonBisRegel", box, sz.farbe)
    gt.zeichne_linie(sz, m, box, t2, 22, ease="in_out_cubic")
    b = gt.text(sz, "VonBis", s2, r.left, top2, alpha=0.8, motion_blur=False)
    bw = gt.wisch(sz, "VonBis", b, (r.left - 4, top2 - s2.cap * 0.5, s2.width + 20, h2 + s2.cap * 1.1), t2, 22,
                  richtung="rechts", soft=16, ease="in_out_cubic")
    g = gt.gruppe(sz, "VonBisG", [rule, bw])
    gt.eintrag(sz, "t2", g, t2, sz.halten("converge_sub"), slot="kopf/2", teile=[s2.text.replace("\n", " ")],
               event="converge_sub", abgang="hoch")


def _kein_callcenter(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("tickets_fall")
    text = sz.bildtexte["t3"]["text"]
    s = gt.satz(c, text.replace(". ", ".\n"), gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM)
    (top,) = sz.spalte([(len(s.lines) - 1) * s.pitch + s.cap], [])
    a = gt.text(sz, "Callcenter", s, r.left, top, follower=gt.zeichen(
        t, dy=64, px_w=c.px_w(1), dur=16, delay=0.9, ease="out_back"))
    gt.eintrag(sz, "t3", a, t, sz.halten("tickets_fall", "workday"), slot="kopf", teile=s.lines,
               event="tickets_fall", text=text, abgang="runter")


def _werktag(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("workday")
    s = gt.satz(c, sz.bildtexte["t4"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    h = (len(s.lines) - 1) * s.pitch + s.cap
    (top,) = sz.spalte([h], [])
    a = gt.text(sz, "Werktag", s, r.left, top, motion_blur=False)
    xf = c.transform("WerktagXf", a, motion_blur=True, quality=10)
    c.keyframes(xf, "Center", {t: (0.5 - c.px_w(36), 0.5), t + 26: (0.5, 0.5)}, ease="out_expo")
    aw = gt.wisch(sz, "Werktag", xf, (r.left - 40, top - s.cap * 0.5, s.width + 80, h + s.cap * 1.0), t, 26,
                  richtung="rechts", soft=50, ease="out_cubic")
    gt.eintrag(sz, "t4", aw, t, sz.halten("workday"), slot="kopf", teile=s.lines, event="workday",
               text=sz.bildtexte["t4"]["text"], abgang="links")


def _black_box(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("blackbox_unfold")
    s = gt.satz(c, sz.bildtexte["t5"]["text"], gt.HEAD, round(r.head_cap * 1.2), track_em=gt.HEAD_TRACK_EM,
                max_w=r.max_w)
    (top,) = sz.spalte([(len(s.lines) - 1) * s.pitch + s.cap], [])
    a = gt.text(sz, "BlackBox", s, r.left, top, follower=cw.FollowerSpec(
        delay=1.4, ease="out_back",
        keys={"CharacterSizeY": {t: 0.0, t + 14: 1.0}, "Opacity1": {t: 0.0, t + 6: 1.0}},
        per_key={"Opacity1": {t: "out_cubic"}}))
    gt.eintrag(sz, "t5", a, t, sz.halten("blackbox_unfold"), slot="kopf", teile=[s.text.replace("\n", " ")],
               event="blackbox_unfold", abgang="hoch")


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    _team(sz)
    _kein_callcenter(sz)
    _werktag(sz)
    _black_box(sz)
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
