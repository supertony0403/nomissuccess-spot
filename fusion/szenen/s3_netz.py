"""Scene 3 „Netzwerk“ — typography, 16:9 and 9:16 (accent: Blau).

* ``dive_into_pixels`` kicker NETZWERK & SICHERHEIT
* ``firewall_wall``    „Firewall“ rises like a wall out of a mask edge; mono subline
                       „FortiGate · OPNsense“ types on under a short blue rule
* ``radar_sweep``      „Zero Trust“ is uncovered by a soft radar beam sweeping left→right
                       (a thin blue beam line leads it); subline „statt klassischem VPN“
* ``net_vanish``       „Im Internet unsichtbar.“ condenses out of blur, glyph by glyph
* ``zero_ports``       a big „0“ drops through a window with vertical motion blur and
                       stops hard; mono label „offene Ports“ beside it
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "s3_netz"


def _kopf_mit_unter(sz: gt.Szene, bid: str, prefix: str) -> tuple[gt.Satz, gt.Satz, float, float]:
    c, r = sz.c, sz.r
    d = sz.bildtexte[bid]
    head = gt.satz(c, d["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    unter = gt.satz(c, d["unter"], gt.MONO, r.kicker_cap + 2, track_em=0.08)
    hh = (len(head.lines) - 1) * head.pitch + head.cap
    top_h, top_u = sz.spalte([hh, unter.cap], [30])
    return head, unter, top_h, top_u


def _firewall(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("firewall_wall")
    head, unter, top_h, top_u = _kopf_mit_unter(sz, "t1", "Fw")
    a = gt.text(sz, "FwKopf", head, r.left, top_h, motion_blur=False)
    rise = c.transform("FwKopfXf", a, motion_blur=True, quality=12, shutter=240.0)
    c.keyframes(rise, "Center", {t: (0.5, 0.5 - c.px_h(head.cap * 1.1)), t + 20: (0.5, 0.5)}, ease="out_expo")
    hb = (r.left - 6, top_h - head.cap * 0.45, head.width + 30, head.cap * 1.75)
    wall = gt.wisch(sz, "FwKopf", rise, hb, t, 20, richtung="hoch")
    rule_box = (r.left, top_u - 14, 46, 2)
    rule, m = gt.flaeche(sz, "FwRegel", rule_box, sz.farbe)
    gt.zeichne_linie(sz, m, rule_box, t + 10, 12)
    b = gt.text(sz, "FwUnter", unter, r.left, top_u, alpha=0.74, follower=gt.tippen(t + 14, delay=0.8, alpha=0.74),
                motion_blur=False)
    g = gt.gruppe(sz, "FwG", [wall, rule, b])
    gt.eintrag(sz, "t1", g, t, sz.halten("firewall_wall", "radar_sweep"), slot="kopf",
               teile=[head.text.replace("\n", " "), unter.text], event="firewall_wall", abgang="links")


def _zero_trust(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("radar_sweep")
    head, unter, top_h, top_u = _kopf_mit_unter(sz, "t2", "Zt")
    a = gt.text(sz, "ZtKopf", head, r.left, top_h, motion_blur=False)
    b = gt.text(sz, "ZtUnter", unter, r.left, top_u, alpha=0.74, motion_blur=False)
    block = gt.gruppe(sz, "ZtBlock", [a, b])
    span = max(head.width, unter.width) + 40
    top, bottom = top_h - head.cap * 0.45, top_u + unter.cap * 1.5
    sweep = 30
    revealed = gt.wisch(sz, "Zt", block, (r.left - 8, top, span, bottom - top), t, sweep, richtung="rechts",
                        soft=40, ease="in_out_cubic")
    # the beam: a thin blue vertical line riding the sweep's edge, then fading
    beam, _ = gt.flaeche(sz, "ZtStrahl", (r.left - 8, top, 2, bottom - top), sz.farbe, 1.0)
    bxf = c.transform("ZtStrahlXf", beam, motion_blur=True, quality=10)
    c.keyframes(bxf, "Center", {t: (0.5, 0.5), t + sweep: (0.5 + c.px_w(span), 0.5)}, ease="in_out_cubic")
    g = gt.Stack(c, "ZtG")
    g.add(revealed)
    lead = g.add(bxf)
    c.keyframes(lead, "Blend", {t: 1.0, t + sweep - 4: 1.0, t + sweep + 8: 0.0}, ease="linear")
    gt.eintrag(sz, "t2", g.top, t, sz.halten("radar_sweep", "net_vanish"), slot="kopf",
               teile=[head.text.replace("\n", " "), unter.text], event="radar_sweep", abgang="links")


def _unsichtbar(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("net_vanish")
    s = gt.satz(c, sz.bildtexte["t3"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    (top,) = sz.spalte([(len(s.lines) - 1) * s.pitch + s.cap], [])
    a = gt.text(sz, "Unsichtbar", s, r.left, top, follower=gt.zeichen(
        t, dy=0, px_w=c.px_w(1), dur=24, delay=1.3, ease="out_cubic",
        extra={"SoftnessX1": {t: 14.0, t + 24: 0.0}, "SoftnessY1": {t: 14.0, t + 24: 0.0}}))
    gt.eintrag(sz, "t3", a, t, sz.halten("net_vanish", "zero_ports"), slot="kopf",
               teile=[s.text.replace("\n", " ")], event="net_vanish", abgang="runter")


def _null(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("zero_ports")
    d = sz.bildtexte["zahl"]
    cap = 236 if sz.vertical else 220
    zero = gt.satz(c, d["text"], gt.HEAD, cap)
    label = gt.satz(c, d["unter"], gt.TEXT, round(r.head_cap * 0.6), track_em=0.0)
    (top,) = sz.spalte([cap + 0.24 * zero.em], [])     # keep the descender room of the grid
    z = gt.text(sz, "Null", zero, r.left, top, motion_blur=False)
    drop = c.transform("NullXf", z, motion_blur=True, quality=16, shutter=300.0)
    c.keyframes(drop, "Center", {t: (0.5, 0.5 + c.px_h(cap * 1.7)), t + 12: (0.5, 0.5)}, ease="out_quart")
    win = (r.left - 10, top - cap * 0.25, zero.width + 20, cap * 1.5)
    fenster = c.rect_mask("NullFenster", center=c.px(win[0] + win[2] / 2, win[1] + win[3] / 2), width=c.px_w(win[2]),
                          height=c.px_h(win[3]), soft_edge=c.px_w(cap * 0.12))
    lx = r.left + zero.width + 34
    ly = top + cap - label.cap
    lab = gt.text(sz, "NullLabel", label, lx, ly, alpha=0.86,
                  follower=gt.zeichen(t + 8, dx=-24, dy=0, px_w=c.px_w(1), dur=14, delay=0.8, alpha=0.86))
    rule_box = (lx, ly - 18, 40, 2)
    rule, m = gt.flaeche(sz, "NullRegel", rule_box, sz.farbe)
    gt.zeichne_linie(sz, m, rule_box, t + 6, 12)
    g = gt.gruppe(sz, "NullG", [(drop, fenster), rule, lab])
    gt.eintrag(sz, "zahl", g, t, sz.halten("zero_ports"), slot="kopf", teile=[d["text"], d["unter"]],
               event="zero_ports", abgang="hoch")


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    r = sz.r
    top = r.oben if sz.vertical else r.unten - r.kicker_cap  # type: ignore[operator]
    gt.kicker(sz, "kicker", top, sz.ev("dive_into_pixels"), sz.halten("dive_into_pixels"))
    _firewall(sz)
    _zero_trust(sz)
    _unsichtbar(sz)
    _null(sz)
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
