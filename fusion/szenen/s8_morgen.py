"""Scene 8 „07:00“ — typography and lockup, 16:9 and 9:16 (accent: the signal gradient).

* ``clock_roll_7`` the HUD clock grows out of its corner into a big rolling counter
                   06:59:58 → 06:59:59 → 07:00:00 (vertical motion blur), landing on the
                   event (the mirror of scene 1, where the counter flew into the HUD)
* ``bell_sleep``   „Niemand wurde geweckt.“ — the line-icon bell from scene 1 sits beside
                   it, nods off (slow tilt) and dissolves (blur, shrink, fade)
* ``logo_fold``    word mark „nomis“ (Manrope 600) + „success“ (Manrope 400, 55 %),
                   tracking −0.015 em, writes itself out under the 3D logo; a 2 px
                   signal-gradient edge draws out from its centre (brand moment 2)
* ``claim``        „Infrastruktur, die nachts niemanden weckt.“
* ``cta``          pill „Erstgespräch · 30 Minuten · kostenlos“: 1.5 px white outline,
                   fully rounded, no shadow; the outline draws open from the centre
* ``url``          „nomissuccess.de“

The lockup is centred (the one allowed exception) and holds to the last frame.
3D logo in the square: y ≈ 700–1050 → 16:9 frame y 280–630, 9:16 frame y 700–1050.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion import hud  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "s8_morgen"
GLOCKE = gt.ICONS / "glocke.png"
GLOCKE_PX, GLOCKE_TOP_PX, GLOCKE_DRAWN_PX = 384, 38, 332
LOGO_BOTTOM_SQUARE = 1050


def _lockup_tops(sz: gt.Szene) -> dict[str, float]:
    if sz.vertical:
        return {"marke": 1100, "cap": 62, "kante": 1194, "claim": 1228, "claim_cap": 26, "pill": 1296,
                "pill_cap": 19, "url": 1398, "url_cap": 24}
    logo_bottom = LOGO_BOTTOM_SQUARE - 420
    return {"marke": logo_bottom + 48, "cap": 54, "kante": logo_bottom + 48 + 54 + 26, "claim": logo_bottom + 162,
            "claim_cap": 24, "pill": logo_bottom + 214, "pill_cap": 17, "url": 0, "url_cap": 22}


def _uhr(sz: gt.Szene) -> dict[str, float]:
    r = sz.r
    roll = sz.ev("clock_roll_7")
    d = sz.bildtexte["uhr"]
    states = [d["von"], "06:59:59", d["text"]]
    cap = 104
    top = 470 if sz.vertical else 300
    ticks = [roll - 20, roll]
    layer, last_xf, width = gt.rollzaehler(sz, "Uhr", states, ticks, r.left, top, cap)
    # grows out of the HUD clock's corner
    h = hud.layout(sz.fmt)
    pivot = sz.px(r.left, top)
    src = sz.px(h.clock_left, h.clock_top)
    start = (0.5 + src[0] - pivot[0], 0.5 + src[1] - pivot[1])
    grow = sz.c.transform("UhrWachsen", layer, pivot=pivot, motion_blur=True, quality=16, shutter=200.0)
    t0, t1 = 2, 30
    sz.c.keyframes(grow, "Center", {t0: start, t1: (0.5, 0.5)}, ease="in_out_cubic")
    sz.c.keyframes(grow, "Size", {t0: h.clock_cap / cap, t1: 1.0}, ease="in_out_cubic")
    gt.eintrag(sz, "uhr", grow, t0, sz.halten("clock_roll_7"), slot="uhr", teile=list(d["text"]),
               event="clock_roll_7", art="landung", landung=(last_xf, "Center"), abgang="hoch")
    return {"top": top, "cap": cap}


def _niemand(sz: gt.Szene, uhr: dict[str, float]) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("bell_sleep")
    s = gt.satz(c, sz.bildtexte["t1"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w - 90)
    top = uhr["top"] + uhr["cap"] + 72
    a = gt.text(sz, "Niemand", s, r.left, top, follower=gt.zeichen(t, dy=-40, px_w=c.px_w(1), dur=18, delay=1.0))
    # the bell from scene 1: beside the last line, nods off and dissolves
    last = s.lines[-1]
    lx = r.left + gt.metrics(gt.HEAD).width_px(last, s.size, c.width, s.tracking) + s.cap * 0.5
    ly = top + (len(s.lines) - 1) * s.pitch
    scale = s.cap * 1.25 / GLOCKE_DRAWN_PX
    icon = c.image("Glocke", GLOCKE)
    pv = (0.5, 1 - GLOCKE_TOP_PX / GLOCKE_PX)
    sway = c.transform("GlockeXf", icon, pivot=pv, motion_blur=True, quality=10)
    c.keyframes(sway, "Angle", {t + 8: 0.0, t + 20: 6.0, t + 34: -14.0}, ease="in_out_cubic")
    soft = c.tool("GlockeWeich", "Blur", {"Input": cw.Link(sway), "XBlurSize": 0.0})
    c.keyframes(soft, "XBlurSize", {t + 34: 0.0, t + 54: 10.0}, ease="in_cubic")
    g = gt.Stack(c, "NiemandG")
    g.add(a)
    bell_center = sz.px(lx + s.cap * 0.62, ly + s.cap * 0.55)
    bl = g.add(soft, center=bell_center, size=scale)
    c.keyframes(bl, "Size", {t: scale * 0.5, t + 10: scale, t + 34: scale, t + 54: scale * 0.7}, ease="out_back",
                per_key={t + 34: "in_cubic"})
    c.keyframes(bl, "Blend", {t: 0.0, t + 6: 1.0, t + 36: 1.0, t + 54: 0.0}, ease="out_cubic",
                per_key={t + 36: "in_cubic"})
    sz.layout["glocke"] = (lx, ly, s.cap * 1.25)
    gt.eintrag(sz, "t1", g.top, t, sz.halten("bell_sleep"), slot="zeile", teile=[s.text.replace("\n", " ")],
               event="bell_sleep", abgang="hoch")


def _lockup(sz: gt.Szene) -> None:
    c = sz.c
    L = _lockup_tops(sz)
    cx = c.width / 2

    # word mark + signal edge
    t = sz.ev("logo_fold")
    marke, mw, teile = gt.wortmarke(sz, "Marke", cx, L["marke"], L["cap"], t, delay=2.0)
    edge_box = (cx - mw / 2, L["kante"], mw, 2)
    edge = c.gradient("Kante", gt.SIGNAL, start=sz.px(edge_box[0], 0), end=sz.px(edge_box[0] + mw, 0))
    em = c.rect_mask("KanteMaske", center=sz.px(cx, L["kante"] + 1), width=0.0, height=c.px_h(2))
    c.apply_mask(edge, em)
    gt.zeichne_linie(sz, em, edge_box, t + 22, 34, von="mitte", ease="in_out_cubic")
    g = gt.gruppe(sz, "MarkeG", [marke, edge])
    sz.layout["marke"] = (cx - mw / 2, L["marke"], mw, L["cap"])
    gt.eintrag(sz, "wortmarke", g, t, sz.len - 1, slot="marke", teile=teile, event="logo_fold",
               text="".join(teile), bis_ende=True)

    # claim
    t = sz.ev("claim")
    s = gt.satz(c, sz.bildtexte["claim"]["text"], gt.TEXT, L["claim_cap"], track_em=0.0, max_w=sz.r.max_w)
    a = gt.text(sz, "Claim", s, cx, L["claim"], alpha=0.86, justify="center",
                follower=gt.zeichen(t, dy=-18, px_w=c.px_w(1), dur=16, delay=0.45, alpha=0.86))
    gt.eintrag(sz, "claim", a, t, sz.len - 1, slot="claim", teile=[s.text.replace("\n", " ")], event="claim",
               bis_ende=True)

    # CTA pill (+ URL beside it in 16:9, below it in 9:16)
    t_cta, t_url = sz.ev("cta"), sz.ev("url")
    ps = gt.satz(c, sz.bildtexte["cta"]["text"], gt.TEXT_MITTEL, L["pill_cap"], track_em=0.01)
    us = gt.satz(c, sz.bildtexte["url"]["text"], gt.TEXT, L["url_cap"], track_em=0.0)
    pad = L["pill_cap"] * 1.6
    ph = round(L["pill_cap"] * 3.2)
    pw = ps.width + 2 * pad
    if sz.vertical:
        px0 = cx - pw / 2
        ux, utop, ujust = cx, L["url"], "center"
    else:
        gap = 34
        row = pw + gap + us.width
        px0 = cx - row / 2
        ux, utop, ujust = px0 + pw + gap, L["pill"] + ph / 2 - L["url_cap"] / 2, "left"
    ptop = L["pill"]
    ring = gt.ring(sz, "Pill", (px0, ptop, pw, ph), 1.5, pill=True)
    outline = c.background("PillKontur", color=gt.WHITE, alpha=0.95)
    c.apply_mask(outline, ring)
    drawn = gt.wisch(sz, "PillZug", outline, (px0 - 4, ptop - 4, pw + 8, ph + 8), t_cta, 22, richtung="rechts",
                     ease="in_out_cubic")
    label = gt.text(sz, "PillText", ps, px0 + pad, ptop + ph / 2 - L["pill_cap"] / 2,
                    follower=gt.zeichen(t_cta + 8, dy=-12, px_w=c.px_w(1), dur=12, delay=0.35))
    g = gt.gruppe(sz, "PillG", [drawn, label])
    sz.layout["pill"] = (px0, ptop, pw, ph)
    gt.eintrag(sz, "cta", g, t_cta, sz.len - 1, slot="cta", teile=[ps.text], event="cta", bis_ende=True)

    u = gt.text(sz, "Url", us, ux, utop, justify=ujust,
                follower=gt.zeichen(t_url, dy=-16, px_w=c.px_w(1), dur=14, delay=0.6))
    gt.eintrag(sz, "url", u, t_url, sz.len - 1, slot="url", teile=[us.text], event="url", bis_ende=True)


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    uhr = _uhr(sz)
    _niemand(sz, uhr)
    _lockup(sz)
    sz.fertig(lesefeld=False)   # dawn behind the lockup stays clean
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
