"""Scene 2 „Schaufenster“ — typography, 16:9 and 9:16.

* ``window_reveal``  kicker WEB & BETRIEB (accent line draws, mono letters type on)
* ``layers_explode`` „Komplett programmiert.“ — glyphs rise in one after another
* ``tower_fall``     „Kein Baukasten.“ — glyphs tip upright like falling blocks, landing
* ``speed_line``     value list, row 1: a line races across the frame and „< 1 s Ladezeit“
                     arrives behind it with heavy motion blur and stops hard
* ``de_pin``         row 2 „Gehostet in Deutschland“ drops in and clicks into place
* ``calendar_ticks`` row 3 „Täglich gesichert“ wipes in; a mini calendar 1–30 is ticked
                     off day by day (numbers give way to mint check marks)
* ``month_carousel`` month cards Jan–Dez glide through a soft window, every month the
                     same bar (one fixed amount); headline „Monatlich. Ein fester Betrag.“
* ``key_turn``       mono label „git clone — Ihr Repository“ types on, cursor blinks

16:9: block bottom-left on the shop-window floor; 9:16: top of the safe area.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "s2_website"


def _zeilen(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t1, t2 = sz.ev("layers_explode"), sz.ev("tower_fall")
    s1 = gt.satz(c, sz.bildtexte["t1"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    s2 = gt.satz(c, sz.bildtexte["t2"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    h1 = (len(s1.lines) - 1) * s1.pitch + s1.cap
    h2 = (len(s2.lines) - 1) * s2.pitch + s2.cap
    top1, top2 = sz.spalte([h1, h2], [s1.pitch - s1.cap])
    a = gt.text(sz, "T1", s1, r.left, top1,
                follower=gt.zeichen(t1, dy=-46, px_w=c.px_w(1), dur=20, delay=1.1))
    sz.layout["kopf_top"] = top1
    gt.eintrag(sz, "t1", a, t1, sz.halten("layers_explode"), slot="kopf/1", teile=[s1.text.replace("\n", " ")],
               event="layers_explode", text=sz.bildtexte["t1"]["text"])
    # „Kein Baukasten.“: each glyph tips upright from −28° and drops a little, out_back
    b = gt.text(sz, "T2", s2, r.left, top2, follower=gt.zeichen(
        t2, dx=-10, dy=34, px_w=c.px_w(1), dur=16, delay=1.6, ease="out_back",
        extra={"CharacterAngleZ": {t2: 28.0, t2 + 16: 0.0}}))
    gt.eintrag(sz, "t2", b, t2, sz.halten("tower_fall"), slot="kopf/2", teile=[s2.text.replace("\n", " ")],
               event="tower_fall", text=sz.bildtexte["t2"]["text"])


def _werte(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    cap, pitch, marker_w, gap = r.wert_cap, round(r.wert_cap * 1.62), 18, 16
    tops = sz.spalte([cap, cap, cap], [pitch - cap, pitch - cap])
    tx = r.left + marker_w + gap
    texts = [sz.bildtexte[k]["text"] for k in ("w1", "w2", "w3")]
    sats = [gt.satz(c, t, gt.HEAD, cap, track_em=gt.HEAD_TRACK_EM) for t in texts]

    def marker(k: int, t: int) -> str:
        box = (r.left, tops[k] + cap / 2 - 1, marker_w, 2)
        layer, m = gt.flaeche(sz, f"W{k}Marke", box, sz.farbe)
        gt.zeichne_linie(sz, m, box, t, 10)
        return layer

    # row 1 — speed line races across, the value arrives behind it and stops hard
    t = sz.ev("speed_line")
    y = tops[0] + cap / 2
    lw = 420
    # a Transform only moves pixels that exist at rest, so the line rests inside the frame
    line, _ = gt.flaeche(sz, "W0Linie", (0, y - 1, lw, 2), (1, 1, 1), 0.9)
    lxf = c.transform("W0LinieXf", line, motion_blur=True, quality=16, shutter=360.0)
    c.keyframes(lxf, "Center", {t: (0.5 - c.px_w(lw), 0.5), t + 12: (0.5 + c.px_w(c.width), 0.5)},
                ease="in_out_cubic")
    val = gt.text(sz, "W0Text", sats[0], tx, tops[0], motion_blur=False)
    vxf = c.transform("W0TextXf", val, motion_blur=True, quality=16, shutter=360.0)
    c.keyframes(vxf, "Center", {t + 3: (0.5 - c.px_w(560), 0.5), t + 10: (0.5, 0.5)}, ease="out_quart")
    g0 = gt.gruppe(sz, "W0G", [lxf, marker(0, t + 8), vxf])
    gt.eintrag(sz, "w1", g0, t, sz.halten("speed_line"), slot="wert/1", teile=[texts[0]], event="speed_line")

    # row 2 — drops in from above and clicks into place (out_back)
    t = sz.ev("de_pin")
    v2 = gt.text(sz, "W1Text", sats[1], tx, tops[1], follower=gt.zeichen(
        t, dy=22, px_w=c.px_w(1), dur=12, delay=0.6, ease="out_back"))
    g1 = gt.gruppe(sz, "W1G", [marker(1, t), v2])
    gt.eintrag(sz, "w2", g1, t, sz.halten("de_pin"), slot="wert/2", teile=[texts[1]], event="de_pin")

    # row 3 — wipes in left→right; a calendar 1–30 is ticked off day by day
    t = sz.ev("calendar_ticks")
    v3 = gt.text(sz, "W2Text", sats[2], tx, tops[2], motion_blur=False)
    v3w = gt.wisch(sz, "W2", v3, (tx - 4, tops[2] - cap * 0.5, sats[2].width + 30, cap * 2.0), t, 16,
                   richtung="rechts", soft=10)
    kal = _kalender(sz, t, tx, tops, cap, sats)
    g2 = gt.gruppe(sz, "W2G", [marker(2, t), v3w, kal])
    gt.eintrag(sz, "w3", g2, t, sz.halten("calendar_ticks"), slot="wert/3", teile=[texts[2]],
               event="calendar_ticks", symbole={"1–30": "KalZahlen", "✓": "KalHaken"})


def _kalender(sz: gt.Szene, t: int, tx: float, tops: list[float], cap: float, sats: list[gt.Satz]) -> str:
    """Mini month grid 1–30 (7 columns, mono). Numbers fade in, then the days are
    ticked off in time-lapse: each number gives way to a mint check mark."""
    c = sz.c
    kcap = 15 if sz.vertical else 13
    trk = 1.15
    rows = [list(range(1 + 7 * i, min(31, 8 + 7 * i))) for i in range(5)]
    zahlen = "\n".join(" ".join(f"{d:2d}" for d in row) for row in rows)
    haken = "\n".join(" ".join("✓ " for _ in row) for row in rows)
    pitch = kcap * 1.75
    sz_ = gt.satz(c, zahlen, gt.MONO, kcap, tracking=trk, pitch=pitch)
    sh = gt.satz(c, haken, gt.MONO, kcap, tracking=trk, pitch=pitch)
    cell = sz_.width / 20
    if sz.vertical:
        x, y = tx, tops[2] + cap + 40
    else:
        x, y = tx + max(s.width for s in sats) + 72, tops[0] + 4
    tick = t + 12
    zf = cw.FollowerSpec(delay=0.45, ease="out_cubic",
                         keys={"Opacity1": {t: 0.0, t + 6: 0.42, tick: 0.42, tick + 3: 0.0}})
    hf = cw.FollowerSpec(delay=0.45, ease="out_cubic", keys={"Opacity1": {tick: 0.0, tick + 3: 1.0}})
    a = gt.text(sz, "KalZahlen", sz_, x, y, follower=zf, motion_blur=False)
    b = gt.text(sz, "KalHaken", sh, x + cell / 2, y, color=gt.FARBEN["mint"], follower=hf, motion_blur=False)
    sz.layout["kalender"] = (x, y, sz_.width, sz_.height)
    return gt.gruppe(sz, "KalG", [a, b])


def _monate(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("month_carousel")
    bis = sz.halten("month_carousel")
    data = sz.bildtexte["monat"]
    s = gt.satz(c, data["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    head_h = (len(s.lines) - 1) * s.pitch + s.cap
    card_w, bar_w, bar_h, mcap = (76, 30, 56, 16) if sz.vertical else (92, 32, 56, 15)
    strip_h = bar_h + 16 + mcap + 8
    if sz.vertical:
        head_top, strip_top = sz.spalte([head_h, strip_h], [56])
    else:
        strip_top, head_top = sz.spalte([strip_h, head_h], [44])
    head = gt.text(sz, "MonatText", s, r.left, head_top,
                   follower=gt.zeichen(t, dy=-40, px_w=c.px_w(1), dur=18, delay=1.0))

    win_l, win_r = r.left, r.left + (840 if sz.vertical else 760)
    total = 12 * card_w
    # the strip rests inside the frame (a Transform only moves pixels that exist at rest);
    # it glides from just right of the window until Dez sits inside the right edge
    strip_left = win_l + 12
    window = cw.Box(win_l, int(strip_top - 12), win_r, int(strip_top + strip_h + 12))
    cards = gt.Stack(c, "Karten")
    base_y = strip_top + bar_h
    for k, m in enumerate(data["monate"]):
        cx = strip_left + (k + 0.5) * card_w
        bar, _ = gt.flaeche(sz, f"Karte{k}Balken", (cx - bar_w / 2, strip_top, bar_w, bar_h), sz.farbe, 0.85)
        cards.add(bar)
        ms = gt.satz(c, m, gt.MONO, mcap, track_em=gt.MONO_TRACK_EM)
        tool = gt.text(sz, f"Karte{k}Monat", ms, cx, base_y + 16, alpha=0.66, justify="center", motion_blur=False)
        sz.maskiert[tool] = window
        cards.add(tool)
    glide = c.transform("KartenXf", cards.top, motion_blur=True, quality=12, shutter=220.0)
    start_shift = win_r - strip_left + 30
    end_shift = (win_r - 24) - (strip_left + total)
    c.keyframes(glide, "Center", {t: (0.5 + c.px_w(start_shift), 0.5), max(t + 60, bis - 24): (0.5 + c.px_w(end_shift), 0.5)},
                ease="out_quart")
    fenster = c.rect_mask("KartenFenster", center=c.px((win_l + win_r) / 2, strip_top + strip_h / 2),
                          width=c.px_w(win_r - win_l - 60), height=c.px_h(strip_h + 24), soft_edge=c.px_w(30))
    base, bm = gt.flaeche(sz, "KartenGrund", (win_l, base_y, win_r - win_l, 1), (1, 1, 1), 0.18)
    gt.zeichne_linie(sz, bm, (win_l, base_y, win_r - win_l, 1), t, 24)
    g = gt.gruppe(sz, "MonatG", [base, (glide, fenster), head])
    sz.layout["karussell"] = window
    gt.eintrag(sz, "monat", g, t, bis, slot="monat", teile=[s.text.replace("\n", " ")] + data["monate"],
               event="month_carousel", text=data["text"])


def _code(sz: gt.Szene) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("key_turn")
    data = sz.bildtexte["code"]
    cap = 22 if sz.vertical else 20
    s = gt.satz(c, data["text"], gt.MONO, cap, track_em=0.04)
    (top,) = sz.spalte([cap], [])
    delay = 0.7
    a = gt.text(sz, "Code", s, r.left, top, alpha=0.92, follower=gt.tippen(t, delay=delay, alpha=0.92),
                motion_blur=False)
    typed = t + round(len(data["text"]) * delay) + 2
    cur, _ = gt.flaeche(sz, "CodeCursor", (r.left + s.width + 8, top - cap * 0.18, cap * 0.62, cap * 1.36),
                        sz.farbe, 1.0)
    blink = {t: 0.0, typed: 0.0}
    f = typed
    on = True
    while f + 16 < sz.len:
        blink[f] = 1.0 if on else 0.0
        f += 16
        on = not on
    c.keyframes(cur, "TopLeftAlpha", blink, ease="step")
    g = gt.gruppe(sz, "CodeG", [a, cur])
    gt.eintrag(sz, "code", g, t, sz.halten("key_turn"), slot="code", teile=[data["text"]], event="key_turn",
               abgang="links", dur=10)


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    r = sz.r
    t = sz.ev("window_reveal")
    top = r.oben if sz.vertical else r.unten - r.kicker_cap  # type: ignore[operator]
    gt.kicker(sz, "kicker", top, t, sz.halten("window_reveal"))
    _zeilen(sz)
    _werte(sz)
    _monate(sz)
    _code(sz)
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
