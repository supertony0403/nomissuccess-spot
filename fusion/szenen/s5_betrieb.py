"""Scene 5 „Betrieb“ — typography, 16:9 and 9:16 (accent: Mint).

* ``invoice_spin``   kicker BETRIEB & MIGRATION (stays above the headline)
* ``paper_plane``    „VMware → Proxmox“: „VMware“ rises in, a mint arrow draws its shaft
                     and snaps its head open, „Proxmox“ follows the arrow in from the left
* ``rollback_path``  „Mit Testlauf und Rückweg.“ — subline with a dashed mint path that
                     draws itself back, right to left
* ``rolling_update`` „Updates im laufenden Betrieb.“ — glyphs roll in from below one after
                     another (containers swapping); a mint status line stays lit, a light
                     pulse runs along it
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion.szenen import gemeinsam_typo as gt  # noqa: E402

SZENE = "s5_betrieb"


def _block(sz: gt.Szene) -> dict[str, float]:
    """Rows: kicker, headline, subline — tops anchored like every block."""
    r = sz.r
    sub_cap = r.sub_cap
    k, h, s = sz.spalte([r.kicker_cap, r.head_cap, sub_cap], [34, 40])
    return {"kicker": k, "kopf": h, "unter": s}


def _migration(sz: gt.Szene, tops: dict[str, float]) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("paper_plane")
    left_t, right_t = sz.bildtexte["t1"]["text"].split("→")
    left_t, right_t = left_t.strip(), right_t.strip()
    s1 = gt.satz(c, left_t, gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM)
    s2 = gt.satz(c, right_t, gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM)
    top = tops["kopf"]
    gap, shaft = r.head_cap * 0.45, r.head_cap * 1.7
    ax0 = r.left + s1.width + gap
    ax1 = ax0 + shaft
    ay = top + r.head_cap * 0.52
    a = gt.text(sz, "Vm", s1, r.left, top, follower=gt.zeichen(t, dy=-36, px_w=c.px_w(1), dur=16, delay=1.0))
    stroke = max(2.0, r.head_cap * 0.045)
    shaft_box = (ax0, ay - stroke / 2, shaft, stroke)
    sh, m = gt.flaeche(sz, "PfeilSchaft", shaft_box, sz.farbe)
    gt.zeichne_linie(sz, m, shaft_box, t + 8, 14, ease="in_out_cubic")
    head_len = r.head_cap * 0.36
    kopf_mask = gt.striche(sz, "PfeilKopf", [(ax1 - head_len, ay - head_len), (ax1, ay), (ax1 - head_len, ay + head_len)],
                           stroke)
    kopf = c.background("PfeilKopf", color=sz.farbe, alpha=1.0)
    c.apply_mask(kopf, kopf_mask)
    kxf = c.transform("PfeilKopfXf", kopf, pivot=sz.px(ax1, ay), motion_blur=True, quality=8)
    c.keyframes(kxf, "Size", {t + 19: 0.2, t + 26: 1.0}, ease="out_back")
    bx = ax1 + gap
    b = gt.text(sz, "Px", s2, bx, top, follower=gt.zeichen(t + 18, dx=-44, dy=0, px_w=c.px_w(1), dur=16, delay=0.9,
                                                            ease="out_quart"))
    g = gt.Stack(c, "MigG")
    g.add(a)
    g.add(sh)
    head_layer = g.add(kxf)
    c.keyframes(head_layer, "Blend", {t + 18: 0.0, t + 19: 1.0}, ease="linear")
    g.add(b)
    sz.layout["kopf_breite"] = bx + s2.width - r.left
    gt.eintrag(sz, "t1", g.top, t, sz.halten("paper_plane"), slot="kopf/1", teile=[left_t, right_t],
               event="paper_plane", abgang="links", symbole={"→": "PfeilSchaft"})


def _rueckweg(sz: gt.Szene, tops: dict[str, float]) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("rollback_path")
    s = gt.satz(c, sz.bildtexte["t2"]["text"], gt.TEXT, r.sub_cap, track_em=0.0)
    top = tops["unter"]
    a = gt.text(sz, "Rueck", s, r.left, top, alpha=0.82, follower=gt.zeichen(
        t, dy=-20, px_w=c.px_w(1), dur=14, delay=0.6, alpha=0.82))
    # dashed path under the subline, drawn right → left
    y = top + r.sub_cap * 1.62
    dash, gap = 10, 7
    n = int((s.width + gap) // (dash + gap))
    prev = None
    for i in range(n):
        x = r.left + i * (dash + gap)
        prev = c.rect_mask(f"RueckStrich{i}", center=c.px(x + dash / 2, y + 1), width=c.px_w(dash), height=c.px_h(2),
                           combine_with=prev)
    path = c.background("RueckPfad", color=sz.farbe, alpha=1.0)
    c.apply_mask(path, prev)  # type: ignore[arg-type]
    width = n * (dash + gap)
    drawn = gt.wisch(sz, "RueckPfad", path, (r.left - 2, y - 4, width + 4, 10), t + 4, 22, richtung="links",
                     ease="in_out_cubic")
    g = gt.gruppe(sz, "RueckG", [a, drawn])
    gt.eintrag(sz, "t2", g, t, sz.halten("rollback_path"), slot="kopf/2", teile=[s.text], event="rollback_path",
               abgang="links")


def _update(sz: gt.Szene, tops: dict[str, float]) -> None:
    c, r = sz.c, sz.r
    t = sz.ev("rolling_update")
    s = gt.satz(c, sz.bildtexte["t3"]["text"], gt.HEAD, r.head_cap, track_em=gt.HEAD_TRACK_EM, max_w=r.max_w)
    hh = (len(s.lines) - 1) * s.pitch + s.cap
    top, line_top = sz.spalte([hh, 2], [30])
    a = gt.text(sz, "Upd", s, r.left, top, follower=gt.zeichen(
        t, dy=-52, px_w=c.px_w(1), dur=14, delay=1.5, ease="out_quart"))
    box = (r.left, line_top, s.width, 2)
    status, m = gt.flaeche(sz, "UpdStatus", box, sz.farbe)
    gt.zeichne_linie(sz, m, box, t, 20)
    pulse, _ = gt.flaeche(sz, "UpdPuls", (r.left, line_top - 1, 70, 4), (1, 1, 1), 0.85, soft=10)
    pxf = c.transform("UpdPulsXf", pulse, motion_blur=True, quality=8)
    run = max(40, int(s.width / 14))
    keys = {}
    f = t + 24
    while f + run < sz.len - 30:
        keys[f] = (0.5 - c.px_w(70), 0.5)
        keys[f + run] = (0.5 + c.px_w(s.width), 0.5)
        f += run + 36
    if len(keys) >= 2:
        c.keyframes(pxf, "Center", keys, ease="in_out_cubic", per_key={k: "step" for k in list(keys)[1::2]})
    mask = c.rect_mask("UpdPulsFenster", center=c.px(r.left + s.width / 2, line_top + 1), width=c.px_w(s.width),
                       height=c.px_h(8))
    g = gt.gruppe(sz, "UpdG", [a, status, (pxf, mask)])
    gt.eintrag(sz, "t3", g, t, sz.halten("rolling_update"), slot="kopf", teile=[s.text.replace("\n", " ")],
               event="rolling_update", abgang="hoch")


def bau(fmt: str, tl: dict) -> gt.Szene:
    sz = gt.Szene(SZENE, fmt, tl)
    tops = _block(sz)
    gt.kicker(sz, "kicker", tops["kicker"], sz.ev("invoice_spin"), sz.halten("invoice_spin"), abgang="hoch")
    _migration(sz, tops)
    _rueckweg(sz, tops)
    _update(sz, tops)
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
