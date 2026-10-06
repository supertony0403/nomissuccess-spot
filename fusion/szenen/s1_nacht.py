"""Scene 1 „03:12“ — kinetic typography (prototype for Task 6), 16:9 and 9:16.

Beats, all read from timeline.json when the comp is built (frames are scene-local):

* ``clock_roll``    rolling counter 03:11:57 → 03:12:00; each digit rolls up through a
                    soft window with real vertical motion blur and lands on the event.
* ``sleep_letters`` „Die Stadt schläft.“ — letters rise in, then sag one after another
                    and dim to 40 % (StyledTextFollower, per-character motion blur).
* ``lights_off``    „Ihr Büro ist dunkel.“ — revealed by a mask wipe, then the light
                    goes out letter by letter.
* ``led_1..led_4``  status column WEBSITE / FIREWALL / SERVER / BACKUPS: dim dots light
                    up mint with a soft halo, mono labels brighten.
* ``bell_morph``    „Weckt sie heute / Nacht jemanden?“ — the question mark shrinks away
                    and a line-icon bell takes its place and jolts once.

Layout: text flush left on the 120 px grid; 16:9 keeps to the left half, 9:16 to the
upper half inside the safe area. Usage: ``python -m fusion.szenen.s1_nacht``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion import comp_writer as cw  # noqa: E402
from fusion import hud  # noqa: E402

SZENE = "s1_nacht"
HEAD = ("Manrope", "SemiBold")
NUM = ("JetBrains Mono", "Light")   # same glyphs as the HUD clock it turns into
MONO = ("JetBrains Mono", "Medium")
MINT = "#10b981"
WHITE = (1.0, 1.0, 1.0)
GLOCKE = cw.REPO / "fusion" / "icons" / "glocke.png"
GLOCKE_PX, GLOCKE_TOP_PX, GLOCKE_DRAWN_PX = 384, 38, 332   # png size, top of the bell, drawn height
START, END = "03:11:57", "03:12:00"


@dataclass(frozen=True)
class Layout:
    left: int
    counter_top: int
    counter_cap: int
    text_top: int
    text_cap: int
    line_pitch: int
    led_top: int
    led_pitch: int
    led_cap: int


def layout(fmt: str) -> Layout:
    if fmt == "9x16":   # upper half, inside the safe area (sides 90, top 215)
        return Layout(left=120, counter_top=470, counter_cap=132, text_top=470, text_cap=64,
                      line_pitch=104, led_top=740, led_pitch=58, led_cap=21)
    return Layout(left=120, counter_top=300, counter_cap=132, text_top=300, text_cap=64,
                  line_pitch=104, led_top=580, led_pitch=58, led_cap=21)


@dataclass(frozen=True)
class Zeiten:
    """Scene-local frames of every beat (derived, never hard-coded)."""

    ende: int
    erscheinen: int
    ticks: tuple[int, int, int]
    roll: int
    flug: int          # counter starts flying into the HUD clock
    ankunft: int       # … and arrives there
    zeile1: int
    sacken: int
    zeile2: int
    licht_aus: int
    leds_ein: int
    leds: tuple[int, int, int, int]
    zeilen_weg: int
    frage: int
    glocke: int


def zeiten(tl: dict) -> Zeiten:
    s0, s1 = cw.scene_span(tl, SZENE)

    def ev(eid: str) -> int:
        return cw.event_frame(tl, eid) - s0

    def wort(vo: str, w: str) -> int:
        return cw.word_frame(tl, vo, w) - s0

    roll = ev("clock_roll")
    erscheinen = max(ev("ribbon_enter") + 6, 10)
    step = max(16, min(40, (roll - (erscheinen + 28) - 4) // 2))
    ticks = (roll - 2 * step, roll - step, roll)
    flug, ankunft = (f - s0 for f in hud.counter_flight(tl))
    zeile1 = max(wort("v01", "Die") - 12, flug + 10)
    sacken = max(ev("sleep_letters"), zeile1 + 20)
    zeile2 = max(wort("v01", "Ihr") - 10, sacken + 10)
    licht_aus = max(ev("lights_off"), zeile2 + 24)
    leds = tuple(ev(f"led_{k}") for k in range(1, 5))
    leds_ein = min(wort("v02", "Aber") + 4, leds[0] - 40)
    frage = wort("v03", "Weckt") - 10
    zeilen_weg = min(wort("v03", "Die") - 8, frage - 22)
    glocke = ev("bell_morph")
    return Zeiten(s1 - s0, erscheinen, ticks, roll, flug, ankunft, zeile1, sacken, zeile2, licht_aus,
                  leds_ein, leds, zeilen_weg, frage, glocke)  # type: ignore[arg-type]


class _Stack:
    """Bottom-up layer stack: each ``add`` merges a layer over the result so far."""

    def __init__(self, c: cw.Comp, prefix: str):
        self.c, self.prefix, self.n = c, prefix, 0
        self.top = c.background(f"{prefix}Leer", alpha=0.0)

    def add(self, layer: str, mask: str | None = None, **merge_kw) -> str:
        self.n += 1
        name = f"{self.prefix}M{self.n}"
        self.c.merge(name, self.top, layer, **merge_kw)
        if mask:
            self.c.apply_mask(name, mask)
        self.top = name
        return name


def _counter(c: cw.Comp, lay: Layout, z: Zeiten, scene: _Stack, fmt: str) -> None:
    """Rolling counter. Mono cells use the HUD clock's tracking, so after landing the
    group can scale down around its top-left corner and become the HUD clock exactly."""
    fam, sty, met = cw.pick_font(*NUM, fallback=("JetBrains Mono", "Regular"))
    size = c.size_for_cap(lay.counter_cap, fam, sty)
    em = met.em_px(size, c.width)
    cell = dict(met.advances)[ord("0")] / met.upm * em * hud.CLOCK_TRACKING
    pitch = lay.counter_cap * 1.75
    states = [START, "03:11:58", "03:11:59", END]
    roll_len = min(18, z.ticks[1] - z.ticks[0] - 4)

    hhmm, ss = _Stack(c, "ZaehlerHM"), _Stack(c, "ZaehlerS")
    for idx, ch in enumerate(START):
        target = hhmm if idx < 5 else ss
        cx = lay.left + (idx + 0.5) * cell
        changes = [k for k in range(1, 4) if states[k][idx] != states[k - 1][idx]]
        if not changes:
            target.add(c.text(f"Z{idx}", ch, font=fam, style=sty, size=size, color=WHITE,
                              center=c.px(cx, lay.counter_top), justify="center"))
            continue
        digits = [states[0][idx]] + [states[k][idx] for k in changes]
        strip = _Stack(c, f"Z{idx}Rolle")
        for j, d in enumerate(digits):
            strip.add(c.text(f"Z{idx}D{j}", d, font=fam, style=sty, size=size, color=WHITE,
                             center=c.px(cx, lay.counter_top + j * pitch), justify="center"))
        xf = c.transform(f"Z{idx}Xf", strip.top, motion_blur=True, quality=16, shutter=300.0)
        keys = {0: (0.5, 0.5)}
        for j, k in enumerate(changes, start=1):
            land = z.ticks[k - 1]
            keys[land - roll_len] = (0.5, 0.5 + (j - 1) * c.px_h(pitch))
            keys[land] = (0.5, 0.5 + j * c.px_h(pitch))
        c.keyframes(xf, "Center", keys, ease="out_quart")
        # tall window with a wide soft edge: digits fade in and out like on a drum
        c.rect_mask(f"Z{idx}Fenster", center=c.px(cx, lay.counter_top + lay.counter_cap / 2),
                    width=c.px_w(cell * 1.15), height=c.px_h(lay.counter_cap * 1.9),
                    soft_edge=c.px_w(lay.counter_cap * 0.35))
        target.add(xf, mask=f"Z{idx}Fenster")

    # seconds leave on the way up — the HUD clock shows HH:MM
    sec = hhmm.add(ss.top)
    c.keyframes(sec, "Blend", {z.flug: 1.0, z.flug + 10: 0.0}, ease="in_cubic")

    h = hud.layout(fmt)
    scale = h.clock_cap / lay.counter_cap
    pivot = c.px(lay.left, lay.counter_top)
    dest = c.px(h.clock_left, h.clock_top)
    arrive = (0.5 + dest[0] - pivot[0], 0.5 + dest[1] - pivot[1])
    group = c.transform("ZaehlerXf", hhmm.top, pivot=pivot, motion_blur=True, quality=24, shutter=200.0)
    a, f, t = z.erscheinen, z.flug, z.ankunft
    c.keyframes(group, "Center", {a: (0.5, 0.5 - c.px_h(26)), a + 30: (0.5, 0.5), f: (0.5, 0.5), t: arrive},
                ease="out_expo", per_key={a + 30: "linear", f: "in_out_cubic"})
    c.keyframes(group, "Size", {f: 1.0, t: scale}, ease="in_out_cubic")
    layer = scene.add(group)
    c.keyframes(layer, "Blend", {a: 0.0, a + 20: 1.0, t: 1.0, t + 6: 0.0},
                ease="out_cubic", per_key={t: "linear"})


def _lines(c: cw.Comp, lay: Layout, z: Zeiten, scene: _Stack) -> None:
    fam, sty, met = cw.pick_font(*HEAD)
    size = c.size_for_cap(lay.text_cap, fam, sty)
    rise = c.px_w(44)            # CharacterOffset is measured in image widths
    top1, top2 = lay.text_top, lay.text_top + lay.line_pitch

    i1, s = z.zeile1, z.sacken
    c.text("Zeile1", "Die Stadt schläft.", font=fam, style=sty, size=size, color=WHITE,
           center=c.px(lay.left, top1), motion_blur=True, quality=8,
           follower=cw.FollowerSpec(
               delay=1.5, ease="out_expo",
               keys={"CharacterOffset": {i1: (0.0, -rise), i1 + 18: (0.0, 0.0), s: (0.0, 0.0),
                                         s + 26: (0.0, -c.px_w(13))},
                     "Opacity1": {i1: 0.0, i1 + 12: 1.0, s: 1.0, s + 26: 0.4}},
               per_key={"CharacterOffset": {s: "in_out_cubic"}, "Opacity1": {i1: "out_cubic", s: "in_out_cubic"}}))

    i2, off = z.zeile2, z.licht_aus
    text2 = "Ihr Büro ist dunkel."
    c.text("Zeile2", text2, font=fam, style=sty, size=size, color=WHITE, center=c.px(lay.left, top2),
           follower=cw.FollowerSpec(delay=2.5, ease="linear", keys={"Opacity1": {off: 1.0, off + 1: 0.13}}))
    c.transform("Zeile2Xf", "Zeile2", motion_blur=True, quality=10)
    c.keyframes("Zeile2Xf", "Center", {i2: (0.5, 0.5 - c.px_h(36)), i2 + 22: (0.5, 0.5)}, ease="out_expo")
    width2 = met.width_px(text2, size, c.width) + 40
    box_top, box_bottom = top2 - 14, top2 + lay.text_cap * 1.45
    cx = lay.left - 20 + width2 / 2
    c.rect_mask("Zeile2Wisch", center=c.px(cx, box_bottom), width=c.px_w(width2), height=0.0,
                soft_edge=c.px_w(6))
    c.keyframes("Zeile2Wisch", "Center", {i2: c.px(cx, box_bottom), i2 + 20: c.px(cx, (box_top + box_bottom) / 2)},
                ease="out_expo")
    c.keyframes("Zeile2Wisch", "Height", {i2: 0.0, i2 + 20: c.px_h(box_bottom - box_top)}, ease="out_expo")

    group = _Stack(c, "Zeilen")
    group.add("Zeile1")
    group.add("Zeile2Xf", mask="Zeile2Wisch")
    xf = c.transform("ZeilenXf", group.top, motion_blur=True, quality=10)
    w = z.zeilen_weg
    c.keyframes(xf, "Center", {w: (0.5, 0.5), w + 18: (0.5, 0.5 + c.px_h(30))}, ease="in_cubic")
    layer = scene.add(xf)
    c.keyframes(layer, "Blend", {w: 1.0, w + 16: 0.0}, ease="in_cubic")


def _leds(c: cw.Comp, lay: Layout, z: Zeiten, scene: _Stack) -> None:
    fam, sty, _ = cw.pick_font(*MONO, fallback=("JetBrains Mono", "Regular"))
    size = c.size_for_cap(lay.led_cap, fam, sty)
    labels = ["WEBSITE", "FIREWALL", "SERVER", "BACKUPS"]
    for k, (label, on) in enumerate(zip(labels, z.leds)):
        y = lay.led_top + k * lay.led_pitch
        dot_c = c.px(lay.left + 6, y + lay.led_cap / 2)
        start = z.leds_ein + 6 * k
        c.ellipse_mask(f"Led{k}Punkt", center=dot_c, width=c.px_w(13), height=c.px_h(13), soft_edge=c.px_w(1))
        c.background(f"Led{k}Aus", color=WHITE, alpha=0.3)
        c.apply_mask(f"Led{k}Aus", f"Led{k}Punkt")
        c.background(f"Led{k}An", color=MINT, alpha=1.0)
        c.apply_mask(f"Led{k}An", f"Led{k}Punkt")
        c.ellipse_mask(f"Led{k}HofMaske", center=dot_c, width=c.px_w(34), height=c.px_h(34), soft_edge=c.px_w(12))
        c.background(f"Led{k}Hof", color=MINT, alpha=0.6)
        c.apply_mask(f"Led{k}Hof", f"Led{k}HofMaske")
        c.text(f"Led{k}Text", label, font=fam, style=sty, size=size, color=(1, 1, 1, 0.0),
               center=c.px(lay.left + 34, y), tracking=1.14)
        c.keyframes(f"Led{k}Text", "Opacity1", {start: 0.0, start + 14: 0.4, on: 0.4, on + 6: 0.92},
                    ease="out_cubic")
        c.keyframes(f"Led{k}Text", "Center", {start: c.px(lay.left + 20, y), start + 16: c.px(lay.left + 34, y)},
                    ease="out_expo")
        aus = scene.add(f"Led{k}Aus")
        c.keyframes(aus, "Blend", {start: 0.0, start + 12: 1.0}, ease="out_cubic")
        hof = scene.add(f"Led{k}Hof")
        c.keyframes(hof, "Blend", {on - 1: 0.0, on + 4: 1.0, on + 40: 0.0}, ease="out_cubic")
        an = scene.add(f"Led{k}An")
        c.keyframes(an, "Blend", {on - 1: 0.0, on + 3: 1.0}, ease="out_cubic")
        scene.add(f"Led{k}Text")


def _question(c: cw.Comp, lay: Layout, z: Zeiten, scene: _Stack) -> None:
    fam, sty, met = cw.pick_font(*HEAD)
    size = c.size_for_cap(lay.text_cap, fam, sty)
    natural = lay.text_cap * met.height / met.cap
    spacing = lay.line_pitch / natural
    rise = c.px_w(40)
    q, b = z.frage, z.glocke

    def spec() -> cw.FollowerSpec:
        return cw.FollowerSpec(delay=1.2, ease="out_expo",
                               keys={"CharacterOffset": {q: (0.0, -rise), q + 18: (0.0, 0.0)},
                                     "Opacity1": {q: 0.0, q + 12: 1.0}},
                               per_key={"Opacity1": {q: "out_cubic"}})

    zeilen = "Weckt sie heute\nNacht jemanden"
    c.text("Frage", zeilen, font=fam, style=sty, size=size, color=WHITE, center=c.px(lay.left, lay.text_top),
           line_spacing=spacing, follower=spec(), motion_blur=True)
    scene.add("Frage")

    # The question mark: an identical text with "?" appended, masked to the "?" only —
    # it is laid out exactly where Text+ would put it and enters with its character delay.
    top2 = lay.text_top + lay.line_pitch
    qx = lay.left + met.width_px("Nacht jemanden", size, c.width)
    qw = met.width_px("?", size, c.width)
    q_center = (qx + qw / 2, top2 + lay.text_cap / 2)
    c.text("FrageZ", zeilen + "?", font=fam, style=sty, size=size, color=WHITE,
           center=c.px(lay.left, lay.text_top), line_spacing=spacing, follower=spec(), motion_blur=True)
    c.rect_mask("FrageZMaske", center=c.px(qx + qw / 2, top2 + lay.text_cap * 0.6), width=c.px_w(qw + 16),
                height=c.px_h(lay.text_cap * 1.9))
    alone = _Stack(c, "FrageZ")
    alone.add("FrageZ", mask="FrageZMaske")
    pivot = c.px(*q_center)
    c.transform("FrageZXf", alone.top, pivot=pivot, motion_blur=True, quality=10)
    c.keyframes("FrageZXf", "Size", {b - 4: 1.0, b + 6: 0.45}, ease="in_cubic")
    layer = scene.add("FrageZXf")
    c.keyframes(layer, "Blend", {b - 2: 1.0, b + 6: 0.0}, ease="in_cubic")

    # the bell: line icon, scales in with a little overshoot, then jolts once and freezes
    c.image("Glocke", GLOCKE)
    pv = (0.5, 1 - GLOCKE_TOP_PX / GLOCKE_PX)
    c.transform("GlockeXf", "Glocke", pivot=pv, motion_blur=True, quality=12, shutter=200.0)
    jolt = b + 8
    c.keyframes("GlockeXf", "Angle", {jolt: 0.0, jolt + 3: 15.0, jolt + 6: -10.0, jolt + 9: 6.0,
                                      jolt + 12: -2.5, jolt + 15: 0.0}, ease="in_out_cubic")
    scale = lay.text_cap * 1.32 / GLOCKE_DRAWN_PX
    bell_center = c.px(q_center[0] + 4, q_center[1] - 2)
    layer = scene.add("GlockeXf", center=bell_center, size=scale)
    c.keyframes(layer, "Size", {b - 2: scale * 0.4, b + 10: scale}, ease="out_back")
    c.keyframes(layer, "Blend", {b - 2: 0.0, b + 4: 1.0}, ease="out_cubic")


def build(fmt: str, tl: dict) -> cw.Comp:
    w, h = cw.FORMATS[fmt]
    z = zeiten(tl)
    lay = layout(fmt)
    c = cw.Comp(w, h, frames=z.ende, fps=int(tl["fps"]))
    scene = _Stack(c, "Szene")
    _counter(c, lay, z, scene, fmt)
    _lines(c, lay, z, scene)
    _leds(c, lay, z, scene)
    _question(c, lay, z, scene)
    c.output(scene.top)
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scene 1 typography comps")
    ap.add_argument("--format", choices=["16x9", "9x16", "beide"], default="beide")
    ap.add_argument("--out", type=Path, default=cw.REPO / "work" / "fusion")
    args = ap.parse_args(argv)
    tl = cw.load_timeline()
    for fmt in (["16x9", "9x16"] if args.format == "beide" else [args.format]):
        print(build(fmt, tl).save(args.out / fmt / f"{SZENE}.comp"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
