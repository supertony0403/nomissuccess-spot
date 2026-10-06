"""HUD layer for the whole spot: mono clock, signal-gradient progress rail, corner ticks.

One comp per format (``work/fusion/<format>/hud.comp``), as long as the spot
(``timeline["dauer_f"]``), transparent, meant for track V4 above everything else.

* Clock (top left, JetBrains Mono, 60 % white): runs in spot time from 03:12 at event
  ``clock_roll`` to 07:00 at ``clock_roll_7`` (a Fusion expression on ``time``). It fades
  in when scene 1's big rolling counter has left and fades out when scene 8's counter
  takes over.
* Progress rail (bottom): a 2 px track at 12 % white; the signal gradient (Blau 0 % →
  Violett 34 % → Rosa 66 % → Mint 100 %) is uncovered left→right until ``logo_fold``,
  led by a small bright head.
* Corner ticks: viewfinder marks; in 9:16 they sit exactly on the safe-area corners.

Usage: ``python -m fusion.hud [--format 16x9|9x16|beide]``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fusion import comp_writer as cw  # type: ignore[no-redef]
else:
    from . import comp_writer as cw

SIGNAL = [(0.0, "#145fe4"), (0.34, "#7c3aed"), (0.66, "#e44b8d"), (1.0, "#10b981")]
MONO = ("JetBrains Mono", "Regular")
START_MIN, END_MIN = 3 * 60 + 12, 7 * 60  # 03:12 → 07:00


@dataclass(frozen=True)
class HudLayout:
    ticks: cw.Box          # corner tick rectangle (px, top-left origin)
    arm: int               # tick arm length (px)
    stroke: int            # line thickness (px)
    clock_left: int
    clock_top: int
    clock_cap: float
    rail_left: int
    rail_right: int
    rail_y: int            # centre line of the rail (px from top)


def layout(fmt: str) -> HudLayout:
    w, h = cw.FORMATS[fmt]
    if fmt == "9x16":
        side, top, bottom = cw.SAFE_9x16
        box = cw.Box(side, top, w - side, h - bottom)
        return HudLayout(box, 22, 2, side + 30, top + 22, 16, side + 30, w - side - 30, box.bottom - 24)
    box = cw.Box(44, 44, w - 44, h - 44)
    gx, _ = cw.GRID_16x9
    return HudLayout(box, 22, 2, gx, 50, 16, gx, w - gx, h - 52)


CLOCK_TRACKING = 1.08
FLIGHT_FRAMES = 24


def counter_flight(tl: dict) -> tuple[int, int]:
    """Spot frames (start, arrival) of scene 1's big counter flying into the HUD clock:
    it leaves after landing on ``clock_roll`` and makes room before „Die Stadt …“."""
    roll = cw.event_frame(tl, "clock_roll")
    start = max(roll + 10, cw.word_frame(tl, "v01", "Die") - 20)
    return start, start + FLIGHT_FRAMES


def clock_in_frame(tl: dict) -> int:
    """First frame of the HUD clock — the hand-over from scene 1's counter."""
    return counter_flight(tl)[1] - 2


def clock_expression(start_f: int, end_f: int) -> str:
    """Fusion expression: HH:MM, linear from 03:12 at ``start_f`` to 07:00 at ``end_f``."""
    span = max(1, end_f - start_f)
    m = f"({START_MIN} + {END_MIN - START_MIN} * math.min(1, math.max(0, (time - {start_f}) / {span})))"
    return f'Text(string.format("%02d:%02d", math.floor({m} / 60), math.floor({m}) % 60))'


def clock_text_at(frame: int, start_f: int, end_f: int) -> str:
    """Python mirror of ``clock_expression`` (for tests and previews)."""
    span = max(1, end_f - start_f)
    m = START_MIN + (END_MIN - START_MIN) * min(1.0, max(0.0, (frame - start_f) / span))
    return f"{int(m // 60):02d}:{int(m) % 60:02d}"


def build(fmt: str, tl: dict) -> cw.Comp:
    w, h = cw.FORMATS[fmt]
    frames = int(tl["dauer_f"])
    lay = layout(fmt)
    c = cw.Comp(w, h, frames=frames, fps=int(tl["fps"]))
    fam, sty, _ = cw.pick_font(*MONO, fallback=MONO)

    t_in, t_out_start = 24, cw.event_frame(tl, "logo_fold")
    roll_1, roll_7 = cw.event_frame(tl, "clock_roll"), cw.event_frame(tl, "clock_roll_7")
    s8_start, _ = cw.scene_span(tl, "s8_morgen")

    # --- corner ticks: 8 thin rectangles chained into one mask over a white plate
    b, arm, st = lay.ticks, lay.arm, lay.stroke
    prev = None
    corners = [(b.left, b.top, 1, 1), (b.right, b.top, -1, 1), (b.left, b.bottom, 1, -1), (b.right, b.bottom, -1, -1)]
    for i, (x, y, dx, dy) in enumerate(corners):
        horiz = c.rect_mask(f"TickH{i}", center=c.px(x + dx * arm / 2, y + dy * st / 2),
                            width=c.px_w(arm), height=c.px_h(st), combine_with=prev)
        prev = c.rect_mask(f"TickV{i}", center=c.px(x + dx * st / 2, y + dy * arm / 2),
                           width=c.px_w(st), height=c.px_h(arm), combine_with=horiz)
    c.background("Ticks", color=(1, 1, 1), alpha=0.28)
    c.apply_mask("Ticks", prev)

    # --- progress rail: track + gradient uncovered by a growing mask + bright head
    rail_w = lay.rail_right - lay.rail_left
    c.rect_mask("SpurMaske", center=c.px(lay.rail_left + rail_w / 2, lay.rail_y),
                width=c.px_w(rail_w), height=c.px_h(st))
    c.background("Spur", color=(1, 1, 1), alpha=0.12)
    c.apply_mask("Spur", "SpurMaske")

    x0, x1 = c.px_w(lay.rail_left), c.px_w(lay.rail_right)
    y_rail = c.px(0, lay.rail_y)[1]
    c.rect_mask("FortschrittMaske", center=(x0, y_rail), width=0.0, height=c.px_h(st))
    c.keyframes("FortschrittMaske", "Center", {0: (x0, y_rail), t_out_start: ((x0 + x1) / 2, y_rail)}, ease="linear")
    c.keyframes("FortschrittMaske", "Width", {0: 0.0, t_out_start: x1 - x0}, ease="linear")
    c.gradient("Signal", SIGNAL, start=(x0, 0.5), end=(x1, 0.5))
    c.apply_mask("Signal", "FortschrittMaske")

    c.ellipse_mask("KopfMaske", center=(x0, y_rail), width=c.px_w(6), height=c.px_h(6), soft_edge=0.0008)
    c.keyframes("KopfMaske", "Center", {0: (x0, y_rail), t_out_start: (x1, y_rail)}, ease="linear")
    c.background("Kopf", color=(1, 1, 1), alpha=0.95)
    c.apply_mask("Kopf", "KopfMaske")
    c.ellipse_mask("HofMaske", center=(x0, y_rail), width=c.px_w(26), height=c.px_h(26), soft_edge=0.006)
    c.keyframes("HofMaske", "Center", {0: (x0, y_rail), t_out_start: (x1, y_rail)}, ease="linear")
    c.background("Hof", color=(1, 1, 1), alpha=0.22)
    c.apply_mask("Hof", "HofMaske")

    # --- clock
    size = c.size_for_cap(lay.clock_cap, font=fam, style=sty)
    c.text("Uhr", "03:12", font=fam, style=sty, size=size, color=(1, 1, 1, 0.6),
           center=c.px(lay.clock_left, lay.clock_top), tracking=CLOCK_TRACKING, justify="left")
    c.expression("Uhr", "StyledText", clock_expression(roll_1, roll_7))
    c_in = clock_in_frame(tl)
    c.keyframes("Uhr", "Opacity1", {0: 0.0, c_in: 0.0, c_in + 6: 0.6, s8_start: 0.6, s8_start + 12: 0.0},
                ease="in_out_cubic")

    base = c.merge_all("Inhalt", ["Ticks", "Spur", "Signal", "Hof", "Kopf", "Uhr"])
    c.background("Leer", alpha=0.0)
    c.merge("HUD", "Leer", base)
    c.keyframes("HUD", "Blend", {t_in: 0.0, t_in + 36: 1.0, t_out_start: 1.0, t_out_start + 30: 0.0},
                ease="in_out_cubic")
    c.output("HUD")
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=["16x9", "9x16", "beide"], default="beide")
    ap.add_argument("--out", type=Path, default=cw.REPO / "work" / "fusion")
    args = ap.parse_args(argv)
    tl = cw.load_timeline()
    for fmt in (["16x9", "9x16"] if args.format == "beide" else [args.format]):
        path = build(fmt, tl).save(args.out / fmt / "hud.comp")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
