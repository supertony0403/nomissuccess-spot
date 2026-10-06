"""Shared typography kit for the scene comps s2–s8 and the end card (Task 6).

Every scene module builds a ``Szene``: a transparent comp as long as the scene plus one
``Eintrag`` per ``script.json → bildtexte`` entry. The records tell the tests what was
built (texts, the gate merge, appear frame, hold end). Rules shared by all scenes:

* **Appear:** a Bildtext's gate (the ``Blend`` of its scene merge) opens exactly on its
  event frame (scene-local). The entrance animation starts on the same frame.
* **Hold:** it stays readable until its VO line ends + 0.4 s, or until the next event of
  the same *slot* (a slot is a place on screen; a list that grows keeps its slot free).
  The scene-8 lockup holds to the last frame.
* **Exit:** every entry leaves with its own short move (Transform with motion blur, and
  a fade). It is never just a cross-fade.
* **Area:** text boxes stay inside the 120 px margin (16:9) or the safe area (9:16).
  Measured with ``FontMetrics``, so the layout needs no render round trip.

Look: Manrope Light/Regular for headlines (tracking −0.015 em like the website),
JetBrains Mono for kickers and labels (tracked +0.16 em). Text is white on night, and
colour is used only on edges and accents in the scene colour. Text is flush left. In
16:9 a block sits bottom-left (the 3D object is centre/right); in 9:16 it sits at the top
inside the safe area, under the HUD clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fusion import comp_writer as cw

REPO = cw.REPO
SCRIPT: dict = json.loads((REPO / "script.json").read_text(encoding="utf-8"))
EVENTS: dict = json.loads((REPO / "szenen" / "events.json").read_text(encoding="utf-8"))["events"]
FARBEN: dict = SCRIPT["farben"]
SIGNAL = list(zip(FARBEN["signal_stops"], FARBEN["signal"]))
AKZENT = {"rosa": FARBEN["rosa"], "blau": FARBEN["blau"], "mint": FARBEN["mint"],
          "spektrum": FARBEN["violett"], "nacht": FARBEN["nacht_text_2"]}
WHITE = (1.0, 1.0, 1.0)
ICONS = REPO / "fusion" / "icons"

HEAD = ("Manrope", "Light")          # 300 — big headlines
HEAD_DUENN = ("Manrope", "ExtraLight")  # 200 — „Hoffnung“
TEXT = ("Manrope", "Regular")        # 400 — sublines, values
TEXT_MITTEL = ("Manrope", "Medium")  # 500 — pill label
MARKE = ("Manrope", "SemiBold")      # 600 — „nomis“
MONO = ("JetBrains Mono", "Medium")
MONO_LEICHT = ("JetBrains Mono", "Light")

HEAD_TRACK_EM = -0.015
MONO_TRACK_EM = 0.14
HOLD_PAD_S = 0.4


def bereich(fmt: str) -> cw.Box:
    """Where text may sit: 9:16 safe area, 16:9 120 px margin on every side."""
    w, h = cw.FORMATS[fmt]
    if fmt == "9x16":
        side, top, bottom = cw.SAFE_9x16
        return cw.Box(side, top, w - side, h - bottom)
    return cw.Box(120, 120, w - 120, h - 120)


@dataclass(frozen=True)
class Raster:
    """Type grid of one format (pixels, top-left origin)."""

    left: int
    max_w: int          # widest a text column may run
    kicker_cap: int
    head_cap: int
    sub_cap: int
    wert_cap: int
    unten: int | None   # 16:9: block bottom edge
    oben: int | None    # 9:16: block top edge (below the HUD clock)


def raster(fmt: str) -> Raster:
    if fmt == "9x16":
        return Raster(left=120, max_w=860, kicker_cap=20, head_cap=54, sub_cap=27, wert_cap=36,
                      unten=None, oben=330)
    return Raster(left=120, max_w=1100, kicker_cap=16, head_cap=56, sub_cap=25, wert_cap=36,
                  unten=936, oben=None)


# --------------------------------------------------------------------------------------
# measured text
# --------------------------------------------------------------------------------------

_METRICS: dict[tuple[str, str], cw.FontMetrics] = {}


def metrics(font: tuple[str, str]) -> cw.FontMetrics:
    if font not in _METRICS:
        _METRICS[font] = cw.FontMetrics.find(*font)
    return _METRICS[font]


def tracking_for(met: cw.FontMetrics, text: str, em: float) -> float:
    """Fusion ``CharacterSpacing`` that adds ``em`` × em between glyphs, like CSS
    ``letter-spacing``. Fusion adds (s − 1) × Size × image width per glyph, and
    em = FUSION_FONT_HEIGHT × Size × width / font height, so s − 1 = em × 0.794 / height."""
    return 1.0 + em * cw.FUSION_FONT_HEIGHT / met.height


def spacing_for_width(met: cw.FontMetrics, text: str, size: float, image_w: int, width: float) -> float:
    """CharacterSpacing at which ``text`` is ``width`` px wide."""
    base = met.width_px(text, size, image_w, 1.0)
    return 1.0 + (width - base) / (size * image_w * max(1, len(text) - 1))


@dataclass(frozen=True)
class Satz:
    """A text measured for a given cap height (wrapped if ``max_w`` was given)."""

    font: tuple[str, str]
    text: str
    cap: float
    size: float
    tracking: float
    pitch: float            # line pitch in px
    line_spacing: float     # Text+ LineSpacing
    width: float
    em: float

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")

    @property
    def height(self) -> float:
        """Cap top of line 1 to the descender of the last line."""
        return (len(self.lines) - 1) * self.pitch + self.cap + 0.24 * self.em


def satz(c: cw.Comp, text: str, font: tuple[str, str], cap: float, *, track_em: float = 0.0,
         max_w: float | None = None, pitch: float | None = None, tracking: float | None = None) -> Satz:
    met = metrics(font)
    size = c.size_for_cap(cap, *font)
    if max_w is not None and "\n" not in text:
        text = umbruch(met, text, size, c.width, max_w, track_em)
    trk = tracking if tracking is not None else tracking_for(met, text, track_em)
    em = met.em_px(size, c.width)
    natural = em * met.height
    pitch = pitch if pitch is not None else 1.36 * cap
    width = max(met.width_px(line, size, c.width, trk) for line in text.split("\n"))
    return Satz(font, text, cap, size, trk, pitch, pitch / natural, width, em)


def umbruch(met: cw.FontMetrics, text: str, size: float, image_w: int, max_w: float, track_em: float) -> str:
    """Word wrap so every line is at most ``max_w`` px wide; a break after a comma or a
    full stop wins over a greedy break when both halves fit (no orphans like „empfehlen.“)."""
    trk = tracking_for(met, text, track_em)
    if met.width_px(text, size, image_w, trk) <= max_w:
        return text
    for sep in (", ", ". "):
        if sep in text:
            head, tail = text.split(sep, 1)
            head += sep.strip()
            if max(met.width_px(head, size, image_w, trk), met.width_px(tail, size, image_w, trk)) <= max_w:
                return f"{head}\n{tail}"
    words, lines, cur = text.split(" "), [], ""
    for word in words:
        cand = f"{cur} {word}" if cur else word
        if cur and met.width_px(cand, size, image_w, trk) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    lines.append(cur)
    if len(lines) == 2:   # balance two lines (no single-word orphan)
        best = min((max(met.width_px(" ".join(words[:i]), size, image_w, trk),
                        met.width_px(" ".join(words[i:]), size, image_w, trk)), i) for i in range(1, len(words)))
        lines = [" ".join(words[:best[1]]), " ".join(words[best[1]:])]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# records for the tests
# --------------------------------------------------------------------------------------


@dataclass
class Eintrag:
    id: str                 # bildtexte id
    text: str               # the script text (or the joined list for lists)
    teile: list[str]        # strings written into Text+ tools
    erscheinen: int         # scene-local frame the gate opens (= the event)
    bis: int                # rule hold end (VO end + 0.4 s / next event of the slot / scene end)
    t_out: int              # frame the exit starts (bis, pulled in to finish inside the scene)
    slot: str
    gate: str               # scene merge whose Blend is the gate
    event: str | None
    art: str = "erscheinen"  # or "landung": ``landung`` = (tool, input) whose last key is the event
    landung: tuple[str, str] | None = None
    symbole: dict[str, str] = field(default_factory=dict)   # glyph drawn as a tool → tool name
    bis_ende: bool = False


class Stack:
    """Bottom-up layer stack: each ``add`` merges a layer over the result so far."""

    def __init__(self, c: cw.Comp, prefix: str):
        self.c, self.prefix, self.n = c, prefix, 0
        self.top = c.background(c._unique(f"{prefix}Leer"), alpha=0.0)

    def add(self, layer: str, mask: str | None = None, **merge_kw: Any) -> str:
        self.n += 1
        name = self.c._unique(f"{self.prefix}M{self.n}")
        self.c.merge(name, self.top, layer, **merge_kw)
        if mask:
            self.c.apply_mask(name, mask)
        self.top = name
        return name


class Szene:
    """One scene comp in one format, with its timing helpers and records."""

    def __init__(self, szene_id: str, fmt: str, tl: dict):
        self.id, self.fmt, self.tl = szene_id, fmt, tl
        self.fps = int(tl["fps"])
        if szene_id == "abspann":
            self.s0, self.s1 = int(tl["abspann"]["start_f"]), int(tl["abspann"]["ende_f"])
        else:
            self.s0, self.s1 = cw.scene_span(tl, szene_id)
        self.len = self.s1 - self.s0
        w, h = cw.FORMATS[fmt]
        self.c = cw.Comp(w, h, frames=self.len, fps=self.fps)
        self.r = raster(fmt)
        self.area = bereich(fmt)
        self.stack = Stack(self.c, "Szene")
        self.eintraege: list[Eintrag] = []
        self.maskiert: dict[str, cw.Box] = {}   # Text+ tools shown only through a mask window
        self.layout: dict[str, Any] = {}          # positions worth checking in tests
        self.bildtexte = {b["id"]: b for b in SCRIPT["bildtexte"].get(szene_id, [])}
        farbe = next((s["farbe"] for s in SCRIPT["szenen"] if s["id"] == szene_id), "nacht")
        self.farbe = AKZENT[farbe]

    # ---------------------------------------------------------------- timing
    @property
    def vertical(self) -> bool:
        return self.fmt == "9x16"

    def ev(self, eid: str) -> int:
        return cw.event_frame(self.tl, eid) - self.s0

    def vo_ende(self, vo_id: str) -> int:
        for line in self.tl["vo"]:
            if line["id"] == vo_id:
                return round(line["ende_s"] * self.fps) - self.s0
        raise KeyError(vo_id)

    def wort(self, vo_id: str, word: str, edge: str = "start") -> int:
        return cw.word_frame(self.tl, vo_id, word, edge) - self.s0

    def halten(self, event: str, naechstes: str | None = None) -> int:
        """Rule hold end: VO line of ``event`` ends + 0.4 s, or the next event of the
        slot (``naechstes``), whichever is first; never past the scene end."""
        vo = next(e["anker"]["vo"] for e in EVENTS[self.id] if e["id"] == event)
        bis = self.vo_ende(vo) + round(HOLD_PAD_S * self.fps)
        if naechstes is not None:
            bis = min(bis, self.ev(naechstes))
        return min(bis, self.len - 1)

    # ---------------------------------------------------------------- pixels
    def px(self, x: float, y: float) -> tuple[float, float]:
        return self.c.px(x, y)

    def spalte(self, hoehen: list[float], luecken: list[float]) -> list[float]:
        """Tops of stacked rows (heights + gaps after each row) — anchored to the block
        bottom in 16:9 and to the block top in 9:16."""
        total = sum(hoehen) + sum(luecken[:len(hoehen) - 1])
        y = self.r.oben if self.vertical else self.r.unten - total  # type: ignore[operator]
        tops = []
        for i, h in enumerate(hoehen):
            tops.append(y)
            y += h + (luecken[i] if i < len(luecken) else 0)
        return tops

    # ---------------------------------------------------------------- output
    def fertig(self, lesefeld: bool = True) -> cw.Comp:
        if lesefeld and self.eintraege:
            self._lesefeld()
        self.c.output(self.stack.top)
        return self.c

    def _lesefeld(self) -> None:
        """A very soft night-coloured field under the text corner (bottom-left in 16:9,
        top in 9:16), on only while text is on screen — keeps white type readable over
        bright 3D frames without a box, blur or glass."""
        c = self.c
        if self.vertical:
            box = (-160, 140, 1300, 900)
        else:
            box = (-260, 560, 1700, 760)
        x, y, w, h = box
        mask = c.ellipse_mask("LesefeldMaske", center=c.px(x + w / 2, y + h / 2), width=c.px_w(w),
                              height=c.px_h(h), soft_edge=c.px_w(min(w, h) * 0.45))
        field = c.background("Lesefeld", color=FARBEN["nacht"], alpha=0.4)
        c.apply_mask(field, mask)
        spans = sorted((e.erscheinen, min(self.len - 1, e.t_out + 16)) for e in self.eintraege)
        merged: list[list[int]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + 40:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        keys: dict[int, float] = {}
        for a, b in merged:
            keys[max(0, a - 6)] = 0.0
            keys[min(self.len - 1, a + 10)] = 1.0
            keys[max(a + 11, b - 12)] = 1.0
            keys[b + 4 if b + 4 < self.len else self.len - 1] = 0.0
        under = Stack(c, "Lese")
        layer = under.add(field)
        if len(keys) >= 2:
            c.keyframes(layer, "Blend", dict(sorted(keys.items())), ease="in_out_cubic")
        first = next(n for n, t in c._tools.items() if n == "SzeneM1")
        c.link(first, "Background", under.top)

    def text_tools(self) -> list[str]:
        return [n for n, t in self.c._tools.items() if t.regid == "TextPlus"]


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------


def text(sz: Szene, name: str, s: Satz, x: float, top: float, *, color: Any = WHITE, alpha: float = 1.0,
         justify: str = "left", follower: cw.FollowerSpec | None = None, motion_blur: bool = True) -> str:
    """Text+ at cap-top ``top``; ``x`` is the left edge (or the centre for ``center``)."""
    r, g, b = cw.hex_rgb(color) if isinstance(color, str) else color[:3]
    return sz.c.text(name, s.text, font=s.font[0], style=s.font[1], size=s.size, color=(r, g, b, alpha),
                     center=sz.px(x, top), tracking=s.tracking, justify=justify, follower=follower,
                     line_spacing=s.line_spacing, motion_blur=motion_blur, quality=8)


def zeichen(t: int, *, dx: float = 0.0, dy: float = -40.0, px_w: float, dur: int = 18, delay: float = 1.2,
            alpha: float = 1.0, order: str = "left_to_right", ease: str = "out_expo",
            extra: dict[str, dict[int, Any]] | None = None) -> cw.FollowerSpec:
    """Per-character entrance: each glyph slides from (dx, dy) px (y up = positive) and
    fades in, ``delay`` frames after its neighbour."""
    keys: dict[str, dict[int, Any]] = {
        "CharacterOffset": {t: (dx * px_w, dy * px_w), t + dur: (0.0, 0.0)},
        "Opacity1": {t: 0.0, t + max(4, int(dur * 0.55)): alpha},
    }
    keys.update(extra or {})
    return cw.FollowerSpec(delay=delay, order=order, ease=ease, keys=keys,
                           per_key={"Opacity1": {t: "out_cubic"}})


def tippen(t: int, *, delay: float = 1.0, alpha: float = 1.0) -> cw.FollowerSpec:
    """Typewriter reveal for mono labels: each character steps on."""
    return cw.FollowerSpec(delay=delay, ease="linear", keys={"Opacity1": {t: 0.0, t + 2: alpha}})


def flaeche(sz: Szene, name: str, box_px: tuple[float, float, float, float], color: Any, alpha: float = 1.0,
            *, soft: float = 0.0, ellipse: bool = False, angle: float = 0.0) -> tuple[str, str]:
    """Solid rectangle/ellipse (left, top, width, height in px). Returns (layer, mask)."""
    c = sz.c
    x, y, w, h = box_px
    center = c.px(x + w / 2, y + h / 2)
    if ellipse:
        mask = c.ellipse_mask(f"{name}Maske", center=center, width=c.px_w(w), height=c.px_h(h),
                              soft_edge=c.px_w(soft))
    else:
        mask = c.rect_mask(f"{name}Maske", center=center, width=c.px_w(w), height=c.px_h(h),
                           soft_edge=c.px_w(soft), angle=angle)
    layer = c.background(name, color=color, alpha=alpha)
    c.apply_mask(layer, mask)
    return layer, mask


def zeichne_linie(sz: Szene, mask: str, box_px: tuple[float, float, float, float], t: int, dur: int,
                  von: str = "links", ease: str = "out_expo") -> None:
    """Animate a rectangle mask so the line draws from one side (or the middle)."""
    c = sz.c
    x, y, w, h = box_px
    cy = y + h / 2
    start_x = {"links": x, "rechts": x + w, "mitte": x + w / 2}[von]
    c.keyframes(mask, "Center", {t: c.px(start_x, cy), t + dur: c.px(x + w / 2, cy)}, ease=ease)
    c.keyframes(mask, "Width", {t: 0.0, t + dur: c.px_w(w)}, ease=ease)


def wisch(sz: Szene, name: str, layer: str, box_px: tuple[float, float, float, float], t: int, dur: int,
          richtung: str = "hoch", soft: float = 0.0, ease: str = "out_expo") -> str:
    """Mask wipe: ``layer`` is revealed by a window that grows from one edge
    (``hoch`` = from the bottom edge up, ``rechts`` = from the left edge to the right)."""
    c = sz.c
    x, y, w, h = box_px
    full = c.px(x + w / 2, y + h / 2)
    if richtung == "hoch":
        m = c.rect_mask(f"{name}Wisch", center=c.px(x + w / 2, y + h), width=c.px_w(w), height=0.0,
                        soft_edge=c.px_w(soft))
        c.keyframes(m, "Center", {t: c.px(x + w / 2, y + h), t + dur: full}, ease=ease)
        c.keyframes(m, "Height", {t: 0.0, t + dur: c.px_h(h)}, ease=ease)
    elif richtung in ("rechts", "links"):
        edge = x if richtung == "rechts" else x + w
        m = c.rect_mask(f"{name}Wisch", center=c.px(edge, y + h / 2), width=0.0, height=c.px_h(h),
                        soft_edge=c.px_w(soft))
        c.keyframes(m, "Center", {t: c.px(edge, y + h / 2), t + dur: full}, ease=ease)
        c.keyframes(m, "Width", {t: 0.0, t + dur: c.px_w(w)}, ease=ease)
    else:
        raise ValueError(richtung)
    g = Stack(c, name)
    g.add(layer, mask=m)
    return g.top


def gruppe(sz: Szene, name: str, layers: list[str | tuple[str, str]]) -> str:
    """Merge layers (optionally ``(layer, mask)``) into one group image."""
    g = Stack(sz.c, name)
    for item in layers:
        if isinstance(item, tuple):
            g.add(item[0], mask=item[1])
        else:
            g.add(item)
    return g.top


ABGANG: dict[str, tuple[float, float]] = {
    "hoch": (0.0, 30.0), "runter": (0.0, -26.0), "links": (-70.0, 0.0), "rechts": (70.0, 0.0),
}


def eintrag(sz: Szene, bid: str, layer: str, t_in: int, bis: int, *, slot: str, teile: list[str],
            abgang: str = "hoch", dur: int = 14, text: str | None = None, event: str | None = None,
            art: str = "erscheinen", landung: tuple[str, str] | None = None, symbole: dict | None = None,
            bis_ende: bool = False, eingang_xf: dict[int, tuple[float, float]] | None = None,
            pivot_px: tuple[float, float] | None = None) -> str:
    """Wrap ``layer`` in its exit move and gate it into the scene; record the entry.

    Exit styles: ``hoch``/``runter``/``links``/``rechts`` (move + fade, motion blur) or
    ``klein`` (shrinks 6 % about ``pivot_px`` + fade). ``bis_ende`` = no exit (lockup)."""
    c = sz.c
    name = cw._safe(f"E{bid[:1].upper()}{bid[1:]}")
    t_out = min(bis, sz.len - 1 - dur)
    pivot = sz.px(*pivot_px) if pivot_px else None
    xf = c.transform(c._unique(f"{name}Xf"), layer, motion_blur=True, quality=10, shutter=200.0, pivot=pivot)
    keys: dict[int, tuple[float, float]] = dict(eingang_xf or {})
    if not bis_ende and abgang in ABGANG:
        dx, dy = ABGANG[abgang]
        keys[t_out] = (0.5, 0.5)
        keys[t_out + dur] = (0.5 + c.px_w(dx), 0.5 + c.px_h(dy))
    elif not bis_ende and abgang == "klein":
        c.keyframes(xf, "Size", {t_out: 1.0, t_out + dur: 0.94}, ease="in_cubic")
    elif not bis_ende:
        raise ValueError(f"unknown exit {abgang!r}")
    if len(keys) >= 2:
        per = {t_out: "in_cubic"} if not bis_ende else {}
        c.keyframes(xf, "Center", dict(sorted(keys.items())), ease="out_expo", per_key=per)
    gate = sz.stack.add(xf)
    blend = {t_in - 1: 0.0, t_in: 1.0} if t_in > 0 else {t_in: 1.0}
    if not bis_ende:
        blend.update({t_out: 1.0, t_out + dur: 0.0})
    if len(blend) >= 2:
        c.keyframes(gate, "Blend", blend, ease="linear", per_key={t_out: "in_cubic"} if not bis_ende else None)
    else:
        c.set_input(gate, "Blend", 1.0)
    sz.eintraege.append(Eintrag(
        id=bid, text=text if text is not None else sz.bildtexte.get(bid, {}).get("text", ""),
        teile=list(teile), erscheinen=t_in, bis=bis, t_out=t_out if not bis_ende else sz.len - 1,
        slot=slot, gate=gate, event=event, art=art, landung=landung, symbole=dict(symbole or {}),
        bis_ende=bis_ende))
    return gate


def kicker(sz: Szene, bid: str, top: float, t: int, bis: int, *, abgang: str = "links") -> str:
    """Mono label in capitals, tracked, with a short accent line in the scene colour that
    draws first; the letters then type on."""
    c, r = sz.c, sz.r
    label = sz.bildtexte[bid]["text"]
    s = satz(c, label, MONO, r.kicker_cap, track_em=MONO_TRACK_EM)
    line_w, gap = 28, 14
    line_box = (r.left, top + r.kicker_cap / 2 - 1, line_w, 2)
    line, m = flaeche(sz, cw._safe(f"K{bid}Linie"), line_box, sz.farbe, 1.0)
    zeichne_linie(sz, m, line_box, t, 12)
    tx = text(sz, cw._safe(f"K{bid}Text"), s, r.left + line_w + gap, top, alpha=0.78,
              follower=tippen(t + 6, delay=1.0, alpha=0.78), motion_blur=False)
    g = gruppe(sz, cw._safe(f"K{bid}G"), [line, tx])
    return eintrag(sz, bid, g, t, bis, slot="kicker", teile=[label], abgang=abgang, dur=12,
                   event=sz.bildtexte[bid].get("event"))


def ring(sz: Szene, name: str, box_px: tuple[float, float, float, float], stroke: float, *,
         pill: bool = False, angle: float = 0.0) -> str:
    """Outline mask of a rectangle (or a fully rounded pill): outer shape minus inner."""
    c = sz.c
    x, y, w, h = box_px

    def shape(prefix: str, x0: float, y0: float, w0: float, h0: float, prev: str | None, subtract: bool) -> str:
        cy = y0 + h0 / 2
        if not pill:
            m = c.rect_mask(f"{prefix}R", center=c.px(x0 + w0 / 2, cy), width=c.px_w(w0), height=c.px_h(h0),
                            angle=angle, combine_with=prev)
            parts = [m]
        else:
            rad = h0 / 2
            m1 = c.rect_mask(f"{prefix}R", center=c.px(x0 + w0 / 2, cy), width=c.px_w(w0 - h0),
                             height=c.px_h(h0), combine_with=prev)
            m2 = c.ellipse_mask(f"{prefix}L", center=c.px(x0 + rad, cy), width=c.px_w(h0), height=c.px_h(h0),
                                combine_with=m1)
            m3 = c.ellipse_mask(f"{prefix}E", center=c.px(x0 + w0 - rad, cy), width=c.px_w(h0),
                                height=c.px_h(h0), combine_with=m2)
            parts = [m1, m2, m3]
        if subtract:
            for p in parts:
                c.set_input(p, "PaintMode", cw.FuID("Subtract"))
        return parts[-1]

    outer = shape(f"{name}Aussen", x, y, w, h, None, False)
    inner = shape(f"{name}Innen", x + stroke, y + stroke, w - 2 * stroke, h - 2 * stroke, outer, True)
    return inner


def bild(sz: Szene, name: str, path: Path) -> str:
    return sz.c.image(name, path)


def wortmarke(sz: Szene, prefix: str, cx: float, top: float, cap: float, t: int, *, delay: float = 2.0,
              rise: float = 26.0) -> tuple[str, float, list[str]]:
    """Word mark „nomis“ (Manrope 600) + „success“ (Manrope 400, 55 %), tracking
    −0.015 em, centred on ``cx``; writes itself out letter by letter from ``t``.
    Returns (layer, width px, texts)."""
    c = sz.c
    teile = SCRIPT["bildtexte"]["s8_morgen"]
    marke = next(b for b in teile if b["id"] == "wortmarke")
    (p1, p2), em_track = marke["teile"], marke["tracking_em"]
    s1 = satz(c, p1["text"], MARKE, cap, track_em=em_track)
    s2 = satz(c, p2["text"], TEXT, cap, track_em=em_track)
    # the joint between the parts gets the same −0.015 em as every other pair
    joint = em_track * s1.em
    total = s1.width + joint + s2.width
    left = cx - total / 2
    n1 = len(p1["text"])
    a = text(sz, f"{prefix}A", s1, left, top, alpha=p1["deckkraft"],
             follower=zeichen(t, dy=-rise, px_w=c.px_w(1), dur=20, delay=delay, alpha=p1["deckkraft"],
                              extra={"SoftnessX1": {t: 6.0, t + 14: 0.0}}))
    b = text(sz, f"{prefix}B", s2, left + s1.width + joint, top, alpha=p2["deckkraft"],
             follower=zeichen(t + round(n1 * delay), dy=-rise, px_w=c.px_w(1), dur=20, delay=delay,
                              alpha=p2["deckkraft"],
                              extra={"SoftnessX1": {t + round(n1 * delay): 6.0, t + round(n1 * delay) + 14: 0.0}}))
    return gruppe(sz, f"{prefix}G", [a, b]), total, [p1["text"], p2["text"]]


def rollzaehler(sz: Szene, prefix: str, states: list[str], ticks: list[int], left: float, top: float,
                cap: float, *, font: tuple[str, str] = MONO_LEICHT, tracking: float = 1.08,
                roll: int = 16) -> tuple[str, str, float]:
    """Mechanical rolling counter: every column that changes is a strip of digits that
    rolls up through a soft window with real vertical motion blur, landing on ``ticks``
    (one per state change). Returns (layer, (tool of the last column), width px)."""
    c = sz.c
    met = metrics(font)
    size = c.size_for_cap(cap, *font)
    # cell pitch as in scene 1's counter (advance × tracking): the big clock stays compact
    cell = dict(met.advances)[ord("0")] / met.upm * met.em_px(size, c.width) * tracking
    pitch = cap * 1.75
    g = Stack(c, f"{prefix}G")
    last_xf = ""
    for idx, ch in enumerate(states[0]):
        cx = left + (idx + 0.5) * cell
        changes = [k for k in range(1, len(states)) if states[k][idx] != states[k - 1][idx]]
        s = satz(c, ch, font, cap, tracking=tracking)
        if not changes:
            g.add(text(sz, f"{prefix}Z{idx}", s, cx, top, justify="center", motion_blur=False))
            continue
        digits = [states[0][idx]] + [states[k][idx] for k in changes]
        strip = Stack(c, f"{prefix}Z{idx}Rolle")
        window = cw.Box(int(cx - cell * 0.6), int(top - cap * 0.45), int(cx + cell * 0.6), int(top + cap * 1.45))
        for j, d in enumerate(digits):
            tool = text(sz, f"{prefix}Z{idx}D{j}", satz(c, d, font, cap, tracking=tracking), cx, top + j * pitch,
                        justify="center", motion_blur=False)
            sz.maskiert[tool] = window
            strip.add(tool)
        xf = c.transform(f"{prefix}Z{idx}Xf", strip.top, motion_blur=True, quality=16, shutter=300.0)
        keys = {0: (0.5, 0.5)}
        for j, k in enumerate(changes, start=1):
            land = ticks[k - 1]
            keys[land - roll] = (0.5, 0.5 + (j - 1) * c.px_h(pitch))
            keys[land] = (0.5, 0.5 + j * c.px_h(pitch))
        c.keyframes(xf, "Center", keys, ease="out_quart")
        m = c.rect_mask(f"{prefix}Z{idx}Fenster", center=c.px(cx, top + cap / 2), width=c.px_w(cell * 1.15),
                        height=c.px_h(cap * 1.9), soft_edge=c.px_w(cap * 0.35))
        g.add(xf, mask=m)
        last_xf = xf
    return g.top, last_xf, cell * len(states[0])


def striche(sz: Szene, name: str, punkte: list[tuple[float, float]], stroke: float) -> str:
    """Union mask of a polyline (px points, top-left origin): one rotated rectangle per
    segment plus round joints, so arrows and check marks are crisp vectors."""
    import math

    c = sz.c
    prev: str | None = None
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(punkte, punkte[1:])):
        length = math.hypot(x1 - x0, y1 - y0)
        angle = -math.degrees(math.atan2(y1 - y0, x1 - x0))   # Fusion: y up, CCW positive
        prev = c.rect_mask(f"{name}S{i}", center=c.px((x0 + x1) / 2, (y0 + y1) / 2), width=c.px_w(length),
                           height=c.px_h(stroke), angle=angle, combine_with=prev)
    for i, (x, y) in enumerate(punkte):
        prev = c.ellipse_mask(f"{name}P{i}", center=c.px(x, y), width=c.px_w(stroke), height=c.px_h(stroke),
                              combine_with=prev)
    assert prev is not None
    return prev


def haken(sz: Szene, name: str, left: float, top: float, h: float, color: Any, t: int, dur: int = 10,
          stroke: float | None = None) -> str:
    """Check mark (proportions of fusion/icons/haken.svg) that draws itself left→right."""
    pts = [(4.6, 12.6), (9.2, 17.2), (19.4, 7.0)]
    k = h / 10.2                               # drawn height of the svg path is 10.2 units
    punkte = [(left + (x - 4.6) * k, top + (y - 7.0) * k) for x, y in pts]
    stroke = stroke or max(2.0, h * 0.13)
    mask = striche(sz, name, punkte, stroke)
    layer = sz.c.background(name, color=color, alpha=1.0)
    sz.c.apply_mask(layer, mask)
    w = (19.4 - 4.6) * k
    return wisch(sz, f"{name}Zug", layer, (left - stroke, top - stroke, w + 2 * stroke, h + 2 * stroke), t, dur,
                 richtung="rechts", ease="in_out_cubic")
