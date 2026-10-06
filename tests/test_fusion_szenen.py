"""Tests for the scene typography comps s2–s8 and the end card (Task 6).

Every check parses the *generated comp text* (Lua-table parser from test_comp_writer)
and recomputes the expectation from the contract files (script.json, events.json,
timeline.json) — the scene modules' own bookkeeping is only used to find the gate merge
and the text tools of an entry.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fusion import comp_writer as cw  # noqa: E402
from test_comp_writer import parse_comp  # noqa: E402

SZENEN = ["s2_website", "s3_netz", "s4_ernstfall", "s5_betrieb", "s6_beweis", "s7_team", "s8_morgen", "abspann"]
FORMATE = ["16x9", "9x16"]
SCRIPT = json.loads((REPO / "script.json").read_text(encoding="utf-8"))
EVENTS = json.loads((REPO / "szenen" / "events.json").read_text(encoding="utf-8"))["events"]
LOCKUP = {"wortmarke", "claim", "cta", "url"}          # s8: holds to the last frame
SYMBOLE = ("→", "✓")                                  # drawn as shapes, not typed
# the only entry whose gate opens before its event: key_turn is 55 frames before the end
# of s2, so „git clone — Ihr Repository“ starts typing 0.8 s early to be readable
VORLAUF = {("s2_website", "code"): 48}


@pytest.fixture(scope="module")
def timeline():
    return cw.load_timeline()


@pytest.fixture(scope="module")
def gebaut(timeline):
    """(szene, fmt) → (Szene object, parsed tools)."""
    out = {}
    for name in SZENEN:
        mod = importlib.import_module(f"fusion.szenen.{name}")
        for fmt in FORMATE:
            sz = mod.bau(fmt, timeline)
            out[(name, fmt)] = (sz, parse_comp(sz.c.to_text())["Tools"])
    return out


CASES = [(n, f) for n in SZENEN for f in FORMATE]


# --------------------------------------------------------------------------------------
# helpers on the parsed comp
# --------------------------------------------------------------------------------------


def _spline(tools: dict, tool: str, inp: str) -> dict[int, float] | None:
    """{frame: value} of an animated number input, None if it is static."""
    node = tools[tool]["Inputs"].get(inp)
    if node is None or "SourceOp" not in node:
        return None
    spline = tools[node["SourceOp"]]
    if spline.type_name == "XYPath":
        spline = tools[spline["Inputs"]["Y"]["SourceOp"]]
    return {int(f): kf[1] for f, kf in spline["KeyFrames"].items()}


def _static(tools: dict, tool: str, inp: str, default=None):
    node = tools[tool]["Inputs"].get(inp)
    if node is None:
        return default
    if "Value" in node:
        return node["Value"]
    keys = _spline(tools, tool, inp)
    return keys


def _text_of(tools: dict, tool: str) -> str:
    node = tools[tool]["Inputs"]["StyledText"]
    if "Value" in node:
        return node["Value"]
    follower = tools[node["SourceOp"]]
    return follower["Inputs"]["Text"]["Value"]["Value"]


def _text_tools(tools: dict) -> list[str]:
    return [n for n, t in tools.items() if t.type_name == "TextPlus"]


def _local_event(timeline: dict, sz, event: str) -> int:
    return cw.event_frame(timeline, event) - sz.s0


def _script_entry(name: str, bid: str) -> dict:
    return next(b for b in SCRIPT["bildtexte"][name] if b["id"] == bid)


def _expected_event(name: str, bid: str) -> str:
    return "endcard_in" if name == "abspann" else _script_entry(name, bid)["event"]


def _text_box(tools: dict, tool: str, w: int, h: int) -> cw.Box:
    """Pixel box of a Text+ at rest, from its inputs and the font file."""
    inp = tools[tool]["Inputs"]
    font, style = inp["Font"]["Value"], inp["Style"]["Value"]
    size = inp["Size"]["Value"]
    met = cw.FontMetrics.find(font, style)
    spacing = _static(tools, tool, "CharacterSpacing", 1.0)
    if isinstance(spacing, dict):                       # animated tracking: widest state
        spacing = max(spacing.values())
    cx, cy = inp["Center"]["Value"][1], inp["Center"]["Value"][2]
    lines = _text_of(tools, tool).split("\n")
    em = met.em_px(size, w)
    pitch = em * met.height * inp["LineSpacing"]["Value"]
    width = max(met.width_px(line, size, w, spacing) for line in lines)
    anchor = inp["HorizontalLeftCenterRight"]["Value"]
    x = cx * w
    left = x if anchor == -1 else x - width / 2 if anchor == 0 else x - width
    top = (1 - cy) * h
    assert inp["VerticalTopCenterBottom"]["Value"] == -1, f"{tool}: layout assumes a top anchor"
    bottom = top + (len(lines) - 1) * pitch + met.cap * em + 0.2 * em
    return cw.Box(math.floor(left), math.floor(top), math.ceil(left + width), math.ceil(bottom))


def _area(fmt: str) -> cw.Box:
    return cw.Box(90, 215, 990, 1515) if fmt == "9x16" else cw.Box(120, 120, 1800, 960)


def _inside(box: cw.Box, area: cw.Box) -> bool:
    return area.left <= box.left and box.right <= area.right and area.top <= box.top and box.bottom <= area.bottom


# --------------------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,fmt", CASES)
def test_comp_is_valid_and_spans_the_scene(gebaut, timeline, name, fmt):
    sz, tools = gebaut[(name, fmt)]
    root = parse_comp(sz.c.to_text())
    span = (timeline["abspann"]["ende_f"] - timeline["abspann"]["start_f"] if name == "abspann"
            else int.__sub__(*reversed(cw.scene_span(timeline, name))))
    assert [root["RenderRange"][1], root["RenderRange"][2]] == [0, span - 1]
    assert (root["Prefs"]["Comp"]["FrameFormat"]["Width"], root["Prefs"]["Comp"]["FrameFormat"]["Height"]) \
        == cw.FORMATS[fmt]
    assert tools["MediaOut1"]["Inputs"]["Input"]["SourceOp"] in tools
    # every link points at an existing tool
    for tname, tool in tools.items():
        for inp, node in (tool.get("Inputs") or {}).items():
            if isinstance(node, dict) and "SourceOp" in node:
                assert node["SourceOp"] in tools, f"{tname}.{inp} → {node['SourceOp']}"


def _quality_values(tools: dict, tool: str) -> list[float]:
    node = tools[tool].get("Inputs", {}).get("Quality")
    if node is None:
        return []
    if "Value" in node:
        return [node["Value"]]
    return list(_spline(tools, tool, "Quality").values())


def test_no_comp_renders_more_than_6_motion_blur_samples(tmp_path, timeline):
    """Render time: Quality ≤ 4, ≤ 6 only on the rolling-counter drums, 1 at rest, and
    no blur on tools that never move (all 20 comps, HUD and scene 1 included)."""
    from fusion import build_all

    for path in build_all.build_all(tmp_path, FORMATE, timeline):
        tools = parse_comp(path.read_text(encoding="utf-8"))["Tools"]
        for name, tool in tools.items():
            qs = _quality_values(tools, name)
            if not qs:
                continue
            assert max(qs) <= cw.MB_QUALITY_FAST, f"{path.name}:{name} Quality {max(qs)}"
            drum = "Z" in name and name.endswith("Xf") and "Rolle" not in name and name[-3:-2].isdigit()
            if not drum:
                assert max(qs) <= cw.MB_QUALITY_MAX, f"{path.name}:{name} Quality {max(qs)}"
            if tool.get("Inputs", {}).get("MotionBlur", {}).get("Value") == 1:
                keys = _spline(tools, name, "Quality")
                assert keys is not None and min(keys.values()) == 1, f"{path.name}:{name} blurs at rest"


def test_motion_blur_only_inside_motion_windows():
    comp = cw.Comp(1920, 1080, frames=300)
    comp.text("T", "x")
    comp.transform("Ruhig", "T", quality=16)
    comp.transform("Bewegt", "T", quality=16)
    comp.keyframes("Bewegt", "Center", {100: (0.5, 0.4), 120: (0.5, 0.5), 200: (0.5, 0.5)})
    tools = parse_comp(comp.to_text())["Tools"]
    assert tools["Ruhig"]["Inputs"]["MotionBlur"]["Value"] == 0
    q = _spline(tools, "Bewegt", "Quality")
    assert q[0] == 1 and q[99] == cw.MB_QUALITY_MAX and q[121] == 1 and max(q.values()) == cw.MB_QUALITY_MAX


def test_build_all_writes_every_comp_of_both_formats(tmp_path, timeline):
    from fusion import build_all

    paths = build_all.build_all(tmp_path, FORMATE, timeline)
    expected = {tmp_path / f / f"{n}.comp" for f in FORMATE for n in ["hud", "s1_nacht", *SZENEN]}
    assert set(paths) == expected
    for p in paths:
        parse_comp(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# every Bildtext is there, appears on its event, holds, and leaves with a move
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,fmt", CASES)
def test_every_bildtext_is_built_with_its_exact_words(gebaut, name, fmt):
    sz, tools = gebaut[(name, fmt)]
    ids = [b["id"] for b in SCRIPT["bildtexte"][name]]
    assert sorted(e.id for e in sz.eintraege) == sorted(ids)
    typed = [_text_of(tools, t).replace("\n", " ") for t in _text_tools(tools)]
    for e in sz.eintraege:
        entry = _script_entry(name, e.id)
        if "teile" in entry:
            want = "".join(p["text"] for p in entry["teile"])
        else:
            want = " ".join([entry["text"], entry.get("unter", ""), entry.get("ersatz", ""),
                             *entry.get("monate", [])])
        for sym in SYMBOLE:
            want = want.replace(sym, " ")
        assert "".join(want.split()) == "".join("".join(e.teile).split()), e.id
        for part in e.teile:
            assert any(part in t for t in typed), f"{e.id}: {part!r} is not in any Text+"
        for sym, tool in e.symbole.items():
            assert tool in tools, f"{e.id}: symbol {sym} should be drawn by {tool}"


@pytest.mark.parametrize("name,fmt", CASES)
def test_every_bildtext_appears_exactly_on_its_event_frame(gebaut, timeline, name, fmt):
    sz, tools = gebaut[(name, fmt)]
    for e in sz.eintraege:
        assert e.event == _expected_event(name, e.id)
        ev = _local_event(timeline, sz, e.event)
        blend = _spline(tools, e.gate, "Blend")
        assert blend is not None, f"{e.id}: gate {e.gate} has no Blend animation"
        opened = min(f for f, v in blend.items() if v > 0)
        assert all(v == 0 for f, v in blend.items() if f < opened)
        if e.art == "landung":
            # a rolling counter is visible before; its last column lands on the event
            land = _spline(tools, e.landung[0], e.landung[1])
            assert max(land) == ev, f"{e.id}: lands on {max(land)}, event {ev}"
            assert opened <= ev
        else:
            lead = VORLAUF.get((name, e.id), 0)
            assert e.vorlauf == lead
            assert opened == ev - lead, f"{e.id}: gate opens on {opened}, event {e.event} is {ev}"


@pytest.mark.parametrize("name,fmt", CASES)
def test_every_bildtext_holds_for_its_sentence_then_leaves_with_a_move(gebaut, timeline, name, fmt):
    sz, tools = gebaut[(name, fmt)]
    fps = timeline["fps"]
    last = sz.len - 1
    for e in sz.eintraege:
        blend = _spline(tools, e.gate, "Blend")
        frames = sorted(blend)
        if name == "s8_morgen" and e.id in LOCKUP:
            assert blend[frames[-1]] == 1.0, f"{e.id}: the lockup must stand to the last frame"
            continue
        # rule: VO line of the event ends + 0.4 s, or the next event that takes the slot
        if name == "abspann":
            want = last
        else:
            vo = next(ev["anker"]["vo"] for ev in EVENTS[name] if ev["id"] == e.event)
            line = next(v for v in timeline["vo"] if v["id"] == vo)
            want = round(line["ende_s"] * fps) - sz.s0 + round(0.4 * fps)
            later = [o.erscheinen for o in sz.eintraege
                     if o.erscheinen > e.erscheinen and (o.slot == e.slot or e.slot.startswith(o.slot + "/"))]
            want = min([want, last, *later])
        assert blend[frames[-1]] == 0.0, f"{e.id}: never leaves"
        t_out = frames[-2]
        dur = frames[-1] - t_out
        assert blend[t_out] == 1.0 and 8 <= dur <= 20
        assert t_out == min(want, last - dur), f"{e.id}: exit starts {t_out}, rule says {want}"
        assert frames[-1] <= last
        # readable all the way: the gate stays fully open between appearing and the exit
        assert all(blend[f] == 1.0 for f in frames if e.erscheinen <= f <= t_out)
        # the exit is a move, not only a fade: the gate's transform animates at t_out
        xf = tools[e.gate]["Inputs"]["Foreground"]["SourceOp"]
        assert tools[xf].type_name == "Transform"
        moves = [_spline(tools, xf, i) for i in ("Center", "Size")]
        assert any(m and t_out in m and t_out + dur in m for m in moves), f"{e.id}: exit has no move"


# --------------------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,fmt", CASES)
def test_all_text_boxes_lie_inside_the_margin_or_safe_area(gebaut, name, fmt):
    sz, tools = gebaut[(name, fmt)]
    w, h = cw.FORMATS[fmt]
    area = _area(fmt)
    texts = _text_tools(tools)
    assert texts
    for tool in texts:
        if tool in sz.maskiert:     # strips behind a window (counter drums, month cards)
            assert _inside(sz.maskiert[tool], area), f"{tool}: window {sz.maskiert[tool]} outside {area}"
            continue
        box = _text_box(tools, tool, w, h)
        assert _inside(box, area), f"{tool} {_text_of(tools, tool)!r}: {box} outside {area}"


@pytest.mark.parametrize("name,fmt", [c for c in CASES if c[0] not in ("s8_morgen", "abspann")])
def test_running_text_is_flush_left_on_the_grid(gebaut, name, fmt):
    """No centred running text outside the s8 lockup and the end card."""
    sz, tools = gebaut[(name, fmt)]
    for tool in _text_tools(tools):
        if tool in sz.maskiert:
            continue
        assert tools[tool]["Inputs"]["HorizontalLeftCenterRight"]["Value"] == -1, tool


@pytest.mark.parametrize("fmt", FORMATE)
def test_headlines_use_light_manrope_with_slightly_negative_tracking(gebaut, fmt):
    sz, tools = gebaut[("s2_website", fmt)]
    t1 = tools["T1"]["Inputs"]
    assert (t1["Font"]["Value"], t1["Style"]["Value"]) == ("Manrope", "Light")
    assert 0.95 < t1["CharacterSpacing"]["Value"] < 1.0
    kick = tools["KkickerText"]["Inputs"]
    assert kick["Font"]["Value"] == "JetBrains Mono" and kick["CharacterSpacing"]["Value"] > 1.06


# --------------------------------------------------------------------------------------
# Fusion semantics measured on the control renders of 06.10.2026
# --------------------------------------------------------------------------------------


def test_tracking_adds_spacing_times_size_times_width_per_glyph():
    """Measured ink widths (16:9, Resolve 21.1.1): the s6 stamp (JetBrains Mono Bold,
    cap 24, spacing 1.161) was 881 px, the s5 kicker (JetBrains Mono Medium, cap 15,
    spacing 1.253) 390 px. A multiplier model predicted 713 / 295 px."""
    comp = cw.Comp(1920, 1080, frames=2)
    for font, cap, spacing, text, measured in (
            (("JetBrains Mono", "Bold"), 24, 1.161, "KEIN MOCKUP · ECHTES MONITORING", 881),
            (("JetBrains Mono", "Medium"), 15, 1.253, "BETRIEB & MIGRATION", 390)):
        met = cw.FontMetrics.find(*font)
        size = comp.size_for_cap(cap, *font)
        assert met.width_px(text, size, 1920, spacing) == pytest.approx(measured, rel=0.02)


def test_background_colour_is_premultiplied():
    comp = cw.Comp(1920, 1080, frames=2)
    comp.background("Feld", color="#e44b8d", alpha=0.1)
    tools = parse_comp(comp.to_text())["Tools"]
    r, _, _ = cw.hex_rgb("#e44b8d")
    assert tools["Feld"]["Inputs"]["TopLeftRed"]["Value"] == pytest.approx(r * 0.1)


def test_exit_never_starts_before_the_entrance_near_the_scene_end(timeline):
    """An event in the last frames of a scene must not flash its text before the event."""
    from fusion.szenen import gemeinsam_typo as gt

    sz = gt.Szene("s6_beweis", "16x9", timeline)
    layer = sz.c.background("Probe", color=(1, 1, 1), alpha=1.0)
    t_in = sz.len - 5
    gt.eintrag(sz, "stempel", layer, t_in, sz.len - 1, slot="probe", teile=[], dur=14)
    sz.fertig()
    tools = parse_comp(sz.c.to_text())["Tools"]
    blend = _spline(tools, sz.eintraege[0].gate, "Blend")
    assert all(v == 0 for f, v in blend.items() if f < t_in)
    assert min(f for f, v in blend.items() if v > 0) == t_in


def test_scene1_lines_never_leave_before_they_went_dark(timeline):
    """L7 in scene 1: with lights_off pushed late, the lines' exit follows it."""
    import copy

    from fusion.szenen import s1_nacht as s1

    tl = copy.deepcopy(timeline)
    _, s1_end = cw.scene_span(tl, "s1_nacht")
    next(e for e in tl["events"] if e["id"] == "lights_off")["f"] = s1_end - 40
    z = s1.zeiten(tl)
    assert z.zeilen_weg > z.licht_aus


@pytest.mark.parametrize("name,fmt", CASES)
def test_no_background_blinks_through_its_alpha(gebaut, name, fmt):
    """Fusion does not premultiply a Background: an animated TopLeftAlpha leaves the full
    colour on screen. Blinking and fading go through a merge's Blend."""
    _, tools = gebaut[(name, fmt)]
    for tname, tool in tools.items():
        if tool.type_name == "Background":
            assert "SourceOp" not in tool["Inputs"].get("TopLeftAlpha", {}), tname


# --------------------------------------------------------------------------------------
# scene specifics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATE)
def test_s2_git_clone_label_stands_fully_typed_for_at_least_1_2_s(gebaut, timeline, fmt):
    sz, tools = gebaut[("s2_website", fmt)]
    fps = timeline["fps"]
    text = _text_of(tools, "Code")
    follower = tools[tools["Code"]["Inputs"]["StyledText"]["SourceOp"]]
    delay = follower["Inputs"]["Delay"]["Value"]
    typing = _spline(tools, follower.name if hasattr(follower, "name") else
                     tools["Code"]["Inputs"]["StyledText"]["SourceOp"], "Opacity1")
    complete = max(typing) + delay * (len(text) - 1)          # last glyph fully on
    entry = next(e for e in sz.eintraege if e.id == "code")
    blend = _spline(tools, entry.gate, "Blend")
    frames = sorted(blend)
    t_out, gone = frames[-2], frames[-1]
    assert (t_out - complete) / fps >= 1.2, f"stands {(t_out - complete) / fps:.2f} s"
    assert gone <= sz.len - 1                                 # leaves inside the scene
    key = _local_event(timeline, sz, "key_turn")
    assert complete <= key < t_out                            # typed when the key turns
    # the cursor is dark until the text is typed, then blinks via its merge
    cursor = next(n for n, t in tools.items() if t.type_name == "Merge"
                  and t["Inputs"].get("Foreground", {}).get("SourceOp") == "CodeCursor")
    blink = _spline(tools, cursor, "Blend")
    assert all(v == 0 for f, v in blink.items() if f <= complete)
    assert any(v == 1 for v in blink.values())


@pytest.mark.parametrize("fmt", FORMATE)
def test_s2_calendar_ticks_off_thirty_days_and_months_run_jan_to_dez(gebaut, timeline, fmt):
    sz, tools = gebaut[("s2_website", fmt)]
    days = _text_of(tools, "KalZahlen").split()
    assert days == [str(d) for d in range(1, 31)]
    assert _text_of(tools, "KalHaken").count("✓") == 30
    months = [_text_of(tools, f"Karte{k}Monat") for k in range(12)]
    assert months == _script_entry("s2_website", "monat")["monate"]
    glide = _spline(tools, "KartenXf", "Center")
    assert glide is not None and min(glide) == _local_event(timeline, sz, "month_carousel")


@pytest.mark.parametrize("fmt", FORMATE)
def test_s6_stamp_is_mint_turned_minus_3_degrees_and_lands_with_a_press(gebaut, timeline, fmt):
    sz, tools = gebaut[("s6_beweis", fmt)]
    t = _local_event(timeline, sz, "stamp")
    assert tools["StempelXf"]["Inputs"]["Angle"]["Value"] == -3.0
    size = _spline(tools, "StempelXf", "Size")
    assert size[t] == pytest.approx(1.06) and size[max(size)] == pytest.approx(1.0) and max(size) - t <= 8
    blur = _spline(tools, "StempelKante", "XBlurSize")
    assert blur[t] > 0 and blur[max(blur)] == 0
    st = tools["Stempel"]["Inputs"]
    mint = cw.hex_rgb(SCRIPT["farben"]["mint"])
    assert (st["Red1"]["Value"], st["Green1"]["Value"], st["Blue1"]["Value"]) == pytest.approx(mint)
    assert _text_of(tools, "Stempel") == "KEIN MOCKUP · ECHTES MONITORING"


@pytest.mark.parametrize("fmt", FORMATE)
def test_s8_pill_has_a_fully_round_1_5px_outline_and_no_shadow(gebaut, fmt):
    sz, tools = gebaut[("s8_morgen", fmt)]
    w, h = cw.FORMATS[fmt]
    outer, inner = tools["PillAussenR"]["Inputs"], tools["PillInnenR"]["Inputs"]
    assert (outer["Height"]["Value"] - inner["Height"]["Value"]) * h == pytest.approx(3.0)
    for cap in ("PillInnenR", "PillInnenL", "PillInnenE"):
        assert tools[cap]["Inputs"]["PaintMode"]["Value"].type_name == "FuID"
    # ends are circles as tall as the pill (radius = full). Fusion measures *both* ellipse
    # sizes in image widths (rendered 06.10.2026: a height in image heights drew a 54×96 cap)
    end = tools["PillAussenL"]["Inputs"]
    assert end["Width"]["Value"] == pytest.approx(end["Height"]["Value"])
    assert end["Width"]["Value"] * w == pytest.approx(outer["Height"]["Value"] * h)
    assert not any("Shadow" in n or t.type_name == "DropShadow" for n, t in tools.items())


@pytest.mark.parametrize("fmt", FORMATE)
def test_s8_wordmark_weights_opacity_and_place_under_the_logo(gebaut, fmt):
    sz, tools = gebaut[("s8_morgen", fmt)]
    a, b = tools["MarkeA"]["Inputs"], tools["MarkeB"]["Inputs"]
    assert (a["Style"]["Value"], b["Style"]["Value"]) == ("SemiBold", "Regular")
    assert (_text_of(tools, "MarkeA"), _text_of(tools, "MarkeB")) == ("nomis", "success")
    fb = _spline(tools, tools["MarkeB"]["Inputs"]["StyledText"]["SourceOp"], "Opacity1")
    assert fb[max(fb)] == pytest.approx(0.55)
    assert a["CharacterSpacing"]["Value"] < 1.0 and b["CharacterSpacing"]["Value"] < 1.0
    # under the 3D logo (square y 700–1050 → 16:9 frame y 280–630, 9:16 frame y 700–1050)
    logo_bottom = 1050 - (420 if fmt == "16x9" else 0)
    _, top, width, _ = sz.layout["marke"]
    assert top > logo_bottom
    centre = (1 - 0) * 0 + sz.layout["marke"][0] + width / 2
    assert centre == pytest.approx(cw.FORMATS[fmt][0] / 2)


@pytest.mark.parametrize("fmt", FORMATE)
def test_abspann_is_opaque_night(gebaut, fmt):
    sz, tools = gebaut[("abspann", fmt)]
    night = tools["Nacht"]["Inputs"]
    assert night["TopLeftAlpha"]["Value"] == 1.0
    assert (night["TopLeftRed"]["Value"], night["TopLeftGreen"]["Value"], night["TopLeftBlue"]["Value"]) \
        == pytest.approx(cw.hex_rgb(SCRIPT["farben"]["nacht"]))
    for i in range(3):
        assert tools[f"Schein{i}"]["Inputs"]["TopLeftAlpha"]["Value"] <= 0.12
