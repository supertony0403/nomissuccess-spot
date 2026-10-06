"""Tests for fusion/comp_writer.py — the Python → Fusion .comp writer.

The unit tests parse the generated text with a small Lua-table parser (below) and
assert on the *structure* (tools, inputs, keyframes, links), not on substrings.
The round-trip test against a running DaVinci Resolve is opt-in:
``RESOLVE_TESTS=1 pytest -m resolve`` (it creates and deletes a scratch timeline).
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fusion import comp_writer as cw  # noqa: E402

# --------------------------------------------------------------------------------------
# Minimal parser for the Lua-table subset Fusion writes (enough to verify our output)
# --------------------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<str>"(?:[^"\\]|\\.)*")
      | (?P<num>-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)
      | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<sym>[{}\[\]=,])
    )""",
    re.VERBOSE,
)


class Typed(dict):
    """A constructor call like ``Input { ... }`` — a dict that remembers its type."""

    def __init__(self, type_name: str, items: dict):
        super().__init__(items)
        self.type_name = type_name


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot tokenize at {pos}: {text[pos:pos + 40]!r}")
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


def _unquote(s: str) -> str:
    body = s[1:-1]
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "r": "\r"}.get(m.group(1), m.group(1)), body)


class _Parser:
    def __init__(self, text: str):
        self.t = _tokenize(text)
        self.i = 0

    def peek(self, k: int = 0):
        return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)

    def take(self, value: str | None = None):
        tok = self.t[self.i]
        if value is not None and tok[1] != value:
            raise ValueError(f"expected {value!r}, got {tok!r} at token {self.i}")
        self.i += 1
        return tok

    def value(self):
        kind, val = self.peek()
        if kind == "str":
            self.take()
            return _unquote(val)
        if kind == "num":
            self.take()
            return float(val) if any(c in val for c in ".eE") else int(val)
        if kind == "name":
            self.take()
            if val in ("true", "false"):
                return val == "true"
            if val == "nil":
                return None
            # constructor: Name { ... }
            return Typed(val, self.table())
        if val == "{":
            return self.table()
        raise ValueError(f"unexpected token {val!r} at {self.i}")

    def table(self) -> dict:
        self.take("{")
        out: dict = {}
        auto = 1
        while self.peek()[1] != "}":
            kind, val = self.peek()
            if val == "[":
                self.take("[")
                key = self.value()
                self.take("]")
                self.take("=")
                out[key] = self.value()
            elif kind == "name" and self.peek(1)[1] == "=":
                self.take()
                self.take("=")
                out[val] = self.value()
            else:
                out[auto] = self.value()
                auto += 1
            if self.peek()[1] == ",":
                self.take(",")
        self.take("}")
        return out


def parse_comp(text: str) -> Typed:
    p = _Parser(text)
    root = p.value()
    assert p.i == len(p.t), "trailing tokens after the composition"
    assert isinstance(root, Typed) and root.type_name == "Composition"
    return root


def tools_of(text: str) -> dict[str, Typed]:
    return parse_comp(text)["Tools"]


def input_value(tool: Typed, name: str):
    inp = tool["Inputs"][name]
    assert inp.type_name == "Input"
    return inp.get("Value")


def link_of(tool: Typed, name: str) -> tuple[str, str]:
    inp = tool["Inputs"][name]
    return inp["SourceOp"], inp["Source"]


def as_list(t) -> list:
    """Lua array table {a, b} → [a, b]."""
    return [t[i] for i in range(1, len(t) + 1)]


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


def test_empty_comp_has_range_frame_format_and_media_out():
    comp = cw.Comp(1920, 1080, frames=120, fps=60)
    root = parse_comp(comp.to_text())
    assert as_list(root["RenderRange"]) == [0, 119]
    assert as_list(root["GlobalRange"]) == [0, 119]
    ff = root["Prefs"]["Comp"]["FrameFormat"]
    assert (ff["Width"], ff["Height"], ff["Rate"]) == (1920, 1080, 60)
    assert root["Tools"]["MediaOut1"].type_name == "MediaOut"


def test_parser_rejects_unbalanced_text():
    """Guard for the guard: the parser must fail on broken braces."""
    good = cw.Comp(1920, 1080, frames=10).to_text()
    with pytest.raises((ValueError, IndexError, AssertionError)):
        parse_comp(good.rstrip().rstrip("}"))


def test_text_tool_writes_font_size_position_colour_and_left_anchor():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("Titel", "Die Stadt schläft.", font="Manrope", style="Bold", size=0.05,
              color=(1.0, 1.0, 1.0, 0.6), center=(0.0625, 0.7), tracking=0.98, justify="left")
    t = tools_of(comp.to_text())["Titel"]
    assert t.type_name == "TextPlus"
    assert input_value(t, "StyledText") == "Die Stadt schläft."
    assert input_value(t, "Font") == "Manrope"
    assert input_value(t, "Style") == "Bold"
    assert input_value(t, "Size") == pytest.approx(0.05)
    assert as_list(input_value(t, "Center")) == pytest.approx([0.0625, 0.7])
    assert input_value(t, "CharacterSpacing") == pytest.approx(0.98)
    assert input_value(t, "HorizontalLeftCenterRight") == -1   # anchor at the left edge
    assert input_value(t, "HorizontalJustificationNew") == 0   # lines flush left
    assert [input_value(t, k) for k in ("Red1", "Green1", "Blue1")] == [1.0, 1.0, 1.0]
    assert input_value(t, "Opacity1") == pytest.approx(0.6)
    # explicit frame size so a comp never depends on the timeline it lands in
    assert (input_value(t, "Width"), input_value(t, "Height")) == (1920, 1080)


@pytest.mark.parametrize("justify,anchor", [("left", -1), ("center", 0), ("right", 1)])
def test_justify_maps_to_fusion_anchor(justify, anchor):
    comp = cw.Comp(1080, 1920, frames=10)
    comp.text("T", "x", justify=justify)
    assert input_value(tools_of(comp.to_text())["T"], "HorizontalLeftCenterRight") == anchor


def test_text_escapes_quotes_backslashes_and_newlines():
    nasty = 'Er sagt "Hallo"\\n\nzweite Zeile'
    comp = cw.Comp(1920, 1080, frames=10)
    comp.text("T", nasty)
    assert input_value(tools_of(comp.to_text())["T"], "StyledText") == nasty


def test_transform_has_motion_blur_settings_and_input_link():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.text("T", "x")
    comp.transform("Xf", "T", center=(0.5, 0.4), size=1.1, angle=3.0, quality=9, shutter=270.0)
    x = tools_of(comp.to_text())["Xf"]
    assert x.type_name == "Transform"
    assert link_of(x, "Input") == ("T", "Output")
    # a transform that never moves gets no motion blur (render time); quality is capped
    assert input_value(x, "MotionBlur") == 0
    assert input_value(x, "Quality") == cw.MB_QUALITY_MAX
    assert input_value(x, "ShutterAngle") == pytest.approx(270.0)
    assert as_list(input_value(x, "Center")) == pytest.approx([0.5, 0.4])
    assert input_value(x, "Size") == pytest.approx(1.1)
    assert input_value(x, "Angle") == pytest.approx(3.0)


def test_merge_background_and_output_wiring():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.background("Leer", color=(0, 0, 0), alpha=0.0)
    comp.text("T", "x")
    comp.merge("M", "Leer", "T")
    comp.output("M")
    tools = tools_of(comp.to_text())
    assert link_of(tools["M"], "Background") == ("Leer", "Output")
    assert link_of(tools["M"], "Foreground") == ("T", "Output")
    assert link_of(tools["MediaOut1"], "Input") == ("M", "Output")
    bg = tools["Leer"]
    assert input_value(bg, "TopLeftAlpha") == 0.0
    assert (input_value(bg, "Width"), input_value(bg, "Height")) == (1920, 1080)


def test_merge_can_place_and_scale_the_foreground():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.background("Leer", alpha=0.0)
    comp.text("T", "x")
    comp.merge("M", "Leer", "T", center=(0.3, 0.7), size=0.25)
    m = tools_of(comp.to_text())["M"]
    assert as_list(input_value(m, "Center")) == pytest.approx([0.3, 0.7])
    assert input_value(m, "Size") == pytest.approx(0.25)


def test_gradient_background_writes_colour_stops():
    comp = cw.Comp(1920, 1080, frames=10)
    stops = [(0.0, "#145fe4"), (0.34, "#7c3aed"), (0.66, "#e44b8d"), (1.0, "#10b981")]
    comp.gradient("Signal", stops, start=(0.0, 0.5), end=(1.0, 0.5))
    g = tools_of(comp.to_text())["Signal"]
    assert input_value(g, "Type").type_name == "FuID"
    assert input_value(g, "Type")[1] == "Gradient"
    colors = input_value(g, "Gradient")["Colors"]
    assert sorted(colors) == [0, 0.34, 0.66, 1]
    assert as_list(colors[0]) == pytest.approx([0x14 / 255, 0x5F / 255, 0xE4 / 255, 1.0])
    assert as_list(input_value(g, "Start")) == [0.0, 0.5]


def test_rect_mask_and_effect_mask_link():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.text("T", "x")
    comp.rect_mask("Fenster", center=(0.3, 0.6), width=0.5, height=0.2, soft_edge=0.004, corner_radius=0.1)
    comp.apply_mask("T", "Fenster")
    tools = tools_of(comp.to_text())
    m = tools["Fenster"]
    assert m.type_name == "RectangleMask"
    assert (input_value(m, "Width"), input_value(m, "Height")) == (0.5, 0.2)
    assert input_value(m, "SoftEdge") == pytest.approx(0.004)
    assert input_value(m, "CornerRadius") == pytest.approx(0.1)
    assert (input_value(m, "MaskWidth"), input_value(m, "MaskHeight")) == (1920, 1080)
    assert link_of(tools["T"], "EffectMask") == ("Fenster", "Mask")


def test_mask_chaining_combines_masks():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.rect_mask("A", center=(0.1, 0.1), width=0.1, height=0.01)
    comp.rect_mask("B", center=(0.1, 0.1), width=0.01, height=0.1, combine_with="A")
    assert link_of(tools_of(comp.to_text())["B"], "EffectMask") == ("A", "Mask")


def test_image_loader_points_at_file(tmp_path):
    png = tmp_path / "glocke.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    comp = cw.Comp(1920, 1080, frames=10)
    comp.image("Glocke", png)
    ld = tools_of(comp.to_text())["Glocke"]
    assert ld.type_name == "Loader"
    clip = ld["Clips"][1]
    assert clip.type_name == "Clip"
    assert clip["Filename"] == str(png)
    assert clip["FormatID"] == "PNGFormat"


def test_image_loader_requires_existing_file(tmp_path):
    comp = cw.Comp(1920, 1080, frames=10)
    with pytest.raises(FileNotFoundError):
        comp.image("Fehlt", tmp_path / "nope.png")


# --------------------------------------------------------------------------------------
# Keyframes and easing
# --------------------------------------------------------------------------------------


def _spline_for(text: str, tool: str, inp: str) -> Typed:
    tools = tools_of(text)
    op, src = link_of(tools[tool], inp)
    assert src == "Value"
    spline = tools[op]
    assert spline.type_name == "BezierSpline"
    return spline


def test_out_cubic_keyframes_have_real_bezier_handles():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x")
    comp.transform("Xf", "T")
    comp.keyframes("Xf", "Size", {0: 1.0, 30: 2.0}, ease="out_cubic")
    kf = _spline_for(comp.to_text(), "Xf", "Size")["KeyFrames"]
    k0, k30 = kf[0], kf[30]
    assert k0[1] == 1.0 and k30[1] == 2.0
    # CSS easeOutCubic = cubic-bezier(0.33, 1, 0.68, 1) mapped into (time, value) space
    assert as_list(k0["RH"]) == pytest.approx([0.33 * 30, 2.0])
    assert as_list(k30["LH"]) == pytest.approx([0.68 * 30, 2.0])
    assert "Flags" not in k0  # not linear


def test_in_out_cubic_handles_are_flat_at_both_ends():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x")
    comp.keyframes("T", "Size", {10: 0.05, 40: 0.15}, ease="in_out_cubic")
    kf = _spline_for(comp.to_text(), "T", "Size")["KeyFrames"]
    assert as_list(kf[10]["RH"]) == pytest.approx([10 + 0.65 * 30, 0.05])
    assert as_list(kf[40]["LH"]) == pytest.approx([10 + 0.35 * 30, 0.15])


def test_linear_keyframes_are_flagged_linear():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x")
    comp.keyframes("T", "Opacity1", {0: 0.0, 9: 1.0}, ease="linear")
    kf = _spline_for(comp.to_text(), "T", "Opacity1")["KeyFrames"]
    assert kf[0]["Flags"]["Linear"] is True
    assert as_list(kf[0]["RH"]) == pytest.approx([3.0, 1 / 3])


def test_middle_keyframe_gets_both_handles_and_per_key_ease():
    comp = cw.Comp(1920, 1080, frames=90)
    comp.text("T", "x")
    comp.keyframes("T", "Size", {0: 0.0, 30: 1.0, 60: 0.0},
                   ease="out_cubic", per_key={30: "in_cubic"})
    kf = _spline_for(comp.to_text(), "T", "Size")["KeyFrames"]
    assert "LH" in kf[30] and "RH" in kf[30]
    # segment 30→60 uses in_cubic = cubic-bezier(0.32, 0, 0.67, 0)
    assert as_list(kf[30]["RH"]) == pytest.approx([30 + 0.32 * 30, 1.0])
    assert as_list(kf[60]["LH"]) == pytest.approx([30 + 0.67 * 30, 1.0])


def test_step_ease_holds_value_until_next_key():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x")
    comp.keyframes("T", "Opacity1", {0: 1.0, 20: 0.2}, ease="step")
    kf = _spline_for(comp.to_text(), "T", "Opacity1")["KeyFrames"]
    assert kf[19][1] == 1.0 and kf[20][1] == 0.2


def test_point_keyframes_use_xypath_with_two_splines():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x")
    comp.keyframes("T", "Center", {0: (0.1, 0.2), 20: (0.3, 0.6)}, ease="out_cubic")
    tools = tools_of(comp.to_text())
    op, src = link_of(tools["T"], "Center")
    assert tools[op].type_name == "XYPath" and src == "Value"
    xs = tools[link_of(tools[op], "X")[0]]["KeyFrames"]
    ys = tools[link_of(tools[op], "Y")[0]]["KeyFrames"]
    assert (xs[0][1], xs[20][1]) == (0.1, 0.3)
    assert (ys[0][1], ys[20][1]) == (0.2, 0.6)


def test_keyframes_replace_a_static_value():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x", size=0.05)
    comp.keyframes("T", "Size", {0: 0.05, 10: 0.06})
    inp = tools_of(comp.to_text())["T"]["Inputs"]["Size"]
    assert "Value" not in inp and "SourceOp" in inp


@pytest.mark.parametrize("bad", ["bounce", "", "OUT_CUBIC"])
def test_unknown_ease_is_rejected(bad):
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x")
    with pytest.raises(ValueError):
        comp.keyframes("T", "Size", {0: 0.0, 10: 1.0}, ease=bad)


def test_keyframes_need_two_keys_and_integer_frames():
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("T", "x")
    with pytest.raises(ValueError):
        comp.keyframes("T", "Size", {0: 1.0})
    with pytest.raises(ValueError):
        comp.keyframes("T", "Size", {0: 1.0, 2.5: 2.0})


def test_ease_curve_is_monotonic_and_hits_targets():
    """Sampling the Bezier we write must start/end exactly and not overshoot for out_cubic."""
    pts = cw.sample_ease("out_cubic", steps=50)
    assert pts[0] == pytest.approx(0.0) and pts[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-9 for a, b in zip(pts, pts[1:]))
    assert pts[10] > 10 / 50  # ease-out runs ahead of linear early on


# --------------------------------------------------------------------------------------
# Follower (character-level animation)
# --------------------------------------------------------------------------------------


def test_follower_spec_creates_follower_modifier_with_keys():
    spec = cw.FollowerSpec(
        delay=2.5,
        order="left_to_right",
        keys={"CharacterOffset": {0: (0.0, -0.03), 18: (0.0, 0.0)}, "Opacity1": {0: 0.0, 12: 1.0}},
        values={"SoftnessY1": 0.0},
        ease="out_expo",
    )
    comp = cw.Comp(1920, 1080, frames=60)
    comp.text("Zeile", "Die Stadt schläft.", follower=spec, motion_blur=True)
    tools = tools_of(comp.to_text())
    zeile = tools["Zeile"]
    fol_name, src = link_of(zeile, "StyledText")
    assert src == "StyledText"
    fol = tools[fol_name]
    assert fol.type_name == "StyledTextFollower"
    assert input_value(fol, "Text").type_name == "StyledText"
    assert input_value(fol, "Text")["Value"] == "Die Stadt schläft."
    assert input_value(fol, "Delay") == pytest.approx(2.5)
    assert input_value(fol, "Order") == cw.FOLLOWER_ORDER["left_to_right"]
    xy = tools[link_of(fol, "CharacterOffset")[0]]
    assert xy.type_name == "XYPath"
    op = tools[link_of(fol, "Opacity1")[0]]
    assert op.type_name == "BezierSpline" and op["KeyFrames"][12][1] == 1.0
    assert input_value(zeile, "MotionBlur") == 1


def test_follower_rejects_unknown_order():
    with pytest.raises(ValueError):
        cw.FollowerSpec(order="sideways")


# --------------------------------------------------------------------------------------
# Validation, naming, expressions, file output
# --------------------------------------------------------------------------------------


def test_duplicate_tool_names_are_rejected():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.text("T", "x")
    with pytest.raises(ValueError):
        comp.text("T", "y")


@pytest.mark.parametrize("bad", ["1abc", "mit leer", "ä", "", "MediaOut1"])
def test_invalid_tool_names_are_rejected(bad):
    comp = cw.Comp(1920, 1080, frames=10)
    with pytest.raises(ValueError):
        comp.text(bad, "x")


def test_link_to_unknown_tool_fails_at_render_time():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.transform("Xf", "Gibtsnicht")
    with pytest.raises(ValueError, match="Gibtsnicht"):
        comp.to_text()


def test_expression_input_is_written():
    comp = cw.Comp(1920, 1080, frames=10)
    comp.text("Uhr", "03:12")
    comp.expression("Uhr", "StyledText", 'Text(string.format("%02d", time))')
    inp = tools_of(comp.to_text())["Uhr"]["Inputs"]["StyledText"]
    assert inp["Expression"] == 'Text(string.format("%02d", time))'
    assert "Value" not in inp


def test_save_writes_utf8_and_returns_path(tmp_path):
    comp = cw.Comp(1080, 1920, frames=10)
    comp.text("T", "Büro")
    out = comp.save(tmp_path / "sub" / "probe.comp")
    assert out.exists()
    assert "Büro" in out.read_text(encoding="utf-8")
    parse_comp(out.read_text(encoding="utf-8"))


def test_numbers_are_finite_and_compact():
    comp = cw.Comp(1920, 1080, frames=10)
    with pytest.raises(ValueError):
        comp.text("T", "x", size=math.nan)
    comp.text("U", "x", size=1 / 3)
    text = comp.to_text()
    assert "T" not in comp.tool_names  # the failed call left nothing behind
    assert not re.search(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])", text, re.IGNORECASE)
    assert input_value(tools_of(text)["U"], "Size") == pytest.approx(1 / 3, abs=1e-9)


# --------------------------------------------------------------------------------------
# Pixel helpers
# --------------------------------------------------------------------------------------


def test_px_converts_top_left_pixels_to_fusion_coordinates():
    comp = cw.Comp(1920, 1080, frames=10)
    assert comp.px(120, 90) == pytest.approx((120 / 1920, 1 - 90 / 1080))
    assert comp.px(960, 540) == pytest.approx((0.5, 0.5))
    tall = cw.Comp(1080, 1920, frames=10)
    assert tall.px(90, 215) == pytest.approx((90 / 1080, 1 - 215 / 1920))


def test_px_size_helpers_and_text_size_for_cap_height():
    comp = cw.Comp(1920, 1080, frames=10)
    assert comp.px_w(192) == pytest.approx(0.1)
    assert comp.px_h(108) == pytest.approx(0.1)
    # calibrated on Resolve 21: JetBrains Mono Size 0.1 → 'H' 84 px at 1920 wide
    size = comp.size_for_cap(84, font="JetBrains Mono")
    assert size == pytest.approx(0.1, rel=0.02)
    assert cw.Comp(1080, 1920, frames=10).size_for_cap(48, font="JetBrains Mono") == pytest.approx(0.1, rel=0.03)


@pytest.fixture(scope="module")
def jb_mono():
    try:
        return cw.FontMetrics.find("JetBrains Mono", "Regular")
    except LookupError as exc:  # pragma: no cover - depends on installed fonts
        pytest.skip(str(exc))


def test_font_metrics_predict_measured_cap_height(jb_mono):
    # Resolve render: JetBrains Mono Size 0.1 → 'H' 84 px at 1920 wide, 48 px at 1080 wide
    assert jb_mono.cap_px(0.1, 1920) == pytest.approx(84, abs=1.5)
    assert jb_mono.cap_px(0.1, 1080) == pytest.approx(48, abs=1.5)


def test_font_metrics_width_of_monospace_text(jb_mono):
    one = jb_mono.width_px("0", 0.1, 1920)
    assert jb_mono.width_px("0000000000", 0.1, 1920) == pytest.approx(10 * one)
    assert one == pytest.approx(0.6 * jb_mono.em_px(0.1, 1920), rel=0.01)  # JB Mono advance = 600/1000 em


def test_font_lookup_refuses_a_substitute():
    with pytest.raises(LookupError):
        cw.FontMetrics.find("Gibt Es Nicht Sans", "Bold")


def test_size_for_cap_falls_back_to_font_file(jb_mono, monkeypatch):
    monkeypatch.delitem(cw.FONT_CAP_PER_SIZE, "JetBrains Mono")
    comp = cw.Comp(1920, 1080, frames=10)
    assert comp.size_for_cap(84, font="JetBrains Mono", style="Regular") == pytest.approx(0.1, rel=0.02)


def test_safe_area_for_vertical_format():
    comp = cw.Comp(1080, 1920, frames=10)
    box = comp.safe_box()
    assert (box.left, box.top, box.right, box.bottom) == (90, 215, 990, 1515)
    wide = cw.Comp(1920, 1080, frames=10)
    assert (wide.safe_box().left, wide.safe_box().top) == (120, 96)


def test_hex_colour_parsing():
    assert cw.hex_rgb("#145fe4") == pytest.approx((0x14 / 255, 0x5F / 255, 0xE4 / 255))
    with pytest.raises(ValueError):
        cw.hex_rgb("145fe4zz")


# --------------------------------------------------------------------------------------
# Round trip through a running Resolve (opt-in)
# --------------------------------------------------------------------------------------


@pytest.mark.resolve
@pytest.mark.skipif(os.environ.get("RESOLVE_TESTS") != "1", reason="set RESOLVE_TESTS=1 to touch Resolve")
def test_round_trip_import_into_resolve(tmp_path):
    from resolve import resolve_api

    try:
        resolve = resolve_api.connect()
        project = resolve_api.current_project(resolve)
    except resolve_api.ResolveNotReachable as exc:
        pytest.skip(str(exc))

    comp = cw.Comp(1920, 1080, frames=120)
    comp.background("Leer", alpha=0.0)
    comp.text("Zeile", "Rundreise", font="JetBrains Mono", style="Regular",
              follower=cw.FollowerSpec(keys={"Opacity1": {0: 0.0, 10: 1.0}}))
    comp.transform("Bewegt", "Zeile", motion_blur=True)
    comp.keyframes("Bewegt", "Center", {0: (0.5, 0.4), 30: (0.5, 0.5)})
    comp.rect_mask("Fenster", center=(0.5, 0.5), width=0.6, height=0.3)
    comp.apply_mask("Bewegt", "Fenster")
    comp.merge("Gesamt", "Leer", "Bewegt")
    comp.output("Gesamt")
    path = comp.save(tmp_path / "rundreise.comp")

    with resolve_api.scratch_timeline(project, "zz pytest comp-writer") as timeline:
        item = timeline.InsertFusionCompositionIntoTimeline()
        assert item is not None
        imported = item.ImportFusionComp(str(path))
        assert imported is not None
        names = {t.Name for t in imported.GetToolList(False).values()}
        assert {"Leer", "Zeile", "Bewegt", "Fenster", "Gesamt", "MediaOut1"} <= names
        regs = {t.Name: t.GetAttrs()["TOOLS_RegID"] for t in imported.GetToolList(False).values()}
        assert regs["Zeile"] == "TextPlus" and regs["Bewegt"] == "Transform"
        assert any(r == "StyledTextFollower" for r in regs.values())


# --------------------------------------------------------------------------------------
# Render diagnosis rule (resolve/diagnose_render.py) — pure helpers
# --------------------------------------------------------------------------------------


def test_media_storage_roots_are_read_from_config(tmp_path):
    from resolve import diagnose_render as dr

    cfg = tmp_path / "config.dat"
    cfg.write_text("Foo = 1\nSite.1.FS.1.Root = /home/tony/Videos\nSite.1.FS.2.Root = /mnt/media\n")
    assert dr.media_storage_roots(cfg) == [Path("/home/tony/Videos"), Path("/mnt/media")]
    assert dr.media_storage_roots(tmp_path / "missing.dat") == []


def test_render_target_must_lie_under_media_storage(tmp_path):
    from resolve import diagnose_render as dr

    roots = [Path("/home/tony/Videos")]
    assert dr.target_allowed(Path("/home/tony/Videos"), roots)
    assert dr.target_allowed(Path("/home/tony/Videos/zz/link-into-repo"), roots)   # string rule
    assert not dr.target_allowed(Path("/home/tony/Videos2/x"), roots)
    assert not dr.target_allowed(Path("/tmp/render"), roots)


# --------------------------------------------------------------------------------------
# HUD and scene 1 against the timeline contract
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def timeline():
    return cw.load_timeline()


def _keys(text: str, tool: str, inp: str) -> dict:
    tools = tools_of(text)
    op, _ = link_of(tools[tool], inp)
    node = tools[op]
    if node.type_name == "XYPath":
        node = tools[link_of(node, "Y")[0]]
    return {f: v[1] for f, v in node["KeyFrames"].items()}


def test_hud_clock_text_matches_the_night(timeline):
    from fusion import hud

    a, b = cw.event_frame(timeline, "clock_roll"), cw.event_frame(timeline, "clock_roll_7")
    assert hud.clock_text_at(a, a, b) == "03:12"
    assert hud.clock_text_at(b, a, b) == "07:00"
    assert hud.clock_text_at(0, a, b) == "03:12" and hud.clock_text_at(b + 500, a, b) == "07:00"
    expr = hud.clock_expression(a, b)
    assert f"(time - {a})" in expr and str(b - a) in expr


@pytest.mark.parametrize("fmt", ["16x9", "9x16"])
def test_hud_comp_spans_the_spot_and_fills_the_rail_by_logo_fold(timeline, fmt):
    from fusion import hud

    text = hud.build(fmt, timeline).to_text()
    root = parse_comp(text)
    assert as_list(root["RenderRange"]) == [0, timeline["dauer_f"] - 1]
    tools = root["Tools"]
    assert "Expression" in tools["Uhr"]["Inputs"]["StyledText"]
    lf = cw.event_frame(timeline, "logo_fold")
    width_keys = _spline_for(text, "FortschrittMaske", "Width")["KeyFrames"]
    assert sorted(width_keys) == [0, lf] and width_keys[0][1] == 0.0
    # 9:16: the ticks sit on the safe-area corners
    if fmt == "9x16":
        lay = hud.layout(fmt)
        assert (lay.ticks.left, lay.ticks.top, lay.ticks.bottom) == (90, 215, 1920 - 405)


@pytest.mark.parametrize("fmt", ["16x9", "9x16"])
def test_scene1_beats_land_on_their_events(timeline, fmt):
    from fusion import hud
    from fusion.szenen import s1_nacht as s1

    z = s1.zeiten(timeline)
    s0, s_end = cw.scene_span(timeline, "s1_nacht")
    assert z.roll == cw.event_frame(timeline, "clock_roll") - s0
    assert z.glocke == cw.event_frame(timeline, "bell_morph") - s0
    text = s1.build(fmt, timeline).to_text()
    assert as_list(parse_comp(text)["RenderRange"]) == [0, s_end - s0 - 1]
    # the seconds-units drum (column 7) lands exactly on clock_roll
    assert max(_keys(text, "Z7Xf", "Center")) == z.roll
    # every LED lights up on its event
    for k, ev in enumerate(z.leds):
        assert ev == cw.event_frame(timeline, f"led_{k + 1}") - s0
        assert any(f == ev - 1 for f in _keys(text, f"SzeneM{_merge_index(text, f'Led{k}An')}", "Blend"))
    # the counter arrives where and when the HUD clock takes over
    assert z.ankunft == hud.counter_flight(timeline)[1] - s0
    assert hud.clock_in_frame(timeline) - s0 <= z.ankunft
    # lights go out after the line is in, the bell after the question
    assert z.zeile2 + 20 <= z.licht_aus and z.frage < z.glocke < z.ende


def _merge_index(text: str, layer: str) -> int:
    tools = tools_of(text)
    for name, t in tools.items():
        if name.startswith("SzeneM") and t.type_name == "Merge" and link_of(t, "Foreground")[0] == layer:
            return int(name[len("SzeneM"):])
    raise AssertionError(f"no scene merge for {layer}")


@pytest.mark.parametrize("fmt", ["16x9", "9x16"])
def test_scene1_type_fits_its_area(fmt):
    """9:16: inside the safe area and the upper half; 16:9: the left half (± 4 px)."""
    from fusion import hud
    from fusion.szenen import s1_nacht as s1

    w, h = cw.FORMATS[fmt]
    comp = cw.Comp(w, h, frames=10)
    lay = s1.layout(fmt)
    right = comp.safe_box().right if fmt == "9x16" else w // 2 + 4
    head = cw.FontMetrics.find(*s1.HEAD)
    num = cw.FontMetrics.find(*s1.NUM)
    size_num = comp.size_for_cap(lay.counter_cap, *s1.NUM)
    cell = dict(num.advances)[ord("0")] / num.upm * num.em_px(size_num, w) * hud.CLOCK_TRACKING
    assert lay.left + 8 * cell <= right
    size = comp.size_for_cap(lay.text_cap, *s1.HEAD)
    for line in ("Die Stadt schläft.", "Ihr Büro ist dunkel.", "Weckt sie heute", "Nacht jemanden?"):
        assert lay.left + head.width_px(line, size, w) <= right, line
    if fmt == "9x16":
        assert comp.safe_box().top <= lay.counter_top
        assert lay.led_top + 3 * lay.led_pitch + lay.led_cap <= h / 2
