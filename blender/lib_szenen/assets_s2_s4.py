"""Image assets for the Blender shots s2_website / s3_netz / s4_ernstfall.

Run once with the project venv (needs numpy, scipy, Pillow)::

    .venv/bin/python blender/lib_szenen/assets_s2_s4.py

Writes into ``blender/assets/s2_s4/`` (committed, so the shot scripts never
need scipy inside Blender):

* ``web_komposit.png`` - the real start page (``webseite-eigen.png``), cropped
  to 1120x700 so the browser-like card at the bottom edge is gone,
* the page decomposed into layers that composite back to the original
  (``grund`` + ``bild`` + ``typo`` == komposit):

  - ``web_grund.png``  opaque base colour (vertical gradient of the hero),
  - ``web_bild.png``   RGBA colour fields of the hero (+ the logo mark),
  - ``web_typo.png``   RGBA real glyphs and buttons,
  - ``web_balken.png`` RGBA bright bars, one per text line / button,
  - ``web_raster.png`` RGBA layout grid (12 columns, baseline rows),
  - ``web_code.png``   RGBA the real Astro source of exactly this hero,

s3 cuts ``web_komposit.png`` itself into textured tiles (UV per tile).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "blender" / "assets" / "s2_s4"
WEBSITE = Path.home() / "Documents" / "Programmierung" / "nomissuccess-website"
SCREENSHOT = WEBSITE / "src" / "assets" / "webseite-eigen.png"
INDEX_ASTRO = WEBSITE / "src" / "pages" / "index.astro"
FONT_MONO = "/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf"
FONT_MONO_BOLD = "/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf"

W, H = 1120, 700          # page crop (aspect 1.6)
# real source lines of index.astro shown on the code layer (1-based, inclusive)
CODE_RANGES = [(131, 152), (396, 402)]

BASE_STOPS = [(0.0, (11, 12, 20)), (0.55, (16, 18, 30)), (1.0, (11, 12, 20))]


def _save(arr: np.ndarray, name: str) -> Path:
    arr = np.clip(arr, 0.0, 1.0)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    p = OUT / name
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode).save(p, optimize=True)
    return p


def load_page() -> np.ndarray:
    img = Image.open(SCREENSHOT).convert("RGB")
    if img.size[0] != W:
        raise ValueError(f"unerwartete Breite {img.size}")
    return np.asarray(img.crop((0, 0, W, H)), dtype=np.float64) / 255.0


def base_gradient() -> np.ndarray:
    ys = np.linspace(0.0, 1.0, H)
    pos = [p for p, _ in BASE_STOPS]
    cols = np.array([c for _, c in BASE_STOPS], dtype=np.float64) / 255.0
    col = np.stack([np.interp(ys, pos, cols[:, k]) for k in range(3)], axis=-1)
    return np.repeat(col[:, None, :], W, axis=1)


def decompose(page: np.ndarray) -> dict[str, np.ndarray]:
    """Split the page into background fields and foreground (glyphs, buttons)."""
    lum = page.max(axis=-1)
    smooth = ndimage.gaussian_filter(lum, 9.0)
    fg = (lum - smooth) > 0.045
    # buttons and the logo are filled shapes; catch them by saturation/brightness
    fg |= lum > 0.55
    fg = ndimage.binary_dilation(fg, iterations=3)
    keep = (~fg).astype(np.float64)
    num = np.stack([ndimage.gaussian_filter(page[..., k] * keep, 14.0) for k in range(3)], axis=-1)
    den = ndimage.gaussian_filter(keep, 14.0)[..., None]
    bg = np.where(den > 1e-3, num / np.maximum(den, 1e-6), page)
    bg = np.where(fg[..., None], bg, page)  # untouched where there is no text

    # foreground over bg: page = bg (1-a) + c a  with a = max over channels
    diff = page - bg
    a = np.clip((diff / np.maximum(1.0 - bg, 1e-3)).max(axis=-1), 0.0, 1.0)
    a = np.where(fg, a, 0.0)
    a = np.where(a < 0.02, 0.0, a)
    c = np.clip(bg + diff / np.maximum(a, 1e-3)[..., None], 0.0, 1.0)
    typo = np.concatenate([np.where(a[..., None] > 0, c, 1.0), a[..., None]], axis=-1)

    base = base_gradient()
    d = np.abs(bg - base).max(axis=-1)
    a1 = np.clip(d / 0.22, 0.0, 1.0)
    c1 = np.clip(base + (bg - base) / np.maximum(a1, 1e-3)[..., None], 0.0, 1.0)
    bild = np.concatenate([np.where(a1[..., None] > 0, c1, base), a1[..., None]], axis=-1)
    return {"grund": base, "bild": bild, "typo": typo, "mask": a, "bg": bg}


def line_boxes(alpha: np.ndarray, gap_x: int = 26, min_h: int = 6) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of text lines / buttons (x0, y0, x1, y1)."""
    m = alpha > 0.25
    m = ndimage.binary_closing(m, structure=np.ones((3, gap_x)))
    lab, n = ndimage.label(m)
    boxes = []
    for sl in ndimage.find_objects(lab):
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        if y1 - y0 >= min_h and x1 - x0 >= 10:
            boxes.append((x0, y0, x1, y1))
    return boxes


def bars(boxes) -> np.ndarray:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in boxes:
        h = y1 - y0
        cy = 0.5 * (y0 + y1)
        bh = max(6.0, 0.58 * h)  # a bar is slimmer than its text line
        r = int(min(bh / 2, 9))
        d.rounded_rectangle((x0, cy - bh / 2, x1, cy + bh / 2), radius=r, fill=(236, 240, 255, 235))
    return np.asarray(im, dtype=np.float64) / 255.0


def raster() -> np.ndarray:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    margin, gutter, cols = 64, 20, 12
    colw = (W - 2 * margin - (cols - 1) * gutter) / cols
    line = (190, 214, 255, 150)
    faint = (190, 214, 255, 26)
    for i in range(cols):
        x0 = margin + i * (colw + gutter)
        d.rectangle((x0, 0, x0 + colw, H), fill=faint)
        d.line((x0, 0, x0, H), fill=line, width=2)
        d.line((x0 + colw, 0, x0 + colw, H), fill=line, width=2)
    for y in range(0, H, 24):
        d.line((0, y, W, y), fill=(190, 214, 255, 46), width=1)
    for y in (68, 190, 270, 440, 620):  # section rhythm of the hero
        d.line((0, y, W, y), fill=(190, 214, 255, 170), width=2)
    return np.asarray(im, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# code layer
# ---------------------------------------------------------------------------

KEYWORDS = ("class", "aria-hidden", "href", "animation", "width", "height", "left", "right", "top",
            "background", "background-color", "min-height", "display", "align-items", "text-align",
            "position", "overflow", "margin-top", "padding-top", "z-index", "transform")


def code_lines() -> list[str]:
    src = INDEX_ASTRO.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for a, b in CODE_RANGES:
        if out:
            out.append("")
        out.extend(src[a - 1 : b])
    return [ln.rstrip().replace("\t", "  ") for ln in out]


def code_image(lines: list[str]) -> np.ndarray:
    """Light-on-dark monospace panel, subtle PRISMA syntax colours."""
    font = ImageFont.truetype(FONT_MONO, 18)
    bold = ImageFont.truetype(FONT_MONO_BOLD, 18) if Path(FONT_MONO_BOLD).exists() else font
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((18, 18, W - 18, H - 18), radius=18, fill=(11, 12, 20, 222))
    x0, y0, lh = 96, 34, 20.6
    num_col = (110, 118, 150, 255)
    base_col = (226, 231, 245, 255)
    tag_col = (122, 168, 255, 255)     # blau, aufgehellt
    attr_col = (190, 160, 255, 255)    # violett, aufgehellt
    str_col = (110, 222, 182, 255)     # mint, aufgehellt
    com_col = (128, 134, 160, 255)
    pink = (240, 140, 186, 255)
    numbers = []
    for a, b in CODE_RANGES:
        numbers.extend(range(a, b + 1))
        numbers.append(None)
    y = y0
    for ln, no in zip(lines, numbers):
        if no is not None:
            d.text((x0 - 62, y), f"{no:>3}", font=font, fill=num_col)
        _draw_code_line(d, (x0, y), ln, font, bold, base_col, tag_col, attr_col, str_col, com_col, pink)
        y += lh
    return np.asarray(im, dtype=np.float64) / 255.0


def _draw_code_line(d, pos, text, font, bold, base, tag, attr, string, comment, pink) -> None:
    x, y = pos
    stripped = text.lstrip()
    if stripped.startswith(("<!--", "/*", "//")) or stripped.startswith("*"):
        d.text((x, y), text, font=font, fill=comment)
        return
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = text.find(ch, i + 1)
            j = n - 1 if j < 0 else j
            tok, col, fnt = text[i : j + 1], string, font
            i = j + 1
        elif ch == "<":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "/-!"):
                j += 1
            tok, col, fnt = text[i:j], tag, bold
            i = j
        elif ch.isalpha() or ch in "-.":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "-_."):
                j += 1
            tok = text[i:j]
            col, fnt = (attr, font) if tok.lstrip(".") in KEYWORDS else (base, font)
            if tok.startswith("."):
                col = pink
            i = j
        else:
            tok, col, fnt = ch, base, font
            i += 1
        d.text((x, y), tok, font=fnt, fill=col)
        x += d.textlength(tok, font=fnt)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page = load_page()
    parts = decompose(page)
    boxes = line_boxes(parts["mask"])
    written = [
        _save(page, "web_komposit.png"),
        _save(parts["grund"], "web_grund.png"),
        _save(parts["bild"], "web_bild.png"),
        _save(parts["typo"], "web_typo.png"),
        _save(bars(boxes), "web_balken.png"),
        _save(raster(), "web_raster.png"),
        _save(code_image(code_lines()), "web_code.png"),
    ]
    # composite check: grund + bild + typo must give back the page
    g, b, t = parts["grund"], parts["bild"], parts["typo"]
    comp = g * (1 - b[..., 3:]) + b[..., :3] * b[..., 3:]
    comp = comp * (1 - t[..., 3:]) + t[..., :3] * t[..., 3:]
    err = np.abs(comp - page).mean()
    print(f"Zeilenboxen: {len(boxes)}; Rekonstruktionsfehler {err:.4f}")
    for p in written:
        print(" ", p.relative_to(REPO))


if __name__ == "__main__":
    main()
