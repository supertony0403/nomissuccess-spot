"""Print textures for the props of s5-s7 (run with the project venv).

    .venv/bin/python blender/lib_szenen/texturen_s5_s7.py

Writes deterministic PNGs (and the invoice layout JSON) to
``blender/assets/s5_s7/``:

* ``rechnung.png`` + ``rechnung_layout.json`` - invoice face; the amount
  window is left blank, the shader rolls the digits into it,
* ``ziffern.png`` - digit strip 0..9 (white on black, bottom = 0),
* ``prospekt_aussen.png`` / ``prospekt_innen.png`` - glossy tri-fold
  brochure (no readable claims: headline bars only),
* ``tickets.png`` - atlas of 30 ticket slips (6 x 5) with ids like #48213,
* ``doku_runbook.png`` / ``doku_zugaenge.png`` / ``doku_konfiguration.png``.

No real numbers, no customers, no vendor names.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "blender" / "assets" / "s5_s7"
MANROPE = Path.home() / ".local" / "share" / "fonts" / "manrope"
MONO = Path("/usr/share/fonts/TTF")

INK = (16, 18, 35)
GREY = (196, 199, 207)
GREY_D = (150, 154, 166)
PAPER = (251, 250, 247)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for base in (MANROPE, MONO):
        p = base / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    raise FileNotFoundError(name)


def bars(d: ImageDraw.ImageDraw, x: int, y: int, widths: list[int], h: int = 14, gap: int = 26, col=GREY) -> int:
    for w in widths:
        d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=col)
        y += gap
    return y


def rechnung() -> None:
    W, H = 1240, 1754
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    # sender block and date block (bars only)
    bars(d, 110, 120, [300, 220, 260], 13, 26, GREY_D)
    bars(d, 860, 120, [270, 200], 13, 26, GREY)
    d.text((110, 300), "Rechnung", font=font("Manrope-SemiBold.ttf", 76), fill=INK)
    bars(d, 112, 412, [380], 14, 26, GREY)
    # table
    y = 560
    d.line([110, y, W - 110, y], fill=GREY_D, width=3)
    small = font("Manrope-Medium.ttf", 30)
    d.text((110, y + 22), "Pos.", font=small, fill=GREY_D)
    d.text((240, y + 22), "Leistung", font=small, fill=GREY_D)
    d.text((W - 260, y + 22), "Betrag", font=small, fill=GREY_D)
    y += 86
    d.line([110, y, W - 110, y], fill=GREY, width=2)
    rows = [("1", "Lizenz", [420, 260]), ("2", None, [360, 300]), ("3", None, [440, 180]), ("4", None, [300, 240])]
    reg = font("Manrope-Medium.ttf", 34)
    for nr, label, ws in rows:
        d.text((118, y + 30), nr, font=reg, fill=INK)
        if label:
            d.text((240, y + 28), label, font=reg, fill=INK)
            bars(d, 240, y + 86, ws[1:], 12, 24, GREY)
        else:
            bars(d, 240, y + 40, ws, 14, 30, GREY)
        d.rounded_rectangle([W - 300, y + 40, W - 120, y + 54], radius=7, fill=GREY)
        y += 150
        d.line([110, y, W - 110, y], fill=(232, 233, 237), width=2)
    # total row: label, euro sign, blank roll window
    u0, u1, v0, v1 = 0.62, 0.95, 0.170, 0.222
    x0, x1 = int(u0 * W), int(u1 * W)
    y0, y1 = int((1 - v1) * H), int((1 - v0) * H)
    d.line([110, y0 - 56, W - 110, y0 - 56], fill=INK, width=4)
    d.text((110, y0 + 4), "Gesamt", font=font("Manrope-SemiBold.ttf", 54), fill=INK)
    d.text((x0 - 64, y0 + 2), "€", font=font("Manrope-SemiBold.ttf", 64), fill=INK)
    # 7 digit columns as  d d . d d d , d d
    pattern = "dd.ddd,dd"
    sep_w, dig_w = 0.35, 1.0
    units = sum(dig_w if c == "d" else sep_w for c in pattern)
    unit = (x1 - x0) / units
    cols, x = [], float(x0)
    big = font("Manrope-SemiBold.ttf", 64)
    for c in pattern:
        w = (dig_w if c == "d" else sep_w) * unit
        if c == "d":
            cols.append([x / W, (x + w) / W])
        else:
            d.text((x + w * 0.12, y0 + 2), c, font=big, fill=INK)
        x += w
    # faint roll windows (paper slightly recessed)
    for a, b in cols:
        d.rounded_rectangle([int(a * W) + 3, y0, int(b * W) - 3, y1], radius=8, fill=(240, 239, 235))
    bars(d, 110, H - 220, [520, 430], 12, 26, GREY)
    im.save(OUT / "rechnung.png", optimize=True)
    lay = {"fenster_uv": [u0, v0, u1, v1], "spalten": [[round(a, 5), round(b, 5)] for a, b in cols]}
    (OUT / "rechnung_layout.json").write_text(json.dumps(lay, indent=1) + "\n", encoding="utf-8")


def ziffern() -> None:
    cw, ch = 160, 200
    im = Image.new("L", (cw, ch * 10), 0)
    d = ImageDraw.Draw(im)
    f = font("Manrope-SemiBold.ttf", 170)
    for k in range(10):
        top = (9 - k) * ch  # bottom cell = 0 (Blender v grows upwards)
        bb = d.textbbox((0, 0), str(k), font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((cw - w) / 2 - bb[0], top + (ch - h) / 2 - bb[1]), str(k), font=f, fill=255)
    im.save(OUT / "ziffern.png", optimize=True)


def prospekt() -> None:
    W, H = 1800, 1272
    pw = W // 3
    rnd = random.Random(7)
    for side in ("aussen", "innen"):
        im = Image.new("RGB", (W, H), (248, 247, 244))
        d = ImageDraw.Draw(im)
        for p in range(3):
            x = p * pw
            hero = (side == "aussen" and p == 2) or (side == "innen" and p == 1)
            if hero:
                # glossy cover: warm saturated photo-like gradient (no person)
                g = Image.new("RGB", (pw, int(H * 0.62)))
                gd = ImageDraw.Draw(g)
                for yy in range(g.height):
                    t = yy / g.height
                    col = (int(255 - 40 * t), int(150 - 70 * t), int(70 + 120 * t))
                    gd.line([0, yy, pw, yy], fill=col)
                for _ in range(6):
                    cx, cy, r = rnd.randint(0, pw), rnd.randint(0, g.height), rnd.randint(80, 220)
                    gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 214, 160))
                g = g.filter(ImageFilter.GaussianBlur(60))
                im.paste(g, (x, 0))
                bars(d, x + 60, int(H * 0.68), [420, 360], 34, 64, (40, 44, 60))
                bars(d, x + 60, int(H * 0.68) + 150, [380, 400, 300], 12, 26, GREY)
                d.rounded_rectangle([x + 60, H - 150, x + 300, H - 92], radius=29, fill=(240, 120, 60))
            else:
                bars(d, x + 60, 110, [380, 300], 28, 54, (40, 44, 60))
                yy = 260
                for _ in range(3):
                    yy = bars(d, x + 60, yy, [rnd.randint(330, 470) for _ in range(5)], 11, 24, GREY)
                    yy += 40
                d.rounded_rectangle([x + 60, yy, x + pw - 60, yy + 300], radius=18, fill=(232, 168, 120))
                bars(d, x + 60, yy + 340, [rnd.randint(300, 460) for _ in range(4)], 11, 24, GREY)
            if p:
                d.line([x, 0, x, H], fill=(225, 224, 220), width=2)
        im.save(OUT / f"prospekt_{side}.png", optimize=True)


def tickets() -> None:
    cols, rows, cw, ch = 6, 5, 400, 600
    im = Image.new("RGB", (cols * cw, rows * ch), (240, 238, 232))
    d = ImageDraw.Draw(im)
    rnd = random.Random(48213)
    ids = [48213]
    while len(ids) < cols * rows:
        n = rnd.randint(47100, 49890)
        if n not in ids:
            ids.append(n)
    mono = font("JetBrainsMono-Bold.ttf", 66)
    tiny = font("JetBrainsMono-Medium.ttf", 26)
    dots = [(228, 75, 141), (238, 160, 60), (150, 154, 166)]
    for i, n in enumerate(ids):
        x, y = (i % cols) * cw, (i // cols) * ch
        d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=(245, 243, 238))
        d.rectangle([x, y, x + cw - 1, y + 92], fill=(226, 228, 234))
        d.text((x + 30, y + 26), "TICKET", font=tiny, fill=GREY_D)
        d.ellipse([x + cw - 64, y + 30, x + cw - 34, y + 60], fill=dots[i % 3])
        d.text((x + 30, y + 130), f"#{n}", font=mono, fill=INK)
        bars(d, x + 32, y + 250, [rnd.randint(200, 330) for _ in range(5)], 12, 34, GREY)
        # perforation line at the bottom
        for k in range(0, cw - 40, 22):
            d.line([x + 20 + k, y + ch - 70, x + 30 + k, y + ch - 70], fill=GREY, width=3)
        bars(d, x + 32, y + ch - 46, [140], 10, 20, GREY_D)
    im.save(OUT / "tickets.png", optimize=True)


def doku() -> None:
    W, H = 1240, 1754
    mono = font("JetBrainsMono-Medium.ttf", 34)
    code = font("JetBrainsMono-Regular.ttf", 26)
    for key, label, seed in (("runbook", "RUNBOOK", 1), ("zugaenge", "ZUGÄNGE", 2), ("konfiguration", "KONFIGURATION", 3)):
        rnd = random.Random(seed)
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 18, H], fill=(228, 75, 141))
        d.text((110, 110), label, font=mono, fill=(228, 75, 141))
        bars(d, 110, 190, [620, 480], 30, 58, (40, 44, 60))
        y = 360
        for _ in range(3):
            y = bars(d, 110, y, [rnd.randint(700, 1000) for _ in range(5)], 12, 30, GREY)
            y += 40
            if rnd.random() < 0.8:
                d.rounded_rectangle([110, y, W - 110, y + 230], radius=16, fill=(22, 24, 38))
                yy = y + 40
                for _ in range(5):
                    w = rnd.randint(260, 820)
                    d.rounded_rectangle([150, yy, 150 + w, yy + 12], radius=6, fill=(110, 200, 170) if rnd.random() < 0.3 else (120, 126, 150))
                    yy += 34
                y += 280
        d.text((110, H - 120), f"{key}.md", font=code, fill=GREY_D)
        im.save(OUT / f"doku_{key}.png", optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rechnung()
    ziffern()
    prospekt()
    tickets()
    doku()
    for p in sorted(OUT.glob("*.png")):
        print(p.relative_to(REPO), Image.open(p).size)


if __name__ == "__main__":
    main()
