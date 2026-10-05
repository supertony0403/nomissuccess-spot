#!/usr/bin/env bash
# Install Manrope (SIL OFL 1.1) as static instances wght 200-800 into
# ~/.local/share/fonts/manrope/ — Fusion's Text+ on Linux picks styles reliably
# from static TTFs, not from the variable font. JetBrains Mono is already
# installed system-wide.
#
# Usage: scripts/fonts.sh            (idempotent; re-downloads and overwrites)
# Note:  a running DaVinci Resolve may only see the new font after a restart.
set -euo pipefail

WURZEL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$WURZEL/.venv/bin/python"
ZIEL="${HOME}/.local/share/fonts/manrope"
TMP="$WURZEL/work/fonts"
URL_VF='https://github.com/google/fonts/raw/main/ofl/manrope/Manrope%5Bwght%5D.ttf'
URL_OFL='https://github.com/google/fonts/raw/main/ofl/manrope/OFL.txt'

[ -x "$PY" ] || { echo "venv fehlt: $PY (uv venv --python 3.12 .venv)" >&2; exit 1; }
"$PY" -c 'import fontTools' 2>/dev/null || { echo "fonttools fehlt in der venv (uv pip install --python .venv/bin/python fonttools)" >&2; exit 1; }
mkdir -p "$ZIEL" "$TMP"

echo "lade Manrope (variabel) + OFL …"
curl -fsSL --retry 2 -o "$TMP/Manrope-VF.ttf" "$URL_VF"
curl -fsSL --retry 2 -o "$TMP/OFL.txt" "$URL_OFL"

"$PY" - "$TMP/Manrope-VF.ttf" "$ZIEL" <<'PY'
"""Pin wght to each static weight and give every instance proper names
(RIBBI convention: Regular/Bold in family "Manrope", the rest as typographic
subfamilies of "Manrope" with legacy family "Manrope <Style>")."""
import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

quelle, ziel = sys.argv[1], Path(sys.argv[2])
STILE = {200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
         600: "SemiBold", 700: "Bold", 800: "ExtraBold"}

achse = next(a for a in TTFont(quelle)["fvar"].axes if a.axisTag == "wght")
if not (achse.minValue <= min(STILE) and max(STILE) <= achse.maxValue):
    sys.exit(f"wght-Achse {achse.minValue}-{achse.maxValue} deckt 200-800 nicht ab")

for gewicht, stil in STILE.items():
    font = instancer.instantiateVariableFont(TTFont(quelle), {"wght": gewicht}, updateFontNames=False)
    version = font["name"].getDebugName(5) or "Version 1.0"
    name = font["name"]
    for nid in (1, 2, 3, 4, 6, 16, 17, 21, 22, 25):
        name.removeNames(nameID=nid)
    ribbi = stil in ("Regular", "Bold")
    eintraege = {
        1: "Manrope" if ribbi else f"Manrope {stil}",
        2: stil if ribbi else "Regular",
        3: f"{version.split(';')[0]};Manrope-{stil};static",
        4: f"Manrope {stil}",
        6: f"Manrope-{stil}",
    }
    if not ribbi:
        eintraege.update({16: "Manrope", 17: stil})
    for nid, text in eintraege.items():
        name.setName(text, nid, 3, 1, 0x409)
        name.setName(text, nid, 1, 0, 0)
    os2 = font["OS/2"]
    os2.usWeightClass = gewicht
    os2.fsSelection &= ~((1 << 0) | (1 << 5) | (1 << 6))
    os2.fsSelection |= (1 << 5) if stil == "Bold" else (1 << 6) if stil == "Regular" else 0
    font["head"].macStyle = (font["head"].macStyle & ~0b11) | (1 if stil == "Bold" else 0)
    pfad = ziel / f"Manrope-{stil}.ttf"
    font.save(pfad)
    print(f"  {pfad.name}  wght {gewicht}  usWeightClass {os2.usWeightClass}")
PY

cp "$TMP/OFL.txt" "$ZIEL/OFL.txt"
fc-cache -f "$ZIEL" >/dev/null
echo "fontconfig:"
fc-list : family style file | grep -i '/manrope/' | sort
anzahl=$(fc-list : file | grep -ci '/manrope/manrope-.*\.ttf' || true)
if [ "$anzahl" -lt 7 ]; then
  echo "FEHLER: nur $anzahl von 7 Manrope-Stilen in fontconfig" >&2
  exit 1
fi
echo "ok: $anzahl Manrope-Stile in $ZIEL (Lizenz: OFL.txt). Resolve sieht sie ggf. erst nach einem Neustart."
