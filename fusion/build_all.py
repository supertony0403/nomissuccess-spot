"""Build every Fusion comp of the spot for both formats.

Writes ``work/fusion/<format>/<szene>.comp`` for the HUD, scenes s1–s8 and the end card
(``abspann``), all timed from ``timeline.json``. Usage::

    .venv/bin/python -m fusion.build_all [--format 16x9|9x16|beide] [--out DIR]
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import comp_writer as cw  # noqa: E402

SZENEN = ["s1_nacht", "s2_website", "s3_netz", "s4_ernstfall", "s5_betrieb", "s6_beweis", "s7_team",
          "s8_morgen", "abspann"]
FORMATE = ["16x9", "9x16"]


def builders() -> dict[str, Callable[[str, dict], cw.Comp]]:
    """Name → ``build(fmt, timeline)`` for the HUD and every scene module."""
    out: dict[str, Callable[[str, dict], cw.Comp]] = {"hud": importlib.import_module("fusion.hud").build}
    for name in SZENEN:
        out[name] = importlib.import_module(f"fusion.szenen.{name}").build
    return out


def build_all(out_dir: Path, formats: list[str], tl: dict | None = None) -> list[Path]:
    tl = tl or cw.load_timeline()
    written = []
    for fmt in formats:
        for name, build in builders().items():
            written.append(build(fmt, tl).save(out_dir / fmt / f"{name}.comp"))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=FORMATE + ["beide"], default="beide")
    ap.add_argument("--out", type=Path, default=cw.REPO / "work" / "fusion")
    args = ap.parse_args(argv)
    formats = FORMATE if args.format == "beide" else [args.format]
    for path in build_all(args.out, formats):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
