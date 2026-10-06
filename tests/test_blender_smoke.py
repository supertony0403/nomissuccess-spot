"""Smoke tests for the Blender shots (Task 3) - render tiny proxies headless.

* every shot renders frames 1 / mid / last that are not empty,
* the frontal orthographic silhouette of the 3D "N" matches logo.png
  (IoU >= 0.85),
* at ``logo_fold`` the N sits inside the centre 1080 square (both crops keep
  it) at the agreed position (centre y ~875 px, height ~300 px of 1920),
* the band is in frame at ``ribbon_enter`` + 0.5 s, and mid-scene most of it
  lies inside the 9:16 column,
* frame bookkeeping of ``nomiss_timeline`` (handles, 1-based frames).

The Blender part needs ``blender`` on PATH and takes about a minute.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "blender" / "lib"))

import nomiss_timeline as TL  # noqa: E402

RES = 480
SCALE = RES / 1920.0
CENTRE = (420 * SCALE, 1500 * SCALE)  # centre square / 9:16 column in px
BLENDER = shutil.which("blender")
needs_blender = pytest.mark.skipif(BLENDER is None, reason="blender nicht installiert")


def run_shot(shot: str, *args: str) -> str:
    cmd = [BLENDER, "-b", "--factory-startup", "-P", str(REPO / "blender" / "shots" / f"{shot}.py"), "--", *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = res.stdout + res.stderr
    assert res.returncode == 0 and "Traceback" not in out, out[-3000:]
    return out


def alpha(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float64)[:, :, 3] / 255.0


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# --------------------------------------------------------------------------
# timeline bookkeeping (no Blender needed)
# --------------------------------------------------------------------------

def test_shot_frames_include_half_second_handles():
    tl = TL.load(TL.FIXTURE)
    for shot_id in ("s1_nacht", "s8_morgen"):
        sc = TL.szene(shot_id, tl)
        sh = TL.shot(shot_id, tl)
        assert sh.handle_f == 30
        assert sh.frames == sc["ende_f"] - sc["start_f"] + 60
        assert sh.start_s == pytest.approx(sc["start_s"] - 0.5, abs=1e-9)
        first, last = TL.scene_frames(shot_id, tl)
        assert (first, last) == (31, sh.frames - 30)


def test_event_frame_is_one_based_relative_to_shot_start():
    tl = TL.load(TL.FIXTURE)
    ev = TL.event("ribbon_enter", tl)
    assert TL.event_frame("ribbon_enter", "s1_nacht", tl) == ev["f"] + 30 + 1
    fold = TL.event("logo_fold", tl)
    s8 = TL.szene("s8_morgen", tl)
    assert TL.event_frame("logo_fold", "s8_morgen", tl) == fold["f"] - (s8["start_f"] - 30) + 1


def test_real_timeline_is_used_when_present(monkeypatch):
    monkeypatch.delenv("NOMISS_TIMELINE", raising=False)
    expected = REPO / "timeline.json" if (REPO / "timeline.json").exists() else TL.FIXTURE
    assert TL.resolve_path() == expected


# --------------------------------------------------------------------------
# Blender renders
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def renders(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("blender")
    out: dict = {"root": root}
    for shot in ("s1_nacht", "s8_morgen"):
        d = root / shot
        run_shot(shot, "--res", str(RES), "--samples", "8", "--frames", "1,mid,last", "--out", str(d))
        out[shot] = sorted(d.glob("*.png"))
    # s8: frontal logo mask + band matte at logo_fold
    run_shot("s8_morgen", "--res", str(RES), "--logo-mask", str(root / "logo_mask.png"))
    run_shot("s8_morgen", "--res", str(RES), "--samples", "4", "--matte", "LogoN", "--frames", "ev:logo_fold", "--out", str(root / "s8_fold"))
    # s1: band matte at ribbon_enter + 0.5 s and mid-scene
    run_shot(
        "s1_nacht", "--res", str(RES), "--samples", "4", "--matte", "Band",
        "--frames", "ev:ribbon_enter+30,mid", "--out", str(root / "s1_band"),
    )
    return out


@needs_blender
@pytest.mark.parametrize("shot", ["s1_nacht", "s8_morgen"])
def test_first_mid_last_frames_not_empty(renders, shot):
    files = renders[shot]
    assert len(files) == 3, files
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB")).astype(np.float64) / 255.0
        assert img.shape == (RES, RES, 3)
        assert img.mean() > 0.02, f"{f.name} ist fast schwarz"
        assert img.std() > 0.01, f"{f.name} ist einfarbig"


@needs_blender
def test_logo_silhouette_matches_logo_png(renders):
    """IoU of the frontal orthographic alpha vs. logo.png alpha >= 0.85."""
    from nomiss_logo import bilinear, load_logo

    mask = alpha(renders["root"] / "logo_mask.png") > 0.5
    ref_alpha, _ = load_logo(REPO / "blender" / "assets" / "logo.png")
    # the mask camera spans 340 logo px across RES px, centred on the
    # band's logo centre (168, 166.5) in logo pixel-centre coordinates
    step = 340.0 / RES
    ii = np.arange(RES) + 0.5 - RES / 2
    px = 168.0 + ii[None, :] * step * np.ones((RES, 1))
    py = 166.5 + ii[:, None] * step * np.ones((1, RES))
    ref = bilinear(ref_alpha, px, py) > 0.5
    iou = (mask & ref).sum() / (mask | ref).sum()
    print(f"IoU Logo = {iou:.4f}")
    (renders["root"] / "iou.json").write_text(json.dumps({"iou": float(iou)}))
    assert iou >= 0.85


@needs_blender
def test_logo_lands_in_centre_square_at_logo_fold(renders):
    files = sorted((renders["root"] / "s8_fold").glob("*.png"))
    assert len(files) == 1
    m = alpha(files[0]) > 0.5
    x0, y0, x1, y1 = bbox(m)
    lo, hi = CENTRE
    assert lo <= x0 and x1 <= hi and lo <= y0 and y1 <= hi, (x0, y0, x1, y1)
    # agreed placement: centre x ~960, centre y ~875, height ~300 px (at 1920)
    cx, cy, h = (x0 + x1) / 2 / SCALE, (y0 + y1) / 2 / SCALE, (y1 - y0) / SCALE
    assert abs(cx - 960) < 25, cx
    assert abs(cy - 875) < 30, cy
    assert 270 < h < 330, h


@needs_blender
def test_band_visible_after_ribbon_enter_and_inside_column_mid_scene(renders):
    files = sorted((renders["root"] / "s1_band").glob("*.png"))
    assert len(files) == 2
    enter, mid = (alpha(f) > 0.5 for f in files)
    assert enter.sum() > 20, "Band bei ribbon_enter+0,5 s nicht sichtbar"
    lo, hi = (int(v) for v in CENTRE)
    inside = mid[:, lo:hi].sum() / max(mid.sum(), 1)
    assert mid.sum() > 200, "Band in der Szenenmitte zu klein"
    assert inside > 0.5, f"nur {inside:.0%} des Bandes in der 9:16-Spalte"
