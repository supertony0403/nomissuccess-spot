"""Tests for the Blender shots s2_website, s3_netz and s4_ernstfall.

* pure motion helpers (no Blender): overshoot <= 3 %, Hermite tracks land
  exactly with zero velocity, the s4 rewind returns to story frame 1,
* per shot a 480x480 proxy render at three event frames:
  - the beauty frames are not empty,
  - the alpha matte of the event's key objects lies inside the centre
    1080x1080 square of the 1920 frame (Review Focus 4: 9:16 / 16:9 crops),
* the Bullet rigid-body bake of the s2 tower is deterministic (two separate
  Blender runs give the same end poses) and the tower really falls.

The Blender tests are skipped when ``blender`` is not on PATH.  They take a
few minutes (the GPU is shared with other renders).
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
sys.path.insert(0, str(REPO / "blender" / "lib_szenen"))
sys.path.insert(0, str(REPO / "blender" / "lib"))

import bewegung  # noqa: E402
import nomiss_timeline as TL  # noqa: E402

BLENDER = shutil.which("blender")
PROXY = 480
LO = 420 * PROXY / 1920   # 105
HI = 1500 * PROXY / 1920  # 375
SHOTS = {
    "s2_website": ("window_land", "layers_explode", "key_turn"),
    "s3_netz": ("firewall_wall", "packets_bounce", "mtls_lock"),
    "s4_ernstfall": ("red_tendrils", "immutable_lock", "shockwave"),
}
needs_blender = pytest.mark.skipif(BLENDER is None, reason="blender nicht im PATH")


# ---------------------------------------------------------------------------
# pure motion helpers
# ---------------------------------------------------------------------------

def test_overshoot_bounded_and_settles():
    ts = np.linspace(0.0, 1.0, 2001)
    v = np.array([bewegung.overshoot(t, 0.03) for t in ts])
    assert v[0] == 0.0 and v[-1] == 1.0
    assert v.max() <= 1.03 + 1e-9
    assert v.max() > 1.0  # it does snap past the target once
    assert np.all(np.diff(v[: int(0.7 * 2000)]) >= -1e-12)  # monotonic approach


def test_bahn_hits_keys_and_stops_softly():
    b = bewegung.Bahn([(1, (0.0, 0.0, 0.0)), (50, (1.0, 2.0, 0.0), True), (100, (3.0, 2.0, 1.0))])
    assert b(1) == (0.0, 0.0, 0.0)
    assert b(50) == (1.0, 2.0, 0.0)
    assert b(100) == (3.0, 2.0, 1.0)
    # zero velocity at the stop key
    d = np.subtract(b(50.01), b(49.99))
    assert np.linalg.norm(d) < 1e-3
    # held outside the key range
    assert b(-10) == b(1) and b(500) == b(100)


def test_rewind_time_returns_to_start():
    f_rw, dur = 586, 72
    assert bewegung.rewind_time(100, f_rw, dur) == 100
    assert bewegung.rewind_time(f_rw, f_rw, dur) == f_rw
    assert bewegung.rewind_time(f_rw + dur, f_rw, dur) == 1.0
    assert bewegung.rewind_time(f_rw + dur + 200, f_rw, dur) == 1.0
    s = [bewegung.rewind_time(f, f_rw, dur) for f in range(f_rw, f_rw + dur + 1)]
    assert all(a >= b for a, b in zip(s, s[1:]))  # strictly backwards
    assert max(abs(a - b) for a, b in zip(s, s[1:])) > 8.0  # fast in the middle


# ---------------------------------------------------------------------------
# Blender runs
# ---------------------------------------------------------------------------

def run_shot(shot: str, out: Path, *extra: str) -> str:
    cmd = [BLENDER, "-b", "--factory-startup", "-P", str(REPO / "blender" / "shots" / f"{shot}.py"), "--",
           "--proxy", str(PROXY), "--out", str(out), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"{shot}: Blender-Fehler\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    assert "Traceback" not in r.stdout + r.stderr, f"{shot}: Python-Fehler\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def event_frames(shot: str) -> dict[str, int]:
    tl = TL.load()
    return {e: TL.event_frame(e, shot, tl) for e in SHOTS[shot]}


@needs_blender
@pytest.mark.parametrize("shot", list(SHOTS))
def test_proxy_frames_not_empty(shot, tmp_path):
    frames = event_frames(shot)
    run_shot(shot, tmp_path, "--frames", ",".join(f"ev:{e}" for e in SHOTS[shot]))
    for ev, f in frames.items():
        img = np.asarray(Image.open(tmp_path / f"{f:04d}.png").convert("RGB"), dtype=np.float64)
        assert img.shape == (PROXY, PROXY, 3)
        assert img.mean() > 3.0, f"{shot}/{ev}: Bild fast schwarz (Mittel {img.mean():.1f})"
        assert img.std() > 4.0, f"{shot}/{ev}: Bild ohne Inhalt (Streuung {img.std():.1f})"


@needs_blender
@pytest.mark.parametrize("shot", list(SHOTS))
def test_key_objects_inside_centre_square(shot, tmp_path):
    frames = event_frames(shot)
    run_shot(shot, tmp_path, "--matte-keys", "--frames", ",".join(f"ev:{e}" for e in SHOTS[shot]))
    for ev, f in frames.items():
        alpha = np.asarray(Image.open(tmp_path / f"{f:04d}.png").convert("RGBA"))[..., 3]
        ys, xs = np.nonzero(alpha > 8)
        assert xs.size > 50, f"{shot}/{ev}: Matte der Schlüsselobjekte ist leer"
        box = (xs.min(), ys.min(), xs.max(), ys.max())
        assert LO - 1 <= box[0] and box[2] <= HI + 1, f"{shot}/{ev}: x {box[0]}..{box[2]} außerhalb {LO:.0f}..{HI:.0f}"
        assert LO - 1 <= box[1] and box[3] <= HI + 1, f"{shot}/{ev}: y {box[1]}..{box[3]} außerhalb {LO:.0f}..{HI:.0f}"


@needs_blender
def test_tower_rigid_body_bake_is_deterministic(tmp_path):
    runs = []
    for k in range(2):
        out = tmp_path / f"lauf{k}.json"
        run_shot("s2_website", tmp_path / f"r{k}", "--frames", "none", "--physik-json", str(out))
        runs.append(json.loads(out.read_text(encoding="utf-8")))
    a, b = runs
    assert a["ende"].keys() == b["ende"].keys()
    assert len(a["ende"]) == 9
    for name in a["ende"]:
        assert np.allclose(a["ende"][name], b["ende"][name], atol=1e-6), f"{name}: Bake nicht deterministisch"
    # the tower really fell: most blocks ended below the window ledge
    fallen = sum(a["ende"][n][2] < a["ledge_z"] - 0.05 for n in a["ende"])
    assert fallen >= 6, f"nur {fallen} von 9 Bausteinen sind vom Sims gefallen"
    # and it stood still before the push (start poses = stacked tower)
    zs = sorted(v[2] for v in a["start"].values())
    assert zs[0] > a["ledge_z"] and all(z2 > z1 for z1, z2 in zip(zs, zs[1:]))
