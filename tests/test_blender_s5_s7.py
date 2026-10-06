"""Tests for the Blender shots s5_betrieb, s6_beweis and s7_team (Task 5).

* pure-Python geometry helpers (paper-plane fold kinematics, dashed paths)
  run without Blender,
* every shot renders a 480 px proxy at three events; the frames must not be
  empty, every event's key objects must project into the centre 1080x1080
  square of the 1920 frame (both delivery crops keep them),
* the rigid-body ticket collapse is baked deterministically and the committed
  bake equals a fresh simulation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "blender" / "lib"))
sys.path.insert(0, str(REPO / "blender" / "lib_szenen"))

import nomiss_timeline as TL  # noqa: E402
import szenen_s5_s7 as S  # noqa: E402

BLENDER = shutil.which("blender")
needs_blender = pytest.mark.skipif(BLENDER is None, reason="blender nicht installiert")

# three events per shot are rendered; all events of the shot are probed
RENDER_EVENTS = {
    "s5_betrieb": ("invoice_spin", "paper_plane", "dashboards"),
    "s6_beweis": ("prospect_paper", "grafana_flyover", "fan_panels"),
    "s7_team": ("converge", "tickets_fall", "blackbox_unfold"),
}
SHOT_EVENTS = {
    "s5_betrieb": ("invoice_spin", "paper_plane", "rollback_path", "rolling_update", "dashboards"),
    "s6_beweis": ("prospect_paper", "grafana_flyover", "stamp", "fan_panels"),
    "s7_team": ("converge", "tickets_fall", "direct_line", "docs_fly", "blackbox_unfold"),
}
LO, HI = 420.0, 1500.0  # centre square in 1920 px


# --------------------------------------------------------------------------
# pure geometry
# --------------------------------------------------------------------------

def _edges(nx: int, ny: int) -> np.ndarray:
    idx = np.arange(nx * ny).reshape(ny, nx)
    h = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    v = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    return np.concatenate([h, v])


def test_fold_keeps_every_rigid_panel_isometric():
    plane = S.DartPlane(width=0.21, height=0.297)
    grid, (nx, ny) = plane.sheet_grid(41, 57)
    edges = _edges(nx, ny)
    sig = plane.signatures(grid)
    same = np.all(sig[edges[:, 0]] == sig[edges[:, 1]], axis=1)
    rest = np.linalg.norm(grid[edges[:, 0]] - grid[edges[:, 1]], axis=1)
    for t in (0.0, 0.3, 0.55, 0.8, 1.0):
        pos = plane.positions(grid, plane.angles_at(t))
        ln = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        assert np.all(np.isfinite(pos))
        # edges inside one rigid panel keep their length exactly
        assert np.max(np.abs(ln[same] - rest[same])) < 1e-9, t


def test_fold_creases_stay_closed_and_wings_land_outside():
    # a wrong hinge or rotation sign tears the sheet open along a crease
    plane = S.DartPlane(width=0.21, height=0.297)
    grid, (nx, ny) = plane.sheet_grid(41, 57)
    edges = _edges(nx, ny)
    sig = plane.signatures(grid)
    crease = ~np.all(sig[edges[:, 0]] == sig[edges[:, 1]], axis=1)
    rest = np.linalg.norm(grid[edges[:, 0]] - grid[edges[:, 1]], axis=1)
    slack = 2 * 7 * 0.00015 + 1e-9  # paper layer offsets
    for t in np.linspace(0.0, 1.0, 21):
        pos = plane.positions(grid, plane.angles_at(t))
        ln = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        assert np.max(ln[crease] - rest[crease]) <= slack, t
    pos = plane.positions(grid, plane.angles_at(1.0))
    fr = plane.flight_frame(pos, grid)
    side = (pos - fr["kiel"]) @ fr["rechts"]
    # outer wing material that no corner fold (1L, 1R, 2L, 2R) touched
    body = ~sig[:, :4].any(axis=1)
    wr = body & (grid[:, 0] > 0.21 * 0.4)
    wl = body & (grid[:, 0] < -0.21 * 0.4)
    assert wr.sum() > 10 and wl.sum() > 10
    # each wing lies completely on its own side, far from the keel
    assert np.all(np.abs(side[wr]) > 0.03) and np.all(np.abs(side[wl]) > 0.03)
    assert np.all(np.sign(side[wr]) == np.sign(side[wr][0]))
    assert np.all(np.sign(side[wl]) == -np.sign(side[wr][0]))
    # wings sit above the keel
    up = (pos - fr["kiel"]) @ fr["oben"]
    assert np.median(up[wr | wl]) > 0


def test_folded_plane_is_slender_and_mirror_symmetric():
    plane = S.DartPlane(width=0.21, height=0.297)
    grid, _ = plane.sheet_grid(41, 57)
    pos = plane.positions(grid, plane.angles_at(1.0))
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    span = hi - lo
    # length along the nose axis (y) stays the sheet height
    assert span[1] == pytest.approx(0.297, abs=1e-3)
    # the folded plane is narrow compared with the flat sheet
    assert span[0] < 0.21 * 0.5
    # left and right wing mirror each other about the keel plane; the paper
    # layer offsets (against z-fighting) may break it by a few millimetres
    frame = plane.flight_frame(pos)
    assert frame["spannweite"] > 0.08
    assert frame["symmetrie_fehler"] < 1e-2
    exact = S.DartPlane(width=0.21, height=0.297, eps=0.0)
    pos0 = exact.positions(grid, exact.angles_at(1.0))
    assert exact.flight_frame(pos0)["symmetrie_fehler"] < 1e-9


def test_dashes_cover_the_path_in_order():
    pts = np.stack([np.linspace(0, 1, 200), np.zeros(200), np.zeros(200)], axis=1)
    dashes = S.dash_layout(pts, dash=0.05, gap=0.03)
    starts = np.array([d[0] for d in dashes])
    assert np.all(np.diff(starts) > 0)
    assert dashes[0][0] == pytest.approx(0.0)
    assert dashes[-1][1] <= 1.0 + 1e-9
    assert len(dashes) == pytest.approx(1.0 / 0.08, abs=1)


def test_camera_path_never_overshoots_a_held_set_piece():
    # a long move into a set piece followed by a small drift must not
    # overshoot the key (Catmull-Rom does: the beat slides out of frame)
    keys = [(1, (0.0, 0.0, 0.0)), (300, (0.2, 0.0, 0.0)), (400, (2.6, 0.1, 0.0)), (480, (2.7, 0.1, 0.0))]
    xs = np.array([S.smooth_path(f, keys)[0] for f in range(1, 481)])
    assert np.all(np.diff(xs) >= -1e-12)
    assert xs.max() <= 2.7 + 1e-9
    assert S.smooth_path(400, keys) == pytest.approx((2.6, 0.1, 0.0))


def test_camera_path_rejects_keys_out_of_order():
    # a re-timed voice can push two beat keys past each other; that must
    # fail loudly instead of producing NaN or a camera jump
    with pytest.raises(ValueError):
        S.smooth_path(10, [(1, (0, 0, 0)), (20, (1, 0, 0)), (20, (2, 0, 0))])
    with pytest.raises(ValueError):
        S.smooth_path(10, [(1, (0, 0, 0)), (30, (1, 0, 0)), (25, (2, 0, 0))])


def test_png_check_detects_truncated_frames(tmp_path):
    from PIL import Image

    good = tmp_path / "0001.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(good)
    bad = tmp_path / "0002.png"
    bad.write_bytes(good.read_bytes()[:-20])
    assert S.png_complete(good)
    assert not S.png_complete(bad)
    assert not S.png_complete(tmp_path / "0003.png")


def test_ticket_bake_runs_at_60_fps():
    # Bullet steps 1/fps per frame; a scratch scene defaults to 24 fps and
    # would make the collapse 2.5x too fast (g/24^2 per frame^2)
    data = json.loads(S.TICKET_BAKE.read_text(encoding="utf-8"))
    # a lone box in free fall inside the same simulation measures the step
    z = np.asarray(data["fallprobe"])[:20]
    acc = -np.diff(z, n=2)
    g60 = 9.81 / 60.0 ** 2
    assert np.median(acc) == pytest.approx(g60, rel=0.05), np.median(acc) / g60
    assert data.get("fps") == 60


def test_screenshot_crops_drop_vendor_logos():
    # header bars (Proxmox logo) and the Grafana sidebar are cut away
    assert S.SCREENSHOTS["proxmox-ui"]["crop"][1] >= 36
    assert S.SCREENSHOTS["pbs-dashboard"]["crop"][1] >= 42
    assert S.SCREENSHOTS["grafana-dashboard"]["crop"][0] >= 100
    assert S.SCREENSHOTS["grafana-monitoring"]["crop"][3] <= 904


# --------------------------------------------------------------------------
# Blender proxy renders
# --------------------------------------------------------------------------

def _run(shot: str, out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [BLENDER, "-b", "--factory-startup", "-P", str(REPO / "blender" / "shots" / f"{shot}.py"), "--", *extra]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1500)
    (out / "blender.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
    assert res.returncode == 0, res.stderr[-3000:] + res.stdout[-3000:]
    return res


@pytest.fixture(scope="module", params=list(RENDER_EVENTS))
def proxy(request, tmp_path_factory):
    if BLENDER is None:
        pytest.skip("blender nicht installiert")
    shot = request.param
    out = tmp_path_factory.mktemp(shot)
    frames = ",".join(f"ev:{e}" for e in RENDER_EVENTS[shot])
    _run(shot, out, "--proxy", "480", "--frames", frames, "--out", str(out), "--probe", str(out / "probe.json"))
    return shot, out, json.loads((out / "probe.json").read_text(encoding="utf-8"))


@needs_blender
def test_probe_uses_timeline_event_frames(proxy):
    shot, _, probe = proxy
    tl = TL.load()
    for ev in SHOT_EVENTS[shot]:
        assert probe["frames"][ev] == TL.event_frame(ev, shot, tl)
    assert probe["shot_frames"] == TL.shot(shot, tl).frames


@needs_blender
def test_proxy_frames_are_not_empty(proxy):
    from PIL import Image

    shot, out, probe = proxy
    for ev in RENDER_EVENTS[shot]:
        f = probe["frames"][ev]
        img = np.asarray(Image.open(out / f"{f:04d}.png").convert("RGB")).astype(np.float64) / 255.0
        assert img.shape == (480, 480, 3)
        lum = img @ np.array([0.2126, 0.7152, 0.0722])
        assert lum.std() > 0.04, (ev, lum.std())
        assert np.percentile(lum, 99.5) > 0.35, (ev, np.percentile(lum, 99.5))


@needs_blender
def test_event_objects_lie_in_centre_square(proxy):
    shot, _, probe = proxy
    for ev in SHOT_EVENTS[shot]:
        boxes = probe["bbox"][ev]
        assert boxes, f"{ev}: keine Schluesselobjekte"
        for name, (x0, y0, x1, y1) in boxes.items():
            assert LO <= x0 and x1 <= HI and LO <= y0 and y1 <= HI, (ev, name, (x0, y0, x1, y1))
            if probe["typ"].get(name) != "EMPTY":  # probe points have no extent
                assert max(x1 - x0, y1 - y0) > 8, (ev, name, "zu klein")


@needs_blender
def test_shot_specific_checks(proxy):
    shot, _, probe = proxy
    extra = probe.get("extra", {})
    if shot == "s6_beweis":
        # +-1 s around the stamp the dashboard stands still (readable)
        assert extra["ruhe_px_pro_bild"] < 2.0, extra["ruhe_px_pro_bild"]
    if shot == "s7_team":
        # every wall of the black box opens outwards
        assert all(extra["box_aussen"].values()), extra["box_aussen"]
        # standing ticket slips are not upside down
        assert extra["zettel_aufrecht"] == 1.0, extra["zettel_aufrecht"]


@needs_blender
def test_ticket_collapse_bake_is_deterministic(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _run("s7_team", tmp_path, "--bake-check", str(a))
    _run("s7_team", tmp_path, "--bake-check", str(b))
    da = json.loads(a.read_text(encoding="utf-8"))
    db = json.loads(b.read_text(encoding="utf-8"))
    assert da["hash"] == db["hash"]
    ma, mb = np.asarray(da["matrizen"]), np.asarray(db["matrizen"])
    assert ma.shape == mb.shape and ma.shape[0] >= 20
    assert np.max(np.abs(ma - mb)) < 1e-6
    # the tower really collapses: no slip stays anywhere near the old top
    heights = ma[:, :, 2, 3]
    assert heights[:, -1].max() < heights[:, 0].max() * 0.4
    committed = json.loads(S.TICKET_BAKE.read_text(encoding="utf-8"))
    assert committed["hash"] == da["hash"]
    assert np.max(np.abs(np.asarray(committed["matrizen"]) - ma)) < 1e-5


@pytest.mark.parametrize("shot", list(RENDER_EVENTS))
def test_full_render_metadata(shot):
    meta = REPO / "renders" / f"{shot}.json"
    if not meta.exists():
        pytest.skip("Vollrender noch nicht gelaufen")
    data = json.loads(meta.read_text(encoding="utf-8"))
    sh = TL.shot(shot)
    assert data["shot"] == shot and data["szene"] == shot
    assert data["frames"] == sh.frames
    assert data["handle_f"] == 30
    assert data["start_s"] == pytest.approx(sh.start_s, abs=1e-3)
    assert (REPO / "renders" / f"{shot}.mov").exists()
