"""Tests for the synthesized SFX library and cue placement (Task 2)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfiltfilt

WURZEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WURZEL / "audio"))

import sfx  # noqa: E402
from ton_gemeinsam import SR, SFX_KATEGORIE  # noqa: E402

FIXTURE = WURZEL / "tests" / "fixtures" / "timeline_audio.json"

TRANSIENT_ARTEN = ("hit", "subdrop", "tick", "click", "lock", "paper", "clatter", "sting")
ENDE_ARTEN = ("whoosh", "riser")
START_ARTEN = ("downer", "rewind", "swell", "shimmer")
ALLE_ARTEN = TRANSIENT_ARTEN + ENDE_ARTEN + START_ARTEN


def _timeline(cues: list[dict], dauer: float = 6.0) -> dict:
    return {"fps": 60, "dauer_s": dauer, "abspann": {"start_s": dauer - 0.5, "ende_s": dauer},
            "szenen": [{"id": "s1_nacht", "start_s": 0.0, "ende_s": dauer - 0.5}],
            "vo": [], "events": [], "cues": cues}


def _huelle(x: np.ndarray, ms: float = 1.0) -> np.ndarray:
    return maximum_filter1d(np.abs(x).max(axis=1), max(1, int(SR * ms / 1000)))


def _onset(x: np.ndarray, t: float, vor: float = 0.03, nach: float = 0.12, schwelle: float = 0.3) -> float:
    i0, i1 = int((t - vor) * SR), int((t + nach) * SR)
    env = _huelle(x)[i0:i1]
    return (i0 + int(np.argmax(env >= schwelle * env.max()))) / SR


def _ende(x: np.ndarray, t: float, vor: float = 0.4, nach: float = 0.4, schwelle_db: float = -30.0) -> float:
    i0, i1 = int((t - vor) * SR), int((t + nach) * SR)
    env = _huelle(x, 3.0)[i0:i1]
    ueber = np.nonzero(env >= env.max() * 10 ** (schwelle_db / 20))[0]
    return (i0 + int(ueber[-1])) / SR


def test_alle_arten_mit_varianten_vorhanden():
    assert set(sfx.ARTEN) == set(ALLE_ARTEN)
    for art in sfx.ARTEN:
        assert 2 <= sfx.VARIANTEN[art] <= 3, art
        for v in range(sfx.VARIANTEN[art]):
            k = sfx.klang(art, v)
            assert k.daten.ndim == 2 and k.daten.shape[1] == 2, art
            assert np.all(np.isfinite(k.daten)), art
            assert 0.01 < np.abs(k.daten).max() <= 1.0, (art, v)
            assert 0 <= k.anker <= len(k.daten), art


def test_varianten_klingen_verschieden():
    for art in sfx.ARTEN:
        a = sfx.klang(art, 0).daten
        b = sfx.klang(art, 1).daten
        n = min(len(a), len(b))
        assert len(a) != len(b) or not np.allclose(a[:n], b[:n], atol=1e-3), art


@pytest.mark.parametrize("art", TRANSIENT_ARTEN)
def test_transiente_liegt_auf_t(art):
    for v in range(sfx.VARIANTEN[art]):
        t = 1.2345 + 0.5 * v
        stems = sfx.kategorien_rendern(_timeline([{"t_s": t, "art": art}]), varianten_ab=v)
        x = stems[SFX_KATEGORIE[art]]
        assert abs(_onset(x, t) - t) <= 0.005, (art, v, _onset(x, t) - t)


@pytest.mark.parametrize("art", ENDE_ARTEN)
def test_whoosh_und_riser_enden_auf_t(art):
    for v in range(sfx.VARIANTEN[art]):
        t = 3.111 + 0.3 * v
        stems = sfx.kategorien_rendern(_timeline([{"t_s": t, "art": art}]), varianten_ab=v)
        x = stems[SFX_KATEGORIE[art]]
        assert abs(_ende(x, t) - t) <= 0.010, (art, v, _ende(x, t) - t)
        i_peak = int(np.argmax(_huelle(x)))
        assert i_peak / SR < t, (art, v)


@pytest.mark.parametrize("art", START_ARTEN)
def test_klang_beginnt_auf_t(art):
    for v in range(sfx.VARIANTEN[art]):
        t = 1.5
        stems = sfx.kategorien_rendern(_timeline([{"t_s": t, "art": art}]), varianten_ab=v)
        x = stems[SFX_KATEGORIE[art]]
        env = _huelle(x)
        spitze = env.max()
        assert env[: int((t - 0.005) * SR)].max() <= spitze * 1e-3, (art, v)
        assert env[int(t * SR): int((t + 0.15) * SR)].max() >= spitze * 10 ** (-40 / 20), (art, v)


def test_kategorie_routing():
    for art in ALLE_ARTEN:
        stems = sfx.kategorien_rendern(_timeline([{"t_s": 2.5, "art": art}]))
        ziel = SFX_KATEGORIE[art]
        assert np.abs(stems[ziel]).max() > 0.01, art
        for name, x in stems.items():
            if name not in (ziel, "sfx_atmo"):
                assert np.abs(x).max() == 0.0, (art, name)


def test_rotation_wiederholt_nicht_identisch():
    cues = [{"t_s": t, "art": "hit"} for t in (1.0, 2.0, 3.0, 4.0)]
    x = sfx.kategorien_rendern(_timeline(cues))["sfx_hit"]
    teile = [x[int(t * SR): int((t + 0.5) * SR)] for t in (1.0, 2.0, 3.0, 4.0)]
    for i in range(len(teile)):
        for j in range(i + 1, len(teile)):
            assert not np.allclose(teile[i], teile[j], atol=1e-3), (i, j)


def test_unbekannte_art_wird_abgelehnt():
    with pytest.raises(ValueError, match="laser"):
        sfx.kategorien_rendern(_timeline([{"t_s": 1.0, "art": "laser"}]))


def test_ohne_cues_werden_events_genutzt():
    tl = _timeline([])
    del tl["cues"]
    tl["events"] = [{"id": "x", "szene": "s1_nacht", "t_s": 2.0, "sfx": "lock"}]
    stems = sfx.kategorien_rendern(tl)
    assert np.abs(stems["sfx_ui"]).max() > 0.01


@pytest.fixture(scope="module")
def fixture_stems():
    tl = json.loads(FIXTURE.read_text())
    return tl, sfx.kategorien_rendern(tl)


def test_stems_laenge_und_format(fixture_stems):
    tl, stems = fixture_stems
    n = round(tl["dauer_s"] * SR)
    assert set(stems) == {"sfx_whoosh", "sfx_hit", "sfx_ui", "sfx_uebergang", "sfx_signatur", "sfx_atmo"}
    for name, x in stems.items():
        assert x.shape == (n, 2), name
        assert np.all(np.isfinite(x)), name


def test_deterministisch(fixture_stems):
    tl, erste = fixture_stems

    def h(stems: dict) -> str:
        return hashlib.sha256(b"".join(stems[k].astype(np.float32).tobytes() for k in sorted(stems))).hexdigest()

    assert h(erste) == h(sfx.kategorien_rendern(tl))


def test_atmo_bett_ueber_ganze_laenge_und_morgen_heller(fixture_stems):
    tl, stems = fixture_stems
    atmo = stems["sfx_atmo"]
    for t in np.arange(1.0, tl["dauer_s"] - 1.5, 1.0):
        seg = atmo[int(t * SR): int((t + 1.0) * SR)]
        assert 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12) > -70, t
    s8 = next(s for s in tl["szenen"] if s["id"] == "s8_morgen")

    def schwerpunkt(seg: np.ndarray) -> float:
        spek = np.abs(np.fft.rfft(seg.mean(axis=1)))
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return float((spek * f).sum() / spek.sum())

    nacht = schwerpunkt(atmo[int(20 * SR): int(60 * SR)])
    morgen = schwerpunkt(atmo[int((s8["start_s"] + 4) * SR): int((s8["ende_s"] - 1) * SR)])
    assert morgen > 1.3 * nacht, (nacht, morgen)


@pytest.mark.parametrize("stem", ["sfx_atmo", "sfx_whoosh"])
def test_hochpass_120hz(stem, fixture_stems):
    x = fixture_stems[1][stem].mean(axis=1)
    tief = sosfiltfilt(butter(6, 80, "lowpass", fs=SR, output="sos"), x)
    assert 10 * np.log10(np.mean(tief ** 2) / np.mean(x ** 2)) < -25


def test_bibliothek_schreiben(tmp_path):
    dateien = sfx.bibliothek_schreiben(tmp_path)
    for art in sfx.ARTEN:
        eigene = [p for p in dateien if p.name.startswith(f"{art}_")]
        assert len(eigene) == sfx.VARIANTEN[art], art
    daten, sr = sf.read(dateien[0])
    assert sr == SR and daten.ndim == 2 and daten.shape[1] == 2
