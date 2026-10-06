"""Tests for music, ducking, stems and mastering (Task 2).

Runs against tests/fixtures/timeline_audio.json with a synthetic stand-in
voice (mix.ersatz_vo_schreiben), so it works before the real timeline exists.
Levels are measured independently here (pyloudnorm filters, own 4x true peak).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt

WURZEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WURZEL / "audio"))

import mix  # noqa: E402
import musik  # noqa: E402
from ton_gemeinsam import SR, STEMS  # noqa: E402

FIXTURE = WURZEL / "tests" / "fixtures" / "timeline_audio.json"


# ---------------------------------------------------------------- helpers

def _k_gewichtet(x: np.ndarray) -> np.ndarray:
    meter = pyln.Meter(SR)
    y = np.array(x, dtype=np.float64)
    for f in meter._filters.values():  # BS.1770 pre-filter + RLB high-pass, per channel like pyloudnorm
        for ch in range(y.shape[1]):
            y[:, ch] = f.apply_filter(y[:, ch])
    return y


def _pegel(y_k: np.ndarray, a: float, b: float) -> float:
    seg = y_k[int(a * SR): int(b * SR)]
    return -0.691 + 10 * np.log10(np.sum(np.mean(seg ** 2, axis=0)) + 1e-20)


def _true_peak_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.abs(resample_poly(x, 4, 1, axis=0)).max())


def _lies(pfad: Path) -> np.ndarray:
    daten, sr = sf.read(pfad, always_2d=True)
    assert sr == SR
    return daten


def _bauen(basis: Path) -> tuple[mix.MixErgebnis, mix.Ziele, dict]:
    basis.mkdir(parents=True, exist_ok=True)
    tl_pfad = basis / "timeline.json"
    shutil.copy(FIXTURE, tl_pfad)
    tl = json.loads(tl_pfad.read_text())
    mix.ersatz_vo_schreiben(tl, basis)
    ziele = mix.Ziele.unter(basis)
    t0, c0 = time.perf_counter(), time.process_time()
    erg = mix.bauen(tl_pfad, basis=basis, ziele=ziele)
    return erg, ziele, {"wand_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0}


def _mini_timeline(basis: Path) -> dict:
    """9 s timeline with two lines, a hit under the first line and a scene change."""
    def zeile(i: int, start: float, n: int) -> dict:
        woerter = [{"wort": f"w{k}", "start_s": round(start + k * 0.36, 3),
                    "ende_s": round(start + k * 0.36 + 0.3, 3)} for k in range(n)]
        return {"id": f"v{i:02d}", "szene": "s1_nacht" if i == 1 else "s8_morgen", "start_s": start,
                "ende_s": woerter[-1]["ende_s"], "datei": f"assets/audio/vo/v{i:02d}.wav", "woerter": woerter}
    tl = {"fps": 60, "dauer_s": 9.0, "abspann": {"start_s": 7.0, "ende_s": 9.0},
          "szenen": [{"id": "s1_nacht", "start_s": 0.0, "ende_s": 4.0},
                     {"id": "s8_morgen", "start_s": 4.0, "ende_s": 7.0}],
          "vo": [zeile(1, 0.8, 6), zeile(2, 4.5, 4)],
          "events": [{"id": "x", "szene": "s1_nacht", "t_s": 2.0, "f": 120, "sfx": "hit"}],
          "cues": [{"t_s": 2.0, "art": "hit"}, {"t_s": 4.0, "art": "whoosh"}]}
    mix.ersatz_vo_schreiben(tl, basis)
    return tl


@pytest.fixture(scope="module")
def gebaut(tmp_path_factory):
    erg, ziele, laufzeit = _bauen(tmp_path_factory.mktemp("mix_a"))  # default modus 'regeln'
    tl = json.loads(FIXTURE.read_text())
    return erg, ziele, laufzeit, tl


# ---------------------------------------------------------------- master

def test_master_lufs(gebaut):
    _, ziele, _, _ = gebaut
    ref = _lies(ziele.referenz)
    assert abs(pyln.Meter(SR).integrated_loudness(ref) - (-14.0)) <= 0.5


def test_true_peak(gebaut):
    _, ziele, _, _ = gebaut
    assert _true_peak_db(_lies(ziele.referenz)) <= -1.0


def test_stems_summieren_zum_referenzmix(gebaut):
    _, ziele, _, tl = gebaut
    n = round(tl["dauer_s"] * SR)
    summe = np.zeros((n, 2))
    for name in STEMS:
        x = _lies(ziele.stems / f"{name}.wav")
        assert x.shape == (n, 2), name
        summe += x
    ref = _lies(ziele.referenz)
    assert ref.shape == (n, 2)
    assert np.abs(summe - ref).max() < 1e-3


def test_musik_mindestens_15_db_unter_jedem_wort(gebaut):
    _, ziele, _, tl = gebaut
    vo_k = _k_gewichtet(_lies(ziele.stems / "vo.wav"))
    mu_k = _k_gewichtet(_lies(ziele.stems / "musik.wav"))
    im_wort, kurzzeit = [], []
    for zeile in tl["vo"]:
        for w in zeile["woerter"]:
            mu = _pegel(mu_k, w["start_s"], w["ende_s"])
            mitte = (w["start_s"] + w["ende_s"]) / 2
            im_wort.append(_pegel(vo_k, w["start_s"], w["ende_s"]) - mu)
            kurzzeit.append(_pegel(vo_k, mitte - 1.5, mitte + 1.5) - mu)  # EBU short-term (3 s) reading
    assert min(im_wort) >= 15.0, min(im_wort)
    assert min(kurzzeit) >= 15.0, min(kurzzeit)


def test_musik_kommt_zwischen_szenen_wieder_hoch(gebaut):
    """The duck must release: between scenes the music is clearly louder than under words."""
    _, ziele, _, tl = gebaut
    mu_k = _k_gewichtet(_lies(ziele.stems / "musik.wav"))
    zeilen = tl["vo"]
    for a, b in zip(zeilen, zeilen[1:]):
        if a["szene"] != b["szene"]:
            luecke = _pegel(mu_k, a["ende_s"] + 0.45, b["start_s"] - 0.1)
            unter = _pegel(mu_k, a["woerter"][-3]["start_s"], a["ende_s"])
            assert luecke > unter + 4.0, (a["id"], b["id"], luecke, unter)


def test_handylautsprecher(gebaut):
    _, ziele, _, _ = gebaut
    ref = _lies(ziele.referenz).mean(axis=1)
    hoch = sosfiltfilt(butter(8, 400, "highpass", fs=SR, output="sos"), ref)
    anteil = np.sqrt(np.mean(hoch ** 2)) / np.sqrt(np.mean(ref ** 2))
    assert anteil >= 0.45, anteil


def test_vo_kette_ist_rotator_gain_limiter_ohne_kompressor(gebaut):
    """VO stem = all-pass(raw) x one static gain x VO peak limiter x linked master safety. Nothing else."""
    erg, ziele, _, _ = gebaut
    vo = _lies(ziele.stems / "vo.wav")
    assert np.array_equal(vo[:, 0], vo[:, 1])
    from scipy.signal import lfilter
    rot = erg.eingang.vo
    for f, q in mix.VO_ROTATOR:  # independent RBJ all-pass cascade
        w0 = 2 * np.pi * f / SR
        al, c = np.sin(w0) / (2 * q), np.cos(w0)
        rot = lfilter([1 - al, -2 * c, 1 + al], [1 + al, -2 * c, 1 - al], rot)
    maske = np.abs(rot) > 1e-3
    statisch = erg.vo_rotiert[maske] / rot[maske]
    assert np.ptp(statisch) / np.mean(statisch) < 1e-6
    erwartet = erg.vo_rotiert * erg.vo_limiter * erg.sicherheit * 10 ** (erg.master_db / 20)
    assert np.allclose(vo[:, 0], erwartet, atol=1e-6)
    unberuehrt = maske & (erg.vo_limiter == 1.0) & (erg.sicherheit == 1.0)
    faktor = vo[unberuehrt, 0] / rot[unberuehrt]
    assert np.ptp(faktor) / np.mean(faktor) < 1e-4


def test_vo_limiter_haelt_decke_und_wird_gemessen(gebaut):
    erg, _, _, _ = gebaut
    master = 10 ** (erg.master_db / 20)
    assert _true_peak_db(erg.vo_rotiert * erg.vo_limiter * master) <= mix.VO_DECKE_DBTP + 0.05
    lim = erg.pegel["vo_limiter"]
    gr = -20 * np.log10(erg.vo_limiter)
    assert abs(lim["gr_max_db"] - gr.max()) < 0.01
    assert abs(lim["aktiv_prozent_gesamt"] - np.mean(gr > mix.GR_AKTIV_DB) * 100) < 0.01
    assert lim["lra_verlust_lu"] <= 1.0
    assert set(lim["regeln"]) == {"aktiv_unter_1_prozent", "gr_max_6_db", "st_abweichung_unter_0_3_lu",
                                  "lra_verlust_max_1_lu"}


def test_default_build_haelt_alle_grenzen_und_vo_regeln(gebaut):
    """Default modus 'regeln': the four voice limiter rules hold (over the spoken words) and no
    hard limit is violated."""
    erg, _, _, _ = gebaut
    p = erg.pegel
    assert p["modus"] == "regeln"
    assert p["verstoesse"] == []
    lim = p["vo_limiter"]
    assert lim["regeln_erfuellt"], lim
    assert lim["aktiv_prozent_sprache"] < 1.0 and lim["gr_max_db"] <= 6.0
    assert lim["st_abweichung_lu"] < 0.3 and lim["lra_verlust_lu"] <= 1.0
    sprache = sum(w["ende_s"] - w["start_s"] for z in gebaut[3]["vo"] for w in z["woerter"])
    assert abs(lim["sprache_s"] - sprache) < 0.5  # denominator = spoken words, not whole lines


def test_brief_modus_trifft_minus_14(gebaut):
    """modus='brief': exactly -14 LUFS; the limiter rules are measured and reported."""
    erg, _, _, tl = gebaut
    ein = erg.eingang
    laut = mix.mischen(ein.vo, ein.musik, ein.sfx, tl, modus="brief")
    assert abs(pyln.Meter(SR).integrated_loudness(laut.master) - (-14.0)) <= 0.06
    assert _true_peak_db(laut.master) <= -1.0
    assert laut.master_db >= erg.master_db - 1e-6
    assert set(laut.pegel["vo_limiter"]["regeln"]) == set(erg.pegel["vo_limiter"]["regeln"])


def test_bett_weicht_nur_kurz_aus(gebaut):
    b = gebaut[0].pegel["bett_platz"]
    assert b["ueber_10db_s"] < 0.5 and b["ueber_3db_s"] < 2.0, b


def test_sprachband_dip_im_mix(gebaut):
    """Atmo: inside phrases its 2-4 kHz band sits ~4 dB lower relative to its low band than in the
    raw render; between scenes nothing changes (all broadband gains cancel in the ratio)."""
    erg, ziele, _, tl = gebaut
    roh = erg.eingang.sfx["sfx_atmo"].mean(axis=1)
    fertig = _lies(ziele.stems / "sfx_atmo.wav").mean(axis=1)
    band = butter(4, (2300, 3700), "bandpass", fs=SR, output="sos")
    tief = butter(4, (300, 1000), "bandpass", fs=SR, output="sos")

    def verhaeltnis(x: np.ndarray, bereiche: list[tuple[float, float]]) -> float:
        b_, t_ = sosfiltfilt(band, x), sosfiltfilt(tief, x)
        idx = np.concatenate([np.arange(int(a * SR), int(e * SR)) for a, e in bereiche])
        return 10 * np.log10(np.mean(b_[idx] ** 2) / np.mean(t_[idx] ** 2))

    from ton_gemeinsam import phrasen
    innen = [(a + 0.1, b - 0.1) for a, b in phrasen(tl) if b - a > 0.6]
    zeilen = tl["vo"]
    pausen = [(a["ende_s"] + 0.45, b["start_s"] - 0.1) for a, b in zip(zeilen, zeilen[1:])
              if b["start_s"] - a["ende_s"] > 0.9]
    assert abs((verhaeltnis(fertig, innen) - verhaeltnis(roh, innen)) - (-4.0)) < 1.0
    assert abs(verhaeltnis(fertig, pausen) - verhaeltnis(roh, pausen)) < 0.5


def test_phasenrotator_ist_allpass():
    from scipy.signal import freqz
    h_ges = np.ones(4096, dtype=complex)
    for f, q in mix.VO_ROTATOR:
        b, a = mix._allpass_koeff(f, q)
        _, h = freqz(b, a, worN=4096, fs=SR)
        h_ges *= h
    assert np.allclose(np.abs(h_ges), 1.0, atol=1e-9)
    x = np.random.default_rng(3).standard_normal(SR * 5) * 0.1
    st = lambda y: pyln.Meter(SR).integrated_loudness(np.stack([y, y], axis=1))  # noqa: E731
    assert abs(st(mix.phasenrotator(x)) - st(x)) < 0.05
    # the function itself (not only its coefficients): flat magnitude and strictly linear,
    # so no hidden compression can sit inside it
    h = mix.phasenrotator(np.r_[1.0, np.zeros(2 ** 16 - 1)])
    assert np.allclose(np.abs(np.fft.rfft(h)), 1.0, atol=1e-6)
    spitzen = x.copy()
    spitzen[::4801] = 0.9
    assert np.allclose(mix.phasenrotator(3 * spitzen), 3 * mix.phasenrotator(spitzen), atol=1e-12)


def test_vo_limiter_fasst_nur_einzelspitze():
    t = np.arange(SR * 6) / SR
    x = 0.1 * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    x[3 * SR: 3 * SR + 24] += 0.8 * np.hanning(24)
    g = mix.vo_limiter_gain(x, -4.5)  # spike ~ -0.6 dBFS -> about 4 dB gain reduction
    gr = -20 * np.log10(g)
    aktiv = np.nonzero(gr > mix.GR_AKTIV_DB)[0]
    assert len(aktiv) and (aktiv[-1] - aktiv[0]) / SR < 0.01
    assert _true_peak_db(x * g) <= -4.5 + 0.05
    m = mix.vo_limiter_metriken(x, g, np.ones(len(x), dtype=bool))
    assert m["regeln_erfuellt"], m


def test_vorlaeufige_zeilen_werden_uebersprungen(tmp_path):
    tl = json.loads(FIXTURE.read_text())
    mix.ersatz_vo_schreiben(tl, tmp_path)
    tl["vo"][3]["datei"] = None
    tl["vo"][4]["vorlaeufig"] = True
    with pytest.warns(UserWarning) as meldungen:
        vo = mix.vo_laden(tl, tmp_path)
    texte = " ".join(str(m.message) for m in meldungen)
    assert "v04" in texte and "v05" in texte
    z = tl["vo"][3]
    assert np.abs(vo[int(z["start_s"] * SR): int(z["ende_s"] * SR)]).max() == 0.0
    z = tl["vo"][5]
    assert np.abs(vo[int(z["start_s"] * SR): int(z["ende_s"] * SR)]).max() > 0.01


def test_vorlaeufige_zeile_im_ganzen_mix(tmp_path):
    """A line without recording is skipped in the whole mix: no duck, no crash."""
    import musik
    import sfx
    tl = _mini_timeline(tmp_path)
    tl["vo"][1]["datei"] = None
    with pytest.warns(UserWarning, match="v02"):
        vo = mix.vo_laden(tl, tmp_path)
        erg = mix.mischen(vo, musik.musik_rendern(tl).mix, sfx.kategorien_rendern(tl), tl)
    z = tl["vo"][1]
    for p in erg.pegel["duck_phrasen"]:
        assert p["ende_s"] < z["start_s"] or p["start_s"] > z["ende_s"], p
    assert np.all(np.isfinite(erg.master))


def test_verstoss_bricht_ab_und_schreibt_keine_stems(tmp_path, monkeypatch):
    _mini_timeline(tmp_path)
    tl = _mini_timeline(tmp_path)
    (tmp_path / "timeline.json").write_text(json.dumps(tl))
    monkeypatch.setattr(mix, "grenzen_pruefen", lambda pegel: ["Testverstoss"])
    ziele = mix.Ziele.unter(tmp_path)
    with pytest.raises(mix.MixFehler, match="Testverstoss"):
        mix.bauen(tmp_path / "timeline.json", basis=tmp_path, ziele=ziele)
    assert not ziele.stems.exists() and not ziele.referenz.exists()
    assert json.loads(ziele.pegel.read_text())["verstoesse"] == ["Testverstoss"]


def test_duck_bericht_zeigt_angewandte_tiefen(tmp_path, monkeypatch):
    """If words stay too close, phrases are deepened 3x by 3 dB and the report shows exactly the
    applied depth (not a 4th, never-applied step), and the violation is flagged."""
    import musik
    import sfx
    tl = _mini_timeline(tmp_path)
    monkeypatch.setattr(mix, "_duck_phrasen", lambda t, v, m: [
        {"start_s": a, "ende_s": b, "tiefe_db": -10.0} for a, b in mix.phrasen(t)])
    monkeypatch.setattr(mix, "_wort_abstaende", lambda t, v, m: [
        (a, 12.0, 12.0) for a, _ in mix.wort_intervalle(t)])
    erg = mix.mischen(mix.vo_laden(tl, tmp_path), musik.musik_rendern(tl).mix, sfx.kategorien_rendern(tl), tl)
    assert {p["tiefe_db"] for p in erg.pegel["duck_phrasen"]} == {-19.0}
    assert any("duck" in v for v in erg.pegel["verstoesse"])


def test_pegel_json(gebaut):
    _, ziele, _, _ = gebaut
    p = json.loads(ziele.pegel.read_text())
    for schluessel in ("master_gain_db", "lufs_integriert", "true_peak_dbtp", "duck_min_abstand_db",
                       "handy_anteil", "gains_db", "sicherheit_aktiv_s"):
        assert schluessel in p, schluessel
    ref = _lies(ziele.referenz)
    assert abs(p["lufs_integriert"] - pyln.Meter(SR).integrated_loudness(ref)) < 0.05


def test_laufzeit_unter_zwei_minuten(gebaut):
    """CPU time of the whole build (wall time depends on what else runs on the machine)."""
    assert gebaut[2]["cpu_s"] < 120.0, gebaut[2]


def test_deterministisch(gebaut, tmp_path):
    _, ziele, _, _ = gebaut
    _, ziele_b, _ = _bauen(tmp_path / "mix_b")
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert h(ziele.referenz) == h(ziele_b.referenz)
    for name in STEMS:
        assert h(ziele.stems / f"{name}.wav") == h(ziele_b.stems / f"{name}.wav"), name


def test_sfx_aenderung_misst_neu(gebaut):
    """Review Focus 3: re-rendering a stem must re-measure, never reuse a cached gain."""
    erg, _, _, tl = gebaut
    ein = erg.eingang
    lauter = {k: (v * 4.0 if k != "sfx_atmo" else v) for k, v in ein.sfx.items()}
    neu = mix.mischen(ein.vo, ein.musik, lauter, tl)
    assert abs(neu.master_db - erg.master_db) > 0.3
    assert abs(pyln.Meter(SR).integrated_loudness(neu.master) - (-14.0)) <= 0.5
    assert _true_peak_db(neu.master) <= -1.0


def test_fehlende_vo_datei_meldet_pfad(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "timeline.json")
    with pytest.raises(FileNotFoundError, match="v01.wav"):
        mix.bauen(tmp_path / "timeline.json", basis=tmp_path, ziele=mix.Ziele.unter(tmp_path))


def test_sprachband_ducken():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SR * 4, 2)) * 0.1
    gain_db = np.zeros(SR * 4)
    gain_db[SR: 3 * SR] = -4.0
    y = mix.sprachband_ducken(x, gain_db)
    band = butter(4, (2300, 3700), "bandpass", fs=SR, output="sos")

    def band_db(s: np.ndarray, a: float, b: float) -> float:
        seg = sosfiltfilt(band, s[:, 0])[int(a * SR): int(b * SR)]
        return 10 * np.log10(np.mean(seg ** 2))

    assert abs((band_db(y, 1.5, 2.5) - band_db(x, 1.5, 2.5)) - (-4.0)) < 0.7
    assert np.allclose(y[: int(0.5 * SR)], x[: int(0.5 * SR)], atol=1e-6)
    tief = butter(4, 600, "lowpass", fs=SR, output="sos")
    assert np.allclose(sosfiltfilt(tief, y[:, 0])[int(1.5 * SR): int(2.5 * SR)],
                       sosfiltfilt(tief, x[:, 0])[int(1.5 * SR): int(2.5 * SR)], atol=2e-3)


# ---------------------------------------------------------------- music

@pytest.fixture(scope="module")
def partitur():
    return musik.partitur_planen(json.loads(FIXTURE.read_text()))


def test_partitur_grenzfaelle():
    with pytest.raises(ValueError, match="Szenen"):
        musik.partitur_planen({"dauer_s": 5.0, "szenen": [], "vo": [], "events": []})
    tl = {"dauer_s": 6.0, "szenen": [{"id": "s1_nacht", "start_s": 0.0, "ende_s": 4.0},
                                     {"id": "s2_website", "start_s": 4.0, "ende_s": 4.0}],
          "vo": [], "events": [{"id": "frueh", "t_s": -0.2, "sfx": "hit"}]}
    m = musik.musik_rendern(tl)
    assert np.all(np.isfinite(m.mix)) and m.mix.shape == (6 * SR, 2)


def test_tempo_nah_an_100_bpm(partitur):
    for ab in partitur.abschnitte:
        assert 95.0 <= ab.bpm <= 105.0, (ab.szene, ab.bpm)


def test_szenenwechsel_auf_taktanfang(partitur):
    tl = json.loads(FIXTURE.read_text())
    takte = np.array(partitur.taktanfaenge)
    for s in tl["szenen"]:
        assert np.abs(takte - s["start_s"]).min() < 1.0 / SR, s["id"]


def test_akzente_auf_hits_und_stings(partitur):
    tl = json.loads(FIXTURE.read_text())
    raster = np.array(partitur.raster16)
    akzente = np.array([a.t_s for a in partitur.akzente])
    for ev in tl["events"]:
        if ev["sfx"] not in ("hit", "sting"):
            continue
        naechster = akzente[np.argmin(np.abs(akzente - ev["t_s"]))]
        assert abs(naechster - ev["t_s"]) <= 0.040, ev["id"]
        r = raster[np.argmin(np.abs(raster - ev["t_s"]))]
        if abs(r - ev["t_s"]) <= 0.040:
            assert abs(naechster - r) < 1e-3, ev["id"]


def test_akzente_klingen_auf_ihrer_zeit(gebaut):
    """Every accent attacks within 5 ms of its own start, and the accent layer is exactly
    the sum of the accents placed at their times (so the attack lands on akz.t_s)."""
    erg, _, _, _ = gebaut
    p = erg.partitur
    erwartet = np.zeros_like(erg.musik_spuren["akzente"])
    gesetzt = 0
    for a in p.akzente:
        teile = musik.akzent_klang(a, p)
        if teile is None:
            continue
        trocken = teile[0]
        env = maximum_filter1d(np.abs(trocken).max(axis=1), 48)
        onset = int(np.argmax(env >= 0.3 * env[: int(0.1 * SR)].max())) / SR
        assert onset <= 0.005, (a.event, onset)
        i = int(round(a.t_s * SR))
        k = min(len(trocken), len(erwartet) - i)
        erwartet[i:i + k] += trocken[:k]
        gesetzt += 1
    assert gesetzt == len(p.akzente)
    assert np.allclose(erg.musik_spuren["akzente"], erwartet, atol=1e-9)


def test_schlussakkord_klingt_unter_abspann_aus(gebaut):
    erg, ziele, _, tl = gebaut
    dauer = tl["dauer_s"]
    assert erg.partitur.schlussakkord_s <= dauer - 3.0
    assert erg.partitur.schlussakkord_s >= tl["vo"][-1]["woerter"][-1]["start_s"]
    mu = _lies(ziele.stems / "musik.wav")
    rms = lambda a, b: 20 * np.log10(np.sqrt(np.mean(mu[int(a * SR): int(b * SR)] ** 2)) + 1e-12)  # noqa: E731
    t0 = erg.partitur.schlussakkord_s
    assert rms(dauer - 1.0, dauer - 0.1) > -60.0
    assert rms(dauer - 1.0, dauer - 0.1) < rms(t0 + 0.2, t0 + 1.2) - 3.0


def test_sub_ist_mono(gebaut):
    erg, _, _, _ = gebaut
    m = erg.eingang.musik
    tief = butter(4, 100, "lowpass", fs=SR, output="sos")
    seite = sosfiltfilt(tief, (m[:, 0] - m[:, 1]) / 2)
    mitte = sosfiltfilt(tief, (m[:, 0] + m[:, 1]) / 2)
    assert 10 * np.log10(np.mean(seite ** 2) / np.mean(mitte ** 2)) < -30


def test_pads_sind_breit(gebaut):
    erg, _, _, _ = gebaut
    p = erg.musik_spuren["pads"]
    hoch = butter(4, 300, "highpass", fs=SR, output="sos")
    seite = sosfiltfilt(hoch, (p[:, 0] - p[:, 1]) / 2)
    mitte = sosfiltfilt(hoch, (p[:, 0] + p[:, 1]) / 2)
    assert 10 * np.log10(np.mean(seite ** 2) / np.mean(mitte ** 2)) > -12
