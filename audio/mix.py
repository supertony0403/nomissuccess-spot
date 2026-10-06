"""Mix: ducking, stems and loudness mastering (Task 2).

Builds everything from timeline.json: renders SFX (sfx.py) and music (musik.py),
loads the VO lines, ducks the music under every spoken phrase, and masters to
-14 LUFS integrated / <= -1 dBTP. Writes eight stems that are already at their
final level, so in Resolve every track sits at 0 dB and the stems sum exactly
to out/mix_referenz.wav.

Rules (brief section 6 + plan):
  - music >= 15 dB below the VO's short-term level during every spoken word
    (sidechain from the word intervals, 60 ms attack, 350 ms release)
  - atmo and whooshes dipped -4 dB at 2-4 kHz while the VO speaks
  - every level is measured anew on every run; nothing is read back from pegel.json

Voice chain (decision of 05.10.2026: no compressor, nothing that squashes the voice):
  1. phase rotator: a cascade of second-order all-pass filters. The magnitude
     response is exactly flat (same loudness, same LRA, same timbre); only the
     phase changes. On the real voice it lowers the peak-to-loudness ratio by just
     ~0.3 dB, but it cuts the share of samples the limiter must touch by ~40 %
     (measured 1.45 % -> 0.89 %), so it stays.
  2. static gain
  3. transparent true-peak look-ahead limiter that only catches peaks
     (4x oversampled detection, 1 ms look-ahead, 1 ms hold, triangular
     smoothing, no release tail, no make-up gain, no compression ratio).
  Its rules are measured on every run (limiter x linked safety, over the spoken
  words) and written to pegel.json ("vo_limiter"):
     gain reduction (> 0.01 dB) on < 1 % of the word samples, max 6 dB,
     mean short-term loudness (3 s) deviation before/after < 0.3 LU,
     loudness range (LRA) loss <= 1 LU.
  The real TTS voice has peaks up to 20.6 dB above its loudness (vowel energy at
  300-1000 Hz), so the rules decide how loud the voice may sit. Two modes:
     modus="regeln" (default): the rules hold; the master gain is capped so the
                    voice never needs more (on the real voice: about -14.2 LUFS).
     modus="brief": exactly -14 LUFS; the limiter does what that requires and the
                    rules are only reported.
  Where voice + bed would exceed the true-peak limit, only the bed (music + all
  SFX stems, linked) gives way for those milliseconds; between the lines the same
  stage is the bed's peak limiter. The voice is never touched by it.
  A linked master safety gain (all stems alike) stays as a last resort; it is
  reported, counted in the voice rules, and expected to be idle.
  Hard limits are checked on every build (grenzen_pruefen); on a violation no
  stem is written and the CLI exits with code 3.

CLI:
  .venv/bin/python audio/mix.py [--timeline timeline.json] [--modus regeln|brief]
  .venv/bin/python audio/mix.py --timeline tests/fixtures/timeline_audio.json --ersatz-vo   (preview)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter

import musik
import sfx
from ton_gemeinsam import (SFX_STEMS, SR, STEMS, WURZEL, Pegelmesser, bp0, handy_anteil, kurzzeit_lufs,
                           lade_timeline, lautheitsspanne, lufs_integriert, n_samples, phrasen, true_peak_db,
                           true_peak_huelle, trapez_db, von_db, wav_lesen_mono, wav_schreiben, wort_intervalle)

ZIEL_LUFS = -14.0
TP_GRENZE_DBTP = -1.0
SICHERHEIT_SCHWELLE_DBTP = -1.15          # last-resort linked safety; expected to stay idle

VO_ZIEL_LUFS = -16.0        # VO alone, before the master gain
MUSIK_ZIEL_LUFS = -19.0     # music alone (unducked), before the master gain
SFX_GAINS_DB = {"sfx_whoosh": -1.0, "sfx_hit": -3.0, "sfx_ui": -1.0, "sfx_uebergang": -2.0,
                "sfx_signatur": 0.0, "sfx_atmo": 0.0}

DUCK_ABSTAND_DB = 15.0      # requirement
DUCK_RESERVE_DB = 2.0       # safety margin on top of the requirement
DUCK_MIN_DB = -6.0          # under speech the music always dips at least this much
DUCK_MAX_DB = -45.0
SPRACHBAND_DB = -4.0
SPRACHBAND_HZ = (2000.0, 4000.0)

# True-peak headroom: limits sit 0.2 dB under -1 dBTP, so finer (8x) meters and the AAC
# encode in Task 7 keep reserve. The 0.3 dB gap between voice ceiling and sum limit is the
# room the bed keeps under voice peaks (smaller gap = deeper bed dips).
VO_DECKE_DBTP = -1.5                      # VO limiter ceiling in the final (mastered) scale
BETT_GRENZE_DBTP = -1.2                   # voice + bed must stay below this; only the bed gives way
# SFX sit a little lower while the voice speaks (broadband, same 60/350 ms phrase envelope):
# keeps the voice in front and leaves the bed-give-way stage less to do under words.
SFX_UNTER_SPRACHE_DB = {"sfx_whoosh": 0.0, "sfx_hit": -3.0, "sfx_ui": -2.0, "sfx_uebergang": -3.0,
                        "sfx_signatur": -2.0, "sfx_atmo": 0.0}
VO_ROTATOR = ((100.0, 0.6), (160.0, 0.6), (250.0, 0.6), (400.0, 0.6), (630.0, 0.6), (1000.0, 0.6))
VO_LIMITER_VOR_S = 0.001
VO_LIMITER_HALTEN_S = 0.001
VO_REGELN = {"aktiv_max_prozent": 1.0, "gr_max_db": 6.0, "st_abweichung_max_lu": 0.3, "lra_verlust_max_lu": 1.0}
GR_AKTIV_DB = 0.01


# ------------------------------------------------------------------ data

@dataclass
class Ziele:
    stems: Path
    referenz: Path
    pegel: Path
    partitur: Path
    bibliothek: Path

    @classmethod
    def unter(cls, basis: Path) -> "Ziele":
        basis = Path(basis)
        return cls(stems=basis / "assets" / "audio" / "stems",
                   referenz=basis / "out" / "mix_referenz.wav",
                   pegel=basis / "work" / "ton" / "pegel.json",
                   partitur=basis / "work" / "ton" / "partitur.json",
                   bibliothek=basis / "assets" / "audio" / "sfx_bibliothek")

    @classmethod
    def standard(cls) -> "Ziele":
        return cls.unter(WURZEL)


@dataclass
class Eingang:
    vo: np.ndarray                  # mono, placed, unity gain (raw files)
    musik: np.ndarray               # stereo music, as rendered
    sfx: dict[str, np.ndarray]      # stereo SFX category stems, as rendered


@dataclass
class MixErgebnis:
    stems: dict[str, np.ndarray]
    master: np.ndarray
    pegel: dict
    sicherheit: np.ndarray          # linked master safety gain (1.0 = untouched)
    vo_limiter: np.ndarray          # VO limiter gain (1.0 = untouched)
    vo_rotiert: np.ndarray          # VO after phase rotator and static gain, before the limiter
    master_db: float                # exact master gain (pegel.json holds a rounded copy)
    eingang: Eingang
    partitur: musik.Partitur | None = None
    musik_spuren: dict[str, np.ndarray] = field(default_factory=dict)


# ------------------------------------------------------------------ voice

def _vo_pfad(datei: str, basis: Path) -> Path:
    p = Path(datei)
    return p if p.is_absolute() else Path(basis) / p


def zeilen_mit_ton(timeline: dict) -> list[dict]:
    """VO lines that have a recording. Lines with datei=null or vorlaeufig=true are skipped
    (with a warning), so a preliminary timeline never crashes the build."""
    out = []
    for z in timeline.get("vo", []):
        if not z.get("datei") or z.get("vorlaeufig") is True:
            warnings.warn(f"VO {z.get('id')} übersprungen (datei={z.get('datei')!r}, "
                          f"vorlaeufig={z.get('vorlaeufig')!r})", stacklevel=2)
            continue
        out.append(z)
    return out


def _nur_mit_ton(timeline: dict) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {**timeline, "vo": zeilen_mit_ton(timeline)}


def vo_laden(timeline: dict, basis: Path = WURZEL) -> np.ndarray:
    """All VO lines placed at their start_s (mono, unity gain). Raises if a file is missing."""
    n = n_samples(timeline["dauer_s"])
    zeilen = zeilen_mit_ton(timeline)
    fehlend = [str(_vo_pfad(z["datei"], basis)) for z in zeilen if not _vo_pfad(z["datei"], basis).is_file()]
    if fehlend:
        raise FileNotFoundError("VO-Datei(en) fehlen: " + ", ".join(fehlend))
    vo = np.zeros(n)
    for z in zeilen:
        x = wav_lesen_mono(_vo_pfad(z["datei"], basis))
        a = int(round(z["start_s"] * SR))
        b = min(n, a + len(x))
        if b > a:
            vo[a:b] += x[: b - a]
    return vo


def ersatz_vo_schreiben(timeline: dict, basis: Path) -> list[Path]:
    """Speech-like stand-in voice (partly phase-aligned glottal harmonics through moving formants,
    consonant noise, syllable level spread). Calibrated against the real TTS voice on the number
    that drives the mix: the limiter threshold at which the voice rules just hold (real 15.6 dB,
    stand-in ~15.5 dB above loudness). Its single highest peak is lower than the real one.
    Written where the timeline expects the files. For tests and previews. Deterministic."""
    import soundfile as sf
    dateien = []
    for zi, z in enumerate(timeline.get("vo", [])):
        if not z.get("datei"):
            continue
        rng = np.random.default_rng(7000 + zi)
        laenge = z["ende_s"] - z["start_s"] + 0.05
        n = int(laenge * SR)
        y = np.zeros(n)
        woerter = z["woerter"]
        for wi, w in enumerate(woerter):
            a = int((w["start_s"] - z["start_s"]) * SR)
            m = max(1, int((w["ende_s"] - w["start_s"]) * SR))
            silben = max(1, int(round((m / SR) / 0.17)))
            f0_basis = 128.0 - 26.0 * wi / max(1, len(woerter) - 1)  # declination over the line
            grenzen = np.linspace(0, m, silben + 1).astype(int)
            for s in range(silben):
                s0, s1 = grenzen[s], grenzen[s + 1]
                k = s1 - s0
                t = np.arange(k) / SR
                f0 = f0_basis * (1.0 + 0.06 * np.sin(np.pi * t / max(t[-1], 1e-3))) * rng.uniform(0.95, 1.06)
                phase = 2 * np.pi * np.cumsum(f0) / SR
                formanten = (rng.uniform(320, 800), rng.uniform(900, 2200), rng.uniform(2400, 3000))
                stimm = np.zeros(k)
                ph_zufall = rng.uniform(0, 2 * np.pi, 200)
                for h in range(1, int(4200 / f0_basis)):
                    fh = h * f0_basis
                    gewicht = sum(np.exp(-0.5 * ((fh - fm) / bw) ** 2) * g
                                  for fm, bw, g in zip(formanten, (110, 170, 260), (1.0, 0.6, 0.3)))
                    # half-coherent phases: glottal pulses, but not every period a full-scale spike
                    stimm += (gewicht + 0.02) / h ** 0.3 * np.cos(h * phase + 0.5 * ph_zufall[h])
                huelle = np.clip(t / 0.025, 0, 1) * np.clip((k / SR - t) / 0.04, 0, 1)
                huelle *= float(np.clip(np.exp(rng.normal(0.0, 0.45)), 0.25, 2.2))  # stress / level spread
                stimm *= huelle
                konsonant = int(min(k, rng.uniform(0.025, 0.06) * SR))
                rausch = rng.standard_normal(konsonant) * np.hanning(konsonant) * 0.25
                stimm[:konsonant] += bp0(rausch, 2500.0, 7000.0, 2)
                y[a + s0: a + s0 + k] += stimm[: max(0, n - (a + s0))]
        y *= von_db(-3.0) / (np.abs(y).max() + 1e-12)
        pfad = _vo_pfad(z["datei"], basis)
        pfad.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(pfad), y.astype(np.float32), SR, subtype="PCM_24")
        dateien.append(pfad)
    return dateien


def _allpass_koeff(f: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    w = 2 * np.pi * f / SR
    al = np.sin(w) / (2 * q)
    c = np.cos(w)
    b = np.array([1 - al, -2 * c, 1 + al]) / (1 + al)
    a = np.array([1 + al, -2 * c, 1 - al]) / (1 + al)
    return b, a


def phasenrotator(x: np.ndarray) -> np.ndarray:
    """All-pass cascade (|H| = 1 at every frequency): lowers the crest factor, changes nothing audible."""
    y = np.asarray(x, dtype=np.float64)
    for f, q in VO_ROTATOR:
        b, a = _allpass_koeff(f, q)
        y = lfilter(b, a, y)
    return y


def vo_limiter_gain(x: np.ndarray, decke_dbtp: float, huelle: np.ndarray | None = None) -> np.ndarray:
    """Transparent true-peak look-ahead limiter gain (<= 1) for a mono signal.
    Window [i - hold, i + look-ahead] on the 4x true-peak requirement, then a triangular
    smoothing whose half-width fits inside that window, so the gain never exceeds the
    requirement (the ceiling holds) and has no release tail. `huelle` = precomputed
    true-peak envelope of x (the voice is limited at several ceilings while mastering)."""
    huelle = true_peak_huelle(x) if huelle is None else huelle
    noetig = np.minimum(1.0, von_db(decke_dbtp) / np.maximum(huelle, 1e-12))
    if noetig.min() >= 1.0:
        return np.ones(len(x))
    vor, halten = int(VO_LIMITER_VOR_S * SR), int(VO_LIMITER_HALTEN_S * SR)
    groesse = vor + halten + 1
    g = minimum_filter1d(noetig, size=groesse, origin=halten - groesse // 2, mode="nearest")
    w = max(1, min(vor, halten) // 2)
    g = uniform_filter1d(uniform_filter1d(g, 2 * w + 1, mode="nearest"), 2 * w + 1, mode="nearest")
    return np.minimum(g, noetig)


def vo_limiter_metriken(vor: np.ndarray, gain: np.ndarray, sprach_maske: np.ndarray) -> dict:
    """Rule measurements for everything that reduces the voice (limiter x linked safety),
    on the VO alone, before/after. The 'active' share is counted over the spoken words only."""
    nach = vor * gain
    gr_db = -20 * np.log10(np.maximum(gain, 1e-12))
    sprache = max(1, int(np.sum(sprach_maske)))
    aktiv = float(np.sum((gr_db > GR_AKTIV_DB) & sprach_maske)) / sprache * 100
    st0, st1 = kurzzeit_lufs(vor), kurzzeit_lufs(nach)
    integ = lufs_integriert(np.stack([vor, vor], axis=1)) - 3.0103  # mono reference
    gueltig = st0 > max(-70.0, integ - 20.0)
    st_abw = float(np.mean(np.abs(st1[gueltig] - st0[gueltig]))) if np.any(gueltig) else 0.0
    lra0, lra1 = lautheitsspanne(vor), lautheitsspanne(nach)
    m = {"aktiv_prozent_sprache": round(aktiv, 3),
         "sprache_s": round(sprache / SR, 2),
         "aktiv_prozent_gesamt": round(float(np.mean(gr_db > GR_AKTIV_DB)) * 100, 3),
         "gr_max_db": round(float(gr_db.max()), 3),
         "st_abweichung_lu": round(st_abw, 3),
         "lra_vor_lu": round(lra0, 3), "lra_nach_lu": round(lra1, 3),
         "lra_verlust_lu": round(lra0 - lra1, 3)}
    m["regeln"] = {"aktiv_unter_1_prozent": aktiv < VO_REGELN["aktiv_max_prozent"],
                   "gr_max_6_db": m["gr_max_db"] <= VO_REGELN["gr_max_db"],
                   "st_abweichung_unter_0_3_lu": st_abw < VO_REGELN["st_abweichung_max_lu"],
                   "lra_verlust_max_1_lu": m["lra_verlust_lu"] <= VO_REGELN["lra_verlust_max_lu"]}
    m["regeln_erfuellt"] = all(m["regeln"].values())
    return m


def _sprach_maske(timeline: dict, n: int) -> np.ndarray:
    """Samples inside spoken words (not whole lines: pauses inside a line are not speech)."""
    maske = np.zeros(n, dtype=bool)
    for a, b in wort_intervalle(timeline):
        maske[max(0, int(a * SR)): max(0, int(b * SR))] = True
    return maske


def _regel_schwelle(vo: np.ndarray, maske: np.ndarray, huelle: np.ndarray) -> float:
    """Smallest limiter threshold (dB above the VO's own integrated loudness) that keeps all
    four rules. Bisection (0.04 dB resolution); the rules get easier with a higher threshold."""
    l_vo = lufs_integriert(np.stack([vo, vo], axis=1)) - 3.0103  # mono reference, like _mastern
    ok = lambda t: vo_limiter_metriken(vo, vo_limiter_gain(vo, l_vo + t, huelle), maske)["regeln_erfuellt"]  # noqa
    lo, hi = 8.0, 26.0
    if not ok(hi):
        return hi
    for _ in range(9):
        mitte = (lo + hi) / 2
        if ok(mitte):
            hi = mitte
        else:
            lo = mitte
    return hi


# ------------------------------------------------------------------ processing

def sprachband_ducken(x: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """Dynamic EQ: the 2-4 kHz band of x follows gain_db (per sample, <= 0 dB).
    Complementary zero-phase split, so 0 dB gives back x exactly."""
    if not np.any(gain_db < 0):
        return x.copy()
    band = bp0(x, *SPRACHBAND_HZ, 4)
    return x + band * (von_db(gain_db) - 1.0)[:, None]


def _gain_aus_bedarf(noetig: np.ndarray, vor_s: float = 0.0015, halten_s: float = 0.05,
                     release_s: float = 0.08) -> np.ndarray:
    """Smooth gain <= the per-sample requirement: look-ahead, hold, exponential release."""
    if noetig.min() >= 1.0:
        return np.ones(len(noetig))
    vor, halten = int(vor_s * SR), int(halten_s * SR)
    # output i sees the window [i - halten, i + vor] (origin verified empirically: g//2 + origin = halten)
    groesse = vor + halten + 1
    g = minimum_filter1d(noetig, size=groesse, origin=halten - groesse // 2, mode="nearest")
    g = uniform_filter1d(g, size=2 * vor + 1, mode="nearest")
    g = np.minimum(g, noetig)
    gr = 1.0 - g
    idx = np.nonzero(gr > 1e-9)[0]
    if idx.size == 0:
        return g
    a = np.exp(-1.0 / (release_s * SR))
    schweif = int(7 * release_s * SR)
    out = gr.copy()
    for ber in np.split(idx, np.nonzero(np.diff(idx) > schweif)[0] + 1):
        vorher = 0.0
        for i in range(int(ber[0]), min(len(gr), int(ber[-1]) + schweif)):
            vorher = max(gr[i], vorher * a)
            out[i] = vorher
    return 1.0 - out


def sicherheits_gain(x: np.ndarray, schwelle_dbtp: float = SICHERHEIT_SCHWELLE_DBTP) -> np.ndarray:
    """Linked master safety gain: only where the 4x true-peak exceeds the threshold."""
    return _gain_aus_bedarf(np.minimum(1.0, von_db(schwelle_dbtp) / np.maximum(true_peak_huelle(x), 1e-12)))


def _ueberabgetastet_bloecke(x: np.ndarray, block_s: float = 5.0, rand_s: float = 0.05):
    """Yield (start, end, 4x-oversampled block) for a (n, ch) signal, block-wise to save memory."""
    from scipy.signal import resample_poly
    n, blk, rand = len(x), int(block_s * SR), int(rand_s * SR)
    for a in range(0, n, blk):
        b = min(n, a + blk)
        a0, b0 = max(0, a - rand), min(n, b + rand)
        os = resample_poly(x[a0:b0], 4, 1, axis=0)
        yield a, b, os[(a - a0) * 4:(a - a0) * 4 + (b - a) * 4]


def bett_platz_gain(vo_st: np.ndarray, bett: np.ndarray, grenze_dbtp: float = BETT_GRENZE_DBTP) -> np.ndarray:
    """Gain for the bed only, so that voice + bed stay below the true-peak limit.
    Exact on the 4x-oversampled sum: only where |vo + bett| > c the bed is scaled to
    g = (s*c - vo) / bett (s = sign of the sum), which exists in [0, 1] because the voice
    itself stays below c. Between the lines (vo = 0) this is the bed's own peak limiter.
    The voice is never touched."""
    c = von_db(grenze_dbtp)
    noetig = np.ones(len(bett))
    beide = np.concatenate([vo_st, bett], axis=1)
    for a, b, os in _ueberabgetastet_bloecke(beide):
        vo_os, bett_os = os[:, :2], os[:, 2:]
        summe = vo_os + bett_os
        ueber = np.abs(summe) > c
        if not ueber.any():
            continue
        g = np.ones_like(summe)
        s = np.sign(summe[ueber])
        g[ueber] = np.clip((s * c - vo_os[ueber]) / np.where(np.abs(bett_os[ueber]) > 1e-12, bett_os[ueber], 1e-12),
                           0.0, 1.0)
        noetig[a:b] = g.min(axis=1).reshape(b - a, 4).min(axis=1)
    return _gain_aus_bedarf(noetig, vor_s=0.0015, halten_s=0.005, release_s=0.025)


def _vo_referenz(pm_vo: Pegelmesser, a: float, b: float) -> float:
    """The voice level a word is compared against: the lower of the level inside the word and
    the EBU short-term level (3 s) around it, so both readings of 'Kurzzeitpegel' hold."""
    mitte = (a + b) / 2
    return min(pm_vo.pegel(a, b), pm_vo.pegel(mitte - 1.5, mitte + 1.5))


def _duck_phrasen(timeline: dict, vo: np.ndarray, musik_st: np.ndarray) -> list[dict]:
    """Per phrase the duck depth that keeps the music DUCK_ABSTAND + reserve below every word."""
    pm_vo = Pegelmesser(np.stack([vo, vo], axis=1))
    pm_mu = Pegelmesser(musik_st)
    woerter = sorted(wort_intervalle(timeline))
    liste = []
    for a, b in phrasen(timeline):
        noetig = DUCK_MIN_DB
        for ws, we in woerter:
            if ws >= a - 1e-6 and we <= b + 1e-6:
                abstand = _vo_referenz(pm_vo, ws, we) - pm_mu.pegel(ws, we)
                noetig = min(noetig, abstand - DUCK_ABSTAND_DB - DUCK_RESERVE_DB)
        liste.append({"start_s": a, "ende_s": b, "tiefe_db": max(DUCK_MAX_DB, noetig)})
    return liste


def _wort_abstaende(timeline: dict, vo_st: np.ndarray, musik_st: np.ndarray) -> list[tuple[float, float, float]]:
    """(word start, VO-in-word minus music, VO-short-term minus music) for every word."""
    pm_vo, pm_mu = Pegelmesser(vo_st), Pegelmesser(musik_st)
    out = []
    for ws, we in wort_intervalle(timeline):
        mu = pm_mu.pegel(ws, we)
        mitte = (ws + we) / 2
        out.append((ws, pm_vo.pegel(ws, we) - mu, pm_vo.pegel(mitte - 1.5, mitte + 1.5) - mu))
    return out


def _mastern(vo_rot: np.ndarray, rest: np.ndarray, modus: str, maske: np.ndarray) -> dict:
    """Master gain + VO limiter + bed give-way + linked safety, iterated until -14 LUFS /
    <= -1 dBTP hold (modus 'brief'), or with the master capped so the voice limiter rules
    hold (modus 'regeln')."""
    vo_st_roh = np.stack([vo_rot, vo_rot], axis=1)
    vo_lufs_mono = lufs_integriert(vo_st_roh) - 3.0103
    huelle_vo = true_peak_huelle(vo_rot)
    master_db = ZIEL_LUFS - lufs_integriert(vo_st_roh + rest)
    deckel_db = None
    if modus == "regeln":
        deckel_db = VO_DECKE_DBTP - _regel_schwelle(vo_rot, maske, huelle_vo) - vo_lufs_mono
    elif modus != "brief":
        raise ValueError(f"unbekannter Modus {modus!r} (brief|regeln)")
    schwelle = SICHERHEIT_SCHWELLE_DBTP
    for _ in range(12):
        if deckel_db is not None:
            master_db = min(master_db, deckel_db)
        vo_gain = vo_limiter_gain(vo_rot, VO_DECKE_DBTP - master_db, huelle_vo)
        vo_st = np.stack([vo_rot * vo_gain] * 2, axis=1) * von_db(master_db)
        bett_gain = bett_platz_gain(vo_st, rest * von_db(master_db))
        pre = vo_st + rest * (bett_gain[:, None] * von_db(master_db))
        sicherheit = sicherheits_gain(pre, schwelle)
        out = pre * sicherheit[:, None]
        lufs, tp = lufs_integriert(out), true_peak_db(out)
        if tp > TP_GRENZE_DBTP:
            schwelle -= 0.2
            continue
        if deckel_db is not None and master_db >= deckel_db - 1e-9 and lufs <= ZIEL_LUFS + 0.05:
            break  # rules mode: the voice caps the master
        if abs(lufs - ZIEL_LUFS) <= 0.05:
            break
        master_db += ZIEL_LUFS - lufs
    return {"master_db": master_db, "vo_gain": vo_gain, "bett_gain": bett_gain, "sicherheit": sicherheit,
            "schwelle": schwelle, "deckel_db": deckel_db}


def grenzen_pruefen(pegel: dict) -> list[str]:
    """Hard limits of the deliverable. Empty list = everything holds."""
    v = []
    if abs(pegel["lufs_integriert"] - ZIEL_LUFS) > 0.5:
        v.append(f"Lautheit {pegel['lufs_integriert']} LUFS (Ziel {ZIEL_LUFS} +-0.5)")
    if pegel["true_peak_dbtp"] > TP_GRENZE_DBTP:
        v.append(f"True Peak {pegel['true_peak_dbtp']} dBTP > {TP_GRENZE_DBTP}")
    for k in ("duck_min_abstand_db", "duck_min_abstand_st3s_db"):
        if pegel[k] is not None and pegel[k] < DUCK_ABSTAND_DB:
            v.append(f"{k} {pegel[k]} dB < {DUCK_ABSTAND_DB}")
    lim = pegel.get("vo_limiter") or {}
    if pegel["modus"] == "regeln" and lim and not lim["regeln_erfuellt"]:
        v.append(f"VO-Limiter-Regeln verletzt: {lim['regeln']}")
    return v


def mischen(vo: np.ndarray, musik_st: np.ndarray, sfx_stems: dict[str, np.ndarray], timeline: dict,
            modus: str = "regeln") -> MixErgebnis:
    """Measure everything, duck, sum, master. Pure function of its inputs (no cached gains)."""
    tl = _nur_mit_ton(timeline)
    n = len(vo)
    maske = _sprach_maske(tl, n)
    hat_vo = bool(np.any(vo != 0))
    vo_rot = phasenrotator(vo)
    vo_lufs_roh = lufs_integriert(np.stack([vo, vo], axis=1)) if hat_vo else float("-inf")
    g_vo = (VO_ZIEL_LUFS - lufs_integriert(np.stack([vo_rot, vo_rot], axis=1))) if hat_vo else 0.0
    vo_rot = vo_rot * von_db(g_vo)
    mu_lufs_roh = lufs_integriert(musik_st)
    g_mu = MUSIK_ZIEL_LUFS - mu_lufs_roh
    mu_g = musik_st * von_db(g_mu)

    sprach_phrasen = phrasen(tl)
    band_db = trapez_db(n, [(a, b, SPRACHBAND_DB) for a, b in sprach_phrasen])
    sfx_g = {}
    for name in SFX_STEMS:
        x = sfx_stems[name] * von_db(SFX_GAINS_DB[name])
        if SFX_UNTER_SPRACHE_DB[name] < 0:
            x = x * von_db(trapez_db(n, [(a, b, SFX_UNTER_SPRACHE_DB[name]) for a, b in sprach_phrasen]))[:, None]
        if name in ("sfx_atmo", "sfx_whoosh"):
            x = sprachband_ducken(x, band_db)
        sfx_g[name] = x

    phrasen_liste = _duck_phrasen(tl, vo_rot, mu_g)
    for runde in range(4):
        duck_db = trapez_db(n, [(p["start_s"], p["ende_s"], p["tiefe_db"]) for p in phrasen_liste])
        mu_d = mu_g * von_db(duck_db)[:, None]
        rest = mu_d + sum(sfx_g.values())
        m = _mastern(vo_rot, rest, modus, maske)
        faktor = von_db(m["master_db"]) * m["sicherheit"][:, None]
        bett = m["bett_gain"][:, None]
        vor_master = {"vo": np.stack([vo_rot * m["vo_gain"]] * 2, axis=1), "musik": mu_d * bett,
                      **{k: v * bett for k, v in sfx_g.items()}}
        stems = {name: vor_master[name] * faktor for name in STEMS}
        abst = _wort_abstaende(tl, stems["vo"], stems["musik"])
        zu_knapp = [t for t, d1, d2 in abst if min(d1, d2) < DUCK_ABSTAND_DB + 0.5]
        if not zu_knapp or runde == 3:
            break  # phrasen_liste now holds exactly the depths that were applied
        for p in phrasen_liste:  # the limiter lowered a few words: deepen exactly those phrases
            if any(p["start_s"] - 1e-6 <= t <= p["ende_s"] for t in zu_knapp):
                p["tiefe_db"] = max(DUCK_MAX_DB, p["tiefe_db"] - 3.0)
    master = sum(stems[name] for name in STEMS)

    lim = vo_limiter_metriken(vo_rot, m["vo_gain"] * m["sicherheit"], maske) if hat_vo else {}
    d_wort = [d for _, d, _ in abst]
    d_st3 = [d for _, _, d in abst]
    pegel = {
        "modus": modus,
        "lufs_integriert": round(lufs_integriert(master), 3),
        "true_peak_dbtp": round(true_peak_db(master), 3),
        "master_gain_db": round(float(m["master_db"]), 3),
        "master_deckel_db": None if m["deckel_db"] is None else round(float(m["deckel_db"]), 3),
        "gains_db": {"vo": round(g_vo + m["master_db"], 3), "musik": round(g_mu + m["master_db"], 3),
                     **{k: round(SFX_GAINS_DB[k] + m["master_db"], 3) for k in SFX_STEMS}},
        "sfx_unter_sprache_db": SFX_UNTER_SPRACHE_DB,
        "roh_lufs": {"vo": round(vo_lufs_roh, 3), "musik": round(mu_lufs_roh, 3),
                     **{k: round(lufs_integriert(sfx_stems[k]), 3) for k in SFX_STEMS}},
        "stem_lufs": {k: round(lufs_integriert(stems[k]), 3) for k in STEMS},
        "duck_min_abstand_db": round(min(d_wort), 3) if d_wort else None,
        "duck_min_abstand_st3s_db": round(min(d_st3), 3) if d_st3 else None,
        "duck_median_abstand_db": round(float(np.median(d_wort)), 3) if d_wort else None,
        "duck_phrasen": [{k: round(v, 3) for k, v in p.items()} for p in phrasen_liste],
        "handy_anteil": round(handy_anteil(master), 4),
        "vo_kette": {"phasenrotator_hz_q": [list(p) for p in VO_ROTATOR],
                     "limiter_decke_dbtp": VO_DECKE_DBTP,
                     "limiter_vor_ms": VO_LIMITER_VOR_S * 1000, "limiter_halten_ms": VO_LIMITER_HALTEN_S * 1000,
                     "kompressor": None,
                     "plr_roh_db": round(true_peak_db(vo) - (vo_lufs_roh - 3.0103), 3) if hat_vo else None,
                     "plr_rotiert_db": (round(true_peak_db(vo_rot) - (lufs_integriert(
                         np.stack([vo_rot, vo_rot], axis=1)) - 3.0103), 3) if hat_vo else None)},
        "vo_limiter": lim,
        "bett_platz": {"grenze_dbtp": BETT_GRENZE_DBTP,
                       "aktiv_s": round(float(np.sum(m["bett_gain"] < 0.99885)) / SR, 3),
                       "ueber_3db_s": round(float(np.sum(m["bett_gain"] < von_db(-3.0))) / SR, 3),
                       "ueber_10db_s": round(float(np.sum(m["bett_gain"] < von_db(-10.0))) / SR, 3),
                       "gr_max_db": round(-20 * np.log10(max(float(m["bett_gain"].min()), 1e-6)), 3)},
        "sicherheit_schwelle_dbtp": round(m["schwelle"], 2),
        "sicherheit_aktiv_s": round(float(np.sum(m["sicherheit"] < 1.0)) / SR, 4),
        "sicherheit_max_db": round(20 * np.log10(float(m["sicherheit"].min())), 3),
    }
    pegel["verstoesse"] = grenzen_pruefen(pegel)
    return MixErgebnis(stems, master, pegel, m["sicherheit"], m["vo_gain"], vo_rot, float(m["master_db"]),
                       Eingang(vo, musik_st, sfx_stems))


# ------------------------------------------------------------------ build

class MixFehler(RuntimeError):
    """The mix misses a hard limit (loudness, true peak, duck distance, voice limiter rules)."""


def bauen(timeline_pfad: Path, basis: Path = WURZEL, ziele: Ziele | None = None,
          modus: str = "regeln") -> MixErgebnis:
    """Render, mix, check the hard limits, then write stems, reference, library and reports.
    On a violation only pegel.json is written (with 'verstoesse') and MixFehler is raised,
    so no stem that misses the spec ever lands in assets/ for the Resolve builder."""
    ziele = ziele or Ziele.standard()
    t0 = time.perf_counter()
    tl = lade_timeline(timeline_pfad)
    vo = vo_laden(tl, basis)  # fail fast before the heavy work
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_musik = pool.submit(musik.musik_rendern, tl)
        f_sfx = pool.submit(sfx.kategorien_rendern, tl)
        mu, sfx_stems = f_musik.result(), f_sfx.result()
    erg = mischen(vo, mu.mix, sfx_stems, tl, modus=modus)
    erg.partitur, erg.musik_spuren = mu.partitur, mu.spuren

    pegel = dict(erg.pegel)
    pegel["timeline"] = str(timeline_pfad)
    pegel["vorlaeufig"] = bool(tl.get("vorlaeufig"))
    if not pegel["verstoesse"]:
        for name in STEMS:
            wav_schreiben(ziele.stems / f"{name}.wav", erg.stems[name])
        wav_schreiben(ziele.referenz, erg.master)
        sfx.bibliothek_schreiben(ziele.bibliothek)
    pegel["laufzeit_s"] = round(time.perf_counter() - t0, 2)
    for pfad, daten in ((ziele.pegel, pegel), (ziele.partitur, mu.partitur.als_dict())):
        pfad.parent.mkdir(parents=True, exist_ok=True)
        pfad.write_text(json.dumps(daten, ensure_ascii=False, indent=1) + "\n")
    erg.pegel = pegel
    if pegel["verstoesse"]:
        raise MixFehler("; ".join(pegel["verstoesse"]) + f" (Details: {ziele.pegel})")
    return erg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build SFX, music, stems and the reference mix")
    ap.add_argument("--timeline", type=Path, default=WURZEL / "timeline.json")
    ap.add_argument("--modus", choices=("regeln", "brief"), default="regeln",
                    help="regeln (default): VO limiter rules hold, master capped if needed; "
                         "brief: -14 LUFS exactly, limiter rules only reported")
    ap.add_argument("--ersatz-vo", action="store_true",
                    help="preview with a synthetic stand-in voice; writes to work/ton/vorschau/")
    args = ap.parse_args(argv)
    if not args.timeline.is_file():
        print(f"{args.timeline} fehlt (wartet auf Task 1). Vorschau: --timeline "
              f"tests/fixtures/timeline_audio.json --ersatz-vo", file=sys.stderr)
        return 2
    if args.ersatz_vo:
        basis = WURZEL / "work" / "ton" / "vorschau"
        ersatz_vo_schreiben(lade_timeline(args.timeline), basis)
        ziele = Ziele.unter(basis)
    else:
        basis, ziele = WURZEL, Ziele.standard()
    try:
        erg = bauen(args.timeline, basis=basis, ziele=ziele, modus=args.modus)
    except MixFehler as fehler:
        print(f"MIX VERLETZT GRENZEN, keine Stems geschrieben: {fehler}", file=sys.stderr)
        return 3
    p = erg.pegel
    print(f"Referenz {ziele.referenz}  (Modus {p['modus']})")
    print(f"  {p['lufs_integriert']:.2f} LUFS  {p['true_peak_dbtp']:.2f} dBTP  Master {p['master_gain_db']:+.2f} dB")
    print(f"  Duck-Abstand min {p['duck_min_abstand_db']} dB im Wort, {p['duck_min_abstand_st3s_db']} dB "
          f"gegen Kurzzeitpegel 3 s (Median {p['duck_median_abstand_db']})  Handy-Anteil {p['handy_anteil']:.3f}")
    lim = p["vo_limiter"]
    if lim:
        print(f"  VO-Limiter: aktiv {lim['aktiv_prozent_sprache']} % der Wörter, max {lim['gr_max_db']} dB, "
              f"ST-Abw. {lim['st_abweichung_lu']} LU, LRA {lim['lra_vor_lu']} -> {lim['lra_nach_lu']} LU, "
              f"Regeln erfüllt: {lim['regeln_erfuellt']}")
    b = p["bett_platz"]
    print(f"  Bett weicht aus: {b['aktiv_s']} s (> 3 dB {b['ueber_3db_s']} s, > 10 dB {b['ueber_10db_s']} s, "
          f"max {b['gr_max_db']} dB)")
    print(f"  Sicherheitsbegrenzung aktiv {p['sicherheit_aktiv_s']} s, max {p['sicherheit_max_db']} dB")
    print(f"  Laufzeit {p['laufzeit_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
