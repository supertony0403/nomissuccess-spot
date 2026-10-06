"""Shared audio constants and DSP helpers for sfx.py, musik.py and mix.py (Task 2).

Deliberately independent of gemeinsam.py (Task 1): the sound engine only needs
the timeline contract, wav io and a few filters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, oaconvolve, resample_poly, sosfilt, sosfiltfilt

SR = 48_000
WURZEL = Path(__file__).resolve().parent.parent

STEMS = ("vo", "musik", "sfx_whoosh", "sfx_hit", "sfx_ui", "sfx_uebergang", "sfx_signatur", "sfx_atmo")
SFX_STEMS = STEMS[2:]

SFX_KATEGORIE = {
    "whoosh": "sfx_whoosh",
    "hit": "sfx_hit", "subdrop": "sfx_hit",
    "tick": "sfx_ui", "click": "sfx_ui", "lock": "sfx_ui", "paper": "sfx_ui", "clatter": "sfx_ui",
    "riser": "sfx_uebergang", "downer": "sfx_uebergang", "rewind": "sfx_uebergang",
    "swell": "sfx_uebergang", "shimmer": "sfx_uebergang",
    "sting": "sfx_signatur",
}

# Speech phrases: words closer than this are one phrase for ducking.
PHRASE_LUECKE_S = 0.6
DUCK_ATTACK_S = 0.060
DUCK_RELEASE_S = 0.350


# ------------------------------------------------------------------ timeline

def lade_timeline(pfad: Path) -> dict:
    return json.loads(Path(pfad).read_text(encoding="utf-8"))


def n_samples(dauer_s: float) -> int:
    return int(round(dauer_s * SR))


def wort_intervalle(timeline: dict) -> list[tuple[float, float]]:
    return [(w["start_s"], w["ende_s"]) for z in timeline.get("vo", []) for w in z["woerter"]]


def phrasen(timeline: dict, luecke_s: float = PHRASE_LUECKE_S) -> list[tuple[float, float]]:
    """Word intervals merged into phrases (gaps shorter than luecke_s are bridged)."""
    out: list[list[float]] = []
    for a, b in sorted(wort_intervalle(timeline)):
        if out and a - out[-1][1] < luecke_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def trapez_db(n: int, phrasen_tiefe: list[tuple[float, float, float]],
              attack_s: float = DUCK_ATTACK_S, release_s: float = DUCK_RELEASE_S) -> np.ndarray:
    """Gain curve in dB (<= 0): per phrase (start, end, depth_db) a trapezoid that is fully
    down at `start` (linear ramp over attack_s before it) and recovers over release_s after `end`.
    Overlapping phrases take the deeper value (min)."""
    g = np.zeros(n)
    t = np.arange(n) / SR
    for a, b, tiefe in phrasen_tiefe:
        if tiefe >= 0:
            continue
        i0 = max(0, int(np.floor((a - attack_s) * SR)))
        i1 = min(n, int(np.ceil((b + release_s) * SR)) + 1)
        if i0 >= i1:
            continue
        tt = t[i0:i1]
        form = np.ones_like(tt)
        auf = tt < a
        form[auf] = np.clip((tt[auf] - (a - attack_s)) / attack_s, 0.0, 1.0)
        ab = tt > b
        form[ab] = np.clip(1.0 - (tt[ab] - b) / release_s, 0.0, 1.0)
        g[i0:i1] = np.minimum(g[i0:i1], tiefe * form)
    return g


# ------------------------------------------------------------------ wav io

def wav_lesen_mono(pfad: Path) -> np.ndarray:
    daten, sr = sf.read(str(pfad), always_2d=True, dtype="float64")
    mono = daten.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        mono = resample_poly(mono, SR // g, sr // g)
    return mono


def wav_schreiben(pfad: Path, daten: np.ndarray) -> None:
    """32-bit float WAV, 48 kHz: exact, so stems at 0 dB sum to the reference mix.
    Written with scipy, not libsndfile: libsndfile adds a PEAK chunk with a timestamp to
    float WAVs, which makes identical audio produce different files (breaks reproducibility)."""
    from scipy.io import wavfile
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(pfad), SR, np.ascontiguousarray(daten, dtype=np.float32))


# ------------------------------------------------------------------ levels

def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def von_db(d: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(d) / 20.0)


# ITU-R BS.1770 K-weighting coefficients for exactly 48 kHz
_K1_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K1_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K2_B = np.array([1.0, -2.0, 1.0])
_K2_A = np.array([1.0, -1.99004745483398, 0.99007225036621])


def k_gewichtet(x: np.ndarray) -> np.ndarray:
    y = lfilter(_K1_B, _K1_A, x, axis=0)
    return lfilter(_K2_B, _K2_A, y, axis=0)


class Pegelmesser:
    """K-weighted levels (LUFS-like, ungated) over arbitrary intervals via cumulative sums."""

    def __init__(self, x: np.ndarray):
        y = k_gewichtet(np.atleast_2d(x.T).T)
        energie = np.sum(y ** 2, axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(energie)])

    def pegel(self, a_s: float, b_s: float) -> float:
        i0 = max(0, int(a_s * SR))
        i1 = min(len(self._cum) - 1, max(i0 + 1, int(b_s * SR)))
        m = (self._cum[i1] - self._cum[i0]) / (i1 - i0)
        return -0.691 + 10.0 * np.log10(m + 1e-20)


def lufs_integriert(x: np.ndarray) -> float:
    import pyloudnorm as pyln
    return float(pyln.Meter(SR).integrated_loudness(np.atleast_2d(x.T).T))


def kurzzeit_lufs(x: np.ndarray, hop_s: float = 0.1) -> np.ndarray:
    """Short-term loudness (3 s K-weighted window, EBU R128) every hop_s seconds."""
    x2 = np.atleast_2d(x.T).T
    y = np.sum(k_gewichtet(x2) ** 2, axis=1)
    e = np.concatenate([[0.0], np.cumsum(y)])
    w, h = int(3.0 * SR), max(1, int(hop_s * SR))
    if len(y) <= w:
        return np.array([-0.691 + 10 * np.log10(e[-1] / max(1, len(y)) + 1e-20)])
    idx = np.arange(0, len(y) - w + 1, h)
    return -0.691 + 10 * np.log10((e[idx + w] - e[idx]) / w + 1e-20)


def lautheitsspanne(x: np.ndarray) -> float:
    """Loudness range (LRA, EBU Tech 3342): 10th..95th percentile of gated short-term loudness."""
    st = kurzzeit_lufs(x)
    st = st[st > -70.0]
    if len(st) < 2:
        return 0.0
    rel = 10 * np.log10(np.mean(10 ** (st / 10))) - 20.0
    st = st[st > rel]
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def true_peak_huelle(x: np.ndarray) -> np.ndarray:
    """Per-sample true-peak envelope (4x oversampling, max over channels and sub-samples)."""
    x2 = np.atleast_2d(x.T).T
    os = np.abs(resample_poly(x2, 4, 1, axis=0)).max(axis=1)
    n = x2.shape[0]
    os = os[: n * 4]
    if len(os) < n * 4:
        os = np.pad(os, (0, n * 4 - len(os)))
    return os.reshape(n, 4).max(axis=1)


def true_peak_db(x: np.ndarray) -> float:
    return db(float(true_peak_huelle(x).max()))


def handy_anteil(x: np.ndarray) -> float:
    """RMS share above 400 Hz (amplitude ratio): phone-speaker test, must be >= 0.45."""
    mono = np.atleast_2d(x.T).T.mean(axis=1)
    hoch = sosfiltfilt(butter(8, 400, "highpass", fs=SR, output="sos"), mono)
    return float(np.sqrt(np.mean(hoch ** 2)) / (np.sqrt(np.mean(mono ** 2)) + 1e-20))


# ------------------------------------------------------------------ filters / dsp

@lru_cache(maxsize=256)
def _sos(art: str, f: float | tuple[float, float], ordnung: int) -> np.ndarray:
    return butter(ordnung, f, art, fs=SR, output="sos")


def hp(x: np.ndarray, f: float, ordnung: int = 2) -> np.ndarray:
    return sosfilt(_sos("highpass", float(f), ordnung), x, axis=0)


def lp(x: np.ndarray, f: float, ordnung: int = 2) -> np.ndarray:
    return sosfilt(_sos("lowpass", float(f), ordnung), x, axis=0)


def bp(x: np.ndarray, f1: float, f2: float, ordnung: int = 2) -> np.ndarray:
    return sosfilt(_sos("bandpass", (float(f1), float(f2)), ordnung), x, axis=0)


def hp0(x: np.ndarray, f: float, ordnung: int = 2) -> np.ndarray:
    """Zero-phase high-pass (no group delay, keeps transients where they are)."""
    return sosfiltfilt(_sos("highpass", float(f), ordnung), x, axis=0)


def lp0(x: np.ndarray, f: float, ordnung: int = 2) -> np.ndarray:
    return sosfiltfilt(_sos("lowpass", float(f), ordnung), x, axis=0)


def bp0(x: np.ndarray, f1: float, f2: float, ordnung: int = 2) -> np.ndarray:
    return sosfiltfilt(_sos("bandpass", (float(f1), float(f2)), ordnung), x, axis=0)


def pan(mono: np.ndarray, position: float | np.ndarray) -> np.ndarray:
    """Constant-power pan, position -1 (left) .. +1 (right); position may vary per sample."""
    w = (np.asarray(position) + 1.0) * np.pi / 4.0
    return np.stack([mono * np.cos(w), mono * np.sin(w)], axis=1)


def stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1)


def huelle_exp(n: int, tau_s: float, attack_s: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.exp(-t / max(tau_s, 1e-6))
    if attack_s > 0:
        e *= np.clip(t / attack_s, 0.0, 1.0)
    return e


def fade(x: np.ndarray, ein_s: float = 0.0, aus_s: float = 0.0) -> np.ndarray:
    y = x.copy()
    n = len(y)
    if ein_s > 0:
        k = min(n, int(ein_s * SR))
        r = np.sin(np.linspace(0, np.pi / 2, k)) ** 2
        y[:k] *= r[:, None] if y.ndim == 2 else r
    if aus_s > 0:
        k = min(n, int(aus_s * SR))
        r = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
        y[n - k:] *= r[:, None] if y.ndim == 2 else r
    return y


def raum_ir(rt60_s: float, seed: int, laenge_s: float | None = None, vorverz_s: float = 0.012,
            hell_hz: float = 6000.0, dunkel_faktor: float = 0.45) -> np.ndarray:
    """Synthetic stereo reverb impulse response: decorrelated noise, exponential decay,
    highs decay faster than lows (dunkel_faktor = high-band RT relative to low band)."""
    laenge_s = laenge_s or rt60_s * 1.1
    n = int(laenge_s * SR)
    rng = np.random.default_rng(seed)
    rausch = rng.standard_normal((n, 2))
    t = np.arange(n) / SR
    tau = rt60_s / 6.91
    tief = lp(rausch, 1800.0, 2) * np.exp(-t / tau)[:, None]
    hoch = hp(rausch, 1800.0, 2) * np.exp(-t / (tau * dunkel_faktor))[:, None]
    ir = lp(tief + hoch, hell_hz, 2)
    ir *= np.clip(t / 0.004, 0, 1)[:, None]  # soft onset, avoids a click
    k = int(vorverz_s * SR)
    ir = np.concatenate([np.zeros((k, 2)), ir])
    return ir / np.sqrt(np.sum(ir ** 2) / 2.0)


def falten(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Stereo convolution; output has the length of x (tail beyond x is dropped)."""
    x2 = np.atleast_2d(x.T).T
    if x2.shape[1] == 1:
        x2 = np.repeat(x2, 2, axis=1)
    y = np.stack([oaconvolve(x2[:, c], ir[:, c])[: len(x2)] for c in range(2)], axis=1)
    return y


def falten_mit_schweif(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    x2 = np.atleast_2d(x.T).T
    if x2.shape[1] == 1:
        x2 = np.repeat(x2, 2, axis=1)
    return np.stack([oaconvolve(x2[:, c], ir[:, c]) for c in range(2)], axis=1)


def momentan_max_lufs(x: np.ndarray) -> float:
    """Max momentary loudness (400 ms K-weighted window, 25 ms hop) of a short sound."""
    x2 = np.atleast_2d(x.T).T
    pad = np.zeros((int(0.4 * SR), x2.shape[1]))
    y = k_gewichtet(np.concatenate([pad, x2, pad]))
    e = np.concatenate([[0.0], np.cumsum(np.sum(y ** 2, axis=1))])
    w = int(0.4 * SR)
    hop = int(0.025 * SR)
    idx = np.arange(0, len(e) - w - 1, hop)
    m = (e[idx + w] - e[idx]) / w
    return float(-0.691 + 10 * np.log10(m.max() + 1e-20))


def auf_lautheit(x: np.ndarray, ziel_lufs: float) -> np.ndarray:
    return x * von_db(ziel_lufs - momentan_max_lufs(x))


@dataclass
class Klang:
    """A rendered sound: stereo samples plus the anchor sample that lands on the cue time.
    ankerart: 'transiente' (attack on t), 'ende' (sound ends on t), 'start' (begins on t)."""
    daten: np.ndarray
    anker: int
    ankerart: str


def platzieren(ziel: np.ndarray, klang_daten: np.ndarray, start: int, gain: float = 1.0) -> None:
    """Add klang_daten into ziel starting at sample `start` (may be negative / overrun)."""
    n = len(ziel)
    a, b = start, start + len(klang_daten)
    ka, kb = 0, len(klang_daten)
    if a < 0:
        ka, a = -a, 0
    if b > n:
        kb -= b - n
        b = n
    if a < b and ka < kb:
        ziel[a:b] += gain * klang_daten[ka:kb]
