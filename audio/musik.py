"""Original score for the spot, generated against timeline.json (Task 2).

~100 BPM, A minor, arranged along the night:
  s1  sub pulse + the clock tick as hi-hat, very sparse
  s2  build: plucks with sidechain pumping, muted kick, clap on 2/4 from bar 3
  s3  build+: brighter plucks in 16ths, 16th hats, kick opens up
  s4  thin: low drone with a minor-second rub, heartbeat sub, sparse ticks; E in the last bar
  s5  drive: full groove, rolling bass
  s6  drive+: groove with syncopated kick and a brighter top
  s7  lift: chords step upwards (F-G-Am-C), wide supersaw pads, plucks an octave up
  s8  warm resolve: soft groove, F-G cadence into A major; the last chord rings out
      under the end card (>= 3 s)

Scene starts sit on bar lines: every scene gets a whole number of beats and its own
tempo (95-105 BPM); the scene times themselves are never moved. Events with sfx
'hit'/'sting' get a musical accent on the nearest 1/16 if that is within +-40 ms,
otherwise exactly on the event.

CLI:  .venv/bin/python audio/musik.py [--timeline timeline.json] [--aus work/ton/musik_vorschau.wav]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ton_gemeinsam import (SR, WURZEL, bp, falten, hp, hp0, huelle_exp, lade_timeline, lp, lp0, n_samples, pan,
                           platzieren, raum_ir, von_db, wav_schreiben)

AKZENT_FENSTER_S = 0.040
SCHLUSS_MIN_KLANG_S = 3.0

# chord -> (bass midi, pad voicing, arpeggio notes). The bass voice sits in octave 2
# (65-110 Hz, audible on phones); a quiet sine one octave below adds the weight.
AKKORDE: dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]] = {
    "Am": (45, (57, 60, 64, 71), (69, 72, 76, 79, 71)),
    "F": (41, (57, 60, 65, 67), (65, 69, 72, 76, 67)),
    "C": (36, (55, 60, 64, 74), (67, 72, 76, 79, 74)),
    "G": (43, (55, 59, 62, 69), (67, 71, 74, 79, 69)),
    "E": (40, (56, 59, 64, 71), (68, 71, 76, 80, 71)),
    "Am_drone": (33, (45, 52, 57), (69, 72, 76)),
    "A": (45, (57, 61, 64, 71), (69, 73, 76, 81, 83)),
}


@dataclass(frozen=True)
class Profil:
    prog: tuple[str, ...]
    kick: str = ""            # '', 'gedaempft', 'halb', 'voll'
    clap_ab_takt: int = -1    # -1: no clap
    hats: str = ""            # 'uhr', 'uhr_selten', 'achtel', 'sechzehntel', 'drive', 'lift', 'shaker'
    bass: str = ""            # 'puls', 'herzschlag', 'achtel', 'rollend', 'lang'
    pluck: str = ""           # 'achtel', 'sechzehntel', 'hoch', 'weich'
    pluck_hell: tuple[float, float] = (0.6, 0.6)  # brightness ramp over the scene
    pad: float = 0.0
    pad_hell: float = 1500.0
    pad_stimmen: int = 3
    pad_breite: float = 0.6
    drone: bool = False
    pumpen: float = 0.0
    synkope: bool = False
    energie: float = 1.0       # scales accents


PROFILE: tuple[Profil, ...] = (
    Profil(prog=("Am",), hats="uhr", bass="puls", pad=0.30, pad_hell=800.0, pad_stimmen=3, energie=0.55),
    Profil(prog=("Am", "F", "C", "G"), kick="gedaempft", clap_ab_takt=2, hats="achtel", bass="achtel",
           pluck="achtel", pluck_hell=(0.45, 0.75), pad=0.45, pad_hell=1500.0, pumpen=0.55, energie=0.8),
    Profil(prog=("Am", "F", "C", "G"), kick="halb", clap_ab_takt=0, hats="sechzehntel", bass="achtel",
           pluck="sechzehntel", pluck_hell=(0.75, 1.0), pad=0.45, pad_hell=2200.0, pumpen=0.6, energie=0.9),
    Profil(prog=("Am_drone",), hats="uhr_selten", bass="herzschlag", pad=0.0, drone=True, energie=0.75),
    Profil(prog=("Am", "F", "C", "G"), kick="voll", clap_ab_takt=0, hats="drive", bass="rollend",
           pluck="sechzehntel", pluck_hell=(0.9, 1.1), pad=0.40, pad_hell=2600.0, pumpen=0.7, energie=1.0),
    Profil(prog=("Am", "F", "C", "G"), kick="voll", clap_ab_takt=0, hats="drive", bass="rollend",
           pluck="sechzehntel", pluck_hell=(1.1, 1.3), pad=0.45, pad_hell=3000.0, pumpen=0.7, synkope=True,
           energie=1.0),
    Profil(prog=("F", "G", "Am", "C"), kick="voll", clap_ab_takt=0, hats="lift", bass="achtel", pluck="hoch",
           pluck_hell=(1.1, 1.3), pad=0.85, pad_hell=4200.0, pad_stimmen=7, pad_breite=1.0, pumpen=0.6,
           energie=1.0),
    Profil(prog=("Am", "F", "C", "G", "F", "G"), kick="", hats="shaker", bass="lang", pluck="weich",
           pluck_hell=(0.6, 0.6), pad=0.7, pad_hell=2200.0, pad_stimmen=5, pad_breite=0.9, energie=0.9),
)


# ------------------------------------------------------------------ score (plan)

@dataclass
class Abschnitt:
    szene: str
    start_s: float
    ende_s: float
    bpm: float
    schlaege: int
    profil_index: int

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm


@dataclass
class Takt:
    start_s: float
    schlaege: int
    beat_s: float
    szene: str
    profil_index: int
    nr_in_szene: int
    takte_in_szene: int
    akkord: str = "Am"

    @property
    def ende_s(self) -> float:
        return self.start_s + self.schlaege * self.beat_s

    def schritt_s(self, schritt: int) -> float:
        return self.start_s + schritt * self.beat_s / 4.0


@dataclass
class Akzent:
    t_s: float
    event: str
    art: str
    quantisiert: bool


@dataclass
class Partitur:
    abschnitte: list[Abschnitt]
    takte: list[Takt]
    akzente: list[Akzent]
    schlussakkord_s: float
    dauer_s: float

    @property
    def taktanfaenge(self) -> list[float]:
        return [t.start_s for t in self.takte]

    @property
    def raster16(self) -> list[float]:
        return [t.schritt_s(k) for t in self.takte for k in range(t.schlaege * 4)]

    def als_dict(self) -> dict:
        return {"abschnitte": [vars(a) for a in self.abschnitte],
                "takte": [{"start_s": round(t.start_s, 4), "schlaege": t.schlaege, "szene": t.szene,
                           "akkord": t.akkord} for t in self.takte],
                "akzente": [vars(a) for a in self.akzente], "schlussakkord_s": self.schlussakkord_s}


def _schlaege_fuer(dauer_s: float) -> tuple[int, float]:
    """Whole number of beats for a scene, tempo as close to 100 BPM as possible,
    preferring whole 4/4 bars, then 2/4 endings."""
    beste = None
    lo, hi = max(1, int(np.floor(dauer_s * 95 / 60))), int(np.ceil(dauer_s * 105 / 60))
    for n in range(lo, hi + 1):
        bpm = n * 60.0 / dauer_s
        if not 95.0 <= bpm <= 105.0:
            continue
        kosten = abs(bpm - 100.0) + (0.0 if n % 4 == 0 else 1.5 if n % 2 == 0 else 4.0)
        if beste is None or kosten < beste[0]:
            beste = (kosten, n, bpm)
    if beste is None:  # very short scene: just fit it
        n = max(1, round(dauer_s * 100 / 60))
        return n, n * 60.0 / dauer_s
    return beste[1], beste[2]


def partitur_planen(timeline: dict) -> Partitur:
    dauer = float(timeline["dauer_s"])
    szenen = sorted(timeline["szenen"], key=lambda s: s["start_s"])
    abschnitte: list[Abschnitt] = []
    for i, s in enumerate(szenen):
        ende = szenen[i + 1]["start_s"] if i + 1 < len(szenen) else s["ende_s"]
        n, bpm = _schlaege_fuer(ende - s["start_s"])
        abschnitte.append(Abschnitt(s["id"], s["start_s"], ende, bpm, n, min(i, len(PROFILE) - 1)))

    takte: list[Takt] = []
    for ab in abschnitte:
        rest, t, eigene = ab.schlaege, ab.start_s, []
        while rest > 0:
            k = min(4, rest)
            eigene.append(Takt(t, k, ab.beat_s, ab.szene, ab.profil_index, len(eigene), 0))
            t += k * ab.beat_s
            rest -= k
        for tk in eigene:
            tk.takte_in_szene = len(eigene)
        takte += eigene
    # end card: continue the last scene's tempo up to the end
    letzter = abschnitte[-1]
    t = letzter.ende_s
    while t < dauer - 1e-6:
        takte.append(Takt(t, 4, letzter.beat_s, "abspann", letzter.profil_index, -1, -1))
        t += 4 * letzter.beat_s

    schluss = _schlussakkord_s(timeline, takte, dauer)
    _akkorde_setzen(takte, schluss)
    akzente = _akzente_planen(timeline, takte)
    return Partitur(abschnitte, takte, akzente, schluss, dauer)


def _schlussakkord_s(timeline: dict, takte: list[Takt], dauer: float) -> float:
    """Last chord: first bar line after the start of the last spoken word that still leaves
    >= 3 s of ring-out; else the first beat in that window; else the last beat before it."""
    worte = [w["start_s"] for z in timeline.get("vo", []) for w in z["woerter"]]
    frueh = max(worte) if worte else takte[-1].start_s
    spaet = dauer - SCHLUSS_MIN_KLANG_S
    for tk in takte:
        if frueh <= tk.start_s <= spaet:
            return tk.start_s
    beats = [tk.start_s + k * tk.beat_s for tk in takte for k in range(tk.schlaege)]
    im_fenster = [b for b in beats if frueh <= b <= spaet]
    if im_fenster:
        return im_fenster[0]
    vorher = [b for b in beats if b <= spaet]
    return vorher[-1] if vorher else 0.0


def _akkorde_setzen(takte: list[Takt], schluss: float) -> None:
    for tk in takte:
        prof = PROFILE[tk.profil_index]
        if tk.start_s >= schluss - 1e-6:
            tk.akkord = "A"
        elif tk.profil_index == 7:
            continue  # set below, right-aligned on the final chord
        elif tk.profil_index == 3 and tk.takte_in_szene >= 3 and tk.nr_in_szene == tk.takte_in_szene - 1:
            tk.akkord = "E"  # dominant pull into the drive
        else:
            tk.akkord = prof.prog[tk.nr_in_szene % len(prof.prog)]
    vor_schluss = [tk for tk in takte if tk.profil_index == 7 and tk.start_s < schluss - 1e-6]
    folge = PROFILE[7].prog
    for i, tk in enumerate(reversed(vor_schluss)):
        tk.akkord = folge[len(folge) - 1 - (i % len(folge))]


def _akzente_planen(timeline: dict, takte: list[Takt]) -> list[Akzent]:
    raster = np.array([t.schritt_s(k) for t in takte for k in range(t.schlaege * 4)])
    akzente = []
    for ev in timeline.get("events", []):
        if ev.get("sfx") not in ("hit", "sting"):
            continue
        r = float(raster[np.argmin(np.abs(raster - ev["t_s"]))])
        if abs(r - ev["t_s"]) <= AKZENT_FENSTER_S:
            akzente.append(Akzent(r, ev.get("id", ""), ev["sfx"], True))
        else:
            akzente.append(Akzent(float(ev["t_s"]), ev.get("id", ""), ev["sfx"], False))
    return sorted(akzente, key=lambda a: a.t_s)


# ------------------------------------------------------------------ instruments

def _norm(x: np.ndarray) -> np.ndarray:
    """Peak-normalise an instrument sample to 1.0 so arrangement gains are readable."""
    return x / (np.abs(x).max() + 1e-12)


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _saege(f: float, n: int, phase0: float) -> np.ndarray:
    dt = f / SR
    ph = (phase0 + dt * np.arange(1, n + 1)) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    u = ph[m] / dt
    y[m] -= u + u - u * u - 1.0
    m = ph > 1.0 - dt
    u = (ph[m] - 1.0) / dt
    y[m] -= u * u + u + u + 1.0
    return y


@lru_cache(maxsize=8)
def _kick(art: str) -> np.ndarray:
    n = int(0.55 * SR)
    t = _t(n)
    decay = {"gedaempft": 0.18, "halb": 0.22, "voll": 0.25}[art]
    f = 50.0 + 130.0 * np.exp(-t / 0.03)
    k = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.clip(t / 0.0005, 0, 1) * np.exp(-t / decay)
    k = np.tanh(1.9 * k) / np.tanh(1.9)
    rng = np.random.default_rng(11)
    klick = hp(rng.standard_normal(n) * huelle_exp(n, 0.0015), 2500.0, 2)
    schlaegel = bp(rng.standard_normal(n), 2800.0, 4500.0, 2) * huelle_exp(n, 0.004)  # beater
    k = k + klick * {"gedaempft": 0.03, "halb": 0.10, "voll": 0.16}[art] + schlaegel * {"gedaempft": 0.0, "halb": 0.25,
                                                                                       "voll": 0.4}[art]
    if art == "gedaempft":
        return _norm(lp(k, 1400.0, 2)) * 0.75
    if art == "halb":
        return _norm(lp(k, 4000.0, 2)) * 0.9
    return _norm(k)


@lru_cache(maxsize=4)
def _clap() -> np.ndarray:
    rng = np.random.default_rng(23)
    n = int(0.4 * SR)
    t = _t(n)
    env = np.zeros(n)
    for a in (0.0, 0.009, 0.019):
        k = int(a * SR)
        env[k:] += np.exp(-t[: n - k] / 0.003)
    k = int(0.026 * SR)
    env[k:] += 0.9 * np.exp(-t[: n - k] / 0.11)
    gemein = rng.standard_normal(n)
    st = np.stack([0.75 * gemein + 0.25 * rng.standard_normal(n), 0.75 * gemein + 0.25 * rng.standard_normal(n)], 1)
    return _norm(bp(st * env[:, None], 900.0, 6500.0, 2))


@lru_cache(maxsize=8)
def _hat(art: str) -> np.ndarray:
    rng = np.random.default_rng({"zu": 31, "offen": 37, "shaker": 41}[art])
    tau = {"zu": 0.035, "offen": 0.22, "shaker": 0.045}[art]
    n = int((tau * 6 + 0.01) * SR)
    t = _t(n)
    if art == "shaker":
        return _norm(bp(rng.standard_normal(n), 4000.0, 10000.0, 2) * np.clip(t / 0.008, 0, 1) * np.exp(-t / tau))
    metall = sum(np.sign(np.sin(2 * np.pi * f * 1.7 * t + rng.uniform(0, 6.28)))
                 for f in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0))
    h = bp(metall * 0.2 + rng.standard_normal(n) * 0.6, 7000.0, 13000.0, 2)
    return _norm(h * np.exp(-t / tau) * np.clip(t / 0.0005, 0, 1))


@lru_cache(maxsize=4)
def _uhr(tock: bool, dunkel: bool) -> np.ndarray:
    n = int(0.08 * SR)
    t = _t(n)
    ping = 2600.0 if tock else 3300.0
    holz = 950.0 if tock else 1200.0
    y = (np.sin(2 * np.pi * ping * t) * np.exp(-t / 0.006)
         + 0.5 * np.sin(2 * np.pi * holz * t) * np.exp(-t / 0.009)
         + hp(np.random.default_rng(5 + tock).standard_normal(n) * huelle_exp(n, 0.0006), 3000.0, 2) * 0.4)
    return _norm(lp(y, 2500.0, 2) if dunkel else y)


@lru_cache(maxsize=256)
def _bass(midi: int, n: int, weich: bool) -> np.ndarray:
    f = _hz(midi)
    t = _t(n)
    ph = 2 * np.pi * f * t
    env = np.clip(t / (0.012 if weich else 0.004), 0, 1) * np.clip((n / SR - t) / 0.03, 0, 1)
    sub = np.sin(ph) + 0.4 * np.sin(ph / 2.0)  # fundamental + quiet sub octave
    mitte = lp(np.tanh(3.0 * np.sin(ph)), 1100.0 if not weich else 600.0, 2)
    return _norm((sub * 0.7 + mitte * (0.45 if not weich else 0.25)) * env)


@lru_cache(maxsize=512)
def _pluck(midi: int, hell: float, weich: bool) -> np.ndarray:
    f = _hz(midi)
    laenge = 1.1 if weich else 0.8
    n = int(laenge * SR)
    t = _t(n)
    k_max = int(min(28, 9000.0 / f))
    tau0 = 0.55 if weich else 0.38
    y = np.zeros((n, 2))
    for kanal, cent in ((0, -5.0), (1, 5.0)):
        fd = f * 2 ** (cent / 1200)
        s = np.zeros(n)
        for k in range(1, k_max + 1):
            tau = tau0 / (1.0 + (k - 1) * 0.5 / max(hell, 0.1))
            s += np.sin(2 * np.pi * fd * k * t + 0.3 * k) / k ** 1.15 * np.exp(-t / tau)
        y[:, kanal] = s
    y *= np.clip(t / (0.006 if weich else 0.0015), 0, 1)[:, None]
    return _norm(hp(y, 180.0, 2))


@lru_cache(maxsize=128)
def _pad(akkord: tuple[int, ...], n_ton: int, hell: float, stimmen: int, breite: float, seed: int) -> np.ndarray:
    """Detuned saw stack per note (polyBLEP), voices spread across the stereo field, low-passed."""
    rng = np.random.default_rng(seed)
    rel = int(0.9 * SR)
    n = n_ton + rel
    y = np.zeros((n, 2))
    cents = np.linspace(-14.0, 14.0, stimmen) if stimmen > 1 else np.array([0.0])
    lagen = np.linspace(-breite, breite, stimmen) if stimmen > 1 else np.array([0.0])
    for midi in akkord:
        f = _hz(midi)
        for c, lage in zip(rng.permutation(cents), lagen):
            s = _saege(f * 2 ** (c / 1200), n, float(rng.uniform()))
            w = (lage + 1.0) * np.pi / 4.0
            y[:, 0] += s * np.cos(w)
            y[:, 1] += s * np.sin(w)
    y /= len(akkord) * np.sqrt(stimmen)
    y = lp(lp(y, hell, 2), hell * 1.3, 2)
    t = _t(n)
    env = np.clip(t / 0.35, 0, 1) * np.clip(1.0 - (t - n_ton / SR) / (rel / SR), 0, 1)
    return _norm(hp(y * env[:, None], 140.0, 2))


@lru_cache(maxsize=8)
def _crash(laenge: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(laenge * SR)
    t = _t(n)
    gemein = rng.standard_normal(n)
    st = np.stack([0.6 * gemein + 0.4 * rng.standard_normal(n), 0.6 * gemein + 0.4 * rng.standard_normal(n)], 1)
    metall = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in (3150.0, 4230.0, 5670.0, 7020.0, 8410.0))
    y = hp(st + 0.15 * metall[:, None], 2800.0, 2) * (np.exp(-t / (laenge / 4.5)) * np.clip(t / 0.001, 0, 1))[:, None]
    return _norm(lp(y, 12000.0, 2))


@lru_cache(maxsize=2)
def _anschlag() -> np.ndarray:
    """Short attack transient for accents (click + beater band): makes every accent land crisply."""
    n = int(0.03 * SR)
    rng = np.random.default_rng(97)
    klick = hp(rng.standard_normal(n) * huelle_exp(n, 0.0012), 2500.0, 2)
    schlaegel = bp(rng.standard_normal(n), 2500.0, 5000.0, 2) * huelle_exp(n, 0.004)
    return _norm(klick + 0.6 * schlaegel)


@lru_cache(maxsize=4)
def _boom() -> np.ndarray:
    n = int(1.2 * SR)
    t = _t(n)
    f = 44.0 + 60.0 * np.exp(-t / 0.06)
    b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.clip(t / 0.0008, 0, 1) * np.exp(-t / 0.35)
    return np.tanh(2.0 * b) / np.tanh(2.0)


def _glocke(f: float, n: int, laenge: float = 1.0) -> np.ndarray:
    t = _t(n)
    verh = (1.0, 2.0, 3.0, 4.2, 5.4)
    amps = (1.0, 0.35, 0.18, 0.1, 0.06)
    y = sum(a * np.sin(2 * np.pi * f * v * t) * np.exp(-t / (laenge * 2.4 / v ** 0.7)) for v, a in zip(verh, amps))
    return y * np.clip(t / 0.002, 0, 1)


# ------------------------------------------------------------------ arrangement

@dataclass
class Musik:
    mix: np.ndarray
    spuren: dict[str, np.ndarray]
    partitur: Partitur
    pegel: dict = field(default_factory=dict)


def _s(t_s: float) -> int:
    return int(round(t_s * SR))


def _pluck_muster(art: str) -> list[tuple[int, int]]:
    """(step, arpeggio index) per bar."""
    if art in ("achtel", "weich"):
        return [(s, i) for s, i in zip(range(0, 16, 2), (0, 1, 2, 3, 2, 1, 4, 2))]
    return [(s, i) for s, i in zip(range(16), (0, 2, 1, 3, 2, 0, 4, 1, 0, 2, 1, 3, 4, 2, 1, 3))]


def musik_rendern(timeline: dict, partitur: Partitur | None = None) -> Musik:
    p = partitur or partitur_planen(timeline)
    n = n_samples(timeline["dauer_s"])
    sp = {k: np.zeros((n, 2)) for k in ("drums", "bass", "plucks", "pads", "drone", "fx", "akzente", "schluss")}
    plate_send = np.zeros((n, 2))
    hall_send = np.zeros((n, 2))
    pump_schlaege: list[tuple[float, float]] = []  # (beat time, depth)
    schluss = p.schlussakkord_s
    vor_schluss = [tk for tk in p.takte if tk.start_s < schluss - 1e-6]

    for tk in vor_schluss:
        prof = PROFILE[tk.profil_index]
        bass_midi, pad_ton, arp = AKKORDE[tk.akkord]
        schritte = tk.schlaege * 4
        st = lambda k: tk.schritt_s(k)  # noqa: E731
        ende_takt = min(tk.ende_s, schluss)
        fortschritt = (tk.nr_in_szene / max(1, tk.takte_in_szene - 1)) if tk.nr_in_szene >= 0 else 1.0
        letzter_takt = tk.nr_in_szene == tk.takte_in_szene - 1

        # --- kick + sidechain beats
        if prof.kick:
            kicks = [0, 4, 8, 12]
            if prof.synkope and tk.nr_in_szene % 4 == 3:
                kicks.append(10)
            if tk.profil_index == 6 and letzter_takt:
                kicks = [0, 4]  # drop out before the morning
            for k in kicks:
                if k < schritte:
                    platzieren(sp["drums"], _kick(prof.kick)[:, None] * np.ones(2), _s(st(k)), 0.85)
        if prof.pumpen > 0:
            for k in range(0, schritte, 4):
                pump_schlaege.append((st(k), prof.pumpen))
        # --- clap on 2 and 4
        if prof.clap_ab_takt >= 0 and tk.nr_in_szene >= prof.clap_ab_takt:
            for k in (4, 12):
                if k < schritte and not (tk.profil_index == 6 and letzter_takt and k == 12):
                    g = 0.36 if prof.kick == "gedaempft" else 0.5
                    platzieren(sp["drums"], _clap(), _s(st(k)), g)
                    platzieren(plate_send, _clap(), _s(st(k)), g * 0.6)
        # --- hats / clock
        for k in range(schritte):
            vel = 0.0
            art = "zu"
            if prof.hats == "uhr":
                if k % 4 == 0:
                    platzieren(sp["drums"], pan(_uhr(k % 8 == 4, False), -0.15 if k % 8 else 0.15), _s(st(k)), 0.16)
                continue
            if prof.hats == "uhr_selten":
                if k in (0, 8):
                    platzieren(sp["drums"], pan(_uhr(k == 8, True), 0.0), _s(st(k)), 0.12)
                continue
            if prof.hats == "achtel":
                if k % 4 == 2:
                    vel = 0.55
                elif k % 4 == 0:
                    platzieren(sp["drums"], pan(_uhr(k % 8 == 4, False), 0.12), _s(st(k)), 0.06)
            elif prof.hats == "sechzehntel":
                vel = (0.35, 0.18, 0.6, 0.22)[k % 4]
            elif prof.hats == "drive":
                if k % 4 == 2:
                    art, vel = "offen", 0.45
                else:
                    vel = (0.4, 0.22, 0.0, 0.28)[k % 4]
            elif prof.hats == "lift":
                if k % 4 == 2:
                    art, vel = "offen", 0.5
                elif k % 2 == 0:
                    vel = 0.35
            elif prof.hats == "shaker":
                if k % 2 == 0 and st(k) < schluss - 1.0:
                    art, vel = "shaker", (0.5 if k % 4 == 2 else 0.3) * (1.0 - 0.6 * fortschritt)
            if vel > 0:
                platzieren(sp["drums"], pan(_hat(art), 0.25 if art != "shaker" else -0.3), _s(st(k)), vel * 0.36)
        # --- bass
        if prof.bass == "puls":
            for k in (0, 8):
                if k < schritte:
                    platzieren(sp["bass"], _bass(bass_midi, _s(0.35), True)[:, None] * np.ones(2), _s(st(k)), 0.55)
        elif prof.bass == "herzschlag":
            for k, g in ((0, 0.6), (3, 0.4)):
                if k < schritte:
                    platzieren(sp["bass"], _bass(bass_midi, _s(0.28), True)[:, None] * np.ones(2), _s(st(k)), g)
        elif prof.bass == "achtel":
            dauer = tk.beat_s * 0.45
            for k in (2, 6, 10, 14):
                if k < schritte:
                    platzieren(sp["bass"], _bass(bass_midi, _s(dauer), False)[:, None] * np.ones(2), _s(st(k)), 0.5)
        elif prof.bass == "rollend":
            dauer = tk.beat_s / 4 * 0.85
            for k in range(schritte):
                if k % 4 == 0:
                    continue
                midi = bass_midi + (12 if k % 4 == 3 else 0)
                platzieren(sp["bass"], _bass(midi, _s(dauer), False)[:, None] * np.ones(2), _s(st(k)),
                           0.42 if k % 4 == 2 else 0.32)
        elif prof.bass == "lang":
            dauer = ende_takt - tk.start_s
            platzieren(sp["bass"], _bass(bass_midi, _s(dauer), True)[:, None] * np.ones(2), _s(tk.start_s), 0.4)
        # --- plucks (+ dotted-eighth echoes, ping-pong)
        if prof.pluck:
            hell = prof.pluck_hell[0] + (prof.pluck_hell[1] - prof.pluck_hell[0]) * fortschritt
            oktave = 12 if prof.pluck == "hoch" else 0
            weich = prof.pluck == "weich"
            for k, i in _pluck_muster(prof.pluck):
                if k >= schritte or st(k) >= schluss - 0.05:
                    continue
                midi = arp[i % len(arp)] + oktave
                ton = _pluck(midi, round(hell, 2), weich)
                g = (0.34 if k % 4 == 0 else 0.27) * (0.8 if weich else 1.0)
                platzieren(sp["plucks"], ton, _s(st(k)), g)
                platzieren(plate_send, ton, _s(st(k)), g * 0.35)
                for e, (eg, seite) in enumerate(((0.32, 1), (0.14, -1)), start=1):
                    echo = ton * np.array([0.35, 1.0]) if seite > 0 else ton * np.array([1.0, 0.35])
                    platzieren(sp["plucks"], lp(echo, 3500.0, 1), _s(st(k + 3 * e)), g * eg)
        # --- pads (the thin s4 only gets a dark E chord in its last bar)
        if prof.pad == 0 and tk.akkord == "E":
            n_ton = _s(ende_takt - tk.start_s)
            ton = _pad(pad_ton, n_ton, 700.0, 3, 0.5, 13)
            platzieren(sp["pads"], ton, _s(tk.start_s), 0.12)
            platzieren(hall_send, ton, _s(tk.start_s), 0.08)
        if prof.pad > 0:
            n_ton = _s(ende_takt - tk.start_s)
            ton = _pad(pad_ton, n_ton, prof.pad_hell, prof.pad_stimmen, prof.pad_breite, 7 + tk.profil_index)
            g = prof.pad * 0.42
            if tk.profil_index == 0:
                g *= min(1.0, 0.35 + 0.65 * fortschritt)  # night pad breathes in
            platzieren(sp["pads"], ton, _s(tk.start_s), g)
            platzieren(hall_send, ton, _s(tk.start_s), g * 0.5)

    # --- s4 drone: low A/E with a slowly beating minor-second rub (B-flat) on top
    for ab in p.abschnitte:
        if not PROFILE[ab.profil_index].drone:
            continue
        a, b = _s(ab.start_s), _s(min(ab.ende_s, schluss))
        m = b - a
        t = _t(m)
        rng = np.random.default_rng(91)
        drone = np.zeros((m, 2))
        for midi, g in ((33, 0.5), (40, 0.3), (45, 0.18)):
            s = _saege(_hz(midi), m, float(rng.uniform()))
            drone += pan(s, -0.3 if midi % 2 else 0.3) * g
        filt = 220.0 + 160.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t))
        drone = lp(drone, 330.0, 2) * (0.7 + 0.3 * filt / 380.0)[:, None]
        reibung = (np.sin(2 * np.pi * _hz(70) * t) + np.sin(2 * np.pi * _hz(69) * t + 1.0)) * 0.035
        wind = bp(rng.standard_normal((m, 2)), 300.0, 1400.0, 2) * 0.05
        drone = drone + pan(reibung, 0.2) + wind
        drone = drone * np.clip(t / 1.5, 0, 1)[:, None] * np.clip((m / SR - t) / 0.4, 0, 1)[:, None]
        sp["drone"][a:b] += drone * 0.45
        hall_send[a:b] += drone * 0.15

    # --- transitions: reverse crash into big downbeats, crash on them
    for i, ab in enumerate(p.abschnitte[1:], start=1):
        ziel = ab.start_s
        if ziel >= schluss:
            continue
        gross = ab.profil_index in (4, 6)
        laenge = min(p.abschnitte[i - 1].beat_s * (4 if gross else 2), ziel - p.abschnitte[i - 1].start_s)
        rev = _crash(round(laenge, 3), 61 + i)[::-1]
        platzieren(sp["fx"], rev, _s(ziel) - len(rev), 0.22 if gross else 0.12)
        if gross:
            platzieren(sp["fx"], _crash(2.6, 71), _s(ziel), 0.2)
            platzieren(hall_send, _crash(2.6, 71), _s(ziel), 0.08)

    # --- accents on hit / sting events
    for akz in p.akzente:
        teile = akzent_klang(akz, p)
        if teile is None:
            continue
        trocken, hall = teile
        platzieren(sp["akzente"], trocken, _s(akz.t_s))
        platzieren(hall_send, hall, _s(akz.t_s))

    # --- final chord: A major add9, rings out under the end card
    a = _s(schluss)
    m = n - a
    if m > 0:
        _, pad_ton, arp = AKKORDE["A"]
        rest_s = m / SR
        akk = _pad(pad_ton, m, 2400.0, 7, 1.0, 5)[:m]
        t = _t(m)
        aus = (np.clip(1.0 - t / rest_s, 0, 1) ** 1.6)[:, None]
        sp["schluss"][a:] += akk * aus * 0.30
        for i, midi in enumerate((69, 76, 73, 83)):
            k = int(0.03 * i * SR)
            gl = _glocke(_hz(midi), m - k, 1.2)
            sp["schluss"][a + k:] += pan(gl, (-0.4, 0.3, -0.1, 0.45)[i]) * 0.05
        sub = _bass(33, m, True) * np.exp(-t / 1.6)
        sp["schluss"][a:] += (sub * 0.45)[:, None]
        hall_send[a:] += akk * aus * 0.18 + sp["schluss"][a:] * 0.1

    # --- sidechain pumping on bass, plucks, pads
    if pump_schlaege:
        g = _pump_kurve(n, pump_schlaege)
        for k in ("bass", "plucks", "pads"):
            sp[k] *= g[:, None]

    # --- reverb returns
    plate = hp(lp(falten(plate_send, raum_ir(1.4, 1301, hell_hz=8000.0)), 9000.0, 2), 280.0, 2)
    hall = hp(lp(falten(hall_send, raum_ir(3.2, 1303, vorverz_s=0.03, hell_hz=6500.0)), 8000.0, 2), 250.0, 2)
    sp["hall"] = plate * 0.35 + hall * 0.45

    summe = sum(sp.values())
    summe = _sub_mono(hp(summe, 28.0, 2))
    # bus: set the level so only rare peaks reach the soft saturation (glue, no audible clipping)
    summe *= 0.6 / (np.percentile(np.abs(summe), 99.97) + 1e-12)
    summe = np.tanh(summe)
    summe = _sub_mono(summe)
    summe[-int(0.25 * SR):] *= np.linspace(1, 0, int(0.25 * SR))[:, None]
    return Musik(summe, sp, p)


def akzent_klang(akz: Akzent, p: Partitur) -> tuple[np.ndarray, np.ndarray] | None:
    """One musical accent (attack click + low boom + chord stab of the current chord + crash),
    starting at sample 0 = akz.t_s. Returns (dry, reverb send) or None after the final chord."""
    if akz.t_s >= p.schlussakkord_s + 0.5:
        return None
    tk = max((t for t in p.takte if t.start_s <= akz.t_s + 1e-6), key=lambda t: t.start_s)
    _, _, arp = AKKORDE[tk.akkord]
    e = PROFILE[tk.profil_index].energie * (1.25 if akz.art == "sting" else 1.0)
    becken = _crash(2.8 if akz.art == "sting" else 1.2, 81 if akz.art == "sting" else 83)
    n = max(int(1.2 * SR), len(becken))
    stich = np.zeros((int(0.8 * SR), 2))
    for midi in sorted(set(arp))[:4]:
        ton = _pluck(midi, 1.4, False)[: len(stich)]
        stich[: len(ton)] += ton
    stich *= (np.exp(-_t(len(stich)) / 0.25))[:, None]
    trocken = np.zeros((n, 2))
    platzieren(trocken, stich, 0, 0.09 * e)
    platzieren(trocken, _boom()[:, None] * np.ones(2), 0, 0.25 * e)
    platzieren(trocken, pan(_anschlag(), 0.0), 0, 0.45 * e)
    platzieren(trocken, becken, 0, (0.16 if akz.art == "sting" else 0.08) * e)
    hall = np.zeros((n, 2))
    platzieren(hall, stich, 0, 0.06 * e)
    platzieren(hall, becken, 0, 0.05 * e)
    return trocken, hall


def _pump_kurve(n: int, schlaege: list[tuple[float, float]]) -> np.ndarray:
    zeiten = np.array([s for s, _ in schlaege])
    tiefen = np.array([d for _, d in schlaege])
    t = np.arange(n) / SR
    i = np.searchsorted(zeiten, t, side="right") - 1
    g = np.ones(n)
    gueltig = i >= 0
    seit = t[gueltig] - zeiten[i[gueltig]]
    g[gueltig] -= 0.62 * tiefen[i[gueltig]] * np.exp(-seit / 0.075) * (seit < 0.6)
    j = np.clip(i + 1, 0, len(zeiten) - 1)
    bis = zeiten[j] - t
    vor = (bis > 0) & (bis < 0.005)
    g[vor] = np.minimum(g[vor], 1.0 - 0.62 * tiefen[j[vor]] * (1.0 - bis[vor] / 0.005))
    return g


def _sub_mono(x: np.ndarray, grenze_hz: float = 120.0) -> np.ndarray:
    """Below grenze_hz both channels carry the mid signal (zero-phase, complementary split)."""
    mitte = lp0((x[:, 0] + x[:, 1]) / 2.0, grenze_hz, 4)
    tief = lp0(x, grenze_hz, 4)
    return x - tief + mitte[:, None]


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the score against a timeline")
    ap.add_argument("--timeline", type=Path, default=WURZEL / "timeline.json")
    ap.add_argument("--aus", type=Path, default=WURZEL / "work" / "ton" / "musik_vorschau.wav")
    args = ap.parse_args()
    mu = musik_rendern(lade_timeline(args.timeline))
    wav_schreiben(args.aus, mu.mix / max(1e-9, np.abs(mu.mix).max()) * 0.7)
    print(f"{args.aus}  ({len(mu.mix) / SR:.1f} s, Schlussakkord {mu.partitur.schlussakkord_s:.2f} s)")


if __name__ == "__main__":
    main()
