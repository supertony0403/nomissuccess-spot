"""Synthesized SFX library and cue rendering per category (Task 2).

Everything is generated from code with fixed seeds (deterministic), 48 kHz stereo.
Each sound carries an anchor sample that lands exactly on the cue time:
  - 'transiente': the attack sits on t_s (hit, tick, click, lock, paper, clatter, sting, subdrop)
  - 'ende':       the sound ends on t_s (whoosh, riser)
  - 'start':      the sound begins on t_s (downer, rewind, swell, shimmer)

Every kind has 2-3 variants that rotate per occurrence, and each occurrence is
additionally detuned/re-seeded so no two cues are sample-identical.

CLI:  .venv/bin/python audio/sfx.py [--ordner assets/audio/sfx_bibliothek]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import istft, stft

from ton_gemeinsam import (SFX_KATEGORIE, SFX_STEMS, SR, WURZEL, Klang, Pegelmesser, bp, falten_mit_schweif,
                           fade, hp, hp0, huelle_exp, k_gewichtet, lp, n_samples, pan, platzieren, raum_ir, saege,
                           von_db, wav_schreiben)

ARTEN = ("whoosh", "hit", "tick", "riser", "downer", "click", "lock", "paper", "clatter",
         "sting", "shimmer", "subdrop", "rewind", "swell")

VARIANTEN = {"whoosh": 3, "hit": 3, "tick": 3, "riser": 3, "downer": 3, "click": 3, "lock": 3,
             "paper": 3, "clatter": 3, "sting": 2, "shimmer": 3, "subdrop": 2, "rewind": 2, "swell": 2}

# Loudness of each kind before the mix: max K-weighted level over a 100 ms window (dB, LUFS-like).
# This is the "pre-fader" level; category gains in mix.py sit on top.
ZIEL_PEGEL = {"whoosh": -19.0, "hit": -13.0, "subdrop": -16.0, "tick": -22.0, "click": -20.0,
              "lock": -17.0, "paper": -21.0, "clatter": -18.0, "sting": -13.0, "shimmer": -21.0,
              "riser": -19.0, "downer": -21.0, "rewind": -18.0, "swell": -20.0}

ANKERART = {**{a: "transiente" for a in ("hit", "subdrop", "tick", "click", "lock", "paper", "clatter", "sting")},
            "whoosh": "ende", "riser": "ende",
            **{a: "start" for a in ("downer", "rewind", "swell", "shimmer")}}

# A-centred pitch material (the score is in A minor and resolves to A major)
A_TON = {"A1": 55.0, "A2": 110.0, "E3": 164.81, "A3": 220.0, "C#4": 277.18, "E4": 329.63, "A4": 440.0,
         "B4": 493.88, "C#5": 554.37, "E5": 659.26, "A5": 880.0, "B5": 987.77, "E6": 1318.51,
         "A6": 1760.0, "B6": 1975.53, "E7": 2637.02, "A7": 3520.0}


# ------------------------------------------------------------------ primitives

def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _rausch_stereo(rng: np.random.Generator, n: int, korr: float = 0.4) -> np.ndarray:
    gemein = rng.standard_normal(n)
    eigen = rng.standard_normal((n, 2))
    return np.sqrt(korr) * gemein[:, None] + np.sqrt(1 - korr) * eigen


def _modal(n: int, freqs, taus, amps, phase: float = 0.0) -> np.ndarray:
    t = _t(n)
    y = np.zeros(n)
    for f, tau, a in zip(freqs, taus, amps):
        y += a * np.sin(2 * np.pi * f * t + phase) * np.exp(-t / tau)
    return y


def _klick(rng: np.random.Generator, n: int, tau_s: float = 0.0012, hp_hz: float = 2000.0) -> np.ndarray:
    return hp(rng.standard_normal(n) * huelle_exp(n, tau_s), hp_hz, 2)


def _band_sweep(x: np.ndarray, fc_von_t, q: float) -> np.ndarray:
    """Time-varying Gaussian band-pass (STFT domain). fc_von_t maps seconds -> centre Hz."""
    nper, nover = 1024, 768
    f, tt, z = stft(x, fs=SR, nperseg=nper, noverlap=nover)
    fc = np.maximum(fc_von_t(tt), 40.0)
    bw = fc / q
    g = np.exp(-0.5 * ((f[:, None] - fc[None, :]) / bw[None, :]) ** 2)
    _, y = istft(z * g, fs=SR, nperseg=nper, noverlap=nover)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def _sinus_glide(f: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def _faktor(vorkommen: int) -> float:
    """Per-occurrence pitch/time factor (+-3.5 %), 1.0 for the first occurrence."""
    return 1.0 + 0.035 * np.sin(2.4 * vorkommen)


def _raum(dry: np.ndarray, rt60: float, seed: int, nass_db: float, hp_hz: float = 150.0,
          hell_hz: float = 7000.0) -> np.ndarray:
    """Dry (mono or stereo) + synthetic room; returns stereo with the tail appended."""
    d2 = dry if dry.ndim == 2 else np.stack([dry, dry], axis=1)
    nass = falten_mit_schweif(d2, raum_ir(rt60, seed, hell_hz=hell_hz))
    nass = hp(nass, hp_hz, 2) * von_db(nass_db)
    out = nass.copy()
    out[: len(d2)] += d2
    return out


# ------------------------------------------------------------------ kinds

def _whoosh(v: int, rng: np.random.Generator, fk: float) -> Klang:
    p = [dict(dauer=0.55, f_lo=380, f_hi=2700, spitze=0.80, pan=(-0.65, 0.6), q=1.4, koerper=-7.0),
         dict(dauer=0.85, f_lo=230, f_hi=1900, spitze=0.78, pan=(0.7, -0.5), q=1.1, koerper=-4.0),
         dict(dauer=1.15, f_lo=420, f_hi=4300, spitze=0.84, pan=(-0.3, 0.8), q=1.8, koerper=-9.0)][v]
    n = int(p["dauer"] * fk * SR)
    u = np.linspace(0.0, 1.0, n)
    sp = p["spitze"]

    def bump(x):
        x = np.asarray(x)
        return np.where(x < sp, (x / sp) ** 1.6, 1.0 - 0.55 * (x - sp) / (1 - sp))

    dauer = n / SR
    rausch = _rausch_stereo(rng, n, 0.5)
    fc = lambda tt, k: (p["f_lo"] + (p["f_hi"] - p["f_lo"]) * bump(np.clip(tt / dauer, 0, 1))) * k * fk  # noqa: E731
    luft = np.stack([_band_sweep(rausch[:, 0], lambda tt: fc(tt, 0.97), p["q"]),
                     _band_sweep(rausch[:, 1], lambda tt: fc(tt, 1.03), p["q"])], axis=1)
    amp = np.where(u < sp, (u / sp) ** 2.2, np.cos(np.pi / 2 * np.clip((u - sp) / (1 - sp), 0, 1)) ** 0.8)
    koerper = lp(rng.standard_normal(n), 520.0, 4) * von_db(p["koerper"])
    koerper_amp = np.where(u < sp, (u / sp) ** 3.0, np.cos(np.pi / 2 * np.clip((u - sp) / (1 - sp), 0, 1)) ** 1.2)
    if v == 2:  # airy sparkle layer
        funkeln = hp(rausch[:, 0] * rausch[:, 1], 6000.0, 2) * 0.25
        luft += np.stack([funkeln, -funkeln], axis=1) * (u ** 3)[:, None]
    roh = luft / (np.abs(luft).max() + 1e-9) * amp[:, None] + (koerper * koerper_amp)[:, None]
    # passby: constant-power pan movement
    w = (np.linspace(p["pan"][0], p["pan"][1], n) + 1.0) * np.pi / 4.0
    roh = roh * np.stack([np.cos(w), np.sin(w)], axis=1) * np.sqrt(2.0)
    roh = hp0(roh, 120.0, 4)
    roh[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))[:, None]
    nass = falten_mit_schweif(roh, raum_ir(0.6, int(rng.integers(1 << 30)))) * von_db(-40.0)
    out = nass
    out[:n] += roh
    return Klang(hp0(out, 120.0, 2), n, "ende")


def _hit(v: int, rng: np.random.Generator, fk: float) -> Klang:
    p = [dict(f0=150, f1=46, tau_p=0.030, decay=0.32, snap=-15.0, metall=False, raum=-14.0),
         dict(f0=195, f1=58, tau_p=0.018, decay=0.17, snap=-10.0, metall=False, raum=-12.0),
         dict(f0=140, f1=50, tau_p=0.026, decay=0.40, snap=-13.0, metall=True, raum=-11.0)][v]
    n = int(0.9 * SR)
    t = _t(n)
    f = (p["f1"] + (p["f0"] - p["f1"]) * np.exp(-t / p["tau_p"])) * fk
    koerper = _sinus_glide(f) * np.clip(t / 0.0006, 0, 1) * np.exp(-t / p["decay"])
    koerper = np.tanh(2.2 * koerper) / np.tanh(2.2)
    klick = _klick(rng, n, 0.0012, 1500.0) * von_db(-8.0)
    snap = bp(rng.standard_normal(n), 1800.0, 6000.0, 2) * huelle_exp(n, 0.012) * von_db(p["snap"])
    dry = koerper + klick + snap
    if p["metall"]:
        teil = 420.0 * fk * np.array([1.0, 2.32, 3.87, 5.1])
        dry += hp(_modal(n, teil, [0.5, 0.36, 0.22, 0.15], [0.5, 0.35, 0.25, 0.15]), 300.0, 2) * von_db(-9.0)
    return Klang(_raum(dry, 0.45, int(rng.integers(1 << 30)), p["raum"]), 0, "transiente")


def _subdrop(v: int, rng: np.random.Generator, fk: float) -> Klang:
    p = [dict(f0=92, f1=32, dauer=1.6, tau=0.7), dict(f0=120, f1=38, dauer=1.25, tau=0.5)][v]
    n = int(p["dauer"] * SR)
    t = _t(n)
    f = p["f1"] * (p["f0"] / p["f1"]) ** (1 - t / (n / SR)) * fk
    sub = _sinus_glide(f) * np.clip(t / 0.003, 0, 1) * np.exp(-t / p["tau"])
    sub = np.tanh(3.0 * sub) / np.tanh(3.0)  # harmonics so it survives phone speakers
    tf = (60 + 70 * np.exp(-t / 0.02)) * fk
    thump = _sinus_glide(tf) * np.exp(-t / 0.05) * np.clip(t / 0.0008, 0, 1) * 0.6
    dry = sub + thump + _klick(rng, n, 0.001, 1200.0) * von_db(-18.0)
    if v == 1:
        dry += lp(rng.standard_normal(n), 220.0, 2) * np.exp(-t / 0.4) * von_db(-14.0)
    return Klang(_raum(dry, 0.8, int(rng.integers(1 << 30)), -18.0, hp_hz=90.0), 0, "transiente")


def _tick(v: int, rng: np.random.Generator, fk: float) -> Klang:
    n = int(0.14 * SR)
    p = [dict(ping=3200, tau=0.005, holz=1150, tau_h=0.010, holz_db=-6.0),
         dict(ping=2450, tau=0.006, holz=900, tau_h=0.014, holz_db=-4.0),
         dict(ping=1850, tau=0.022, holz=3700, tau_h=0.012, holz_db=-9.0)][v]
    dry = (_modal(n, [p["ping"] * fk], [p["tau"]], [1.0])
           + _modal(n, [p["holz"] * fk], [p["tau_h"]], [von_db(p["holz_db"])])
           + _klick(rng, n, 0.0006, 3000.0) * von_db(-6.0 if v < 2 else -16.0))
    lage = float(rng.uniform(-0.18, 0.18))
    return Klang(_raum(pan(dry, lage), 0.25, int(rng.integers(1 << 30)), -18.0, hp_hz=600.0), 0, "transiente")


def _click(v: int, rng: np.random.Generator, fk: float) -> Klang:
    n = int(0.12 * SR)
    p = [dict(res=(1650, 2900), taus=(0.009, 0.006), thock=230, tau_t=0.018),
         dict(res=(1200, 2400), taus=(0.011, 0.007), thock=180, tau_t=0.025),
         dict(res=(1750, 3100), taus=(0.008, 0.005), thock=240, tau_t=0.016)][v]
    einzel = lambda gain: (_klick(rng, n, 0.0015, 2000.0) * 0.8  # noqa: E731
                           + _modal(n, np.array(p["res"]) * fk, p["taus"], [0.7, 0.45])
                           + _modal(n, [p["thock"] * fk], [p["tau_t"]], [von_db(-6.0)])) * gain
    dry = einzel(1.0)
    if v == 2:  # spring release: a second, quieter click
        zweit = einzel(0.35)
        k = int(0.028 * SR)
        dry[k:] += zweit[: n - k]
    return Klang(_raum(dry, 0.3, int(rng.integers(1 << 30)), -17.0, hp_hz=300.0), 0, "transiente")


def _lock(v: int, rng: np.random.Generator, fk: float) -> Klang:
    p = [dict(abstand=0.085, ring=(2850, 4630), body=165, bestaetigung=False),
         dict(abstand=0.110, ring=(2300, 3900), body=140, bestaetigung=False),
         dict(abstand=0.075, ring=(3100, 5200), body=180, bestaetigung=True)][v]
    vor = int(0.005 * SR)
    anker = vor + int(p["abstand"] * SR)
    n = anker + int(0.6 * SR)
    dry = np.zeros(n)
    m = n - vor
    stufe1 = (_klick(rng, m, 0.0008, 2500.0) * 0.5 + _modal(m, [3400 * fk], [0.006], [0.45])) * von_db(-9.0)
    dry[vor:] += stufe1
    m2 = n - anker
    ring = _modal(m2, np.array(p["ring"]) * fk, [0.07, 0.045], [von_db(-8.0), von_db(-13.0)])
    body = _modal(m2, [p["body"] * fk], [0.05], [von_db(-3.0)])
    clunk = _klick(rng, m2, 0.002, 1000.0) + ring + body
    k = int(0.032 * SR)
    clunk[k:] += (_klick(rng, m2 - k, 0.0007, 2500.0) * 0.2)  # small rattle after the latch
    if p["bestaetigung"]:
        clunk += _modal(m2, [1320.0 * fk], [0.09], [von_db(-15.0)])
    dry[anker:] += clunk
    return Klang(_raum(dry, 0.35, int(rng.integers(1 << 30)), -16.0, hp_hz=200.0), anker, "transiente")


def _paper(v: int, rng: np.random.Generator, fk: float) -> Klang:
    n = int((0.75 if v != 2 else 0.55) * SR)
    y = np.zeros((n, 2))
    if v in (0, 2):
        t, amp, i = 0.0, 1.0, 0
        while t < n / SR - 0.03:
            m = n - int(t * SR)
            lappen = bp(rng.standard_normal(m), 900.0 * fk, 6500.0, 2) * huelle_exp(m, 0.012, 0.0015) * amp
            lage = float(np.clip(-0.3 + 0.05 * i + rng.uniform(-0.2, 0.2), -0.8, 0.8))
            y[int(t * SR):] += pan(lappen, lage)
            t += rng.uniform(0.022, 0.055) if v == 0 else (0.015 if i < 4 else rng.uniform(0.03, 0.06))
            amp *= rng.uniform(0.72, 0.9)
            i += 1
        if v == 2:  # quick flick then air
            luft = bp(rng.standard_normal(n), 1500.0, 7000.0, 2) * np.sin(np.pi * np.clip(_t(n) / (n / SR), 0, 1)) * 0.15
            y += pan(luft, 0.3)
    else:  # slide + crumple
        t = _t(n)
        gleiten = bp(rng.standard_normal(n), 1500.0 * fk, 8000.0, 2) * np.exp(-t / 0.25) * 0.35
        knistern = np.zeros(n)
        orte = rng.integers(0, n, 120)
        knistern[orte] = rng.uniform(-1, 1, 120) * np.exp(-orte / SR / 0.3)
        knistern = bp(knistern, 2000.0, 9000.0, 2) * 3.0
        start = _klick(rng, n, 0.003, 1200.0) * 1.2
        y = pan(gleiten + start, -0.2) + pan(knistern, 0.25)
    return Klang(_raum(y, 0.3, int(rng.integers(1 << 30)), -18.0, hp_hz=400.0), 0, "transiente")


def _clatter(v: int, rng: np.random.Generator, fk: float) -> Klang:
    p = [dict(bloecke=8, f=(600, 2200), tau=(0.012, 0.035), thud=True),     # wood
         dict(bloecke=11, f=(1400, 4200), tau=(0.008, 0.020), thud=False),  # plastic
         dict(bloecke=7, f=(1000, 5000), tau=(0.006, 0.014), thud=False)][v]  # light slips
    n = int(1.4 * SR)
    y = np.zeros((n, 2))
    for b in range(p["bloecke"]):
        t0 = 0.0 if b == 0 else float(rng.uniform(0.02, 0.22))
        a = 1.0 if b == 0 else float(rng.uniform(0.3, 0.7))
        delta = float(rng.uniform(0.07, 0.16))
        e = float(rng.uniform(0.5, 0.68))
        moden = rng.uniform(*p["f"], 3) * fk
        taus = rng.uniform(*p["tau"], 3)
        lage = float(rng.uniform(-0.7, 0.7)) if b else 0.0
        t, k = t0, 0
        while delta > 0.008 and k < 10 and t < 1.3:
            m = n - int(t * SR)
            if m <= 0:
                break
            jitter = 1.0 + rng.uniform(-0.03, 0.03, 3)
            schlag = (_modal(m, moden * jitter, taus, [1.0, 0.6, 0.4]) * 0.6
                      + _klick(rng, m, 0.0007, 1500.0) * 0.5)
            if p["thud"] and k < 2:
                schlag += _modal(m, [rng.uniform(180, 260)], [0.025], [0.5])
            if v == 2:
                schlag = schlag * 0.6 + bp(rng.standard_normal(m), 2000, 7000, 2) * huelle_exp(m, 0.006) * 0.3
            y[int(t * SR):] += pan(schlag * a, lage + 0.05 * k)
            t += delta
            delta *= e
            a *= e ** 1.4
            k += 1
    return Klang(_raum(y, 0.4, int(rng.integers(1 << 30)), -15.0, hp_hz=250.0), 0, "transiente")


def _glocke(n: int, f0: float, laenge: float, rng: np.random.Generator) -> np.ndarray:
    verh = np.array([1.0, 2.0, 2.76, 4.07, 5.4])
    amps = np.array([1.0, 0.5, 0.35, 0.2, 0.12])
    taus = np.array([2.2, 1.2, 0.8, 0.5, 0.35]) * laenge * (440.0 / f0) ** 0.3
    return _modal(n, f0 * verh, taus, amps) * np.clip(_t(n) / 0.0015, 0, 1)


def _sting(v: int, rng: np.random.Generator, fk: float) -> Klang:
    """Brand signature: bright bell chord (A-E-B sus voicing, key-neutral) over a sub."""
    p = [dict(toene=("A4", "E5", "B5", "A5"), versatz=(0.0, 0.018, 0.036, 0.05), lagen=(-0.35, 0.25, -0.15, 0.4),
              laenge=1.0, dauer=3.6, rt=2.8, nass=-8.0, sub_von=55.0),
         dict(toene=("E4", "A4", "E5", "B5", "E6"), versatz=(0.0, 0.0, 0.022, 0.044, 0.066),
              lagen=(0.1, -0.3, 0.3, -0.45, 0.5), laenge=1.4, dauer=5.0, rt=3.5, nass=-6.0, sub_von=62.0)][v]
    n = int(p["dauer"] * SR)
    y = np.zeros((n, 2))
    for ton, vs, lage in zip(p["toene"], p["versatz"], p["lagen"]):
        k = int(vs * SR)
        g = _glocke(n - k, A_TON[ton] * fk, p["laenge"], rng)
        y[k:] += pan(g, lage) * 0.5
    t = _t(n)
    sub_f = 55.0 + (p["sub_von"] - 55.0) * np.exp(-t / 0.4)
    sub = (_sinus_glide(sub_f) + 0.35 * _sinus_glide(2 * sub_f)) * np.clip(t / 0.004, 0, 1) * np.exp(-t / 0.9)
    sub = np.tanh(1.8 * sub) / np.tanh(1.8) * 0.9
    schlaegel = _klick(rng, n, 0.002, 3000.0) * 0.45
    dry = y + (sub + schlaegel)[:, None]
    nass = falten_mit_schweif(hp(y, 300.0, 2), raum_ir(p["rt"], int(rng.integers(1 << 30)), hell_hz=9000.0))
    out = nass * von_db(p["nass"])
    out[:n] += dry
    return Klang(out, 0, "transiente")


def _shimmer(v: int, rng: np.random.Generator, fk: float) -> Klang:
    p = [dict(koerner=60, dauer=1.4), dict(koerner=40, dauer=1.0), dict(koerner=90, dauer=2.0)][v]
    n = int((p["dauer"] + 0.2) * SR)
    y = np.zeros((n, 2))
    toene = [A_TON[k] for k in ("A5", "B5", "E6", "A6", "B6", "E7", "A7")]
    zeiten = np.sort(p["dauer"] * rng.uniform(0, 1, p["koerner"]) ** 1.8)
    zeiten[0] = 0.0
    for i, t0 in enumerate(zeiten):
        laenge = float(rng.uniform(0.03, 0.11))
        m = int(laenge * SR)
        f = toene[int(rng.integers(len(toene)))] * fk
        korn = np.sin(2 * np.pi * f * _t(m)) * np.hanning(m) * (1.0 - 0.75 * t0 / p["dauer"])
        if i == 0:
            korn *= 1.3
        a = int(t0 * SR)
        y[a:a + m] += pan(korn, float(rng.uniform(-0.8, 0.8)))[: n - a]
    luft = hp(rng.standard_normal((n, 2)), 6000.0, 2) * huelle_exp(n, 0.5, 0.015)[:, None] * von_db(-16.0)
    dry = y * 0.4 + luft
    nass = falten_mit_schweif(dry, raum_ir(2.0, int(rng.integers(1 << 30)), hell_hz=11000.0))
    out = nass * von_db(-5.0)
    out[:n] += dry
    return Klang(out, 0, "start")


def _riser(v: int, rng: np.random.Generator, fk: float) -> Klang:
    dauer = [2.0, 2.4, 1.6][v] * fk
    n = int(dauer * SR)
    u = np.linspace(0, 1, n)
    if v == 0:  # noise riser, band sweeps up
        r = rng.standard_normal((n, 2))
        fc = lambda tt: 400.0 * (7000.0 / 400.0) ** np.clip(tt / dauer, 0, 1)  # noqa: E731
        y = np.stack([_band_sweep(r[:, c], fc, 2.0) for c in range(2)], axis=1) * (u ** 2.5)[:, None]
    elif v == 1:  # tonal riser: detuned saws two octaves up, filter opening
        f = 110.0 * 4.0 ** (u ** 1.3)
        stimmen = [(saege(f * d, n, rng.uniform()), lage) for d, lage in
                   ((0.994, -0.6), (1.0, 0.0), (1.006, 0.6))]
        y = sum(pan(s, lage) for s, lage in stimmen) / 3.0
        y = np.stack([_band_sweep(y[:, c], lambda tt: 500.0 * 12.0 ** np.clip(tt / dauer, 0, 1), 0.7)
                      for c in range(2)], axis=1)
        y += hp(rng.standard_normal((n, 2)), 3000.0, 2) * 0.15
        y *= (u ** 2.2)[:, None]
    else:  # reverse cymbal
        becken = hp(_rausch_stereo(rng, n, 0.3), 3000.0, 2) * huelle_exp(n, dauer / 4.0)[:, None]
        metall = _modal(n, [3150.0, 4230.0, 5670.0, 7020.0], [dauer / 5] * 4, [0.3, 0.25, 0.2, 0.15])
        y = (becken + metall[:, None])[::-1].copy()
    y = hp0(y, 120.0, 2)
    y[-int(0.006 * SR):] *= np.linspace(1, 0, int(0.006 * SR))[:, None]
    return Klang(y, n, "ende")


def _downer(v: int, rng: np.random.Generator, fk: float) -> Klang:
    dauer = [0.9, 0.8, 1.1][v] * fk
    n = int(dauer * SR)
    t = _t(n)
    u = t / dauer
    if v == 0:
        f = 900.0 * (90.0 / 900.0) ** u
        ton = _sinus_glide(f) + 0.3 * _sinus_glide(2 * f)
        rausch = lp(rng.standard_normal(n), 1200.0, 2) * 0.2
        y = pan(ton + rausch, 0.0)
    elif v == 1:  # power-down / tape stop on A3
        rate = (1 - u) ** 2
        ton = saege(220.0 * rate + 1.0, n)
        y = np.stack([_band_sweep(ton, lambda tt: 4000.0 * (1 - np.clip(tt / dauer, 0, 0.97)) ** 2 + 150.0, 0.8)] * 2,
                     axis=1)
    else:
        r = _rausch_stereo(rng, n, 0.4)
        y = np.stack([_band_sweep(r[:, c], lambda tt: 3000.0 * (200.0 / 3000.0) ** np.clip(tt / dauer, 0, 1), 1.5)
                      for c in range(2)], axis=1)
    amp = np.clip(t / 0.004, 0, 1) * (1 - u) ** 1.4
    y = hp(y * amp[:, None], 100.0, 2)
    return Klang(_raum(y, 0.9, int(rng.integers(1 << 30)), -14.0), 0, "start")


def _rewind(v: int, rng: np.random.Generator, fk: float) -> Klang:
    p = [dict(material=1.4, r0=2.2, r1=0.35), dict(material=1.0, r0=3.0, r1=0.5)][v]
    m = int(p["material"] * SR)
    tm = _t(m)
    material = np.zeros(m)
    for f in rng.choice([440.0, 523.25, 659.26, 783.99, 880.0], 6):
        a = int(rng.integers(0, m - SR // 8))
        material[a:] += np.sin(2 * np.pi * f * tm[: m - a]) * np.exp(-tm[: m - a] / 0.25) * 0.5
    material += bp(rng.standard_normal(m), 400.0, 5000.0, 2) * 0.15
    material += _sinus_glide(60 + 80 * np.exp(-tm / 0.02)) * np.exp(-tm / 0.15) * 0.6
    material = material[::-1]
    # variable playback rate (fast -> slow = pitch falls), plus tape flutter
    n_max = int(4 * SR)
    rate = np.linspace(p["r0"], p["r1"], n_max) * fk
    rate *= 1.0 + 0.012 * np.sin(2 * np.pi * 7.5 * _t(n_max))
    pos = np.cumsum(rate)
    n = int(np.searchsorted(pos, m - 1))
    y = np.interp(pos[:n], np.arange(m), material)
    scrub = bp(rng.standard_normal(n), 1000.0, 4000.0, 2) * 0.12 * (0.6 + 0.4 * np.sin(2 * np.pi * 11 * _t(n)))
    y = fade(y + scrub, 0.008, 0.05)
    st = np.stack([y, np.roll(y, int(0.0007 * SR))], axis=1)
    st[: int(0.0007 * SR), 1] = 0.0
    return Klang(_raum(hp(st, 100.0, 2), 0.7, int(rng.integers(1 << 30)), -14.0), 0, "start")


def _swell(v: int, rng: np.random.Generator, fk: float) -> Klang:
    dauer = [3.2, 2.4][v] * fk
    n = int(dauer * SR)
    t = _t(n)
    u = t / dauer
    toene = ("A3", "E4", "B4", "C#5", "E5") if v == 0 else ("A3", "E4", "A4", "C#5")
    y = np.zeros((n, 2))
    for i, ton in enumerate(toene):
        f = A_TON[ton] * fk
        for d, lage in ((0.997, -0.5), (1.003, 0.5)):
            y += pan(np.sin(2 * np.pi * f * d * t + rng.uniform(0, 6.28)), lage * (1 if i % 2 else -1))
    y /= 2 * len(toene)
    luft = _rausch_stereo(rng, n, 0.2) * (0.25 if v == 0 else 0.5)
    y = y + luft * 0.3
    y = np.stack([_band_sweep(y[:, c], lambda tt: 300.0 * (5000.0 / 300.0) ** np.clip(tt / dauer, 0, 1) ** 1.3, 0.6)
                  for c in range(2)], axis=1)
    hoch = 0.85
    amp = np.where(u < hoch, (u / hoch) ** 1.6, np.cos(np.pi / 2 * np.clip((u - hoch) / (1 - hoch), 0, 1)))
    y *= amp[:, None]
    # soft bloom on the start: the light "arrives"
    bluete = _glocke(n, A_TON["E5"] * fk, 0.8, rng) * 0.6 + _glocke(n, A_TON["A4"] * fk, 0.8, rng) * 0.4
    y += pan(bluete, 0.15) * von_db(-14.0)
    return Klang(_raum(hp(y, 120.0, 2), 2.5, int(rng.integers(1 << 30)), -9.0), 0, "start")


_SYNTH = {"whoosh": _whoosh, "hit": _hit, "subdrop": _subdrop, "tick": _tick, "click": _click, "lock": _lock,
          "paper": _paper, "clatter": _clatter, "sting": _sting, "shimmer": _shimmer, "riser": _riser,
          "downer": _downer, "rewind": _rewind, "swell": _swell}


def _pegel_100ms(x: np.ndarray) -> float:
    y = k_gewichtet(np.concatenate([np.zeros((SR // 10, 2)), x, np.zeros((SR // 10, 2))]))
    e = np.concatenate([[0.0], np.cumsum(np.sum(y ** 2, axis=1))])
    w = SR // 10
    m = (e[w:] - e[:-w]) / w
    return float(-0.691 + 10 * np.log10(m.max() + 1e-20))


def klang(art: str, variante: int, vorkommen: int = 0) -> Klang:
    """Render one sound. Deterministic in (art, variante, vorkommen)."""
    if art not in _SYNTH:
        raise ValueError(f"unbekannte SFX-Art: {art!r} (bekannt: {', '.join(ARTEN)})")
    variante %= VARIANTEN[art]
    rng = np.random.default_rng(1_000_003 * (ARTEN.index(art) + 1) + 1009 * variante + vorkommen)
    k = _SYNTH[art](variante, rng, _faktor(vorkommen))
    daten = k.daten * von_db(ZIEL_PEGEL[art] - _pegel_100ms(k.daten))
    spitze = np.abs(daten).max()
    if spitze > 0.98:
        daten *= 0.98 / spitze
    return Klang(daten, k.anker, k.ankerart)


# ------------------------------------------------------------------ atmo

def _vogel(rng: np.random.Generator) -> np.ndarray:
    silben = int(rng.integers(2, 6))
    teile = []
    f_basis = rng.uniform(3000, 5200)
    for _ in range(silben):
        m = int(rng.uniform(0.04, 0.09) * SR)
        u = np.linspace(0, 1, m)
        f = f_basis * (1 + rng.choice([-0.25, 0.3]) * u) * (1 + 0.04 * np.sin(2 * np.pi * rng.uniform(25, 45) * u))
        teile.append(_sinus_glide(f) * np.sin(np.pi * u) ** 2)
        teile.append(np.zeros(int(rng.uniform(0.02, 0.07) * SR)))
    return np.concatenate(teile)


def atmo_rendern(timeline: dict, n: int, seed: int = 4242) -> np.ndarray:
    """Room-tone bed over the full length: quiet night tone in s1-s7, brighter morning in s8."""
    rng = np.random.default_rng(seed)
    t = _t(n)
    # night: dark noise bed, slow breathing, distant traffic swells, a little air
    rosa = lp(_rausch_stereo(rng, n, 0.4), 900.0, 2)
    rosa *= (1.0 + 0.25 * np.sin(2 * np.pi * 0.045 * t + np.array([[0.0, 1.7]]).T).T)
    brumm = bp(rng.standard_normal((n, 2)), 150.0, 420.0, 2) * 0.5
    nacht = rosa + brumm + lp(hp(rng.standard_normal((n, 2)), 2500.0, 2), 6000.0, 2) * von_db(-36.0)
    pos = float(rng.uniform(3, 8))
    while pos < n / SR - 4:
        m = int(4.0 * SR)
        a = int(pos * SR)
        auto = lp(bp(rng.standard_normal(m), 300.0, 900.0, 2), 1200.0, 2) * np.hanning(m) * 0.8
        lage = np.linspace(rng.choice([-0.8, 0.8]), rng.uniform(-0.3, 0.3), m)
        nacht[a:a + m] += pan(auto, lage)[: n - a]
        pos += float(rng.uniform(11, 19))
    # morning: brighter air, breeze, sparse birds
    brise = lp(hp(_rausch_stereo(rng, n, 0.3), 400.0, 2), 7000.0, 2)
    brise *= (1.0 + 0.3 * np.sin(2 * np.pi * 0.08 * t + np.array([[0.4, 2.2]]).T).T)
    morgen = brise + hp(_rausch_stereo(rng, n, 0.2), 5000.0, 2) * von_db(-10.0)
    pos = 0.3
    while pos < n / SR - 0.5:
        ruf = _vogel(rng) * 0.9
        a = int(pos * SR)
        morgen[a:a + len(ruf)] += pan(ruf, float(rng.uniform(-0.8, 0.8)))[: n - a]
        pos += float(rng.uniform(0.6, 2.2))

    def auf(x: np.ndarray, ziel: float) -> np.ndarray:
        a, b = min(SR, n // 4), min(n, int(41.0 * SR))
        seg = x[a:b] if b - a >= int(0.4 * SR) else x
        if not np.any(seg):
            return x
        return x * von_db(ziel - Pegelmesser(seg).pegel(0, len(seg) / SR))

    nacht = auf(hp0(nacht, 120.0, 4), -45.0)
    morgen = auf(hp0(morgen, 120.0, 4), -40.0)
    szenen = timeline.get("szenen", [])
    morgen_start = next((s["start_s"] for s in szenen if s["id"] == "s8_morgen"), None)
    if morgen_start is None and len(szenen) > 1:
        morgen_start = szenen[-1]["start_s"]
    w = np.zeros(n)
    if morgen_start is not None:
        w = np.clip((t - morgen_start) / 3.0, 0.0, 1.0)
        w = np.sin(w * np.pi / 2) ** 2
    bett = nacht * np.sqrt(1 - w)[:, None] + morgen * np.sqrt(w)[:, None]
    return fade(bett, 1.0, 1.5)


# ------------------------------------------------------------------ cues -> stems

def cues_aus_timeline(timeline: dict) -> list[dict]:
    if "cues" in timeline:
        return list(timeline["cues"])
    return [{"t_s": e["t_s"], "art": e["sfx"], "szene": e.get("szene"), "quelle": f"event:{e.get('id')}"}
            for e in timeline.get("events", []) if e.get("sfx")]


def kategorien_rendern(timeline: dict, varianten_ab: int = 0) -> dict[str, np.ndarray]:
    """All SFX category stems (stereo, full length, pre-fader). Includes the atmo bed."""
    n = n_samples(timeline["dauer_s"])
    stems = {k: np.zeros((n, 2)) for k in SFX_STEMS}
    cues = cues_aus_timeline(timeline)
    unbekannt = sorted({c["art"] for c in cues} - set(ARTEN))
    if unbekannt:
        raise ValueError(f"unbekannte SFX-Art(en) in der Timeline: {', '.join(unbekannt)}")
    zaehler: Counter[str] = Counter()
    for cue in sorted(cues, key=lambda c: (c["t_s"], c["art"])):
        art = cue["art"]
        k = zaehler[art]
        zaehler[art] += 1
        kl = klang(art, k + varianten_ab, vorkommen=k)
        start = int(round(cue["t_s"] * SR)) - kl.anker
        platzieren(stems[SFX_KATEGORIE[art]], kl.daten, start)
    stems["sfx_atmo"] = atmo_rendern(timeline, n)
    return stems


def bibliothek_schreiben(ordner: Path) -> list[Path]:
    """Write every kind x variant as a WAV (+ index.json with anchors) for use in Fairlight."""
    ordner = Path(ordner)
    dateien, index = [], {}
    for art in ARTEN:
        for v in range(VARIANTEN[art]):
            kl = klang(art, v)
            pfad = ordner / f"{art}_{v + 1}.wav"
            wav_schreiben(pfad, kl.daten)
            dateien.append(pfad)
            index[pfad.name] = {"art": art, "variante": v + 1, "kategorie": SFX_KATEGORIE[art],
                                "anker_s": round(kl.anker / SR, 4), "ankerart": kl.ankerart,
                                "dauer_s": round(len(kl.daten) / SR, 4)}
    (ordner / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    return dateien


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesized SFX library (nomissuccess spot)")
    ap.add_argument("--ordner", type=Path, default=WURZEL / "assets" / "audio" / "sfx_bibliothek")
    args = ap.parse_args()
    dateien = bibliothek_schreiben(args.ordner)
    print(f"{len(dateien)} Klänge nach {args.ordner}")


if __name__ == "__main__":
    main()
