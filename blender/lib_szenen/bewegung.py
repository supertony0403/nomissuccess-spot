"""Pure-Python motion helpers for the shots s2-s4 (no bpy, unit-testable).

* :class:`Bahn` - cubic Hermite track through timed key poses,
* :func:`overshoot` - accelerate into a target, one bounded bounce, settle,
* :func:`rewind_time` - story time of s4: forward, then an eased rewind.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def in_out_cubic(t: float) -> float:
    t = 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else t
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


class Bahn:
    """Cubic Hermite interpolation through ``(frame, value[, stop])`` keys.

    * value: float or vector-like,
    * ``stop=True`` -> zero velocity at that key (lands / starts softly),
    * otherwise the velocity is the Catmull-Rom estimate, so motion flows.
    Before the first and after the last key the value is held.
    """

    def __init__(self, keys: Sequence[tuple]):
        ks = sorted(keys, key=lambda k: k[0])
        if len({k[0] for k in ks}) != len(ks):
            raise ValueError("Bahn: doppelte Schlüsselbilder")
        self.f = np.array([k[0] for k in ks], dtype=np.float64)
        self.v = np.array([np.atleast_1d(np.asarray(k[1], dtype=np.float64)) for k in ks])
        self.stop = [bool(k[2]) if len(k) > 2 else False for k in ks]
        self.scalar = np.ndim(ks[0][1]) == 0
        n = len(ks)
        self.m = np.zeros_like(self.v)
        for i in range(1, n - 1):
            if not self.stop[i]:
                self.m[i] = (self.v[i + 1] - self.v[i - 1]) / (self.f[i + 1] - self.f[i - 1])

    def __call__(self, frame: float):
        f = self.f
        if frame <= f[0]:
            out = self.v[0]
        elif frame >= f[-1]:
            out = self.v[-1]
        else:
            i = int(np.searchsorted(f, frame) - 1)
            h = f[i + 1] - f[i]
            t = (frame - f[i]) / h
            t2, t3 = t * t, t * t * t
            out = (
                (2 * t3 - 3 * t2 + 1) * self.v[i]
                + (t3 - 2 * t2 + t) * h * self.m[i]
                + (-2 * t3 + 3 * t2) * self.v[i + 1]
                + (t3 - t2) * h * self.m[i + 1]
            )
        return float(out[0]) if self.scalar else tuple(float(x) for x in out)


def overshoot(t: float, amount: float = 0.03) -> float:
    """0 -> 1 with an accelerating approach, a single bounce of ``amount`` and settle.

    The first 70 % of ``t`` accelerate into the target (in-cubic), then the
    value overshoots by at most ``amount`` and settles back with a damped
    half sine (no further oscillation).
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    hit = 0.7
    if t < hit:
        return (t / hit) ** 3
    u = (t - hit) / (1.0 - hit)
    return 1.0 + amount * math.sin(math.pi * u) * (1.0 - u)


def rewind_time(f: float, f_rewind: float, duration: float) -> float:
    """Story frame for shot frame ``f``: forward until ``f_rewind``, then an
    eased rewind that reaches story frame 1 after ``duration`` frames."""
    if f <= f_rewind:
        return float(f)
    u = (f - f_rewind) / duration
    if u >= 1.0:
        return 1.0
    return f_rewind - (f_rewind - 1.0) * in_out_cubic(u)
