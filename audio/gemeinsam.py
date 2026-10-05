"""Shared helpers for the audio pipeline: paths, WAV I/O, Whisper, text matching.

Everything stimme.py and plan.py (and the mix stage) must agree on lives here,
so a path, the sample rate or the frame rate can never drift between scripts.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

# ------------------------------------------------------------------ paths
WURZEL = Path(__file__).resolve().parent.parent
SCRIPT = WURZEL / "script.json"
EVENTS = WURZEL / "szenen" / "events.json"
TIMELINE = WURZEL / "timeline.json"
VO_DIR = WURZEL / "assets" / "audio" / "vo"
WORK_VO = WURZEL / "work" / "vo"
TAKES_DIR = WORK_VO / "takes"
AUSWAHL = WORK_VO / "auswahl.json"
AUSRICHTUNG = WORK_VO / "ausrichtung.json"
BERICHT = WORK_VO / "timeline_bericht.txt"
ANFRAGEN_LOG = WORK_VO / "anfragen.log"

SR = 48_000          # delivery sample rate of every WAV the pipeline writes
FPS = 60
WHISPER_SR = 16_000


# ------------------------------------------------------------------ json
def lade_json(pfad: Path) -> dict:
    return json.loads(Path(pfad).read_text(encoding="utf-8"))


def lade_script() -> dict:
    return lade_json(SCRIPT)


def schreibe_json(pfad: Path, daten: object) -> None:
    """Write atomically (tmp + rename) so a crash never leaves half a contract."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps(daten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(pfad)


def sha256_datei(pfad: Path) -> str:
    return hashlib.sha256(Path(pfad).read_bytes()).hexdigest()


# ------------------------------------------------------------------ audio i/o
def lese_mono(pfad: Path) -> tuple[np.ndarray, int]:
    """float64 mono samples and their sample rate."""
    daten, sr = sf.read(str(pfad), dtype="float64", always_2d=False)
    if daten.ndim == 2:
        daten = daten.mean(axis=1)
    return daten, sr


def resample(x: np.ndarray, sr_von: int, sr_nach: int) -> np.ndarray:
    if sr_von == sr_nach:
        return x.copy()
    from scipy.signal import resample_poly
    g = gcd(sr_von, sr_nach)
    return resample_poly(x, sr_nach // g, sr_von // g)


def schreibe_wav(pfad: Path, daten: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> None:
    """Refuses to clip: the VO chain has no limiter, so a peak above full scale is a bug."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    spitze = float(np.max(np.abs(daten))) if daten.size else 0.0
    if spitze >= 1.0 and not subtype.startswith("FLOAT"):
        raise ValueError(f"{pfad.name}: Spitze {spitze:.3f} >= 0 dBFS — würde clippen")
    sf.write(str(pfad), daten, sr, subtype=subtype)


def db(x: float) -> float:
    return 20.0 * float(np.log10(max(x, 1e-12)))


# ------------------------------------------------------------------ whisper
_whisper: dict[str, object] = {}


def whisper_modell(name: str = "small"):
    """faster-whisper on CPU (int8), loaded once per process and size."""
    if name not in _whisper:
        from faster_whisper import WhisperModel
        _whisper[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _whisper[name]


def dekodiere_16k(pfad: Path) -> np.ndarray:
    """16 kHz mono float32 via ffmpeg. faster-whisper's own PyAV decoder breaks
    with the installed av version (`metadata_errors`), so it always gets an array."""
    roh = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(pfad), "-ac", "1", "-ar", str(WHISPER_SR),
         "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(roh, np.float32).copy()


def transkribiere(pfad: Path, modell: str = "small") -> dict:
    """Word-level transcript, cached next to the audio as <stem>.whisper-<modell>.json
    and keyed by the file hash. No initial_prompt with the script text: that would
    bias Whisper towards hearing exactly what we want to verify."""
    pfad = Path(pfad)
    cache = pfad.with_name(f"{pfad.stem}.whisper-{modell}.json")
    sha = sha256_datei(pfad)
    if cache.exists():
        alt = lade_json(cache)
        if alt.get("sha256") == sha and alt.get("modell") == modell:
            return alt
    segmente, _ = whisper_modell(modell).transcribe(
        dekodiere_16k(pfad), language="de", word_timestamps=True, beam_size=5,
        condition_on_previous_text=False)
    woerter, texte = [], []
    for s in segmente:
        texte.append(s.text.strip())
        for w in s.words or []:
            woerter.append({"wort": w.word, "start": round(float(w.start), 3),
                            "ende": round(float(w.end), 3), "p": round(float(w.probability), 3)})
    ergebnis = {"modell": modell, "sha256": sha, "text": " ".join(texte).strip(), "woerter": woerter}
    schreibe_json(cache, ergebnis)
    return ergebnis


# ------------------------------------------------------------------ text matching
def normalisiere(text: str) -> str:
    """Lowercase, ß -> ss, every non-word character (punctuation, hyphen, quotes) -> space."""
    t = text.lower().replace("ß", "ss")
    t = re.sub(r"[^\w\s]|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def skript_woerter(text: str) -> list[dict]:
    """Spoken words of a script line: display form (edge punctuation stripped) plus the
    normalised sub-words it consists of ("Plugin-Sammlung" -> ["plugin", "sammlung"])."""
    out = []
    for roh in text.split():
        norm = normalisiere(roh).split()
        if norm:
            out.append({"wort": re.sub(r"^\W+|\W+$", "", roh), "norm": norm})
    return out


_EINER = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
          "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_ZEHNER = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig", 7: "siebzig",
           8: "achtzig", 9: "neunzig"}


def _praefix(n: int) -> str:
    """Number as a multiplier prefix: "ein" instead of "eins" (einhundert, hunderteintausend)."""
    w = zahl_de(n)
    return w[:-1] if w.endswith("eins") else w


def zahl_de(n: int) -> str:
    """German cardinal as one word (0 <= n < 1 000 000)."""
    if n < 0:
        return "minus" + zahl_de(-n)
    if n < 20:
        return _EINER[n]
    if n < 100:
        z, e = divmod(n, 10)
        return (_praefix(e) + "und" if e else "") + _ZEHNER[z]
    if n < 1000:
        h, r = divmod(n, 100)
        return ("hundert" if h == 1 else _praefix(h) + "hundert") + (zahl_de(r) if r else "")
    if n < 1_000_000:
        t, r = divmod(n, 1000)
        return ("tausend" if t == 1 else _praefix(t) + "tausend") + (zahl_de(r) if r else "")
    return str(n)


_UHRZEIT = re.compile(r"^(\d{1,2})[.:](\d{2})$")
_URL_PUNKT = re.compile(r"(?<=[^\W\d_])\.(?=[^\W\d_])")   # "nomissuccess.de" -> "... punkt de"


def _verschmelze(roh: list[dict]) -> list[dict]:
    """faster-whisper marks a new word with leading whitespace. A token without it
    (".12" after " 3", "-Success" after " Miss") continues the previous word."""
    out: list[dict] = []
    for w in roh:
        text = str(w["wort"])
        if out and text and not text[0].isspace():
            out[-1]["wort"] += text
            out[-1]["ende"] = max(out[-1]["ende"], float(w["ende"]))
        else:
            out.append({"wort": text, "start": float(w["start"]), "ende": float(w["ende"])})
    return out


def expandiere_whisper(roh: list[dict]) -> list[tuple[str, float, float]]:
    """Whisper words -> normalised spoken words with times, in script spelling where
    Whisper used digits: "3.12 Uhr" -> drei/uhr/zwölf (spoken order, the clock form
    reorders), "30" -> dreissig, "nomissuccess.de" -> nomissuccess/punkt/de.
    A token's time span is split over its words in proportion to their length."""
    token = _verschmelze(roh)
    out: list[tuple[str, float, float]] = []
    i = 0
    while i < len(token):
        t = token[i]
        start, ende = t["start"], t["ende"]
        kern = re.sub(r"^\W+|[^\w]+$", "", t["wort"].strip())
        m = _UHRZEIT.match(kern)
        if m:
            woerter = [zahl_de(int(m[1]))]
            if i + 1 < len(token) and normalisiere(token[i + 1]["wort"]) == "uhr":
                woerter.append("uhr")
                ende = max(ende, token[i + 1]["ende"])
                i += 1
            if int(m[2]):
                woerter.append(zahl_de(int(m[2])))
            woerter = [normalisiere(w) for w in woerter]
        else:
            woerter = [normalisiere(zahl_de(int(w))) if w.isdigit() and len(w) <= 6 else w
                       for w in normalisiere(_URL_PUNKT.sub(" punkt ", t["wort"])).split()]
        i += 1
        if not woerter:
            continue
        gewichte = np.array([max(len(w), 1) for w in woerter], dtype=float)
        kanten = start + (ende - start) * np.concatenate([[0.0], np.cumsum(gewichte) / gewichte.sum()])
        out.extend((w, round(float(a), 4), round(float(b), 4)) for w, a, b in zip(woerter, kanten, kanten[1:]))
    return out


def wortfehlerrate(soll: list[str], ist: list[str]) -> float:
    """Word error rate (Levenshtein over words) relative to the reference length."""
    if not soll:
        return 0.0 if not ist else 1.0
    d = list(range(len(ist) + 1))
    for i, a in enumerate(soll, 1):
        vorher, d[0] = d[0], i
        for j, b in enumerate(ist, 1):
            vorher, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, vorher + (a != b))
    return d[len(ist)] / len(soll)
