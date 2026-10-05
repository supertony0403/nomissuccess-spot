"""Voice-over via Gemini TTS: one request per block (A, B, C), checked by Whisper.

The free tier's daily quota per model is scarce, so this script is frugal:
  * exactly one request per block (lines joined with paragraph breaks); plan.py
    later splits the block into lines by word alignment,
  * a second take only for a block whose Whisper WER against the script exceeds
    MAX_WER (a weak small-model transcript is re-checked with `medium` first),
  * >= PAUSE_S between requests, at most MAX_VERSUCHE attempts for transient
    errors, and a daily-quota 429 (quotaId "...PerDay...") ends the run at once
    while every finished take is kept,
  * takes are cached in work/vo/takes/<block>_take<n>.wav with a sidecar JSON
    (text, style, voice, model); a re-run makes no request while it matches.

The API key is read at runtime from ~/.config/ddr-video/gemini.env and travels
only in the x-goog-api-key header. It is never printed, logged or written, and
every error text is masked before it is shown.

Usage:  .venv/bin/python audio/stimme.py [--nur-bewerten] [--whisper small]
Then:   .venv/bin/python audio/plan.py
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

from gemeinsam import (AUSWAHL, TAKES_DIR, WURZEL, expandiere_whisper, lade_script, lese_mono,
                       schreibe_json, skript_woerter, transkribiere, wortfehlerrate)

API = "https://generativelanguage.googleapis.com/v1beta/models/{modell}:generateContent"
SCHLUESSEL_DATEI = Path.home() / ".config" / "ddr-video" / "gemini.env"
PAUSE_S = 21.0           # free tier: 3 requests per minute
MAX_WER = 0.12           # above this a block earns its (only) second take
MAX_TAKES = 2
MAX_VERSUCHE = 2         # per take, only for per-minute 429 / 5xx / network errors
TIMEOUT_S = 180
ZIEL_WPS = 2.4           # tie-break between equally correct takes


class KontingentErschoepft(RuntimeError):
    """Daily free-tier quota is gone — retrying today only burns time."""


# ------------------------------------------------------------------ key
def lies_schluessel() -> str:
    pfad = SCHLUESSEL_DATEI
    if not pfad.exists():
        raise SystemExit(f"Schlüsseldatei fehlt: {pfad}")
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        if zeile.strip().startswith("GEMINI_API_KEY="):
            wert = zeile.split("=", 1)[1].strip().strip('"').strip("'")
            if wert:
                return wert
    raise SystemExit(f"GEMINI_API_KEY fehlt oder ist leer in {pfad}")


def maskiere(text: str, schluessel: str | None) -> str:
    return text.replace(schluessel, "***") if schluessel else text


# ------------------------------------------------------------------ pacing
class Drossel:
    """Keeps at least `pause_s` between the end of one request and the next."""

    def __init__(self, pause_s: float) -> None:
        self.pause_s = pause_s
        self._letzte: float | None = None

    def vor_anfrage(self) -> None:
        jetzt = time.monotonic()
        if self._letzte is not None:
            rest = self.pause_s - (jetzt - self._letzte)
            if rest > 0:
                time.sleep(rest)
        self._letzte = time.monotonic()

    def nach_anfrage(self) -> None:
        self._letzte = time.monotonic()


class Sitzung:
    """One run: lazily read key, request counter, pacing."""

    def __init__(self, pause_s: float = PAUSE_S) -> None:
        self.drossel = Drossel(pause_s)
        self.anfragen = 0
        self._schluessel: str | None = None

    @property
    def schluessel(self) -> str:
        if self._schluessel is None:
            self._schluessel = lies_schluessel()
        return self._schluessel


# ------------------------------------------------------------------ gemini
def kontingent_info(roh: str) -> tuple[str, float]:
    """quotaId(s) and retryDelay from a 429 body (google.rpc error details)."""
    try:
        details = json.loads(roh)["error"].get("details", [])
    except (ValueError, KeyError, TypeError, AttributeError):
        return "", 0.0
    kontingent, verzoegerung = "", 0.0
    for d in details:
        typ = d.get("@type", "")
        if typ.endswith("QuotaFailure"):
            kontingent = ",".join(v.get("quotaId", "") for v in d.get("violations", []))
        elif typ.endswith("RetryInfo"):
            try:
                verzoegerung = float(str(d.get("retryDelay", "0s")).rstrip("s"))
            except ValueError:
                pass
    return kontingent, verzoegerung


def _protokolliere(protokoll: Path | None, etikett: str, ergebnis: str) -> None:
    """One line per HTTP request (never the key) — the quota audit trail."""
    if protokoll is None:
        return
    protokoll.parent.mkdir(parents=True, exist_ok=True)
    stempel = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with protokoll.open("a", encoding="utf-8") as f:
        f.write(f"{stempel}  {etikett}  {ergebnis}\n")


def _pcm_als_wav(pcm: bytes, mime: str) -> bytes:
    rate = int(m[1]) if (m := re.search(r"rate=(\d+)", mime)) else 24_000
    puffer = io.BytesIO()
    with wave.open(puffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return puffer.getvalue()


def tts(text: str, stimme: dict, sitzung: Sitzung, etikett: str = "",
        protokoll: Path | None = None) -> tuple[bytes, dict]:
    """One generateContent call. Returns (WAV bytes with RIFF header, usageMetadata)."""
    koerper = {
        "contents": [{"parts": [{"text": text, "speech_metadata": {"style": stimme["stil"]}}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": stimme["voice"]}}},
        },
    }
    schluessel = sitzung.schluessel
    anfrage = urllib.request.Request(
        API.format(modell=stimme["modell"]),
        data=json.dumps(koerper, ensure_ascii=False).encode("utf-8"),
        headers={"x-goog-api-key": schluessel, "Content-Type": "application/json"},
        method="POST",
    )
    letzter_fehler = ""
    for versuch in range(1, MAX_VERSUCHE + 1):
        sitzung.drossel.vor_anfrage()
        sitzung.anfragen += 1
        warte = 10.0
        try:
            with urllib.request.urlopen(anfrage, timeout=TIMEOUT_S) as antwort:
                daten = json.loads(antwort.read())
            _protokolliere(protokoll, etikett, f"versuch={versuch} HTTP 200")
            break
        except urllib.error.HTTPError as e:
            roh = maskiere(e.read().decode("utf-8", "replace"), schluessel)
            if e.code == 429:
                kontingent, verzoegerung = kontingent_info(roh)
                _protokolliere(protokoll, etikett, f"versuch={versuch} HTTP 429 {kontingent}")
                if "PerDay" in kontingent:
                    raise KontingentErschoepft(f"Tageskontingent erschöpft ({kontingent})") from None
                letzter_fehler = f"HTTP 429 ({kontingent or 'Rate-Limit'})"
                warte = max(warte, verzoegerung + 1.0)
            elif e.code in (500, 502, 503, 504):
                _protokolliere(protokoll, etikett, f"versuch={versuch} HTTP {e.code}")
                letzter_fehler = f"HTTP {e.code}: {roh[:300]}"
            else:
                _protokolliere(protokoll, etikett, f"versuch={versuch} HTTP {e.code}")
                raise RuntimeError(f"HTTP {e.code}: {roh[:500]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            _protokolliere(protokoll, etikett, f"versuch={versuch} {type(e).__name__}")
            letzter_fehler = f"Netzwerk: {type(e).__name__}"
        finally:
            sitzung.drossel.nach_anfrage()
        if versuch < MAX_VERSUCHE:
            print(f"    {letzter_fehler} — warte {warte:.0f} s, dann Versuch {versuch + 1}/{MAX_VERSUCHE}")
            time.sleep(warte)
    else:
        raise RuntimeError(f"TTS nach {MAX_VERSUCHE} Versuchen gescheitert — {letzter_fehler}")

    try:
        teil = daten["candidates"][0]["content"]["parts"][0]["inlineData"]
        audio = base64.b64decode(teil["data"])
        mime = str(teil.get("mimeType", ""))
    except (KeyError, IndexError, TypeError, ValueError):
        grund = "?"
        if isinstance(daten, dict):
            grund = (daten.get("candidates") or [{}])[0].get("finishReason", "?")
        raise RuntimeError(f"Antwort ohne Audio (finishReason={grund})") from None
    if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        if "l16" in mime.lower() or "pcm" in mime.lower():
            audio = _pcm_als_wav(audio, mime)
        else:
            raise RuntimeError(f"Antwort ist kein WAV (mimeType={mime or '?'})")
    return audio, daten.get("usageMetadata", {})


# ------------------------------------------------------------------ takes
def blocktexte(script: dict) -> list[dict]:
    texte = {z["id"]: z["text"] for z in script["vo"]}
    return [{"id": b["id"], "zeilen": list(b["zeilen"]),
             "text": "\n\n".join(texte[z] for z in b["zeilen"])} for b in script["bloecke"]]


def take_pfad(block_id: str, nummer: int, takes_dir: Path = TAKES_DIR) -> Path:
    return Path(takes_dir) / f"{block_id}_take{nummer}.wav"


def take_aktuell(block: dict, nummer: int, stimme: dict, takes_dir: Path = TAKES_DIR) -> bool:
    """A cached take only counts if it was rendered from today's text, style, voice
    and model — otherwise an edit in script.json would silently be ignored."""
    wav = take_pfad(block["id"], nummer, takes_dir)
    meta = wav.with_suffix(".json")
    if not (wav.exists() and meta.exists()):
        return False
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return (m.get("text"), m.get("stil"), m.get("voice"), m.get("modell")) == (
        block["text"], stimme["stil"], stimme["voice"], stimme["modell"])


def sichere_take(block: dict, nummer: int, stimme: dict, sitzung: Sitzung,
                 takes_dir: Path = TAKES_DIR) -> tuple[Path, bool]:
    """Return (path, newly_rendered). Makes a request only if no current take exists."""
    ziel = take_pfad(block["id"], nummer, takes_dir)
    if take_aktuell(block, nummer, stimme, takes_dir):
        return ziel, False
    wav, nutzung = tts(block["text"], stimme, sitzung, etikett=f"{block['id']}_take{nummer}",
                       protokoll=Path(takes_dir).parent / "anfragen.log")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    tmp = ziel.with_name(ziel.stem + ".part.wav")
    tmp.write_bytes(wav)
    daten, sr = lese_mono(tmp)
    dauer = len(daten) / sr
    if dauer < 0.5:
        tmp.unlink()
        raise RuntimeError(f"{ziel.name}: nur {dauer:.2f} s Audio")
    tmp.replace(ziel)
    schreibe_json(ziel.with_suffix(".json"), {
        "text": block["text"], "modell": stimme["modell"], "voice": stimme["voice"], "stil": stimme["stil"],
        "zeilen": block["zeilen"], "dauer_s": round(dauer, 3), "abtastrate": sr, "nutzung": nutzung,
        "erzeugt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"  erzeugt {ziel.name}  ({dauer:.2f} s, {nutzung.get('totalTokenCount', '?')} Tokens)")
    return ziel, True


# ------------------------------------------------------------------ judging
def bewerte(block: dict, pfad: Path, modell: str = "small") -> dict:
    tr = transkribiere(pfad, modell)
    soll = [n for w in skript_woerter(block["text"]) for n in w["norm"]]
    ist = expandiere_whisper(tr["woerter"])
    daten, sr = lese_mono(pfad)
    sprechzeit = (ist[-1][2] - ist[0][1]) if ist else 0.0
    return {
        "take": pfad.stem, "whisper": modell,
        "wer": round(wortfehlerrate(soll, [w[0] for w in ist]), 4),
        "dauer_s": round(len(daten) / sr, 3),
        "woerter": len(soll),
        "woerter_pro_s": round(len(soll) / sprechzeit, 3) if sprechzeit else 0.0,
        "transkript": tr["text"],
    }


def bewerte_mit_zweitmodell(block: dict, pfad: Path, erst: str, zweit: str) -> dict:
    """The small model mishears brand names; before a scarce request is spent on a
    second take, a bad score is re-checked with the larger model."""
    b = bewerte(block, pfad, erst)
    if b["wer"] > MAX_WER and zweit and zweit != erst:
        print(f"    {pfad.stem}: WER {b['wer']:.3f} mit {erst} — Gegenprobe mit {zweit}")
        b2 = bewerte(block, pfad, zweit)
        b2["wer_" + erst] = b["wer"]
        if b2["wer"] <= b["wer"]:
            return b2
        b["wer_" + zweit] = b2["wer"]
    return b


def waehle(bewertungen: dict[str, dict]) -> str:
    """Best word match first; on a tie the more natural pace wins."""
    return min(bewertungen, key=lambda k: (bewertungen[k]["wer"], abs(bewertungen[k]["woerter_pro_s"] - ZIEL_WPS)))


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nur-bewerten", action="store_true", help="niemals eine API-Anfrage stellen")
    p.add_argument("--whisper", default="small", help="Whisper-Modell für Prüfung und Ausrichtung")
    p.add_argument("--zweitmodell", default="medium", help="Gegenprobe vor einem zweiten Take ('' = aus)")
    a = p.parse_args(argv)

    script = lade_script()
    stimme = script["stimme"]
    bloecke = blocktexte(script)
    sitzung = Sitzung()
    abbruch = ""
    print(f"TTS {stimme['modell']} · {stimme['voice']} · {len(bloecke)} Blöcke")

    for b in bloecke:
        if take_aktuell(b, 1, stimme):
            print(f"  {b['id']}_take1 im Cache")
            continue
        if a.nur_bewerten or abbruch:
            print(f"  {b['id']}_take1 fehlt")
            continue
        try:
            sichere_take(b, 1, stimme, sitzung)
        except KontingentErschoepft as e:
            abbruch = str(e)
            print(f"  ABBRUCH: {e}", file=sys.stderr)
        except RuntimeError as e:
            abbruch = f"{b['id']}_take1: {e}"
            print(f"  FEHLER: {abbruch}", file=sys.stderr)

    auswahl, fehlend = {}, []
    for b in bloecke:
        takes = [n for n in range(1, MAX_TAKES + 1) if take_aktuell(b, n, stimme)]
        if not takes:
            fehlend.append(b["id"])
            continue
        bew = {f"take{n}": bewerte_mit_zweitmodell(b, take_pfad(b["id"], n), a.whisper, a.zweitmodell)
               for n in takes}
        beste = min(x["wer"] for x in bew.values())
        if beste > MAX_WER and len(takes) < MAX_TAKES and not (a.nur_bewerten or abbruch):
            print(f"  {b['id']}: WER {beste:.3f} > {MAX_WER} — zweiter Take")
            try:
                pfad, _ = sichere_take(b, len(takes) + 1, stimme, sitzung)
                bew[pfad.stem.split("_")[-1]] = bewerte_mit_zweitmodell(b, pfad, a.whisper, a.zweitmodell)
            except KontingentErschoepft as e:
                abbruch = str(e)
                print(f"  ABBRUCH: {e}", file=sys.stderr)
            except RuntimeError as e:
                print(f"  FEHLER beim zweiten Take: {e}", file=sys.stderr)
        sieger = waehle(bew)
        s = bew[sieger]
        auswahl[b["id"]] = {
            "take": f"{b['id']}_{sieger}", "datei": str(take_pfad(b["id"], int(sieger[4:])).relative_to(WURZEL)),
            "whisper": s["whisper"], "wer": s["wer"], "zeilen": b["zeilen"], "bewertungen": bew,
        }
        warnung = "  (über Schwelle!)" if s["wer"] > MAX_WER else ""
        print(f"  {b['id']}: {sieger}  WER {s['wer']:.3f}{warnung}  {s['dauer_s']:.1f} s  "
              f"{s['woerter_pro_s']:.2f} W/s  [{s['whisper']}]")
        print(f"      „{s['transkript']}“")

    schreibe_json(AUSWAHL, auswahl)
    print(f"API-Anfragen in diesem Lauf: {sitzung.anfragen}")
    if abbruch:
        print(f"Kontingent/Abbruch: {abbruch}", file=sys.stderr)
    if fehlend:
        print(f"FEHLT: kein Take für Block {', '.join(fehlend)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
