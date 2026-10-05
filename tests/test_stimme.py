"""Tests for audio/stimme.py — no real API request is ever made here.

urllib.request.urlopen is replaced in every test; a call that is not expected
fails the test. The key file is a temporary file with a sentinel value.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import wave

import numpy as np
import pytest

import stimme

SCHLUESSEL = "TEST-SCHLUESSEL-a1b2c3d4e5f6-NIE-AUSGEBEN"
STIMME = {"modell": "gemini-3.8-flash-lite-tts", "voice": "de-de-techagent-6",
          "stil": "moderner Werbespot, klar, warm und selbstbewusst, zügiges natürliches Tempo"}
BLOCK = {"id": "A", "zeilen": ["v01", "v02"], "text": "Drei Uhr zwölf.\n\nDie Stadt schläft."}


# ------------------------------------------------------------------ helpers
def wav_bytes(sekunden: float = 1.0, sr: int = 24_000) -> bytes:
    t = np.arange(int(sekunden * sr)) / sr
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    puffer = io.BytesIO()
    with wave.open(puffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return puffer.getvalue()


class Antwort(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def gute_antwort() -> Antwort:
    daten = {"candidates": [{"content": {"parts": [{"inlineData": {
        "mimeType": "audio/wav", "data": base64.b64encode(wav_bytes()).decode()}}]}}],
        "usageMetadata": {"totalTokenCount": 42}}
    return Antwort(json.dumps(daten).encode())


def http_fehler(code: int, koerper: dict | str) -> urllib.error.HTTPError:
    roh = koerper if isinstance(koerper, str) else json.dumps(koerper)
    return urllib.error.HTTPError(stimme.API, code, "Fehler", {}, io.BytesIO(roh.encode()))


def quota_429(quota_id: str, verzoegerung: str = "17s") -> dict:
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_requests",
                         "quotaId": quota_id}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": verzoegerung}]}}


class FalscheLeitung:
    """Stands in for urllib.request.urlopen: replays scripted results, records requests."""

    def __init__(self, *ergebnisse):
        self.ergebnisse = list(ergebnisse)
        self.anfragen = []

    def __call__(self, anfrage, timeout=None):
        self.anfragen.append(anfrage)
        if not self.ergebnisse:
            raise AssertionError("unerwartete API-Anfrage")
        e = self.ergebnisse.pop(0)
        if isinstance(e, BaseException):
            raise e
        return e


@pytest.fixture
def umgebung(tmp_path, monkeypatch):
    schluessel_datei = tmp_path / "gemini.env"
    schluessel_datei.write_text(f"# test\nGEMINI_API_KEY={SCHLUESSEL}\n", encoding="utf-8")
    monkeypatch.setattr(stimme, "SCHLUESSEL_DATEI", schluessel_datei)
    monkeypatch.setattr(stimme.time, "sleep", lambda s: None)
    return tmp_path / "takes"


def sitzung() -> stimme.Sitzung:
    return stimme.Sitzung(pause_s=0.0)


# ------------------------------------------------------------------ key hygiene
def test_schluessel_nur_im_header_nie_in_url(umgebung, monkeypatch, capsys):
    leitung = FalscheLeitung(gute_antwort())
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    s = sitzung()
    pfad, neu = stimme.sichere_take(BLOCK, 1, STIMME, s, umgebung)
    assert neu and pfad.exists()
    anfrage = leitung.anfragen[0]
    assert SCHLUESSEL not in anfrage.full_url
    assert anfrage.get_header("X-goog-api-key") == SCHLUESSEL
    koerper = json.loads(anfrage.data)
    assert koerper["contents"][0]["parts"][0]["text"] == BLOCK["text"]
    assert koerper["contents"][0]["parts"][0]["speech_metadata"] == {"style": STIMME["stil"]}
    stimme_cfg = koerper["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert stimme_cfg == {"voiceName": STIMME["voice"]}
    assert "gemini-3.8-flash-lite-tts:generateContent" in anfrage.full_url
    aus = capsys.readouterr()
    assert SCHLUESSEL not in aus.out + aus.err
    for datei in umgebung.rglob("*"):
        if datei.is_file():
            assert SCHLUESSEL.encode() not in datei.read_bytes(), datei
    assert s.anfragen == 1


def test_schluessel_nie_in_ausgabe_oder_fehlermeldung(umgebung, monkeypatch, capsys):
    """Even if an error body echoed the key, it must be masked before it is printed or raised."""
    echo = {"error": {"code": 400, "message": f"API key not valid: {SCHLUESSEL}"}}
    monkeypatch.setattr(stimme.urllib.request, "urlopen", FalscheLeitung(http_fehler(400, echo)))
    with pytest.raises(RuntimeError) as fehler:
        stimme.sichere_take(BLOCK, 1, STIMME, sitzung(), umgebung)
    assert SCHLUESSEL not in str(fehler.value)
    assert fehler.value.__cause__ is None and fehler.value.__suppress_context__
    aus = capsys.readouterr()
    assert SCHLUESSEL not in aus.out + aus.err
    log = umgebung.parent / "anfragen.log"
    if log.exists():
        assert SCHLUESSEL not in log.read_text(encoding="utf-8")


# ------------------------------------------------------------------ quota
def test_429_tageskontingent_wirft_kontingent_erschoepft_ohne_wiederholung(umgebung, monkeypatch):
    leitung = FalscheLeitung(http_fehler(429, quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier")))
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    s = sitzung()
    with pytest.raises(stimme.KontingentErschoepft, match="PerDay"):
        stimme.sichere_take(BLOCK, 1, STIMME, s, umgebung)
    assert len(leitung.anfragen) == 1 and s.anfragen == 1
    assert not stimme.take_pfad("A", 1, umgebung).exists()


def test_429_minutenlimit_wartet_einmal_und_wiederholt(umgebung, monkeypatch):
    geschlafen = []
    monkeypatch.setattr(stimme.time, "sleep", geschlafen.append)
    leitung = FalscheLeitung(http_fehler(429, quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "17s")),
                             gute_antwort())
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    s = sitzung()
    pfad, neu = stimme.sichere_take(BLOCK, 1, STIMME, s, umgebung)
    assert neu and pfad.exists() and len(leitung.anfragen) == 2 and s.anfragen == 2
    assert max(geschlafen) >= 17.0


def test_kontingent_info_liest_quota_id_und_verzoegerung():
    roh = json.dumps(quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "33.5s"))
    assert stimme.kontingent_info(roh) == ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", 33.5)
    assert stimme.kontingent_info("kein json") == ("", 0.0)


# ------------------------------------------------------------------ cache
def test_cache_verhindert_zweite_anfrage(umgebung, monkeypatch):
    leitung = FalscheLeitung(gute_antwort())
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    s = sitzung()
    _, neu1 = stimme.sichere_take(BLOCK, 1, STIMME, s, umgebung)
    _, neu2 = stimme.sichere_take(BLOCK, 1, STIMME, s, umgebung)     # FalscheLeitung is empty now
    assert (neu1, neu2) == (True, False)
    assert len(leitung.anfragen) == 1 and s.anfragen == 1
    meta = json.loads(stimme.take_pfad("A", 1, umgebung).with_suffix(".json").read_text(encoding="utf-8"))
    assert (meta["text"], meta["stil"], meta["voice"], meta["modell"]) == (
        BLOCK["text"], STIMME["stil"], STIMME["voice"], STIMME["modell"])


def test_cache_ungueltig_wenn_text_sich_aendert(umgebung, monkeypatch):
    leitung = FalscheLeitung(gute_antwort(), gute_antwort())
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    s = sitzung()
    stimme.sichere_take(BLOCK, 1, STIMME, s, umgebung)
    geaendert = {**BLOCK, "text": BLOCK["text"] + " Ihr Büro ist dunkel."}
    assert not stimme.take_aktuell(geaendert, 1, STIMME, umgebung)
    _, neu = stimme.sichere_take(geaendert, 1, STIMME, s, umgebung)
    assert neu and len(leitung.anfragen) == 2


# ------------------------------------------------------------------ gemini 3.1 (style prefix, raw PCM)
STIMME_31 = {"modell": "gemini-3.1-flash-tts-preview", "voice": "Charon", "stil_modus": "prefix",
             "stil": "Sprich als moderner Werbesprecher, klar, warm und selbstbewusst, in zügigem, natürlichem Tempo"}


def pcm_antwort(samples: np.ndarray, mime: str = "audio/l16; rate=24000; channels=1") -> Antwort:
    daten = {"candidates": [{"content": {"parts": [{"inlineData": {
        "mimeType": mime, "data": base64.b64encode(samples.astype("<i2").tobytes()).decode()}}]}}],
        "usageMetadata": {"totalTokenCount": 7}}
    return Antwort(json.dumps(daten).encode())


def rampe(n: int = 24_000) -> np.ndarray:
    return (np.sin(np.arange(n) / 9.0) * 12_000).astype(np.int16)


def test_prefix_modus_schickt_stil_im_text_ohne_speech_metadata(umgebung, monkeypatch):
    leitung = FalscheLeitung(pcm_antwort(rampe()))
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    stimme.sichere_take(BLOCK, 1, STIMME_31, sitzung(), umgebung)
    anfrage = leitung.anfragen[0]
    teil = json.loads(anfrage.data)["contents"][0]["parts"][0]
    assert teil == {"text": f"{STIMME_31['stil']}: {BLOCK['text']}"}            # no speech_metadata key
    cfg = json.loads(anfrage.data)["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert cfg == {"voiceName": "Charon"}
    assert "gemini-3.1-flash-tts-preview:generateContent" in anfrage.full_url
    meta = json.loads(stimme.take_pfad("A", 1, umgebung).with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["text"] == BLOCK["text"] and meta["stil_modus"] == "prefix"   # cache key = script text only


def test_metadata_modus_bleibt_fuer_3_8(umgebung, monkeypatch):
    leitung = FalscheLeitung(gute_antwort())
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    stimme.sichere_take(BLOCK, 1, STIMME, sitzung(), umgebung)
    teil = json.loads(leitung.anfragen[0].data)["contents"][0]["parts"][0]
    assert teil == {"text": BLOCK["text"], "speech_metadata": {"style": STIMME["stil"]}}


@pytest.mark.parametrize("mime, rate, kanaele", [
    ("audio/l16; rate=24000; channels=1", 24_000, 1),
    ("audio/L16;codec=pcm;rate=16000", 16_000, 1),
    ("audio/l16; rate=48000; channels=2", 48_000, 2),
])
def test_rohes_pcm_l16_wird_wav(umgebung, monkeypatch, mime, rate, kanaele):
    import soundfile as sf
    pcm = rampe(rate * kanaele)                                   # 1 s of audio
    monkeypatch.setattr(stimme.urllib.request, "urlopen", FalscheLeitung(pcm_antwort(pcm, mime)))
    pfad, _ = stimme.sichere_take(BLOCK, 1, STIMME_31, sitzung(), umgebung)
    assert pfad.read_bytes()[:4] == b"RIFF"
    info = sf.info(str(pfad))
    assert (info.samplerate, info.channels, info.subtype) == (rate, kanaele, "PCM_16")
    daten, _ = sf.read(str(pfad), dtype="int16", always_2d=True)
    assert np.array_equal(daten.reshape(-1), pcm)                 # byte-exact, little-endian
    meta = json.loads(pfad.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["abtastrate"] == rate and meta["dauer_s"] == pytest.approx(1.0)


def test_rohdaten_ohne_riff_und_ohne_pcm_mimetype_sind_fehler(umgebung, monkeypatch):
    monkeypatch.setattr(stimme.urllib.request, "urlopen", FalscheLeitung(pcm_antwort(rampe(), "audio/mpeg")))
    with pytest.raises(RuntimeError, match="kein WAV"):
        stimme.sichere_take(BLOCK, 1, STIMME_31, sitzung(), umgebung)


def test_ungerade_pcm_laenge_ist_fehler(umgebung, monkeypatch):
    daten = {"candidates": [{"content": {"parts": [{"inlineData": {
        "mimeType": "audio/l16; rate=24000; channels=1", "data": base64.b64encode(b"\x01\x02\x03").decode()}}]}}]}
    monkeypatch.setattr(stimme.urllib.request, "urlopen", FalscheLeitung(Antwort(json.dumps(daten).encode())))
    with pytest.raises(RuntimeError, match="PCM"):
        stimme.sichere_take(BLOCK, 1, STIMME_31, sitzung(), umgebung)


def test_cache_unterscheidet_modell_voice_und_stilmodus(umgebung, monkeypatch):
    monkeypatch.setattr(stimme.urllib.request, "urlopen", FalscheLeitung(gute_antwort()))
    stimme.sichere_take(BLOCK, 1, STIMME, sitzung(), umgebung)                 # a 3.8 take
    assert stimme.take_aktuell(BLOCK, 1, STIMME, umgebung)
    assert not stimme.take_aktuell(BLOCK, 1, STIMME_31, umgebung)
    assert not stimme.take_aktuell(BLOCK, 1, {**STIMME, "modell": STIMME_31["modell"]}, umgebung)
    assert not stimme.take_aktuell(BLOCK, 1, {**STIMME, "voice": "Charon"}, umgebung)
    assert not stimme.take_aktuell(BLOCK, 1, {**STIMME, "stil": STIMME_31["stil"]}, umgebung)
    assert not stimme.take_aktuell(BLOCK, 1, {**STIMME, "stil_modus": "prefix"}, umgebung)
    leitung = FalscheLeitung(pcm_antwort(rampe()))
    monkeypatch.setattr(stimme.urllib.request, "urlopen", leitung)
    _, neu = stimme.sichere_take(BLOCK, 1, STIMME_31, sitzung(), umgebung)  # model change -> new request
    assert neu and len(leitung.anfragen) == 1
    assert stimme.take_aktuell(BLOCK, 1, STIMME_31, umgebung)
    assert not stimme.take_aktuell(BLOCK, 1, STIMME, umgebung)


def test_alter_sidecar_ohne_stilmodus_gilt_als_metadata(umgebung):
    pfad = stimme.take_pfad("A", 1, umgebung)
    pfad.parent.mkdir(parents=True)
    pfad.write_bytes(wav_bytes())
    pfad.with_suffix(".json").write_text(json.dumps(
        {"text": BLOCK["text"], "stil": STIMME["stil"], "voice": STIMME["voice"], "modell": STIMME["modell"]}),
        encoding="utf-8")
    assert stimme.take_aktuell(BLOCK, 1, STIMME, umgebung)
    assert not stimme.take_aktuell(BLOCK, 1, {**STIMME, "stil_modus": "prefix"}, umgebung)


def test_wer_referenz_ist_nur_der_skripttext(umgebung, monkeypatch):
    """The style prefix is an instruction, not script: if the voice spoke it, that is an error."""
    pfad = stimme.take_pfad("A", 1, umgebung)
    pfad.parent.mkdir(parents=True)
    pfad.write_bytes(wav_bytes())

    def transkript(text):
        woerter = [{"wort": " " + w, "start": i * 0.3, "ende": i * 0.3 + 0.25} for i, w in enumerate(text.split())]
        return lambda p, modell="small": {"modell": modell, "text": text, "woerter": woerter}

    monkeypatch.setattr(stimme, "transkribiere", transkript("Drei Uhr zwölf. Die Stadt schläft."))
    assert stimme.bewerte(BLOCK, pfad)["wer"] == 0.0
    monkeypatch.setattr(stimme, "transkribiere", transkript(f"{STIMME_31['stil']}: Drei Uhr zwölf. Die Stadt schläft."))
    assert stimme.bewerte(BLOCK, pfad)["wer"] > 1.0


def test_drossel_haelt_pause_zwischen_anfragen(monkeypatch):
    uhr = iter([100.0, 100.0, 105.0, 105.0])
    geschlafen = []
    monkeypatch.setattr(stimme.time, "monotonic", lambda: next(uhr))
    monkeypatch.setattr(stimme.time, "sleep", geschlafen.append)
    d = stimme.Drossel(21.0)
    d.vor_anfrage()          # first request: no wait
    d.vor_anfrage()          # 5 s later: wait the remaining 16 s
    assert geschlafen == [pytest.approx(16.0)]


def test_blocktexte_verbinden_zeilen_mit_absatz():
    script = {"vo": [{"id": "v01", "text": "Eins."}, {"id": "v02", "text": "Zwei."}, {"id": "v03", "text": "Drei."}],
              "bloecke": [{"id": "A", "zeilen": ["v01", "v02"]}, {"id": "B", "zeilen": ["v03"]}]}
    bloecke = stimme.blocktexte(script)
    assert [b["text"] for b in bloecke] == ["Eins.\n\nZwei.", "Drei."]
    assert [b["zeilen"] for b in bloecke] == [["v01", "v02"], ["v03"]]
