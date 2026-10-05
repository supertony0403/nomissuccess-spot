"""Tests for the timeline contract (audio/plan.py) and the word alignment.

The timeline is built from the real script.json + szenen/events.json, with a
synthetic, deterministic alignment per line (no audio, no Whisper). The
alignment tests use real Whisper output from the voice probe (fixture) and a
hand-written transcript with merged/split tokens.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

import plan
from gemeinsam import (EVENTS, FPS, expandiere_whisper, lade_json, lade_script,
                       normalisiere, skript_woerter, wortfehlerrate, zahl_de)

FIXTURES = Path(__file__).parent / "fixtures"
EPS = 1e-6


# ------------------------------------------------------------------ helpers
def synthetische_ausrichtung(script: dict) -> dict:
    """Plausible per-line alignment: word length ~ characters, 80 ms lead-in,
    140 ms tail, file length padded to whole frames (like plan.schneide_zeilen)."""
    aus = {}
    for zeile in script["vo"]:
        t = 0.08
        woerter = []
        for w in skript_woerter(zeile["text"]):
            dauer = 0.12 + 0.055 * len(w["wort"])
            woerter.append({"wort": w["wort"], "start_s": round(t, 4), "ende_s": round(t + dauer, 4)})
            t += dauer + 0.04
        ende = woerter[-1]["ende_s"] + 0.14
        frames = -(-round(ende * 48_000) // (48_000 // FPS))
        aus[zeile["id"]] = {"dauer_s": frames / FPS, "woerter": woerter}
    return aus


def ausrichtung_aus_whisper(zeilen_ids: list[str], fixture: str, script: dict) -> dict:
    """Run the real aligner on a fixture transcript and convert the take times
    into per-line file times the way plan.py does (cut = first word - 80 ms)."""
    roh = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    texte = {z["id"]: z["text"] for z in script["vo"]}
    zuordnung = plan.richte_aus([(zid, texte[zid]) for zid in zeilen_ids], roh["woerter"], roh["dauer_s"])
    aus = {}
    for zid in zeilen_ids:
        woerter = zuordnung[zid]
        anfang = max(0.0, woerter[0]["start_s"] - 0.08)
        ende = woerter[-1]["ende_s"] + 0.14
        aus[zid] = {
            "dauer_s": math.ceil((ende - anfang) * FPS) / FPS,
            "woerter": [{"wort": w["wort"], "start_s": round(w["start_s"] - anfang, 4),
                         "ende_s": round(w["ende_s"] - anfang, 4)} for w in woerter],
        }
    return aus


@pytest.fixture(scope="module")
def script() -> dict:
    return lade_script()


@pytest.fixture(scope="module")
def events_doc() -> dict:
    return lade_json(EVENTS)


@pytest.fixture(scope="module")
def timeline(script, events_doc) -> dict:
    return plan.baue_timeline(script, events_doc, synthetische_ausrichtung(script))


def szene_von(tl: dict, sid: str) -> dict:
    if sid == "abspann":
        return {"id": "abspann", **tl["abspann"]}
    return next(s for s in tl["szenen"] if s["id"] == sid)


# ------------------------------------------------------------------ scenes
def test_szenen_lueckenlos_und_aufsteigend(timeline, script):
    szenen = timeline["szenen"]
    assert [s["id"] for s in szenen] == [s["id"] for s in script["szenen"]]
    assert szenen[0]["start_s"] == 0.0 and szenen[0]["start_f"] == 0
    for a, b in zip(szenen, szenen[1:]):
        assert a["ende_f"] == b["start_f"], f"Lücke/Überlappung zwischen {a['id']} und {b['id']}"
        assert a["ende_s"] == pytest.approx(b["start_s"], abs=EPS)
    for s in szenen:
        assert s["ende_f"] > s["start_f"], s["id"]
    assert szenen[-1]["ende_f"] == timeline["abspann"]["start_f"]
    assert timeline["abspann"]["ende_f"] == timeline["dauer_f"]
    assert timeline["abspann"]["ende_s"] - timeline["abspann"]["start_s"] == pytest.approx(2.0, abs=1 / FPS)


def test_jede_vo_zeile_liegt_in_ihrer_szene(timeline):
    for vo in timeline["vo"]:
        s = szene_von(timeline, vo["szene"])
        assert s["start_s"] - EPS <= vo["start_s"] < vo["ende_s"] <= s["ende_s"] + EPS, vo["id"]
        assert s["start_f"] <= vo["start_f"] < vo["ende_f"] <= s["ende_f"], vo["id"]
    for a, b in zip(timeline["vo"], timeline["vo"][1:]):
        assert a["ende_s"] <= b["start_s"] + EPS, f"{a['id']} überlappt {b['id']}"


def test_woerter_monoton_und_im_clip(timeline):
    for vo in timeline["vo"]:
        vorher = vo["start_s"]
        for w in vo["woerter"]:
            assert w["start_s"] >= vorher - EPS, (vo["id"], w)
            assert w["ende_s"] >= w["start_s"], (vo["id"], w)
            vorher = w["ende_s"]
        assert vorher <= vo["ende_s"] + EPS, vo["id"]


def test_timing_regeln(timeline):
    vo = {v["id"]: v for v in timeline["vo"]}
    toleranz = 1.5 / FPS                                   # frame-snapped placement
    assert vo["v01"]["woerter"][0]["start_s"] == pytest.approx(plan.VORLAUF_S, abs=toleranz)
    zeilen = timeline["vo"]
    for a, b in zip(zeilen, zeilen[1:]):
        luecke = b["woerter"][0]["start_s"] - a["woerter"][-1]["ende_s"]
        soll = plan.ZEILEN_LUECKE_S if a["szene"] == b["szene"] else plan.SZENEN_LUECKE_S
        assert luecke == pytest.approx(soll, abs=2 * toleranz), (a["id"], b["id"], luecke)
        if a["szene"] != b["szene"]:
            grenze = szene_von(timeline, b["szene"])["start_s"]
            assert grenze - a["woerter"][-1]["ende_s"] == pytest.approx(plan.SZENEN_LUECKE_S / 2, abs=toleranz)
    s8 = szene_von(timeline, "s8_morgen")
    assert s8["ende_s"] - vo["v23"]["woerter"][-1]["ende_s"] == pytest.approx(plan.NACHLAUF_S, abs=toleranz)


def test_frames_sind_gerundete_sekunden(timeline):
    def pruefe(s: float, f: int, was: str) -> None:
        assert f == round(s * FPS), f"{was}: f={f}, s={s}"

    pruefe(timeline["dauer_s"], timeline["dauer_f"], "dauer")
    pruefe(timeline["abspann"]["start_s"], timeline["abspann"]["start_f"], "abspann.start")
    pruefe(timeline["abspann"]["ende_s"], timeline["abspann"]["ende_f"], "abspann.ende")
    for s in timeline["szenen"]:
        pruefe(s["start_s"], s["start_f"], s["id"])
        pruefe(s["ende_s"], s["ende_f"], s["id"])
    for v in timeline["vo"]:
        pruefe(v["start_s"], v["start_f"], v["id"])
        pruefe(v["ende_s"], v["ende_f"], v["id"])
    for e in timeline["events"]:
        pruefe(e["t_s"], e["f"], e["id"])
    for c in timeline["cues"]:
        pruefe(c["t_s"], c["f"], c["quelle"])


# ------------------------------------------------------------------ events
def test_alle_events_aufgeloest_und_in_ihrer_szene(timeline, events_doc):
    soll = {e["id"] for liste in events_doc["events"].values() for e in liste}
    ist = {e["id"] for e in timeline["events"]}
    assert ist == soll
    for e in timeline["events"]:
        s = szene_von(timeline, e["szene"])
        assert s["start_s"] - EPS <= e["t_s"] <= s["ende_s"] + EPS, (e["id"], e["t_s"], s)
        assert set(e) >= {"id", "szene", "t_s", "f", "sfx", "ebene", "bild"}


def test_cue_je_event_mit_sfx(timeline):
    mit_sfx = [e for e in timeline["events"] if e.get("sfx")]
    assert len(timeline["cues"]) == len(mit_sfx)
    quellen = {c["quelle"]: c for c in timeline["cues"]}
    for e in mit_sfx:
        c = quellen[f"event:{e['id']}"]
        assert c["art"] == e["sfx"] and c["t_s"] == e["t_s"] and c["szene"] == e["szene"]


def test_event_zeit_ist_wortzeit_plus_offset(timeline):
    vo = {v["id"]: v for v in timeline["vo"]}
    ev = {e["id"]: e for e in timeline["events"]}
    zwoelf = next(w for w in vo["v01"]["woerter"] if w["wort"] == "zwölf")
    assert ev["clock_roll"]["t_s"] == pytest.approx(zwoelf["ende_s"], abs=1e-3)
    schlaeft = next(w for w in vo["v01"]["woerter"] if w["wort"] == "schläft")
    assert ev["sleep_letters"]["t_s"] == pytest.approx(schlaeft["start_s"] - 0.1, abs=1e-3)
    s1 = szene_von(timeline, "s1_nacht")
    assert ev["ribbon_enter"]["t_s"] == pytest.approx(s1["start_s"] + 0.25, abs=1e-3)
    assert ev["endcard_in"]["t_s"] == pytest.approx(timeline["abspann"]["start_s"], abs=1e-3)


def test_n_tes_vorkommen(script, events_doc):
    doc = copy.deepcopy(events_doc)
    doc["events"]["s2_website"].append(
        {"id": "zweites_keine", "anker": {"vo": "v05", "wort": "keine", "n": 2}, "offset_s": 0.0,
         "sfx": "", "ebene": "fusion", "bild": "test"})
    tl = plan.baue_timeline(script, doc, synthetische_ausrichtung(script))
    v05 = next(v for v in tl["vo"] if v["id"] == "v05")
    keine = [w for w in v05["woerter"] if normalisiere(w["wort"]) == "keine"]
    ev = next(e for e in tl["events"] if e["id"] == "zweites_keine")
    assert ev["t_s"] == pytest.approx(keine[1]["start_s"], abs=1e-3)
    assert not any(c["quelle"] == "event:zweites_keine" for c in tl["cues"])   # no sfx -> no cue


def test_nicht_aufloesbarer_anker_nennt_event_und_wort(script, events_doc):
    doc = copy.deepcopy(events_doc)
    doc["events"]["s3_netz"][0]["anker"] = {"vo": "v09", "wort": "Quantencomputer"}
    with pytest.raises(plan.AnkerFehler) as fehler:
        plan.baue_timeline(script, doc, synthetische_ausrichtung(script))
    meldung = str(fehler.value)
    assert "dive_into_pixels" in meldung and "Quantencomputer" in meldung and "v09" in meldung


def test_event_ausserhalb_der_szene_ist_fehler(script, events_doc):
    doc = copy.deepcopy(events_doc)
    doc["events"]["s1_nacht"][0]["offset_s"] = -5.0          # before the spot starts
    with pytest.raises(plan.AnkerFehler, match="ribbon_enter"):
        plan.baue_timeline(script, doc, synthetische_ausrichtung(script))


def test_fehlende_zeile_ist_fehler(script, events_doc):
    aus = synthetische_ausrichtung(script)
    del aus["v14"]
    with pytest.raises(ValueError, match="v14"):
        plan.baue_timeline(script, events_doc, aus)


def test_pruefe_timeline_findet_ueberlappung(timeline):
    kaputt = copy.deepcopy(timeline)
    kaputt["szenen"][2]["start_f"] += 3
    kaputt["szenen"][2]["start_s"] = kaputt["szenen"][2]["start_f"] / FPS
    assert any("s3_netz" in f for f in plan.pruefe_timeline(kaputt))
    assert plan.pruefe_timeline(timeline) == []


# ------------------------------------------------------------------ cutting lines from a block take
def _synthetischer_take(pfad: Path, bursts: list[tuple[float, float]], dauer: float = 3.0, sr: int = 24_000) -> None:
    import soundfile as sf
    t = np.arange(int(dauer * sr)) / sr
    x = 1e-4 * np.random.default_rng(7).standard_normal(len(t))           # ~ -80 dBFS floor
    for a, b in bursts:
        m = (t >= a) & (t < b)
        x[m] += 0.3 * np.sin(2 * np.pi * 220 * t[m]) + 0.1 * np.sin(2 * np.pi * 1300 * t[m])
    sf.write(str(pfad), x, sr, subtype="PCM_16")


def _datei_s(e: dict, take_s: float) -> float:
    """Take time -> time in the line file (inverse of plan.schneide_block's offset)."""
    return take_s - e["quelle_von_s"] + e["stille_vorne_s"]


def _take_s(e: dict, datei_s: float) -> float:
    return datei_s - e["stille_vorne_s"] + e["quelle_von_s"]


def test_schnitt_folgt_der_energie_nicht_whispers_raendern(tmp_path):
    """Whisper puts the first word at 0.0 although speech starts at 0.30 (seen in the
    real take A) and ends the next line 0.2 s late — the cut must follow the audio."""
    import pyloudnorm
    import soundfile as sf

    take = tmp_path / "T_take1.wav"
    _synthetischer_take(take, [(0.30, 1.00), (1.60, 2.40)])
    zuordnung = {
        "a": [{"wort": "Eins", "start_s": 0.0, "ende_s": 0.5, "art": "erkannt"},
              {"wort": "zwei", "start_s": 0.5, "ende_s": 0.95, "art": "erkannt"}],
        "b": [{"wort": "drei", "start_s": 1.70, "ende_s": 2.0, "art": "erkannt"},
              {"wort": "vier", "start_s": 2.0, "ende_s": 2.60, "art": "erkannt"}],
    }
    zeilen, info = plan.schneide_block("T", take, zuordnung, ["a", "b"], ziel_dir=tmp_path / "vo")
    a, b = zeilen["a"], zeilen["b"]
    tol = 0.011                                                          # one 10 ms envelope window
    for e, (von, bis) in ((a, (0.30, 1.00)), (b, (1.60, 2.40))):        # where the bursts really are
        assert _take_s(e, e["woerter"][0]["start_s"]) == pytest.approx(von, abs=tol)
        assert _take_s(e, e["woerter"][-1]["ende_s"]) == pytest.approx(bis, abs=tol)
        assert e["woerter"][0]["start_s"] == pytest.approx(plan.VOR_S, abs=tol)
        assert e["stille_vorne_s"] == 0.0                               # the take had room for the lead-in
    for name, e in (("a", a), ("b", b)):
        y, sr = sf.read(str(tmp_path / "vo" / f"{name}.wav"))
        info_datei = sf.info(str(tmp_path / "vo" / f"{name}.wav"))
        assert sr == 48_000 and info_datei.subtype == "FLOAT" and len(y) % (48_000 // FPS) == 0
        assert e["dauer_s"] == pytest.approx(len(y) / sr)
        assert np.max(np.abs(y[:48])) < 0.02 and np.max(np.abs(y[-48:])) < 1e-3          # faded edges
        assert e["woerter"][-1]["ende_s"] + plan.NACH_S <= e["dauer_s"] + 1e-6
    ya, _ = sf.read(str(tmp_path / "vo" / "a.wav"))
    ende_a = int(round((_datei_s(a, 1.00) + 0.02) * 48_000))
    assert np.max(np.abs(ya[ende_a:])) < 1e-3                                    # nothing of line b in a
    # one gain for the whole block: take x gain measures -18 LUFS, and every file is
    # exactly that scaled take (away from the fades)
    from gemeinsam import lese_mono, resample
    x, sr_take = lese_mono(take)
    x48 = resample(x, sr_take, 48_000)
    gain = 10 ** (info["verstaerkung_db"] / 20)
    assert pyloudnorm.Meter(48_000).integrated_loudness(x48 * gain) == pytest.approx(plan.ZIEL_LUFS, abs=0.05)
    for name, e in (("a", a), ("b", b)):
        y, _ = sf.read(str(tmp_path / "vo" / f"{name}.wav"))
        i0 = int(round(e["quelle_von_s"] * 48_000))
        innen = slice(2000, 2000 + 20_000)                               # 0.04 s ... 0.46 s into the file
        # rtol: verstaerkung_db is stored rounded to 0.01 dB
        assert np.allclose(y[innen], x48[i0:][innen] * gain, rtol=2e-3, atol=1e-6), name


def test_schnitt_polstert_fehlenden_vorlauf_mit_stille(tmp_path):
    take = tmp_path / "T_take1.wav"
    _synthetischer_take(take, [(0.02, 0.80)], dauer=0.85)
    zuordnung = {"a": [{"wort": "Eins", "start_s": 0.02, "ende_s": 0.80, "art": "erkannt"}]}
    zeilen, _ = plan.schneide_block("T", take, zuordnung, ["a"], ziel_dir=tmp_path / "vo")
    e = zeilen["a"]
    assert e["woerter"][0]["start_s"] == pytest.approx(plan.VOR_S, abs=0.011)
    assert e["stille_vorne_s"] > 0.04
    assert e["dauer_s"] >= e["woerter"][-1]["ende_s"] + plan.NACH_S - 1e-6


# ------------------------------------------------------------------ provisional lines (quota gone)
def test_endgueltige_timeline_hat_keine_vorlaeufigen_zeilen(timeline):
    assert timeline["vorlaeufig"] == []
    assert all(not v.get("vorlaeufig") and v["datei"] == f"assets/audio/vo/{v['id']}.wav" for v in timeline["vo"])


def test_geschaetzte_zeilen_sind_markiert_und_kalibriert(script, events_doc):
    gemessen = synthetische_ausrichtung(script)
    a_ids = script["bloecke"][0]["zeilen"]
    basis = {z: gemessen[z] for z in a_ids}
    fehlend = [z["id"] for z in script["vo"] if z["id"] not in basis]
    geschaetzt = plan.schaetze_zeilen(script, fehlend, basis)
    assert set(geschaetzt) == set(fehlend)
    for zid, e in geschaetzt.items():
        assert e["vorlaeufig"] is True and e["datei"] is None
        assert e["dauer_s"] * FPS == pytest.approx(round(e["dauer_s"] * FPS), abs=1e-3)   # whole frames
        assert e["woerter"][0]["start_s"] == pytest.approx(plan.VOR_S, abs=1e-6)
        assert e["woerter"][-1]["ende_s"] <= e["dauer_s"]
        echt = gemessen[zid]                    # same synthetic voice -> estimate within 15 %
        sp_echt = echt["woerter"][-1]["ende_s"] - echt["woerter"][0]["start_s"]
        sp_sch = e["woerter"][-1]["ende_s"] - e["woerter"][0]["start_s"]
        assert sp_sch == pytest.approx(sp_echt, rel=0.15), zid
    tl = plan.baue_timeline(script, events_doc, {**basis, **geschaetzt})
    assert tl["vorlaeufig"] == fehlend
    vo = {v["id"]: v for v in tl["vo"]}
    assert vo["v09"]["vorlaeufig"] is True and vo["v09"]["datei"] is None
    assert not vo["v01"].get("vorlaeufig") and vo["v01"]["datei"] == "assets/audio/vo/v01.wav"
    assert plan.pruefe_timeline(tl) == []


# ------------------------------------------------------------------ alignment (Review Focus 1)
def test_zahlwoerter():
    assert zahl_de(3) == "drei" and zahl_de(12) == "zwölf" and zahl_de(7) == "sieben"
    assert zahl_de(30) == "dreißig" and zahl_de(21) == "einundzwanzig" and zahl_de(100) == "hundert"
    assert zahl_de(0) == "null" and zahl_de(1) == "eins" and zahl_de(2026) == "zweitausendsechsundzwanzig"


def test_whisper_uhrzeit_wird_skriptreihenfolge():
    roh = [{"wort": " 3", "start": 0.0, "ende": 0.44}, {"wort": ".12", "start": 0.44, "ende": 0.88},
           {"wort": " Uhr.", "start": 0.88, "ende": 1.28}, {"wort": " Die", "start": 1.4, "ende": 1.52}]
    woerter = expandiere_whisper(roh)
    assert [w[0] for w in woerter] == ["drei", "uhr", "zwölf", "die"]
    assert woerter[0][1] == 0.0 and woerter[2][2] == pytest.approx(1.28)
    assert all(a[2] <= b[1] + EPS for a, b in zip(woerter, woerter[1:]))


def test_whisper_ziffern_werden_zahlwoerter():
    roh = [{"wort": " dauert", "start": 0.0, "ende": 0.3}, {"wort": " 30", "start": 0.3, "ende": 0.7},
           {"wort": " Minuten.", "start": 0.7, "ende": 1.1}, {"wort": " 7", "start": 1.5, "ende": 1.8}]
    assert [w[0] for w in expandiere_whisper(roh)] == ["dauert", "dreissig", "minuten", "sieben"]


def test_ueberlappende_whisper_woerter_werden_monoton(script):
    roh = [{"wort": " Die", "start": 1.0, "ende": 1.5}, {"wort": " Frage", "start": 1.3, "ende": 1.8},
           {"wort": " ist", "start": 1.7, "ende": 1.9}, {"wort": " nur,", "start": 1.9, "ende": 2.2}]
    z = plan.richte_aus([("x", "Die Frage ist nur:")], roh, 3.0)["x"]
    for a, b in zip(z, z[1:]):
        assert b["start_s"] >= a["ende_s"] - EPS and b["ende_s"] >= b["start_s"], (a, b)


def test_probe_transkript_hat_keine_wortfehler(script):
    roh = json.loads((FIXTURES / "whisper_probe_v01_v03.json").read_text(encoding="utf-8"))
    soll = [n for z in script["vo"][:3] for w in skript_woerter(z["text"]) for n in w["norm"]]
    ist = [w[0] for w in expandiere_whisper(roh["woerter"])]
    assert wortfehlerrate(soll, ist) == 0.0


def test_ausrichtung_probe_loest_zwoelf_auf(script):
    """Real Whisper output writes „3 .12 Uhr“ for „Drei Uhr zwölf“ (merged AND reordered)."""
    roh = json.loads((FIXTURES / "whisper_probe_v01_v03.json").read_text(encoding="utf-8"))
    texte = {z["id"]: z["text"] for z in script["vo"]}
    z = plan.richte_aus([(i, texte[i]) for i in ("v01", "v02", "v03")], roh["woerter"], roh["dauer_s"])
    v01 = {w["wort"]: w for w in z["v01"]}
    assert set(v01) >= {"Drei", "Uhr", "zwölf"}
    assert 0.0 <= v01["Drei"]["start_s"] < v01["Uhr"]["start_s"] < v01["zwölf"]["start_s"]
    assert v01["zwölf"]["ende_s"] == pytest.approx(1.28, abs=0.05)       # end of „Uhr.“ token
    assert v01["zwölf"]["ende_s"] <= v01["Die"]["start_s"]
    assert v01["dunkel"]["ende_s"] == pytest.approx(3.32, abs=0.01)
    assert z["v03"][-1]["wort"] == "jemanden" and z["v03"][-1]["ende_s"] == pytest.approx(11.82, abs=0.01)
    alle = [w for zid in ("v01", "v02", "v03") for w in z[zid]]
    assert all(a["ende_s"] <= b["start_s"] + EPS and b["ende_s"] >= b["start_s"] for a, b in zip(alle, alle[1:]))


def test_timeline_mit_probe_ausrichtung_loest_alle_anker(script, events_doc):
    aus = synthetische_ausrichtung(script)
    aus.update(ausrichtung_aus_whisper(["v01", "v02", "v03"], "whisper_probe_v01_v03.json", script))
    tl = plan.baue_timeline(script, events_doc, aus)
    assert plan.pruefe_timeline(tl) == []
    ev = {e["id"]: e for e in tl["events"]}
    v01 = next(v for v in tl["vo"] if v["id"] == "v01")
    zwoelf = next(w for w in v01["woerter"] if w["wort"] == "zwölf")
    assert ev["clock_roll"]["t_s"] == pytest.approx(zwoelf["ende_s"], abs=1e-3)
    assert ev["clock_roll"]["t_s"] < ev["sleep_letters"]["t_s"] < ev["lights_off"]["t_s"] < ev["led_1"]["t_s"]


def test_ausrichtung_getrennte_und_verschmolzene_marke(script, events_doc):
    """„No Miss Success“ (split), „Nomissuccess.de.“ (merged with URL), „30“, „7 Uhr“
    and a swallowed word („kostet“) — every anchor of scene 8 still resolves."""
    roh = json.loads((FIXTURES / "whisper_synthetisch_v21_v23.json").read_text(encoding="utf-8"))
    texte = {z["id"]: z["text"] for z in script["vo"]}
    z = plan.richte_aus([(i, texte[i]) for i in ("v21", "v22", "v23")], roh["woerter"], roh["dauer_s"])
    v22 = {w["wort"]: w for w in z["v22"]}
    assert v22["nomissuccess"]["start_s"] == pytest.approx(4.30) and v22["nomissuccess"]["ende_s"] == pytest.approx(5.20)
    v23 = z["v23"]
    namen = [w["wort"] for w in v23]
    assert namen[-3:] == ["nomissuccess", "punkt", "de"]
    marke, punkt, de = v23[-3:]
    assert 11.80 <= marke["start_s"] < punkt["start_s"] < de["start_s"] < de["ende_s"] <= 13.30 + EPS
    dreissig = next(w for w in v23 if w["wort"] == "dreißig")
    assert (dreissig["start_s"], dreissig["ende_s"]) == pytest.approx((9.52, 9.98))
    assert dreissig["art"] == "erkannt"                                    # „30“ read as the number word
    kostet = next(w for w in v23 if w["wort"] == "kostet")
    assert kostet["art"] == "interpoliert"                                 # swallowed: fills the whole gap
    assert (kostet["start_s"], kostet["ende_s"]) == pytest.approx((10.52, 10.90))
    assert z["v21"][0]["art"] == "erkannt" and z["v21"][1]["wort"] == "Uhr"
    assert z["v21"][1]["ende_s"] == pytest.approx(0.86)

    aus = synthetische_ausrichtung(script)
    aus.update(ausrichtung_aus_whisper(["v21", "v22", "v23"], "whisper_synthetisch_v21_v23.json", script))
    tl = plan.baue_timeline(script, events_doc, aus)
    assert plan.pruefe_timeline(tl) == []
    ev = {e["id"]: e for e in tl["events"]}
    assert ev["clock_roll_7"]["t_s"] < ev["dawn"]["t_s"] < ev["bell_sleep"]["t_s"] < ev["logo_fold"]["t_s"]
    assert ev["logo_fold"]["t_s"] < ev["claim"]["t_s"] < ev["cta"]["t_s"] < ev["url"]["t_s"]
