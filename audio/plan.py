"""Timeline planner — the picture follows the voice.

1. Aligns the chosen take of each block (work/vo/auswahl.json) word by word with
   script.json: difflib over normalised words, after Whisper's digits, clock
   times, merged and split tokens were expanded to script spelling. Script words
   Whisper did not produce get interpolated times between their neighbours.
2. Cuts each block into lines -> assets/audio/vo/<id>.wav (48 kHz mono 32-bit
   float): first word - 80 ms ... last word + 140 ms, never into the neighbouring
   line, 20 ms fades, file padded to whole frames, one gain per block to -18 LUFS
   integrated (no limiter, no compressor). The raw TTS has a peak-to-loudness
   ratio of ~21 dB, so at -18 LUFS its peaks sit above 0 dBFS: float keeps them
   unclipped, and sample/true peak per block are recorded for the mix stage.
3. Places lines and scenes by the timing rules below, resolves every event of
   szenen/events.json to an absolute time/frame and derives one SFX cue per event.

Writes work/vo/ausrichtung.json, timeline.json (THE contract) and
work/vo/timeline_bericht.txt.

If a block has no take yet (daily TTS quota gone), plan.py stops — unless it is
run with --vorlaeufig: then the missing lines get durations estimated from the
measured lines (speaking-rate model fitted on them), are marked
"vorlaeufig": true with "datei": null, and timeline.json lists them in its
top-level "vorlaeufig". No audio is invented for them.

Usage:  .venv/bin/python audio/plan.py                 # align + cut + plan
        .venv/bin/python audio/plan.py --vorlaeufig    # same, missing blocks estimated
        .venv/bin/python audio/plan.py --nur-timeline  # re-plan from ausrichtung.json
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from gemeinsam import (AUSRICHTUNG, AUSWAHL, BERICHT, EVENTS, FPS, SR, TIMELINE, VO_DIR, WURZEL, db,
                       expandiere_whisper, lade_json, lade_script, lese_mono, normalisiere, resample,
                       schreibe_json, schreibe_wav, skript_woerter, transkribiere)

# ------------------------------------------------------------------ timing rules
VORLAUF_S = 1.6            # scene 1: first word starts here
ZEILEN_LUECKE_S = 0.30     # last word -> first word, same scene
SZENEN_LUECKE_S = 1.10     # last word -> first word across a cut; the cut sits in the middle
NACHLAUF_S = 2.8           # scene 8 holds after the last word
ABSPANN_S = 2.0            # end card

# ------------------------------------------------------------------ cutting
VOR_S = 0.08               # lead-in before the first word
NACH_S = 0.14              # tail after the last word
BLENDE_S = 0.02            # fade in/out
ZIEL_LUFS = -18.0          # integrated, per block (one gain per block)
SUCH_S = 0.25              # energy search window around Whisper's line edges
SCHWELLE_DB = -35.0        # "speech" = 10 ms RMS within 35 dB of the block's loudest window
                           # (-40 caught breaths before the next line as speech)
FENSTER_S = 0.01

# brand names reported with what Whisper heard; phrases are matched as a whole
MARKEN = (("nomissuccess", "punkt", "de"), ("zero", "trust"), ("nomissuccess",), ("fortigate",),
          ("opnsense",), ("vmware", "lizenz"), ("proxmox",), ("ransomware",))


class AnkerFehler(ValueError):
    """An event of szenen/events.json cannot be placed (unknown word, outside its scene)."""


def _f(s: float) -> int:
    return int(round(s * FPS))


def _s(f: int) -> float:
    return round(f / FPS, 6)


# ------------------------------------------------------------------ alignment
def _verteile(von: float, bis: float, gewichte: list[int]) -> list[tuple[float, float]]:
    g = np.array([max(x, 1) for x in gewichte], dtype=float)
    kanten = von + (bis - von) * np.concatenate([[0.0], np.cumsum(g) / g.sum()])
    return [(float(a), float(b)) for a, b in zip(kanten, kanten[1:])]


def richte_aus(zeilen: list[tuple[str, str]], whisper_roh: list[dict],
               dauer_s: float | None = None) -> dict[str, list[dict]]:
    """Give every script word of a block a start/end in take time.

    `zeilen` = [(vo_id, text), ...] in spoken order, `whisper_roh` = faster-whisper
    words ({"wort", "start", "ende"}). Returns {vo_id: [{"wort", "start_s", "ende_s",
    "art"}]} with words in script spelling; art = erkannt (exact match), ersetzt
    (1:1 mismatch), verteilt (n:m span split by length), interpoliert (not heard).
    """
    token: list[dict] = []
    norm: list[str] = []
    besitzer: list[int] = []
    for zid, text in zeilen:
        for w in skript_woerter(text):
            for n in w["norm"]:
                norm.append(n)
                besitzer.append(len(token))
            token.append({"zid": zid, "wort": w["wort"]})

    hyp = expandiere_whisper(whisper_roh)
    zeiten: list[tuple[float, float] | None] = [None] * len(norm)
    art: list[str] = ["interpoliert"] * len(norm)
    sm = SequenceMatcher(None, norm, [h[0] for h in hyp], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                zeiten[i1 + k] = (hyp[j1 + k][1], hyp[j1 + k][2])
                art[i1 + k] = "erkannt" if op == "equal" else "ersetzt"
        elif op == "replace":
            for k, z in enumerate(_verteile(hyp[j1][1], hyp[j2 - 1][2], [len(n) for n in norm[i1:i2]])):
                zeiten[i1 + k] = z
                art[i1 + k] = "verteilt"

    ende_gesamt = dauer_s if dauer_s is not None else (hyp[-1][2] if hyp else 0.0)
    k = 0
    while k < len(norm):                       # unheard words: share the gap between neighbours
        if zeiten[k] is not None:
            k += 1
            continue
        m = k
        while m < len(norm) and zeiten[m] is None:
            m += 1
        links = zeiten[k - 1][1] if k > 0 else 0.0
        rechts = max(links, zeiten[m][0] if m < len(norm) else max(ende_gesamt, links))
        for i, z in enumerate(_verteile(links, rechts, [len(n) for n in norm[k:m]])):
            zeiten[k + i] = z
        k = m

    vorher = 0.0
    for k, (a, b) in enumerate(zeiten):         # enforce monotonic, non-negative spans
        a = max(a, vorher)
        zeiten[k] = (a, max(a, b))
        vorher = zeiten[k][1]

    rang = {"erkannt": 0, "ersetzt": 1, "verteilt": 2, "interpoliert": 3}
    teile: dict[int, list[int]] = {}
    for k, ti in enumerate(besitzer):
        teile.setdefault(ti, []).append(k)
    out: dict[str, list[dict]] = {zid: [] for zid, _ in zeilen}
    for ti, t in enumerate(token):
        ks = teile[ti]
        out[t["zid"]].append({
            "wort": t["wort"],
            "start_s": round(zeiten[ks[0]][0], 4),
            "ende_s": round(zeiten[ks[-1]][1], 4),
            "art": max((art[k] for k in ks), key=rang.__getitem__),
        })
    return out


def markenbericht(zuordnung: dict[str, list[dict]], whisper_roh: list[dict]) -> list[dict]:
    """What Whisper actually heard where the script says a brand name (longest
    phrase first, each script word reported once)."""
    rang = {"erkannt": 0, "ersetzt": 1, "verteilt": 2, "interpoliert": 3}
    out = []
    for zid, woerter in zuordnung.items():
        norm = [" ".join(normalisiere(w["wort"]).split()) for w in woerter]
        belegt: set[int] = set()
        for phrase in MARKEN:
            ziel = " ".join(phrase)
            for i in range(len(woerter)):
                for j in range(i + 1, len(woerter) + 1):
                    if " ".join(norm[i:j]) != ziel or belegt & set(range(i, j)):
                        continue
                    von, bis = woerter[i]["start_s"], woerter[j - 1]["ende_s"]
                    gehoert = [r["wort"].strip() for r in whisper_roh if r["start"] < bis and r["ende"] > von]
                    out.append({"zeile": zid, "wort": " ".join(w["wort"] for w in woerter[i:j]),
                                "art": max((w["art"] for w in woerter[i:j]), key=rang.__getitem__),
                                "whisper": " ".join(gehoert) or "—", "start_s": von})
                    belegt |= set(range(i, j))
    return sorted(out, key=lambda m: (m["zeile"], m.pop("start_s")))


# ------------------------------------------------------------------ cutting
def _laute_fenster(pegel: np.ndarray, schwelle: float, von_s: float, bis_s: float) -> np.ndarray:
    k0 = max(0, int(math.floor(von_s / FENSTER_S)))
    k1 = min(len(pegel), int(math.ceil(bis_s / FENSTER_S)))
    if k1 <= k0:
        return np.array([], dtype=int)
    return np.flatnonzero(pegel[k0:k1] > schwelle) + k0


def schneide_block(block_id: str, take: Path, zuordnung: dict[str, list[dict]],
                   zeilen_ids: list[str], ziel_dir: Path = VO_DIR) -> tuple[dict, dict]:
    """Cut one block take into line files. Returns (lines, block info) for ausrichtung.json."""
    import pyloudnorm

    x, sr = lese_mono(take)
    x48 = resample(x, sr, SR)
    dauer = len(x48) / SR
    lufs = float(pyloudnorm.Meter(SR).integrated_loudness(x48))
    if not math.isfinite(lufs):
        raise ValueError(f"{take.name}: Lautheit nicht messbar (Stille?)")
    gain_db = ZIEL_LUFS - lufs
    gain = 10 ** (gain_db / 20)

    w = int(FENSTER_S * SR)
    n = len(x48) // w
    pegel = 20 * np.log10(np.sqrt(np.mean(x48[: n * w].reshape(n, w) ** 2, axis=1)) + 1e-12)
    schwelle = float(pegel.max()) + SCHWELLE_DB

    # 1) refine each line's outer edges with the energy envelope, never past the
    #    midpoint to a neighbour line. Whisper's edges drift: it put the first word
    #    of take A at 0.00 s although the voice starts at 0.30 s, so the search spans
    #    the whole first/last word plus SUCH_S outwards.
    woerter = {z: [dict(w) for w in zuordnung[z]] for z in zeilen_ids}
    kanten = []
    for i, z in enumerate(zeilen_ids):
        erstes, letztes = woerter[z][0], woerter[z][-1]
        ws, we = erstes["start_s"], letztes["ende_s"]
        mitte_l = (woerter[zeilen_ids[i - 1]][-1]["ende_s"] + ws) / 2 if i > 0 else 0.0
        mitte_r = (we + woerter[zeilen_ids[i + 1]][0]["start_s"]) / 2 if i + 1 < len(zeilen_ids) else dauer
        laut = _laute_fenster(pegel, schwelle, max(mitte_l, ws - SUCH_S), max(erstes["ende_s"], ws + FENSTER_S))
        anfang = float(laut[0]) * FENSTER_S if laut.size else ws
        laut = _laute_fenster(pegel, schwelle, min(letztes["start_s"], we - FENSTER_S), min(we + SUCH_S, mitte_r))
        ende = float(laut[-1] + 1) * FENSTER_S if laut.size else we
        woerter[z][0]["start_s"] = round(min(anfang, woerter[z][0]["ende_s"]), 4)
        woerter[z][-1]["ende_s"] = round(max(ende, woerter[z][-1]["start_s"]), 4)
        kanten.append((woerter[z][0]["start_s"], woerter[z][-1]["ende_s"]))

    # 2) cut with lead-in/tail, clamped so no file reaches into the neighbour's speech
    zeilen, spitzen = {}, []
    pro_frame = SR // FPS
    blende = int(BLENDE_S * SR)
    for i, z in enumerate(zeilen_ids):
        links = kanten[i - 1][1] if i > 0 else 0.0
        rechts = kanten[i + 1][0] if i + 1 < len(zeilen_ids) else dauer
        a = int(round(max(kanten[i][0] - VOR_S, links, 0.0) * SR))
        b = int(round(min(kanten[i][1] + NACH_S, rechts, dauer) * SR))
        y = x48[a:b] * gain
        y[:blende] *= np.linspace(0.0, 1.0, blende)
        y[-blende:] *= np.linspace(1.0, 0.0, blende)
        # every file = >= 80 ms before the first word + speech + >= 140 ms after the
        # last word; where the take has no room (take edge, close neighbour) pad silence
        vorne = max(0, int(round((VOR_S - (kanten[i][0] - a / SR)) * SR)))
        hinten = max(0, int(round((NACH_S - (b / SR - kanten[i][1])) * SR)))
        y = np.concatenate([np.zeros(vorne), y, np.zeros(hinten)])
        y = np.concatenate([y, np.zeros((-len(y)) % pro_frame)])
        spitze = float(np.max(np.abs(y)))
        spitzen.append(spitze)
        schreibe_wav(ziel_dir / f"{z}.wav", y.astype(np.float32), SR, "FLOAT")
        start = (a - vorne) / SR                     # take time of the file's first sample
        zeilen[z] = {
            "block": block_id, "take": take.stem, "quelle_von_s": round(a / SR, 4), "quelle_bis_s": round(b / SR, 4),
            "stille_vorne_s": round(vorne / SR, 4),
            "dauer_s": round(len(y) / SR, 6), "spitze_dbfs": round(db(spitze), 2),
            "datei": str((ziel_dir / f"{z}.wav").relative_to(WURZEL)) if ziel_dir.is_relative_to(WURZEL) else str(ziel_dir / f"{z}.wav"),
            "woerter": [{"wort": w["wort"], "start_s": round(w["start_s"] - start, 4),
                         "ende_s": round(w["ende_s"] - start, 4), "art": w["art"]} for w in woerter[z]],
        }
    true_peak = db(float(np.max(np.abs(resample(x48 * gain, SR, 4 * SR)))))
    info = {"take": take.stem, "lufs_vorher": round(lufs, 2), "verstaerkung_db": round(gain_db, 2),
            "spitze_dbfs": round(db(max(spitzen)), 2), "true_peak_dbtp": round(true_peak, 2),
            "plr_db": round(true_peak - ZIEL_LUFS, 1), "dauer_s": round(dauer, 3)}
    return zeilen, info


# ------------------------------------------------------------------ estimate (no take yet)
_SATZENDE = re.compile(r"[.?!:;…]\W*$")
_KOMMA = re.compile(r"[,–—]\W*$")
STANDARD_MODELL = {"s_pro_zeichen": 0.055, "s_pro_wort": 0.10, "satzpause_s": 0.35, "kommapause_s": 0.12}


def _merkmale(text: str) -> list[tuple[str, int, bool, bool]]:
    """(word, characters, sentence break after it, comma after it) per spoken word."""
    out = []
    for roh in text.split():
        if normalisiere(roh):
            out.append((re.sub(r"^\W+|\W+$", "", roh), len(normalisiere(roh).replace(" ", "")),
                        bool(_SATZENDE.search(roh)), bool(_KOMMA.search(roh))))
    return out


def schaetzmodell(script: dict, gemessen: dict) -> dict:
    """Fit seconds per character / word / sentence pause / comma pause on the measured
    lines (non-negative least squares). Falls back to STANDARD_MODELL (~2.4 words/s)."""
    from scipy.optimize import nnls

    texte = {z["id"]: z["text"] for z in script["vo"]}
    zeilen, ziel = [], []
    for zid, e in gemessen.items():
        m = _merkmale(texte[zid])
        innen = m[:-1]
        zeilen.append([sum(x[1] for x in m), len(m), sum(x[2] for x in innen), sum(x[3] for x in innen)])
        ziel.append(e["woerter"][-1]["ende_s"] - e["woerter"][0]["start_s"])
    if len(zeilen) >= 4:
        koeff, _ = nnls(np.array(zeilen, dtype=float), np.array(ziel, dtype=float))
        modell = dict(zip(STANDARD_MODELL, (round(float(k), 4) for k in koeff)))
        tempo = sum(z[1] for z in zeilen) / sum(ziel)
        if koeff[0] + koeff[1] > 0 and 1.2 <= tempo <= 4.5:
            return {**modell, "quelle": f"angepasst an {len(zeilen)} gemessene Zeilen ({tempo:.2f} W/s)"}
    return {**STANDARD_MODELL, "quelle": "Standardwerte (keine gemessenen Zeilen)"}


def schaetze_zeilen(script: dict, zeilen_ids: list[str], gemessen: dict) -> dict:
    """Provisional line entries for lines without a take: word times from the
    speaking-rate model, 80 ms lead-in, 140 ms tail, length on whole frames."""
    modell = schaetzmodell(script, gemessen)
    texte = {z["id"]: z["text"] for z in script["vo"]}
    bloecke = {z: b["id"] for b in script["bloecke"] for z in b["zeilen"]}
    out = {}
    for zid in zeilen_ids:
        m = _merkmale(texte[zid])
        t, woerter = VOR_S, []
        for i, (wort, zeichen, satz, komma) in enumerate(m):
            dauer = modell["s_pro_zeichen"] * zeichen + modell["s_pro_wort"]
            woerter.append({"wort": wort, "start_s": round(t, 4), "ende_s": round(t + dauer, 4), "art": "geschaetzt"})
            t += dauer
            if i + 1 < len(m):
                t += modell["satzpause_s"] if satz else modell["kommapause_s"] if komma else 0.0
        out[zid] = {"block": bloecke.get(zid), "vorlaeufig": True, "datei": None,
                    "dauer_s": round(math.ceil((woerter[-1]["ende_s"] + NACH_S) * FPS) / FPS, 6),
                    "woerter": woerter, "schaetzmodell": modell}
    return out


def richte_und_schneide(script: dict, vorlaeufig: bool = False) -> dict:
    """Align + cut every block from work/vo/auswahl.json; returns ausrichtung.json content.
    Blocks without a take stop the run, or with `vorlaeufig` get estimated lines."""
    if not AUSWAHL.exists() and not vorlaeufig:
        raise SystemExit("work/vo/auswahl.json fehlt — zuerst audio/stimme.py laufen lassen")
    auswahl = lade_json(AUSWAHL) if AUSWAHL.exists() else {}
    texte = {z["id"]: z["text"] for z in script["vo"]}
    aus = {"_hinweis": "erzeugt von audio/plan.py; Wortzeiten relativ zum Anfang der Zeilendatei",
           "ziel_lufs": ZIEL_LUFS, "bloecke": {}, "zeilen": {}, "marken": [], "vorlaeufig": []}
    ohne_take = [b for b in script["bloecke"] if not auswahl.get(b["id"])]
    if ohne_take and not vorlaeufig:
        raise SystemExit(f"Block {', '.join(b['id'] for b in ohne_take)}: noch kein Take (Tageskontingent?) — "
                         f"audio/stimme.py erneut laufen lassen, oder --vorlaeufig für einen gekennzeichneten "
                         f"Schätz-Zeitplan")
    for blk in script["bloecke"]:
        wahl = auswahl.get(blk["id"])
        if not wahl:
            continue
        take = WURZEL / wahl["datei"]
        meta = lade_json(take.with_suffix(".json"))
        soll = "\n\n".join(texte[z] for z in blk["zeilen"])
        if meta.get("text") != soll:
            raise SystemExit(f"{take.name} passt nicht mehr zum Skript — audio/stimme.py neu laufen lassen")
        tr = transkribiere(take, wahl.get("whisper", "small"))
        x, sr = lese_mono(take)
        zuordnung = richte_aus([(z, texte[z]) for z in blk["zeilen"]], tr["woerter"], len(x) / sr)
        aus["marken"].extend(markenbericht(zuordnung, tr["woerter"]))
        zeilen, info = schneide_block(blk["id"], take, zuordnung, blk["zeilen"])
        alle = [w for z in blk["zeilen"] for w in zuordnung[z]]
        info.update({"whisper": tr["modell"], "wer": wahl.get("wer"),
                     "woerter": len(alle),
                     "nicht_gehoert": [f"{z}:{w['wort']}" for z in blk["zeilen"] for w in zuordnung[z]
                                       if w["art"] == "interpoliert"]})
        aus["bloecke"][blk["id"]] = info
        aus["zeilen"].update(zeilen)
    if ohne_take:
        fehlend = [z for b in ohne_take for z in b["zeilen"]]
        aus["zeilen"].update(schaetze_zeilen(script, fehlend, dict(aus["zeilen"])))
        aus["vorlaeufig"] = [z["id"] for z in script["vo"] if z["id"] in fehlend]
    aus["zeilen"] = {z["id"]: aus["zeilen"][z["id"]] for z in script["vo"] if z["id"] in aus["zeilen"]}
    return aus


# ------------------------------------------------------------------ timeline
def baue_timeline(script: dict, events_doc: dict, zeilen: dict) -> dict:
    """Place lines and scenes, resolve events. `zeilen` = {vo_id: {"dauer_s", "woerter":
    [{"wort", "start_s", "ende_s"}] relative to the line file}} (ausrichtung["zeilen"])."""
    szenen_ids = [s["id"] for s in script["szenen"]]
    fehlend = [z["id"] for z in script["vo"] if z["id"] not in zeilen or not zeilen[z["id"]].get("woerter")]
    if fehlend:
        raise ValueError(f"Ausrichtung fehlt für {', '.join(fehlend)}")
    reihenfolge = [szenen_ids.index(z["szene"]) for z in script["vo"]]
    if reihenfolge != sorted(reihenfolge) or set(reihenfolge) != set(range(len(szenen_ids))):
        raise ValueError("vo-Zeilen müssen in Szenenreihenfolge stehen und jede Szene muss Text haben")

    szene_start = {szenen_ids[0]: 0}
    vo = []
    vorige, wortende = None, 0.0
    for z in script["vo"]:
        a = zeilen[z["id"]]
        erster = a["woerter"][0]["start_s"]
        if vorige is None:
            wunsch = VORLAUF_S
        elif z["szene"] == vorige:
            wunsch = wortende + ZEILEN_LUECKE_S
        else:
            grenze = _f(wortende + SZENEN_LUECKE_S / 2)
            szene_start[z["szene"]] = grenze
            wunsch = grenze / FPS + SZENEN_LUECKE_S / 2
        start_f = _f(wunsch - erster)
        if start_f < 0:
            raise ValueError(f"{z['id']}: Vorlauf der Datei ({erster:.3f} s) länger als der verfügbare Platz")
        start = start_f / FPS
        woerter = [{"wort": w["wort"], "start_s": round(start + w["start_s"], 4),
                    "ende_s": round(start + w["ende_s"], 4)} for w in a["woerter"]]
        # files are whole frames; dauer_s is stored rounded (6 decimals), so allow 1e-3 frame
        ende_f = start_f + math.ceil(a["dauer_s"] * FPS - 1e-3)
        geschaetzt = bool(a.get("vorlaeufig"))
        vo.append({"id": z["id"], "szene": z["szene"], "text": z["text"],
                   "start_s": _s(start_f), "ende_s": _s(ende_f), "start_f": start_f, "ende_f": ende_f,
                   "datei": None if geschaetzt else f"assets/audio/vo/{z['id']}.wav",
                   "vorlaeufig": geschaetzt, "woerter": woerter})
        vorige, wortende = z["szene"], woerter[-1]["ende_s"]

    ende_f = _f(wortende + NACHLAUF_S)
    titel = {s["id"]: s for s in script["szenen"]}
    szenen = []
    for i, sid in enumerate(szenen_ids):
        a = szene_start[sid]
        b = szene_start[szenen_ids[i + 1]] if i + 1 < len(szenen_ids) else ende_f
        szenen.append({"id": sid, "titel": titel[sid].get("titel", ""), "farbe": titel[sid].get("farbe", ""),
                       "start_s": _s(a), "ende_s": _s(b), "start_f": a, "ende_f": b})
    abspann_f = (ende_f, ende_f + _f(ABSPANN_S))
    abspann = {"start_s": _s(abspann_f[0]), "ende_s": _s(abspann_f[1]),
               "start_f": abspann_f[0], "ende_f": abspann_f[1]}

    events = loese_events(events_doc, szenen, abspann, vo)
    vorlaeufig = [v["id"] for v in vo if v["vorlaeufig"]]
    hinweis = "erzeugt von audio/plan.py aus script.json + szenen/events.json + Sprachaufnahme — nicht von Hand ändern"
    if vorlaeufig:
        hinweis += (f". VORLÄUFIG: {', '.join(vorlaeufig)} haben noch keine Aufnahme (Dauer geschätzt, datei=null); "
                    f"alle Zeiten ab der ersten geschätzten Zeile verschieben sich mit der echten Stimme")
    tl = {
        "_hinweis": hinweis, "vorlaeufig": vorlaeufig,
        "fps": FPS, "dauer_s": abspann["ende_s"], "dauer_f": abspann["ende_f"],
        "stimme": {k: script.get("stimme", {}).get(k) for k in ("modell", "voice")},
        "abspann": abspann, "szenen": szenen, "vo": vo, "events": events,
        "cues": [{"t_s": e["t_s"], "f": e["f"], "art": e["sfx"], "szene": e["szene"], "quelle": f"event:{e['id']}"}
                 for e in events if e.get("sfx")],
    }
    fehler = pruefe_timeline(tl)
    if fehler:
        raise ValueError("Timeline verletzt den Vertrag:\n  " + "\n  ".join(fehler))
    return tl


def loese_events(events_doc: dict, szenen: list[dict], abspann: dict, vo: list[dict]) -> list[dict]:
    bereiche = {s["id"]: s for s in szenen} | {"abspann": {"id": "abspann", **abspann}}
    zeilen = {v["id"]: v for v in vo}
    out, fehler = [], []
    for sid, liste in events_doc["events"].items():
        if sid not in bereiche:
            fehler.append(f"Szene '{sid}' aus events.json gibt es nicht")
            continue
        s = bereiche[sid]
        for ev in liste:
            eid, anker = ev.get("id", "?"), ev.get("anker", {})
            if anker.get("szenenstart"):
                basis = s["start_s"]
            elif anker.get("szenenende"):
                basis = s["ende_s"]
            else:
                vid, wort = anker.get("vo"), str(anker.get("wort", ""))
                n, kante = int(anker.get("n", 1)), anker.get("kante", "start")
                if vid not in zeilen:
                    fehler.append(f"Event '{eid}' ({sid}): Zeile '{vid}' gibt es nicht (gesucht: '{wort}')")
                    continue
                if kante not in ("start", "ende"):
                    fehler.append(f"Event '{eid}' ({sid}): kante '{kante}' ist weder 'start' noch 'ende'")
                    continue
                treffer = [w for w in zeilen[vid]["woerter"] if normalisiere(w["wort"]) == normalisiere(wort)]
                if len(treffer) < n or n < 1:
                    vorhanden = " ".join(w["wort"] for w in zeilen[vid]["woerter"])
                    fehler.append(f"Event '{eid}' ({sid}): Wort '{wort}' (Vorkommen {n}) nicht in {vid} "
                                  f"gefunden — {vid}: „{vorhanden}“")
                    continue
                basis = treffer[n - 1]["start_s" if kante == "start" else "ende_s"]
            t = round(basis + float(ev.get("offset_s", 0.0)), 4)
            f = _f(t)
            if not (s["start_f"] <= f < s["ende_f"]):
                fehler.append(f"Event '{eid}' landet bei {t:.3f} s (Frame {f}) außerhalb seiner Szene {sid} "
                              f"({s['start_s']:.3f}–{s['ende_s']:.3f} s)")
                continue
            out.append({"id": eid, "szene": sid, "t_s": t, "f": f, "sfx": ev.get("sfx", ""),
                        "ebene": ev.get("ebene", ""), "bild": ev.get("bild", ""),
                        "anker": anker, "offset_s": ev.get("offset_s", 0.0)})
    if fehler:
        raise AnkerFehler("Nicht auflösbare Events:\n  " + "\n  ".join(fehler))
    return sorted(out, key=lambda e: (e["t_s"], e["id"]))


def pruefe_timeline(tl: dict) -> list[str]:
    """Contract check: gap-free scenes, frames = round(s*60), every line inside its
    scene, monotonic words, events inside their scene. Returns the violations."""
    fehler: list[str] = []

    def frames(obj: dict, schluessel: str, was: str) -> None:
        if obj[f"{schluessel}_f"] != _f(obj[f"{schluessel}_s"]):
            fehler.append(f"{was}: {schluessel}_f={obj[f'{schluessel}_f']} ≠ round({obj[f'{schluessel}_s']}·{FPS})")

    if tl.get("fps") != FPS:
        fehler.append(f"fps {tl.get('fps')} ≠ {FPS}")
    frames(tl, "dauer", "dauer")
    ab = tl["abspann"]
    for k in ("start", "ende"):
        frames(ab, k, "abspann")
    szenen = tl["szenen"]
    if not szenen or szenen[0]["start_f"] != 0:
        fehler.append("erste Szene beginnt nicht bei 0")
    for s in szenen:
        for k in ("start", "ende"):
            frames(s, k, s["id"])
        if s["ende_f"] <= s["start_f"]:
            fehler.append(f"{s['id']}: Länge ≤ 0")
    for a, b in zip(szenen, szenen[1:]):
        if a["ende_f"] != b["start_f"]:
            fehler.append(f"Lücke/Überlappung zwischen {a['id']} und {b['id']} "
                          f"({a['ende_f']} ≠ {b['start_f']})")
    if szenen and szenen[-1]["ende_f"] != ab["start_f"]:
        fehler.append(f"{szenen[-1]['id']} endet nicht am Abspann")
    if ab["ende_f"] != tl["dauer_f"]:
        fehler.append("Abspann endet nicht am Spotende")

    bereiche = {s["id"]: s for s in szenen} | {"abspann": ab}
    vorher = None
    for v in tl["vo"]:
        for k in ("start", "ende"):
            frames(v, k, v["id"])
        s = bereiche.get(v["szene"])
        if s is None:
            fehler.append(f"{v['id']}: Szene {v['szene']} unbekannt")
            continue
        if not (s["start_f"] <= v["start_f"] < v["ende_f"] <= s["ende_f"]):
            fehler.append(f"{v['id']} ({v['start_s']:.3f}–{v['ende_s']:.3f} s) liegt nicht ganz in {v['szene']} "
                          f"({s['start_s']:.3f}–{s['ende_s']:.3f} s)")
        if vorher is not None and v["start_f"] < vorher["ende_f"]:
            fehler.append(f"{v['id']} überlappt {vorher['id']}")
        t = v["start_s"] - 1e-6
        for w in v["woerter"]:
            if w["start_s"] < t or w["ende_s"] < w["start_s"]:
                fehler.append(f"{v['id']}: Wort '{w['wort']}' nicht monoton ({w['start_s']}–{w['ende_s']})")
            t = w["ende_s"] - 1e-6
        if v["woerter"] and v["woerter"][-1]["ende_s"] > v["ende_s"] + 1e-6:
            fehler.append(f"{v['id']}: letztes Wort endet nach dem Clip")
        vorher = v

    for e in tl["events"]:
        if e["f"] != _f(e["t_s"]):
            fehler.append(f"Event {e['id']}: f ≠ round(t·{FPS})")
        s = bereiche.get(e["szene"])
        if s is None or not (s["start_f"] <= e["f"] < s["ende_f"]):
            fehler.append(f"Event {e['id']} liegt nicht in {e['szene']}")
    for c in tl["cues"]:
        if c["f"] != _f(c["t_s"]):
            fehler.append(f"Cue {c['quelle']}: f ≠ round(t·{FPS})")
    return fehler


# ------------------------------------------------------------------ report
def _zeit(s: float) -> str:
    m, r = divmod(s, 60)
    return f"{int(m)}:{r:05.2f}"


def bericht(tl: dict, ausrichtung: dict | None = None) -> str:
    z = ["nomissuccess „Nachtschicht“ — Zeitplan (audio/plan.py)"]
    if tl.get("vorlaeufig"):
        z += [f"VORLÄUFIG — ohne Aufnahme, Dauer geschätzt (*): {', '.join(tl['vorlaeufig'])}"]
    z += [
         f"Gesamt {_zeit(tl['dauer_s'])}  ({tl['dauer_f']} Frames @ {tl['fps']} fps), "
         f"Spot bis Abspann {_zeit(tl['abspann']['start_s'])}, Abspann {tl['abspann']['ende_s'] - tl['abspann']['start_s']:.2f} s",
         "", f"{'Szene':<14}{'Titel':<14}{'Start':>9}{'Ende':>10}{'Dauer':>9}  {'Zeilen':<10}{'Events':>6}"]
    for s in tl["szenen"] + [{"id": "abspann", "titel": "made by", **tl["abspann"]}]:
        ids = [v["id"] for v in tl["vo"] if v["szene"] == s["id"]]
        n_ev = sum(1 for e in tl["events"] if e["szene"] == s["id"])
        zeilen = f"{ids[0]}–{ids[-1]}" if len(ids) > 1 else (ids[0] if ids else "—")
        z.append(f"{s['id']:<14}{s.get('titel', ''):<14}{_zeit(s['start_s']):>9}{_zeit(s['ende_s']):>10}"
                 f"{s['ende_s'] - s['start_s']:>8.2f}s  {zeilen:<10}{n_ev:>6}")

    woerter = sum(len(v["woerter"]) for v in tl["vo"])
    sprechzeit = sum(v["woerter"][-1]["ende_s"] - v["woerter"][0]["start_s"] for v in tl["vo"])
    spanne = tl["vo"][-1]["woerter"][-1]["ende_s"] - tl["vo"][0]["woerter"][0]["start_s"]
    z += ["", f"Sprechtempo: {woerter} Wörter / {sprechzeit:.1f} s reine Zeilenzeit = {woerter / sprechzeit:.2f} Wörter/s; "
              f"inkl. aller Pausen {woerter / spanne:.2f} Wörter/s",
          "", f"{'Zeile':<6}{'Szene':<14}{'Start':>9}{'Ende':>10}{'Dauer':>8}{'Wörter':>8}{'W/s':>6}"]
    for v in tl["vo"]:
        sp = v["woerter"][-1]["ende_s"] - v["woerter"][0]["start_s"]
        z.append(f"{v['id']:<6}{v['szene']:<14}{_zeit(v['start_s']):>9}{_zeit(v['ende_s']):>10}"
                 f"{v['ende_s'] - v['start_s']:>7.2f}s{len(v['woerter']):>8}{len(v['woerter']) / sp:>6.2f}"
                 + ("  * geschätzt" if v.get("vorlaeufig") else ""))
    z += ["", f"Events: {len(tl['events'])} (alle aufgelöst), Cues: {len(tl['cues'])}"]
    if ausrichtung:
        z += ["", "Blöcke (Lautheit: eine Verstärkung je Block auf "
              f"{ausrichtung.get('ziel_lufs', ZIEL_LUFS):.0f} LUFS):"]
        for bid, b in ausrichtung.get("bloecke", {}).items():
            nicht = ", ".join(b.get("nicht_gehoert", [])) or "—"
            z.append(f"  {bid}: {b['take']}  WER {b.get('wer')}  {b['lufs_vorher']:.1f} LUFS → "
                     f"{b['verstaerkung_db']:+.1f} dB, Spitze {b['spitze_dbfs']:+.1f} dBFS, "
                     f"True Peak {b.get('true_peak_dbtp', float('nan')):+.1f} dBTP (float, ungeclippt)  "
                     f"[{b.get('whisper')}]  nicht gehört/interpoliert: {nicht}")
        if ausrichtung.get("marken"):
            z += ["", "Markennamen (Skript → was Whisper hörte):"]
            z += [f"  {m['zeile']} {m['wort']:<16} → {m['whisper']}  ({m['art']})" for m in ausrichtung["marken"]]
    return "\n".join(z) + "\n"


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nur-timeline", action="store_true",
                   help="Audio nicht neu schneiden, nur timeline.json aus ausrichtung.json neu planen")
    p.add_argument("--vorlaeufig", action="store_true",
                   help="Blöcke ohne Take mit geschätzten, markierten Zeilen füllen (datei=null)")
    a = p.parse_args(argv)

    script = lade_script()
    events_doc = lade_json(EVENTS)
    if a.nur_timeline:
        ausrichtung = lade_json(AUSRICHTUNG)
    else:
        ausrichtung = richte_und_schneide(script, vorlaeufig=a.vorlaeufig)
        schreibe_json(AUSRICHTUNG, ausrichtung)
    try:
        tl = baue_timeline(script, events_doc, ausrichtung["zeilen"])
    except ValueError as e:
        print(f"FEHLER: {e}", file=sys.stderr)
        return 2
    schreibe_json(TIMELINE, tl)
    text = bericht(tl, ausrichtung)
    BERICHT.parent.mkdir(parents=True, exist_ok=True)
    BERICHT.write_text(text, encoding="utf-8")
    print(text)
    print(f"geschrieben: {TIMELINE.relative_to(WURZEL)}, {AUSRICHTUNG.relative_to(WURZEL)}, "
          f"{BERICHT.relative_to(WURZEL)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
