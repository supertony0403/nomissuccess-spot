# nomissuccess — Spot „Nachtschicht“ (2026)

Werbespot für [nomissuccess.de](https://nomissuccess.de): eine Nacht von 03:12 bis 07:00, in der die IT
arbeitet, bis morgens niemand geweckt wurde. **2:28,7** (8921 Frames, 60 fps), 16:9 und 9:16, je mit Ton und stumm.

Gebaut in **DaVinci Resolve Studio 21** (Projekt „nomiss“, Timelines `nomissuccess 16x9` / `nomissuccess 9x16`),
3D-Shots aus **Blender 5.2**, kinetische Typo als **Fusion**-Comps, Stimme **Gemini TTS**
(`gemini-3.1-flash-tts-preview`, Charon), Musik und Geräusche komplett aus Code.

Brief: [`brief/PROMPT.md`](brief/PROMPT.md) · Bauplan: [`docs/superpowers/plans/2026-10-05-nomissuccess-spot.md`](docs/superpowers/plans/2026-10-05-nomissuccess-spot.md)

## Ergebnis (nicht im Repo, `out/` ist gitignored)

| Datei | Format | Ton |
|---|---|---|
| `out/nomissuccess-nachtschicht-16x9.mp4` | 1920×1080, 60 fps, H.264 | −14,3 LUFS, −1,1 dBTP |
| `out/nomissuccess-nachtschicht-9x16.mp4` | 1080×1920, 60 fps, H.264 | −14,3 LUFS, −1,1 dBTP |
| `out/…-stumm.mp4` | wie oben | ohne Ton |
| `~/Videos/nomissuccess-spot/*-master.mov` | ProRes 422 HQ + PCM | Master |

Kontaktbögen: `out/kontakt-16x9.jpg`, `out/kontakt-9x16.jpg`.

## Wie alles zusammenhängt

`script.json` (Text) + `szenen/events.json` (Bild-Events, an gesprochene Wörter gekoppelt)
→ `audio/plan.py` → **`timeline.json`** (der Vertrag: Szenen, Wörter, Events, SFX-Cues in Sekunden und Frames).
Alles andere liest nur `timeline.json`:

1. `audio/stimme.py` — Gemini-TTS in drei Blöcken, Whisper-Ausrichtung (Schlüssel nur in `~/.config/ddr-video/gemini.env`)
2. `audio/plan.py` — Zeitplan; `audio/mix.py` — SFX, Musik, Ducking, Master (−14 LUFS / −1 dBTP), Stems
3. `blender/shots/sN_*.py` — 3D-Shots (1440² bzw. 1920², 60 fps, je Szene + 0,5 s Handles) → `renders/` (Symlink auf `/mnt/steam-library`)
4. `fusion/build_all.py` — 20 Typo-/HUD-/Abspann-Comps für beide Formate → `work/fusion/`
5. `resolve/shots_kodieren.py` → `resolve/bauen.py --frisch` → `resolve/rendern.py` — Timelines, Master, MP4s

## Fallen (alle gefunden am 05./06.10.2026)

- Resolve rendert nur in Ordner unter **Media Storage** (`~/Videos`), sonst gibt `AddRenderJob()` stumm `''` zurück. Ein Symlink in `~/Videos` wird akzeptiert.
- Resolve unter Linux hat **kein H.264** → ProRes-Master, MP4 per ffmpeg.
- **Grafikspeicher**: Blender-Renders und Resolve auf derselben Arc B580 → `ImportFusionComp` stürzt ab. Erst Blender fertig, dann Resolve.
- **Fusion-Bewegungsunschärfe** mit Quality 8–24 machte den Render 5× langsamer → Obergrenze 4, nur in Bewegung.
- Große Medien gehören auf `/mnt/steam-library`, die Systemplatte lief voll.
- Wiedergabe-Framerate lässt sich per API nicht setzen: in Resolve Shift+9 → 60.
