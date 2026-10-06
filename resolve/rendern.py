"""Render the two timelines in Resolve and deliver the MP4s.

1. Resolve renders one ProRes 422 HQ ``.mov`` with PCM audio per timeline into
   ``~/Videos/nomissuccess-spot/`` (Resolve 21 only renders into Media Storage; any
   other TargetDir makes ``AddRenderJob()`` return ``''`` — see diagnose_render.py).
   It renders into a candidate ``…-master-neu.mov``; only a candidate that passes the
   checks (codec, size, 8921 frames at 60 fps, PCM audio of the full length) replaces
   the master. No MP4 is made from a broken master.
2. ffmpeg makes ``out/nomissuccess-nachtschicht-<fmt>.mp4`` (libx264 crf 16 slow,
   yuv420p, 60 fps, AAC 320k, +faststart) and ``…-stumm.mp4`` (video copied, no audio);
   both must have exactly 8921 frames.
3. Loudness: the master's mix must match ``out/mix_referenz.wav`` (a muted track or a
   moved fader in Resolve shows up here). The MP4 must be -14 ± 0.5 LUFS and ≤ -1.05
   dBTP (0.05 dB reserve: ebur128 prints one decimal). Otherwise the reference mix is
   re-muxed (video copied, never re-rendered), with -0.3 dB if AAC overshoot still
   breaks the true-peak ceiling.
4. Alpha check at ``clock_roll``: the master must equal the Blender frame in most of
   the picture (the Fusion layers are transparent) but not everywhere (the type is there).

``--fallback`` composes the picture with ffmpeg instead of Resolve (shots + 12-frame
dissolves; type only if a PNG sequence with alpha exists in ``work/resolve/ebenen/<fmt>/``)
into its own ``…-fallback.mov`` — clearly marked as such in the report.

Usage::

    .venv/bin/python resolve/rendern.py                  # render + deliver both formats
    .venv/bin/python resolve/rendern.py --nur-ausliefern # MP4s from existing masters
    .venv/bin/python resolve/rendern.py --fallback       # ffmpeg composition, no Resolve
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolve import bauen  # noqa: E402
from resolve import resolve_api as ra  # noqa: E402

REPO = bauen.REPO
OUT = REPO / "out"
VIDEOS_DIR = bauen.VIDEOS_DIR
REFERENZ = OUT / "mix_referenz.wav"
FPS = 60
LUFS_ZIEL, LUFS_TOL = -14.0, 0.5
# ceiling -1 dBTP; ebur128 prints one decimal and AAC overshoots, so keep 0.05 dB reserve
TP_MAX = -1.05
TP_KORREKTUR_DB = -0.3  # gain on the reference mix when the AAC true peak breaks the ceiling
MIX_TOL_LUFS, MIX_TOL_TP = 0.2, 0.3  # Resolve master vs. reference mix
# alpha check: >= 60 % of the picture equals the Blender frame (layers transparent) and
# >= 0.2 % differs strongly (type on screen; the scene-1 counter + HUD measured 1.2 %)
ALPHA_GLEICH_MIN, ALPHA_TYPO_MIN = 0.6, 0.002
MASTER_BYTES = 9_000_000_000  # ProRes 422 HQ, 2.07 Mpx at 60 fps for 148.7 s ~ 8.2 GB
RESERVE_BYTES = 3_000_000_000
BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
X264 = ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-r", "60", *BT709]
AAC = ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"]


def master_pfad(fmt: str) -> Path:
    return VIDEOS_DIR / f"nomissuccess-nachtschicht-{fmt}-master.mov"


def kandidat_pfad(fmt: str) -> Path:
    """Resolve renders here; only a checked candidate replaces the master."""
    return VIDEOS_DIR / f"nomissuccess-nachtschicht-{fmt}-master-neu.mov"


def fallback_pfad(fmt: str) -> Path:
    return VIDEOS_DIR / f"nomissuccess-nachtschicht-{fmt}-fallback.mov"


def mp4_pfad(fmt: str, stumm: bool = False) -> Path:
    return OUT / f"nomissuccess-nachtschicht-{fmt}{'-stumm' if stumm else ''}.mp4"


def run(cmd: list[str], timeout: int = 4 * 3600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{Path(cmd[0]).name if cmd[0] != 'nice' else cmd[3]} failed "
                           f"(rc={exc.returncode}): {exc.stderr.strip()[-800:]}") from exc


def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
               "stream=codec_type,codec_name,profile,width,height,nb_read_packets,r_frame_rate,sample_rate,"
               "channels,duration,color_space,color_range:format=duration", "-of", "json", str(path)], timeout=900).stdout
    return json.loads(out)


def video_frames(info: dict) -> int:
    return next((int(s.get("nb_read_packets") or 0) for s in info.get("streams", [])
                 if s.get("codec_type") == "video"), 0)


# --------------------------------------------------------------------------------------
# Loudness
# --------------------------------------------------------------------------------------

def lautheit(path: Path) -> dict:
    """Integrated loudness (LUFS) and true peak (dBTP) of the first audio stream."""
    err = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-map", "0:a:0",
                          "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=1800).stderr
    summary = err[err.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary)
    tp = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    if not i or not tp:
        raise RuntimeError(f"ebur128 summary not found for {path}")
    return {"lufs": float(i.group(1)), "dbtp": float(tp.group(1))}


def lautheit_ok(m: dict) -> bool:
    return abs(m["lufs"] - LUFS_ZIEL) <= LUFS_TOL and m["dbtp"] <= TP_MAX


def mix_abweichung(master: dict, referenz: dict) -> str | None:
    """None if the master's mix matches the reference mix, else a description."""
    d_lufs, d_tp = master["lufs"] - referenz["lufs"], master["dbtp"] - referenz["dbtp"]
    if abs(d_lufs) <= MIX_TOL_LUFS and abs(d_tp) <= MIX_TOL_TP:
        return None
    return f"Master-Mix weicht von mix_referenz.wav ab: {d_lufs:+.1f} LU, {d_tp:+.1f} dB TP"


def ton_neu_muxen(mp4: Path, gain_db: float = 0.0) -> None:
    """Replace the MP4's audio with the reference mix (video stream copied)."""
    tmp = mp4.with_name(f".{mp4.name}")
    af = ["-af", f"volume={gain_db:.2f}dB"] if gain_db else []
    run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-i", str(REFERENZ), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", *af, *AAC, "-write_tmcd", "0", "-movflags", "+faststart", str(tmp)])
    tmp.replace(mp4)


def lautheit_sichern(mp4: Path) -> dict:
    """Measure the MP4. On a deviation re-mux the reference mix (never re-render); if the
    AAC true peak then still breaks the ceiling, re-mux it with -0.3 dB. The result is
    whatever was measured last — the caller treats a miss as a failure."""
    report: dict[str, Any] = {"gemessen": lautheit(mp4), "massnahme": "keine"}
    m = report["gemessen"]
    if not lautheit_ok(m):
        ton_neu_muxen(mp4)
        report["massnahme"] = "Ton aus out/mix_referenz.wav neu gemuxt"
        m = lautheit(mp4)
        if m["dbtp"] > TP_MAX:
            ton_neu_muxen(mp4, TP_KORREKTUR_DB)
            report["massnahme"] += f", {TP_KORREKTUR_DB:+.1f} dB gegen AAC-Überschwinger"
            m = lautheit(mp4)
    report["ergebnis"] = m
    return report


# --------------------------------------------------------------------------------------
# Checks and delivery
# --------------------------------------------------------------------------------------

def master_pruefen(master: Path, fmt: str, dauer_f: int) -> dict:
    info = probe(master)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    w, h = bauen.FORMATE[fmt]
    problems = []
    if (video.get("width"), video.get("height")) != (w, h):
        problems.append(f"size {video.get('width')}x{video.get('height')}")
    if video_frames(info) != dauer_f:
        problems.append(f"{video_frames(info)} frames, expected {dauer_f}")
    if video.get("r_frame_rate") != f"{FPS}/1":
        problems.append(f"rate {video.get('r_frame_rate')}")
    if not str(video.get("profile", "")).upper().endswith("HQ"):
        problems.append(f"video profile {video.get('profile')}, not ProRes 422 HQ")
    if not audio:
        problems.append("no audio")
    else:
        if not str(audio[0].get("codec_name", "")).startswith("pcm_"):
            problems.append(f"audio is {audio[0].get('codec_name')}, not PCM")
        dauer = float(audio[0].get("duration") or 0)
        if abs(dauer * FPS - dauer_f) > 1:
            problems.append(f"audio {dauer:.3f} s, expected {dauer_f / FPS:.3f} s")
    return {"codec": f"{video.get('codec_name')} {video.get('profile')}", "frames": video_frames(info),
            "farbe": f"{video.get('color_space')}/{video.get('color_range')}",
            "audio": [f"{a.get('codec_name')} {a.get('sample_rate')} Hz {a.get('channels')} ch" for a in audio],
            "probleme": problems}


def kandidat_uebernehmen(kandidat: Path, fmt: str, dauer_f: int) -> tuple[Path, dict]:
    """Check a rendered candidate; only a clean one replaces the master."""
    pruefung = master_pruefen(kandidat, fmt, dauer_f)
    if pruefung["probleme"]:
        return kandidat, pruefung
    master = master_pfad(fmt)
    kandidat.replace(master)
    return master, pruefung


def mp4_machen(master: Path, fmt: str, dauer_f: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    mp4, stumm = mp4_pfad(fmt), mp4_pfad(fmt, stumm=True)
    tmp = mp4.with_name(f".{mp4.name}")
    run(["nice", "-n", "5", "ffmpeg", "-v", "error", "-y", "-i", str(master), "-map", "0:v:0", "-map", "0:a:0",
         *X264, *AAC, "-write_tmcd", "0", "-movflags", "+faststart", str(tmp)])
    tmp.replace(mp4)
    loud = lautheit_sichern(mp4)
    run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-map", "0:v:0", "-c:v", "copy", "-an",
         "-write_tmcd", "0", "-movflags", "+faststart", str(stumm)])
    info, info_stumm = probe(mp4), probe(stumm)
    problems = [f"{p.name}: {video_frames(i)} frames, expected {dauer_f}"
                for p, i in ((mp4, info), (stumm, info_stumm)) if video_frames(i) != dauer_f]
    if any(s.get("codec_type") == "audio" for s in info_stumm.get("streams", [])):
        problems.append(f"{stumm.name} still has audio")
    return {"mp4": str(mp4), "stumm": str(stumm), "frames": video_frames(info),
            "dauer_s": float(info["format"]["duration"]), "lautheit": loud, "mp4_probleme": problems}


# --------------------------------------------------------------------------------------
# Alpha check (Fusion layers must let V1 show through, and the type must be there)
# --------------------------------------------------------------------------------------

def frame_rgb(path: Path, frame: int, vf: str = "") -> Any:
    import numpy as np
    chain = f"select=eq(n\\,{frame}),{vf + ',' if vf else ''}format=rgb24"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", chain, "-frames:v", "1",
                          "-f", "rawvideo", "-"], capture_output=True, timeout=600, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8)


def alpha_frames(timeline: dict, event_id: str = "clock_roll") -> tuple[dict, int, int]:
    """(scene, spot frame, source frame in that scene's shot movie) for the check."""
    event = next(e for e in timeline["events"] if e["id"] == event_id)
    szene = next(s for s in timeline["szenen"] if s["id"] == event["szene"])
    frame = int(event["f"])
    return szene, frame, frame - int(szene["start_f"]) + bauen.HANDLE_F


def alpha_pruefung(master: Path, fmt: str, timeline: dict) -> dict:
    """Compare the master with the Blender frame at the same spot frame. Opaque Fusion
    layers make little of it equal; missing type leaves almost nothing strongly different."""
    import numpy as np
    w, h = bauen.FORMATE[fmt]
    szene, frame, quelle = alpha_frames(timeline)
    shot = REPO / "renders" / f"{szene['id']}.mov"
    a = frame_rgb(master, frame).astype(int)
    b = frame_rgb(shot, quelle, f"scale=1920:1920:flags=lanczos,crop={w}:{h}").astype(int)
    if a.size != b.size or a.size == 0:
        return {"frame": frame, "ok": False, "fehler": f"sizes {a.size} vs {b.size}"}
    diff = np.abs(a - b).reshape(-1, 3).max(axis=1)
    gleich, typo = float((diff <= 16).mean()), float((diff > 48).mean())
    return {"frame": frame, "gleich_anteil": round(gleich, 4), "typo_anteil": round(typo, 4),
            "ok": alpha_urteil(gleich, typo),
            "master_mittel": round(float(a.mean()), 1), "shot_mittel": round(float(b.mean()), 1)}


def alpha_urteil(gleich: float, typo: float) -> bool:
    """Transparent layers (most of the picture is the shot) and type actually on screen."""
    return gleich >= ALPHA_GLEICH_MIN and typo >= ALPHA_TYPO_MIN


# --------------------------------------------------------------------------------------
# Resolve render
# --------------------------------------------------------------------------------------

def prores_hq_codec(project: Any) -> str:
    codecs = project.GetRenderCodecs("mov") or {}
    for label, codec in codecs.items():
        if "422 HQ" in label or codec.lower().replace("_", "") in ("prores422hq",):
            return codec
    raise RuntimeError(f"no ProRes 422 HQ in {codecs}")


def platz_pruefen(anzahl_master: int, ziel: Path = VIDEOS_DIR) -> None:
    """Refuse to start when the disk cannot take the masters (the system disk ran full on 06.10.)."""
    ziel.mkdir(parents=True, exist_ok=True)
    frei = shutil.disk_usage(ziel).free
    noetig = anzahl_master * MASTER_BYTES + RESERVE_BYTES
    if frei < noetig:
        raise RuntimeError(f"zu wenig Platz in {ziel}: {frei / 1e9:.1f} GB frei, {noetig / 1e9:.1f} GB nötig")


def resolve_rendern(formate: list[str], timeout_s: int = 3 * 3600) -> dict[str, Path]:
    platz_pruefen(len(formate))
    resolve = ra.connect()
    project = ra.current_project(resolve)
    if project.IsRenderingInProgress():
        raise RuntimeError("Resolve is already rendering — not queueing")
    ra._guard("start")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    names = {bauen.timeline_name(f) for f in formate}
    for job in project.GetRenderJobList() or []:  # our old jobs only
        if job.get("TimelineName") in names:
            project.DeleteRenderJob(job["JobId"])
    codec = prores_hq_codec(project)
    jobs: dict[str, str] = {}
    previous = project.GetCurrentTimeline()
    with ra.keep_page(resolve):
        for fmt in formate:
            timeline = ra.find_timeline(project, bauen.timeline_name(fmt))
            if timeline is None:
                raise RuntimeError(f"timeline {bauen.timeline_name(fmt)!r} missing — run bauen.py first")
            project.SetCurrentTimeline(timeline)
            kandidat = kandidat_pfad(fmt)
            kandidat.unlink(missing_ok=True)  # our own leftover candidate; the master stays
            w, h = bauen.FORMATE[fmt]
            if not project.SetCurrentRenderFormatAndCodec("mov", codec):
                raise RuntimeError(f"SetCurrentRenderFormatAndCodec(mov, {codec}) failed")
            project.SetCurrentRenderMode(1)  # single clip
            base = {"SelectAllFrames": True, "TargetDir": str(VIDEOS_DIR), "CustomName": kandidat.stem,
                    "UseUniqueFilenames": False, "ExportVideo": True, "ExportAudio": True,
                    "FormatWidth": w, "FormatHeight": h}  # no ExportAlpha: refused for ProRes 422 HQ (21.1.1)
            ok = project.SetRenderSettings({**base, "AudioCodec": "lpcm", "AudioBitDepth": 24,
                                            "AudioSampleRate": 48000})
            if not ok:  # one refused key fails the whole call; the master's audio is checked by ffprobe
                print(f"{fmt}: audio render settings refused, using Resolve's mov default", flush=True)
                ok = project.SetRenderSettings(base)
            job = project.AddRenderJob() if ok else ""
            ra._guard(f"AddRenderJob {fmt}")
            if not job:
                raise RuntimeError(f"AddRenderJob returned '' for {fmt} (SetRenderSettings ok={ok})")
            jobs[fmt] = job
        if not project.StartRendering(list(jobs.values()), False):
            raise RuntimeError("StartRendering returned False")
        started, last = time.time(), 0.0
        while project.IsRenderingInProgress():
            if time.time() - started > timeout_s:
                project.StopRendering()
                raise RuntimeError("render timed out — stopped")
            if time.time() - last > 60:
                state = {f: project.GetRenderJobStatus(j) for f, j in jobs.items()}
                written = sum(kandidat_pfad(f).stat().st_size for f in jobs if kandidat_pfad(f).exists())
                minutes = max(1e-6, (time.time() - started) / 60)
                # ~55 MB of ProRes 422 HQ per second of 1080p60 video
                print(time.strftime("%H:%M:%S"), {f: (s.get("JobStatus"), s.get("CompletionPercentage"))
                                                  for f, s in state.items()},
                      f"{written / 1e6:.0f} MB, {written / 1e6 / minutes:.0f} MB/min "
                      f"(~{written / 55e6 / minutes:.1f} s Video/min)", flush=True)
                last = time.time()
            time.sleep(2)
    if previous is not None:
        project.SetCurrentTimeline(previous)
    failed = {f: project.GetRenderJobStatus(j) for f, j in jobs.items()}
    failed = {f: s for f, s in failed.items() if s.get("JobStatus") != "Complete"}
    if failed:
        raise RuntimeError(f"render jobs not complete: {failed}")
    print(f"render: {time.time() - started:.0f} s", flush=True)
    return {fmt: kandidat_pfad(fmt) for fmt in jobs}


# --------------------------------------------------------------------------------------
# Fallback: ffmpeg composition (no Resolve)
# --------------------------------------------------------------------------------------

def fallback_filter(fmt: str, timeline: dict, overlay_input: int | None) -> str:
    """Filter graph: shots cropped to the format, joined by 12-frame centred dissolves,
    black after the last scene, optional RGBA overlay (BT.709, no 601 auto-conversion).
    Input i (0-based) is scene i's shot movie; ``overlay_input`` the PNG sequence."""
    w, h = bauen.FORMATE[fmt]
    half = bauen.XFADE_F // 2
    szenen = timeline["szenen"]
    chains = []
    for index, s in enumerate(szenen):
        laenge = s["ende_f"] - s["start_f"]
        head = half if index > 0 else 0
        tail = half if index < len(szenen) - 1 else 0
        chains.append(f"[{index}:v]trim=start_frame={bauen.HANDLE_F - head}:end_frame={bauen.HANDLE_F + laenge + tail},"
                      f"setpts=PTS-STARTPTS,scale=1920:1920:flags=lanczos,crop={w}:{h},format=yuv422p10le[c{index}]")
    acc = "c0"
    for index, s in enumerate(szenen[1:], start=1):
        offset = (s["start_f"] - half) / FPS
        chains.append(f"[{acc}][c{index}]xfade=transition=fade:duration={bauen.XFADE_F / FPS}:offset={offset:.6f}[x{index}]")
        acc = f"x{index}"
    tail_f = timeline["dauer_f"] - szenen[-1]["ende_f"]
    chains.append(f"[{acc}]tpad=stop={tail_f}:color=black[base]")
    if overlay_input is not None:
        chains.append(f"[{overlay_input}:v]scale=out_color_matrix=bt709:out_range=tv,format=yuva422p10le[ov]")
        chains.append("[base][ov]overlay=format=yuv422p10[vout]")
    else:
        chains.append("[base]null[vout]")
    return ";".join(chains)


def fallback(fmt: str, timeline: dict) -> tuple[Path, bool]:
    """ProRes master from the shots (validated like the build) + optional overlay sequence
    ``work/resolve/ebenen/<fmt>/%05d.png`` (frame 0 = spot frame 0) + reference mix.
    Returns (path, has_type)."""
    script = bauen.lade_json(REPO / "script.json")
    bauen.plan(fmt, timeline, script, mit_comps=False)  # shots: frames, handles, start
    inputs: list[str] = []
    for s in timeline["szenen"]:
        inputs += ["-i", str(REPO / "renders" / f"{s['id']}.mov")]
    ebenen = REPO / "work" / "resolve" / "ebenen" / fmt
    overlay = len(timeline["szenen"]) if (ebenen / "00000.png").exists() else None
    if overlay is not None:
        inputs += ["-framerate", str(FPS), "-i", str(ebenen / "%05d.png")]
    audio_index = len(timeline["szenen"]) + (1 if overlay is not None else 0)
    inputs += ["-i", str(REFERENZ)]
    out = fallback_pfad(fmt)
    platz_pruefen(1)
    run(["nice", "-n", "5", "ffmpeg", "-v", "error", "-y", *inputs,
         "-filter_complex", fallback_filter(fmt, timeline, overlay),
         "-map", "[vout]", "-map", f"{audio_index}:a:0", "-frames:v", str(timeline["dauer_f"]),
         "-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-pix_fmt", "yuv422p10le", "-r", str(FPS),
         *BT709, "-c:a", "pcm_s24le", str(out)])
    return out, overlay is not None


# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", action="append", choices=list(bauen.FORMATE), help="default: both")
    ap.add_argument("--nur-ausliefern", action="store_true", help="skip the Resolve render, use existing masters")
    ap.add_argument("--fallback", action="store_true", help="compose with ffmpeg instead of Resolve")
    args = ap.parse_args(argv)
    formate = args.format or list(bauen.FORMATE)
    timeline = bauen.lade_json(REPO / "timeline.json")
    dauer_f = int(timeline["dauer_f"])
    bericht: dict[str, Any] = {"weg": "ffmpeg-fallback" if args.fallback else "resolve", "formate": {}}
    referenz = lautheit(REFERENZ)
    bericht["referenz"] = referenz
    typo: dict[str, bool] = {}
    if args.fallback:
        kandidaten = {}
        for fmt in formate:
            kandidaten[fmt], typo[fmt] = fallback(fmt, timeline)
    elif args.nur_ausliefern:
        kandidaten = {fmt: master_pfad(fmt) for fmt in formate}
    else:
        kandidaten = resolve_rendern(formate)
    for fmt, kandidat in kandidaten.items():
        if not kandidat.exists():
            raise RuntimeError(f"render output missing: {kandidat}")
        if kandidat in (master_pfad(fmt), fallback_pfad(fmt)):
            master, pruefung = kandidat, master_pruefen(kandidat, fmt, dauer_f)
        else:
            master, pruefung = kandidat_uebernehmen(kandidat, fmt, dauer_f)
        entry: dict[str, Any] = {"master": str(master), "pruefung": pruefung}
        bericht["formate"][fmt] = entry
        if pruefung["probleme"]:  # no MP4 from a broken master
            print(json.dumps({fmt: entry}, ensure_ascii=False, indent=1), flush=True)
            continue
        entry["lautheit_master"] = lautheit(master)
        if not args.fallback:
            abweichung = mix_abweichung(entry["lautheit_master"], referenz)
            if abweichung:
                pruefung["probleme"].append(abweichung)
                print(json.dumps({fmt: entry}, ensure_ascii=False, indent=1), flush=True)
                continue
        entry.update(mp4_machen(master, fmt, dauer_f))
        if args.fallback and not typo.get(fmt):
            entry["alpha"] = {"ok": False, "fehler": "Fallback ohne Typo-Ebenen: keine Fusion-Typo im Bild"}
        else:
            entry["alpha"] = alpha_pruefung(master, fmt, timeline)
        print(json.dumps({fmt: entry}, ensure_ascii=False, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "render_bericht.json").write_text(json.dumps(bericht, ensure_ascii=False, indent=2) + "\n")
    bad = [f for f, e in bericht["formate"].items()
           if e["pruefung"]["probleme"] or e.get("mp4_probleme")
           or not lautheit_ok(e.get("lautheit", {}).get("ergebnis", {"lufs": 0, "dbtp": 0}))
           or not e.get("alpha", {}).get("ok")]
    if bad:
        print(f"PROBLEME in {bad} — siehe out/render_bericht.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
