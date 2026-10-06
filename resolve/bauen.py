"""Build the DaVinci Resolve project "nomiss": bins, two timelines, comps, audio, markers.

Layout per timeline ("nomissuccess 16x9" 1920x1080, "nomissuccess 9x16" 1080x1920,
60 fps, start 01:00:00:00, input sizing "Scale full frame with crop"):

* V1 "Blender"  - one ProRes shot per scene (``renders/<shot>.mov``, 1440x1440) from the
  scene start, source from frame ``handle_f`` (30). Hard cuts at the scene boundaries,
  each softened by a 12-frame centred Cross Dissolve where both handles allow it.
* V2 "Typo"     - the scene's Fusion comp (``work/fusion/<fmt>/<scene>.comp``) on a black
  ProRes carrier clip trimmed to the scene length (``ImportFusionComp``; the comp's own
  alpha lets V1 show through).
* V3 "HUD"      - ``hud.comp`` over the full length.
* V4 "Abspann"  - ``abspann.comp`` from ``abspann.start_f`` to the end.
* A1 VO, A2 music, A3-A8 SFX categories - the stems at 0 dB from frame 0.
* Timeline markers named after ``script.json -> szenen[].titel`` (+ end card).

Idempotent: before a rebuild the builder exports each of its timelines as ``.drt`` to
``work/resolve/`` and deletes it (only these two names, never while rendering), and it
replaces only media-pool clips whose file is one of the files it places (all inside this
repo). Foreign clips, timelines and bins are never deleted; running twice leaves the
media-pool clip count and every track's item count unchanged.

Usage::

    .venv/bin/python resolve/bauen.py --trocken            # plan only, no Resolve
    .venv/bin/python resolve/bauen.py --nur-traeger        # create the carrier movies
    .venv/bin/python resolve/bauen.py                      # build both timelines
    .venv/bin/python resolve/bauen.py --teilweise          # test build: skip missing media
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolve import resolve_api as ra  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VIDEOS_DIR = Path.home() / "Videos" / "nomissuccess-spot"

FORMATE: dict[str, tuple[int, int]] = {"16x9": (1920, 1080), "9x16": (1080, 1920)}
BINS = {"blender": "01 Blender", "fusion": "02 Fusion", "audio": "03 Audio"}
STEMS = ["vo", "musik", "sfx_whoosh", "sfx_hit", "sfx_ui", "sfx_uebergang", "sfx_signatur", "sfx_atmo"]
VIDEO_SPUREN = {1: "Blender", 2: "Typo", 3: "HUD", 4: "Abspann"}
HANDLE_F = 30
XFADE_F = 12
START_TC = "01:00:00:00"
TRAEGER_F = 9000  # 150 s at 60 fps, longer than the spot (8921 frames)
NEU_SUFFIX = " neu"  # timelines are built under "<name> neu" and renamed only on success
MARKER_FARBEN = {"nacht": "Blue", "rosa": "Pink", "blau": "Sky", "mint": "Mint", "spektrum": "Lavender"}


def timeline_name(fmt: str) -> str:
    return f"nomissuccess {fmt}"


class BauFehler(RuntimeError):
    """The plan or the Resolve state does not allow a correct build."""


# --------------------------------------------------------------------------------------
# Plan (pure, no Resolve)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pfade:
    repo: Path = REPO

    @property
    def renders(self) -> Path:
        return self.repo / "renders"

    @property
    def fusion(self) -> Path:
        return self.repo / "work" / "fusion"

    @property
    def stems(self) -> Path:
        return self.repo / "assets" / "audio" / "stems"

    @property
    def arbeit(self) -> Path:
        return self.repo / "work" / "resolve"

    def traeger(self, fmt: str) -> Path:
        return self.arbeit / f"traeger_{fmt}.mov"

    def eigen(self, path: Path | str) -> bool:
        """Clips the builder may recognise as its own: inside the repo or ~/Videos/nomissuccess-spot."""
        p = Path(str(path))
        return any(p == root or root in p.parents for root in (self.repo, VIDEOS_DIR))


@dataclass(frozen=True)
class Clip:
    """One placement on a timeline track. Frames are relative to the timeline start;
    ``quelle_start`` is the 0-based source frame of the first placed frame."""
    name: str
    spur: str          # "video" | "audio"
    track: int
    medium: Path
    bin: str           # key of BINS
    record_f: int
    quelle_start: int
    laenge: int
    comp: Path | None = None
    ganz: bool = False  # place the whole medium (stems), no source range


@dataclass(frozen=True)
class Blende:
    track: int
    schnitt_f: int     # timeline frame of the cut (relative), the dissolve is centred on it
    dauer: int
    links: str
    rechts: str


@dataclass(frozen=True)
class Marker:
    f: int
    name: str
    notiz: str
    farbe: str
    dauer: int


@dataclass
class Plan:
    fmt: str
    breite: int
    hoehe: int
    fps: int
    dauer_f: int
    clips: list[Clip] = field(default_factory=list)
    blenden: list[Blende] = field(default_factory=list)
    marker: list[Marker] = field(default_factory=list)
    ausgelassen: list[str] = field(default_factory=list)
    teilweise: bool = False

    @property
    def name(self) -> str:
        return timeline_name(self.fmt)

    def spur_anzahl(self) -> dict[tuple[str, int], int]:
        counts: dict[tuple[str, int], int] = {}
        for clip in self.clips:
            counts[(clip.spur, clip.track)] = counts.get((clip.spur, clip.track), 0) + 1
        return counts


def lade_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def shot_meta(pfade: Pfade, shot: str) -> dict | None:
    info = pfade.renders / f"{shot}.json"
    return lade_json(info) if info.exists() else None


def plan(fmt: str, timeline: dict, script: dict, pfade: Pfade = Pfade(), teilweise: bool = False,
         mit_comps: bool = True) -> Plan:
    """All placements for one format, frame-exact from ``timeline.json``.

    Raises BauFehler on any missing or inconsistent input unless ``teilweise`` (then the
    missing parts are listed in ``Plan.ausgelassen`` and left out)."""
    if fmt not in FORMATE:
        raise BauFehler(f"unknown format {fmt!r}")
    breite, hoehe = FORMATE[fmt]
    fps = int(timeline["fps"])
    dauer_f = int(timeline["dauer_f"])
    p = Plan(fmt, breite, hoehe, fps, dauer_f, teilweise=teilweise)
    titel = {s["id"]: s.get("titel", s["id"]) for s in script.get("szenen", [])}
    szenen = sorted(timeline["szenen"], key=lambda s: int(s["start_f"]))

    def fehlt(was: str) -> None:
        if not teilweise:
            raise BauFehler(was)
        p.ausgelassen.append(was)

    # contract checks: gap-free scenes, end card right after the last scene
    for a, b in zip(szenen, szenen[1:]):
        if int(a["ende_f"]) != int(b["start_f"]):
            raise BauFehler(f"gap/overlap between {a['id']} and {b['id']}")
    abspann = timeline["abspann"]
    if int(szenen[0]["start_f"]) != 0 or int(szenen[-1]["ende_f"]) != int(abspann["start_f"]):
        raise BauFehler("scenes do not run from 0 to the end card")
    if int(abspann["ende_f"]) != dauer_f:
        raise BauFehler("end card does not end at dauer_f")

    # V1: Blender shots
    v1: list[tuple[dict, dict]] = []
    for szene in szenen:
        sid, start, ende = szene["id"], int(szene["start_f"]), int(szene["ende_f"])
        laenge = ende - start
        mov, meta = pfade.renders / f"{sid}.mov", shot_meta(pfade, sid)
        if meta is None or not mov.exists():
            fehlt(f"shot {sid}: {mov.name} / {sid}.json missing")
            continue
        if meta.get("kodierer") != "shots_kodieren" or (meta.get("breite"), meta.get("hoehe")) != (1440, 1440) \
                or int(meta.get("fps", 0)) != fps:
            fehlt(f"shot {sid}: {mov.name} is not a shots_kodieren encode (1440², BT.709) — run shots_kodieren.py")
            continue
        handle = int(meta.get("handle_f", -1))
        frames = int(meta.get("frames", -1))
        if handle != HANDLE_F or frames != laenge + 2 * HANDLE_F:
            raise BauFehler(f"shot {sid}: frames={frames} handle_f={handle}, "
                            f"expected {laenge + 2 * HANDLE_F} frames with {HANDLE_F} handles")
        if abs(float(meta.get("start_s", 1e9)) - (float(szene["start_s"]) - HANDLE_F / fps)) > 0.5 / fps:
            raise BauFehler(f"shot {sid}: start_s {meta.get('start_s')} != scene start - handle")
        p.clips.append(Clip(sid, "video", 1, mov, "blender", start, handle, laenge))
        v1.append((szene, meta))

    # cross dissolves where both clips have enough handle around the cut
    for (a, meta_a), (b, meta_b) in zip(v1, v1[1:]):
        if int(a["ende_f"]) != int(b["start_f"]):
            continue  # a shot is missing in between (teilweise)
        len_a = int(a["ende_f"]) - int(a["start_f"])
        tail_a = int(meta_a["frames"]) - int(meta_a["handle_f"]) - len_a
        head_b = int(meta_b["handle_f"])
        if min(tail_a, head_b) >= XFADE_F // 2:
            p.blenden.append(Blende(1, int(b["start_f"]), XFADE_F, a["id"], b["id"]))

    # V2-V4: Fusion comps on the carrier
    traeger = pfade.traeger(fmt)
    comp_dir = pfade.fusion / fmt
    lagen = [(2, s["id"], int(s["start_f"]), int(s["ende_f"])) for s in szenen]
    lagen += [(3, "hud", 0, dauer_f), (4, "abspann", int(abspann["start_f"]), int(abspann["ende_f"]))]
    if not mit_comps:
        p.ausgelassen.append("Fusion comps (V2-V4) skipped on request")
        lagen = []
    for track, cid, start, ende in lagen:
        comp = comp_dir / f"{cid}.comp"
        if not comp.exists():
            fehlt(f"comp {fmt}/{comp.name} missing")
            continue
        if ende - start > TRAEGER_F:
            raise BauFehler(f"comp {cid} longer than the carrier")
        p.clips.append(Clip(f"{cid} ({fmt})", "video", track, traeger, "fusion", start, 0, ende - start, comp=comp))

    # A1-A8: stems at 0 dB from frame 0
    for index, stem in enumerate(STEMS, start=1):
        wav = pfade.stems / f"{stem}.wav"
        if not wav.exists():
            fehlt(f"stem {wav.name} missing")
            continue
        p.clips.append(Clip(stem, "audio", index, wav, "audio", 0, 0, dauer_f, ganz=True))

    # comps and stems are generated from timeline.json: an older file was built from an
    # older contract (e.g. a moved event) and must not go into the final build
    contract = pfade.repo / "timeline.json"
    if contract.exists():
        t_contract = contract.stat().st_mtime
        for clip in p.clips:
            source = clip.comp if clip.comp is not None else (clip.medium if clip.spur == "audio" else None)
            if source is not None and source.stat().st_mtime < t_contract:
                fehlt(f"{source.relative_to(pfade.repo)} is older than timeline.json — regenerate it")

    # markers
    for szene in szenen:
        start, ende = int(szene["start_f"]), int(szene["ende_f"])
        p.marker.append(Marker(start, titel.get(szene["id"], szene["id"]), szene["id"],
                               MARKER_FARBEN.get(szene.get("farbe", ""), "Blue"), ende - start))
    p.marker.append(Marker(int(abspann["start_f"]), "Abspann", "abspann", "Cream",
                           int(abspann["ende_f"]) - int(abspann["start_f"])))
    return p


# --------------------------------------------------------------------------------------
# Carrier movies (ffmpeg, no Resolve)
# --------------------------------------------------------------------------------------

def traeger_erzeugen(fmt: str, pfade: Pfade = Pfade()) -> Path:
    """Black ProRes Proxy movie at the timeline resolution: the timeline clip a Fusion
    comp of any length hangs on (InsertFusionCompositionIntoTimeline gives 300 frames)."""
    path = pfade.traeger(fmt)
    if path.exists():
        return path
    breite, hoehe = FORMATE[fmt]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}")
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"color=c=black:s={breite}x{hoehe}:r=60:d={TRAEGER_F / 60}",
                    "-frames:v", str(TRAEGER_F), "-c:v", "prores_ks", "-profile:v", "0",
                    "-pix_fmt", "yuv422p10le", "-f", "mov", str(tmp)], check=True, timeout=1800)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------
# Resolve side
# --------------------------------------------------------------------------------------

def alle_ordner(folder: Any) -> list[Any]:
    found = [folder]
    for sub in folder.GetSubFolderList() or []:
        found.extend(alle_ordner(sub))
    return found


def clip_pfad(clip: Any) -> str:
    return str(clip.GetClipProperty("File Path") or "")


def eigene_clips(pool: Any, pfade: Pfade) -> list[tuple[Any, Any]]:
    """(folder, clip) for every media-pool clip whose file lies in the repo / ~/Videos/nomissuccess-spot."""
    return [(folder, clip) for folder in alle_ordner(pool.GetRootFolder())
            for clip in folder.GetClipList() or [] if pfade.eigen(clip_pfad(clip))]


def bin_holen(pool: Any, name: str) -> Any:
    root = pool.GetRootFolder()
    for sub in root.GetSubFolderList() or []:
        if sub.GetName() == name:
            return sub
    folder = pool.AddSubFolder(root, name)
    if folder is None:
        raise BauFehler(f"AddSubFolder({name!r}) failed")
    return folder


@dataclass
class Semantik:
    """How this Resolve interprets AppendToTimeline ranges (measured on the first clip)."""
    end_inklusiv: bool | None = None
    record_relativ: bool | None = None


class Bauer:
    def __init__(self, resolve: Any, project: Any, pfade: Pfade = Pfade(), log=print,
                 medien_probe: Callable[[Path], dict] | None = None):
        self.resolve = resolve
        self.medien_probe = medien_probe or ffprobe_medium
        self.project = project
        self.pool = project.GetMediaPool()
        self.pfade = pfade
        self.log = log
        self.sem = Semantik()
        self.media: dict[Path, Any] = {}
        self.zoom_noetig: dict[str, bool] = {}

    # -- guards -----------------------------------------------------------------------
    def nicht_rendernd(self, wofuer: str) -> None:
        if self.project.IsRenderingInProgress():
            raise BauFehler(f"Resolve is rendering — not {wofuer}")

    # -- timelines --------------------------------------------------------------------
    def timeline_sichern_und_loeschen(self, name: str) -> Path | None:
        timeline = ra.find_timeline(self.project, name)
        if timeline is None:
            return None
        self.nicht_rendernd(f"deleting timeline {name!r}")
        self.pfade.arbeit.mkdir(parents=True, exist_ok=True)
        backup = self.pfade.arbeit / f"{name.replace(' ', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.drt"
        if not timeline.Export(str(backup), self.resolve.EXPORT_DRT, self.resolve.EXPORT_NONE):
            raise BauFehler(f"DRT export of {name!r} failed — not deleting it")
        for job in self.project.GetRenderJobList() or []:
            if job.get("TimelineName") == name:
                self.project.DeleteRenderJob(job["JobId"])
        if not self.pool.DeleteTimelines([timeline]):
            raise BauFehler(f"DeleteTimelines({name!r}) failed")
        self.log(f"timeline {name!r}: backup {backup.name}, deleted")
        return backup

    def temp_verwerfen(self, p: Plan) -> None:
        """Delete our own half-built '<name> neu' timeline, if any (never while rendering)."""
        temp = ra.find_timeline(self.project, p.name + NEU_SUFFIX)
        if temp is not None and not self.project.IsRenderingInProgress():
            self.pool.DeleteTimelines([temp])
            self.log(f"timeline {p.name + NEU_SUFFIX!r}: discarded")

    # -- preflight ----------------------------------------------------------------------
    def vorpruefung(self, plaene: list[Plan]) -> None:
        """Check every input file before the project is touched: shot movies (frames and
        size against their JSON), carriers, stems (length) and comps (MediaOut, range)."""
        problems: list[str] = []
        seen: set[Path] = set()
        for p in plaene:
            for clip in p.clips:
                if clip.comp is not None:
                    problems += comp_probleme(clip.comp, clip.laenge)
                if clip.medium in seen:
                    continue
                seen.add(clip.medium)
                try:
                    info = self.medien_probe(clip.medium)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    problems.append(f"{clip.medium.name}: unreadable ({exc})")
                    continue
                if clip.ganz:
                    if abs(float(info.get("dauer_s", 0)) * p.fps - clip.laenge) > 1:
                        problems.append(f"{clip.medium.name}: {info.get('dauer_s')} s, expected {clip.laenge} frames")
                elif clip.comp is not None:
                    if (info.get("breite"), info.get("hoehe")) != (p.breite, p.hoehe) or info.get("frames", 0) < TRAEGER_F:
                        problems.append(f"{clip.medium.name}: {info}, expected {p.breite}x{p.hoehe} >= {TRAEGER_F} frames")
                else:
                    meta = shot_meta(self.pfade, clip.name) or {}
                    want = (int(meta.get("frames", -1)), int(meta.get("breite", -1)), int(meta.get("hoehe", -1)))
                    got = (info.get("frames"), info.get("breite"), info.get("hoehe"))
                    if got != want:
                        problems.append(f"{clip.medium.name}: frames/size {got}, JSON says {want}")
                    pngs = self.pfade.renders / clip.name
                    if pngs.is_dir():
                        newest = max((f.stat().st_mtime for f in pngs.glob("*.png")), default=0.0)
                        if newest != meta.get("quelle_mtime"):
                            problems.append(f"{clip.name}: PNGs changed after the encode — run shots_kodieren.py")
        if problems:
            raise BauFehler("preflight failed, nothing touched:\n  " + "\n  ".join(problems))

    # -- media ------------------------------------------------------------------------
    def medien_importieren(self, plaene: list[Plan]) -> None:
        """One media-pool clip per file we place, in its bin.

        A clip of one of our files that no timeline uses any more (our timelines are
        already deleted) is removed and re-imported, so a re-encoded movie is never served
        from a stale entry. A clip still in use (the other format's timeline when only one
        format is rebuilt, or anyone else's timeline) is reused and only moved into its
        bin; unused duplicates of it are removed. Clips of other files are never touched."""
        wanted: dict[Path, str] = {}
        for p in plaene:
            for clip in p.clips:
                wanted.setdefault(clip.medium, clip.bin)
        by_str = {str(path): path for path in wanted}
        found: dict[Path, list[tuple[Any, Any]]] = {}
        for folder, clip in eigene_clips(self.pool, self.pfade):
            if clip_pfad(clip) in by_str:
                found.setdefault(by_str[clip_pfad(clip)], []).append((folder, clip))
        stale: list[Any] = []
        reuse: dict[Path, tuple[Any, Any]] = {}
        for path, entries in found.items():
            used = [(f, c) for f, c in entries if _usage(c) > 0]
            if used:
                reuse[path] = used[0]
                stale += [c for _, c in entries if c is not used[0][1] and _usage(c) == 0]
            else:
                stale += [c for _, c in entries]
        if stale:
            if not self.pool.DeleteClips(stale):
                raise BauFehler("DeleteClips of our own unused clips failed")
            self.log(f"media pool: removed {len(stale)} own unused clip(s) for re-import")
        previous = self.pool.GetCurrentFolder()
        try:
            for key, bin_name in BINS.items():
                paths = [path for path, b in wanted.items() if b == key]
                if not paths:
                    continue
                folder = bin_holen(self.pool, bin_name)
                for path in paths:
                    if path in reuse:
                        old_folder, clip = reuse[path]
                        if old_folder.GetUniqueId() != folder.GetUniqueId():
                            self.pool.MoveClips([clip], folder)
                        self.media[path] = clip
                fresh = [path for path in paths if path not in reuse]
                if not fresh:
                    continue
                self.pool.SetCurrentFolder(folder)
                items = self.pool.ImportMedia([str(path) for path in fresh]) or []
                ra._guard(f"ImportMedia {bin_name}")
                imported = {clip_pfad(item): item for item in items}
                for path in fresh:
                    item = imported.get(str(path))
                    if item is None:
                        raise BauFehler(f"ImportMedia did not import {path}")
                    self.media[path] = item
        finally:
            if previous is not None:
                self.pool.SetCurrentFolder(previous)
        for p in plaene:
            for clip in p.clips:
                if clip.spur == "video" and not clip.comp:
                    frames = int(float(self.media[clip.medium].GetClipProperty("Frames") or 0))
                    if frames != clip.quelle_start + clip.laenge + HANDLE_F:
                        raise BauFehler(f"{clip.medium.name}: media pool reports {frames} frames")

    # -- build one timeline -----------------------------------------------------------
    def timeline_anlegen(self, p: Plan, name: str) -> Any:
        self.pool.SetCurrentFolder(self.pool.GetRootFolder())
        timeline = self.pool.CreateEmptyTimeline(name)
        if timeline is None:
            raise BauFehler(f"CreateEmptyTimeline({name!r}) failed")
        self.project.SetCurrentTimeline(timeline)
        settings = {
            "useCustomSettings": "1",
            "timelineFrameRate": str(p.fps),
            "timelineResolutionWidth": str(p.breite),
            "timelineResolutionHeight": str(p.hoehe),
            "timelineOutputResMatchTimelineRes": "1",
            "timelineInputResMismatchBehavior": "scaleToCrop",
        }
        for key, value in settings.items():
            timeline.SetSetting(key, value)
        if not timeline.SetStartTimecode(START_TC):
            self.log(f"warning: SetStartTimecode({START_TC}) returned False")
        got = {k: str(timeline.GetSetting(k)) for k in settings}
        if (got["timelineResolutionWidth"], got["timelineResolutionHeight"]) != (str(p.breite), str(p.hoehe)):
            raise BauFehler(f"{p.name}: resolution not taken: {got}")
        if float(got["timelineFrameRate"] or 0) != float(p.fps):
            raise BauFehler(f"{p.name}: frame rate is {got['timelineFrameRate']}, expected {p.fps}")
        self.zoom_noetig[p.fmt] = got["timelineInputResMismatchBehavior"] != "scaleToCrop"
        if self.zoom_noetig[p.fmt]:
            self.log(f"{p.name}: input sizing stayed {got['timelineInputResMismatchBehavior']!r} — zoom per clip")
        while timeline.GetTrackCount("video") < len(VIDEO_SPUREN):
            if not timeline.AddTrack("video"):
                raise BauFehler("AddTrack(video) failed")
        while timeline.GetTrackCount("audio") < len(STEMS):
            if not timeline.AddTrack("audio", "stereo"):
                raise BauFehler("AddTrack(audio) failed")
        for index, name in VIDEO_SPUREN.items():
            timeline.SetTrackName("video", index, name)
        for index, stem in enumerate(STEMS, start=1):
            timeline.SetTrackName("audio", index, stem)
        return timeline

    def _append(self, timeline: Any, clip: Clip, start_frame: int) -> Any:
        info: dict[str, Any] = {"mediaPoolItem": self.media[clip.medium], "trackIndex": clip.track,
                                "mediaType": 1 if clip.spur == "video" else 2}
        rec = start_frame + clip.record_f if not self.sem.record_relativ else clip.record_f
        info["recordFrame"] = rec
        if not clip.ganz:
            info["startFrame"] = clip.quelle_start
            info["endFrame"] = clip.quelle_start + clip.laenge - (1 if self.sem.end_inklusiv else 0)
        items = self.pool.AppendToTimeline([info]) or []
        ra._guard(f"AppendToTimeline {clip.name}")
        if not items:
            raise BauFehler(f"AppendToTimeline failed for {clip.name}")
        return items[0]

    def platzieren(self, timeline: Any, clip: Clip, start_frame: int) -> Any:
        item = self._append(timeline, clip, start_frame)
        if self.sem.record_relativ is None:
            offset = int(item.GetStart()) - (start_frame + clip.record_f)
            if offset not in (0, start_frame):
                timeline.DeleteClips([item], False)
                raise BauFehler(f"{clip.name}: placed at {item.GetStart()}, expected {start_frame + clip.record_f}")
            self.sem.record_relativ = offset == start_frame and start_frame != 0
            if self.sem.record_relativ:
                timeline.DeleteClips([item], False)
                item = self._append(timeline, clip, start_frame)
        if not clip.ganz and self.sem.end_inklusiv is None:
            dauer = int(item.GetDuration())
            if dauer not in (clip.laenge, clip.laenge + 1):
                timeline.DeleteClips([item], False)
                raise BauFehler(f"{clip.name}: duration {dauer}, expected {clip.laenge}")
            self.sem.end_inklusiv = dauer == clip.laenge + 1
            if self.sem.end_inklusiv:
                timeline.DeleteClips([item], False)
                item = self._append(timeline, clip, start_frame)
        start, dauer = int(item.GetStart()), int(item.GetDuration())
        if clip.ganz:
            ok = start == start_frame + clip.record_f and abs(dauer - clip.laenge) <= 1
        else:
            ok = (start == start_frame + clip.record_f and dauer == clip.laenge
                  and int(item.GetLeftOffset()) == clip.quelle_start)
        if not ok:
            raise BauFehler(f"{clip.name}: at {start}+{dauer} (offset {item.GetLeftOffset()}), "
                            f"expected {start_frame + clip.record_f}+{clip.laenge} from {clip.quelle_start}")
        return item

    def comp_anhaengen(self, item: Any, clip: Clip) -> None:
        # a crash inside ImportFusionComp (seen 06.10. under GPU memory pressure) must cost nothing
        self.resolve.GetProjectManager().SaveProject()
        comp = item.ImportFusionComp(str(clip.comp))
        ra._guard(f"ImportFusionComp {clip.name}")
        if comp is None:
            raise BauFehler(f"ImportFusionComp failed for {clip.comp}")
        names = item.GetFusionCompNameList() or []
        if not names:
            raise BauFehler(f"{clip.name}: no Fusion comp on the item after import")
        item.LoadFusionCompByName(names[-1])
        item.SetProperty("CompositeMode", self.resolve.COMPOSITE_NORMAL)
        item.SetProperty("Opacity", 100.0)

    def timeline_bauen(self, p: Plan, name: str) -> dict:
        timeline = self.timeline_anlegen(p, name)
        start_frame = int(timeline.GetStartFrame())
        v1: dict[str, Any] = {}
        for clip in sorted(p.clips, key=lambda c: (c.spur != "video", c.track, c.record_f)):
            item = self.platzieren(timeline, clip, start_frame)
            if clip.comp is not None:
                self.comp_anhaengen(item, clip)
            elif clip.spur == "video" and self.zoom_noetig.get(p.fmt):
                zoom = max(p.breite, p.hoehe) / min(p.breite, p.hoehe)
                item.SetProperty("ZoomX", zoom)
                item.SetProperty("ZoomY", zoom)
            if clip.spur == "video" and clip.track == 1:
                v1[clip.name] = item
        # dissolves last: they must not move any edit point
        before = {name: (int(i.GetStart()), int(i.GetEnd())) for name, i in v1.items()}
        blenden_ok = 0
        for blende in p.blenden:
            created = v1[blende.links].AddTransition({
                "type": "Cross Dissolve", "category": "simple", "position": "end",
                "alignment": "center", "duration": blende.dauer})
            if created is None:
                if not p.teilweise:
                    raise BauFehler(f"Cross Dissolve {blende.links} -> {blende.rechts} not created")
                self.log(f"warning: Cross Dissolve {blende.links} -> {blende.rechts} not created (hard cut)")
            else:
                blenden_ok += 1
        after = {name: (int(i.GetStart()), int(i.GetEnd())) for name, i in v1.items()}
        if after != before:
            raise BauFehler(f"transitions moved edit points: {before} -> {after}")
        for m in p.marker:
            if not timeline.AddMarker(m.f, m.farbe, m.name, m.notiz, m.dauer):
                if not p.teilweise:
                    raise BauFehler(f"marker {m.name!r} at {m.f} not added")
                self.log(f"warning: marker {m.name!r} at {m.f} not added")
        counts = {f"{spur[0].upper()}{track}": len([i for i in timeline.GetItemListInTrack(spur, track) or []
                                                    if not _ist_transition(i)])
                  for (spur, track) in sorted(p.spur_anzahl())}
        expected = {f"{spur[0].upper()}{track}": n for (spur, track), n in sorted(p.spur_anzahl().items())}
        if counts != expected:
            raise BauFehler(f"{p.name}: track item counts {counts}, expected {expected}")
        return {"timeline": p.name, "spuren": counts, "blenden": blenden_ok,
                "marker": len(timeline.GetMarkers() or {}), "ausgelassen": p.ausgelassen,
                "zoom_pro_clip": self.zoom_noetig.get(p.fmt, False),
                "semantik": {"end_inklusiv": self.sem.end_inklusiv, "record_relativ": self.sem.record_relativ}}

    def bauen(self, plaene: list[Plan], frisch: bool = False) -> list[dict]:
        """Build every plan under '<name> neu', swap in only when all succeeded.
        ``frisch``: after the preflight, back up and delete our old timelines first, so
        every clip is re-imported (use when movies were re-encoded under a live timeline)."""
        self.nicht_rendernd("building")
        ra._guard("start")
        if any(c.comp is not None for p in plaene for c in p.clips):
            fonts = Path("~/.local/share/fonts/manrope").expanduser()
            if not ra.ensure_fonts(self.resolve, fonts, "Manrope"):
                raise BauFehler(f"Manrope is not available to Fusion (fonts in {fonts})")
            if "JetBrains Mono" not in (self.resolve.Fusion().FontManager.GetFontList() or {}):
                raise BauFehler("JetBrains Mono is not available to Fusion")
        self.vorpruefung(plaene)  # everything that can fail on our inputs fails here, untouched
        for p in plaene:
            self.temp_verwerfen(p)  # leftover of an earlier aborted run
            if frisch:
                self.timeline_sichern_und_loeschen(p.name)
        self.medien_importieren(plaene)
        results = []
        try:
            for p in plaene:
                results.append(self.timeline_bauen(p, p.name + NEU_SUFFIX))
        except BaseException:
            for p in plaene:  # the old timelines stay untouched; drop the half-built new ones
                self.temp_verwerfen(p)
            raise
        for p in plaene:  # swap only after every new timeline is complete
            self.timeline_sichern_und_loeschen(p.name)
            neu = ra.find_timeline(self.project, p.name + NEU_SUFFIX)
            if neu is None or not neu.SetName(p.name):
                raise BauFehler(f"could not rename {p.name + NEU_SUFFIX!r} to {p.name!r}")
            self.log(f"timeline {p.name!r}: built")
        first = ra.find_timeline(self.project, plaene[0].name)
        if first is not None:
            self.project.SetCurrentTimeline(first)
        self.resolve.GetProjectManager().SaveProject()
        ra._guard("end")
        return results


def ffprobe_medium(path: Path) -> dict:
    """frames/breite/hoehe of the first video stream, dauer_s of the file (ffprobe)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                          "stream=codec_type,width,height,nb_read_packets:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=600, check=True).stdout
    data = json.loads(out)
    video = next((st for st in data.get("streams", []) if st.get("codec_type") == "video"), {})
    return {"frames": int(video.get("nb_read_packets") or 0), "breite": video.get("width"),
            "hoehe": video.get("height"), "dauer_s": float(data.get("format", {}).get("duration") or 0)}


def comp_probleme(comp: Path, laenge: int) -> list[str]:
    """A comp must output through MediaOut1 and cover the clip it sits on."""
    try:
        text = comp.read_text(errors="replace")
    except OSError as exc:
        return [f"{comp.name}: {exc}"]
    problems = []
    if "MediaOut1 = MediaOut" not in text:
        problems.append(f"{comp.parent.name}/{comp.name}: no MediaOut1")
    m = re.search(r"GlobalRange = \{ (-?\d+), (-?\d+) \}", text)
    if not m or int(m.group(2)) - int(m.group(1)) + 1 < laenge:
        problems.append(f"{comp.parent.name}/{comp.name}: GlobalRange {m.groups() if m else None} shorter than {laenge}")
    return problems


def _usage(clip: Any) -> int:
    """How many timeline items use the clip (media-pool column "Usage")."""
    try:
        return int(float(clip.GetClipProperty("Usage") or 0))
    except (TypeError, ValueError):
        return 0


def _ist_transition(item: Any) -> bool:
    try:
        return item.GetType() == "transition"
    except AttributeError:
        return False


def plaene_laden(formate: list[str], pfade: Pfade = Pfade(), teilweise: bool = False,
                 mit_comps: bool = True) -> list[Plan]:
    timeline = lade_json(pfade.repo / "timeline.json")
    script = lade_json(pfade.repo / "script.json")
    return [plan(fmt, timeline, script, pfade, teilweise, mit_comps) for fmt in formate]


def zusammenfassung(p: Plan) -> str:
    lines = [f"{p.name}: {p.breite}x{p.hoehe} @ {p.fps}, {p.dauer_f} frames"]
    for clip in sorted(p.clips, key=lambda c: (c.spur, c.track, c.record_f)):
        src = "whole" if clip.ganz else f"src {clip.quelle_start}..{clip.quelle_start + clip.laenge - 1}"
        comp = f" + {clip.comp.name}" if clip.comp else ""
        lines.append(f"  {clip.spur[0].upper()}{clip.track} {clip.record_f:5}..{clip.record_f + clip.laenge - 1:5} "
                     f"{clip.medium.name} {src}{comp}")
    lines += [f"  X  dissolve {b.dauer}f at {b.schnitt_f} ({b.links} -> {b.rechts})" for b in p.blenden]
    lines += [f"  M  {m.f:5} {m.name} ({m.farbe}, {m.dauer}f)" for m in p.marker]
    lines += [f"  !  left out: {a}" for a in p.ausgelassen]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", action="append", choices=list(FORMATE), help="default: both")
    ap.add_argument("--trocken", action="store_true", help="print the plan, do not touch Resolve")
    ap.add_argument("--teilweise", action="store_true", help="skip missing shots/comps (test build)")
    ap.add_argument("--nur-traeger", action="store_true", help="only create the carrier movies")
    ap.add_argument("--ohne-comps", action="store_true", help="leave out V2-V4 (no ImportFusionComp)")
    ap.add_argument("--frisch", action="store_true",
                    help="delete our timelines (after backup) before importing, so every clip is re-imported")
    args = ap.parse_args(argv)
    formate = args.format or list(FORMATE)
    if args.nur_traeger:
        for fmt in formate:
            print(traeger_erzeugen(fmt))
        return 0
    plaene = plaene_laden(formate, teilweise=args.teilweise or args.trocken, mit_comps=not args.ohne_comps)
    if args.trocken:
        for p in plaene:
            print(zusammenfassung(p))
        return 0
    for fmt in formate:
        traeger_erzeugen(fmt)
    resolve = ra.connect()
    project = ra.current_project(resolve)
    for result in Bauer(resolve, project).bauen(plaene, frisch=args.frisch):
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
