"""Tests for resolve/bauen.py (and the shot check of resolve/shots_kodieren.py) against a
fake Resolve: positions frame-exact from timeline.json, handles, dissolves, markers, and
idempotency (Review Focus 5: a second run duplicates neither media nor track items, and
never touches foreign media or timelines)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from resolve import bauen as b  # noqa: E402
from resolve import resolve_api as ra  # noqa: E402
from resolve import shots_kodieren as sk  # noqa: E402

TL = json.loads((REPO / "timeline.json").read_text())
SCRIPT = json.loads((REPO / "script.json").read_text())
FPS = 60
TC_START = 3600 * FPS  # 01:00:00:00


# --------------------------------------------------------------------------------------
# Fake Resolve (the subset bauen.py uses, with Resolve's observable behaviour)
# --------------------------------------------------------------------------------------

class FakeClip:
    _ids = 0

    def __init__(self, pool: "FakePool", path: str, frames: int):
        FakeClip._ids += 1
        self.uid, self.pool, self.path, self.frames = f"clip{FakeClip._ids}", pool, path, frames

    def GetClipProperty(self, key):
        return {"File Path": self.path, "Frames": str(self.frames), "Usage": str(self.pool.usage(self))}.get(key, "")

    def GetName(self):
        return Path(self.path).name

    def GetUniqueId(self):
        return self.uid


class FakeFolder:
    _ids = 0

    def __init__(self, name):
        FakeFolder._ids += 1
        self.uid, self.name, self.subs, self.clips = f"folder{FakeFolder._ids}", name, [], []

    def GetName(self):
        return self.name

    def GetUniqueId(self):
        return self.uid

    def GetSubFolderList(self):
        return list(self.subs)

    def GetClipList(self):
        return list(self.clips)


class FakeItem:
    def __init__(self, timeline, clip, kind, track, start, duration, source_start):
        self.timeline, self.clip, self.kind, self.track = timeline, clip, kind, track
        self.start, self.duration, self.source_start = start, duration, source_start
        self.comps: list[str] = []
        self.props: dict = {}

    def GetType(self):
        return self.kind

    def GetStart(self, subframePrecision=False):
        return self.start

    def GetEnd(self, subframePrecision=False):
        return self.start + self.duration

    def GetDuration(self, subframePrecision=False):
        return self.duration

    def GetLeftOffset(self, subframePrecision=False):
        return self.source_start

    def GetMediaPoolItem(self):
        return self.clip

    def ImportFusionComp(self, path):
        if not Path(path).exists():
            return None
        self.comps.append(f"Composition {len(self.comps) + 1}")
        self.comp_path = path
        return object()

    def GetFusionCompNameList(self):
        return list(self.comps)

    def LoadFusionCompByName(self, name):
        return object() if name in self.comps else None

    def SetProperty(self, key, value):
        self.props[key] = value
        return True

    def AddTransition(self, options):
        assert options["position"] == "end" and options["alignment"] == "center"
        half = int(options["duration"]) // 2
        track = self.timeline.tracks["video"][self.track - 1]
        right = [i for i in track if i.kind == "video" and i.start == self.GetEnd()]
        tail = self.clip.frames - (self.source_start + self.duration)
        if not right or tail < half or right[0].source_start < half:
            return None
        transition = FakeItem(self.timeline, None, "transition", self.track, self.GetEnd() - half,
                              int(options["duration"]), 0)
        track.append(transition)
        return transition


class FakeTimeline:
    _ids = 0

    def __init__(self, name):
        FakeTimeline._ids += 1
        self.uid, self.name = f"tl{FakeTimeline._ids}", name
        self.settings = {"timelineFrameRate": "60", "timelineResolutionWidth": "3840",
                         "timelineResolutionHeight": "2160", "timelineInputResMismatchBehavior": "scaleToFit"}
        self.start_frame = TC_START
        self.tracks = {"video": [[]], "audio": [[]], "subtitle": []}
        self.track_names: dict = {}
        self.markers: dict = {}
        self.exports: list[str] = []

    def GetName(self):
        return self.name

    def SetName(self, name):
        if any(tl.name == name for tl in self.project_timelines()):
            return False
        self.name = name
        return True

    def project_timelines(self):
        return self.owner.timelines

    def SetSetting(self, key, value):
        self.settings[key] = str(value)
        return True

    def GetSetting(self, key):
        return self.settings.get(key, "")

    def SetStartTimecode(self, tc):
        h, m, s, f = (int(x) for x in tc.split(":"))
        self.start_frame = ((h * 60 + m) * 60 + s) * FPS + f
        return True

    def GetStartFrame(self):
        return self.start_frame

    def GetTrackCount(self, kind):
        return len(self.tracks[kind])

    def AddTrack(self, kind, sub=None):
        self.tracks[kind].append([])
        return True

    def SetTrackName(self, kind, index, name):
        self.track_names[(kind, index)] = name
        return True

    def GetItemListInTrack(self, kind, index):
        return list(self.tracks[kind][index - 1])

    def AddMarker(self, frame, color, name, note, duration, customData=None):
        if frame in self.markers:
            return False
        self.markers[frame] = {"color": color, "name": name, "note": note, "duration": duration}
        return True

    def GetMarkers(self):
        return dict(self.markers)

    def Export(self, path, kind, sub):
        Path(path).write_text("drt")
        self.exports.append(path)
        return True

    def DeleteClips(self, items, ripple=False):
        for track_list in self.tracks.values():
            for track in track_list:
                for item in items:
                    if item in track:
                        track.remove(item)
        return True

    def items(self):
        return [i for track_list in self.tracks.values() for track in track_list for i in track]


class FakePool:
    def __init__(self, project, frames_of, end_inclusive, record_relative):
        self.project, self.frames_of = project, frames_of
        self.end_inclusive, self.record_relative = end_inclusive, record_relative
        self.root = FakeFolder("Master")
        self.current = self.root
        self.imports = 0
        self.resolve_symlinks = False

    def folders(self, folder=None):
        folder = folder or self.root
        out = [folder]
        for sub in folder.subs:
            out += self.folders(sub)
        return out

    def all_clips(self):
        return [c for f in self.folders() for c in f.clips]

    def usage(self, clip):
        return sum(1 for tl in self.project.timelines for i in tl.items() if i.clip is clip)

    def GetRootFolder(self):
        return self.root

    def GetCurrentFolder(self):
        return self.current

    def SetCurrentFolder(self, folder):
        self.current = folder
        return True

    def AddSubFolder(self, parent, name):
        folder = FakeFolder(name)
        parent.subs.append(folder)
        return folder

    def ImportMedia(self, paths):
        out = []
        for path in paths:
            if self.resolve_symlinks:  # Resolve may store the path behind a symlink
                path = os.path.realpath(path)
            clip = FakeClip(self, path, self.frames_of(path))
            self.current.clips.append(clip)
            out.append(clip)
            self.imports += 1
        return out

    def DeleteClips(self, clips):
        for clip in clips:
            for folder in self.folders():
                if clip in folder.clips:
                    folder.clips.remove(clip)
            for tl in self.project.timelines:
                tl.DeleteClips([i for i in tl.items() if i.clip is clip])
        return True

    def MoveClips(self, clips, target):
        for clip in clips:
            for folder in self.folders():
                if clip in folder.clips:
                    folder.clips.remove(clip)
            target.clips.append(clip)
        return True

    def CreateEmptyTimeline(self, name):
        if any(tl.name == name for tl in self.project.timelines):
            return None
        tl = FakeTimeline(name)
        tl.owner = self.project
        self.project.timelines.append(tl)
        return tl

    def DeleteTimelines(self, timelines):
        for tl in timelines:
            self.project.timelines.remove(tl)
            if self.project.current is tl:
                self.project.current = None
        return True

    def AppendToTimeline(self, infos):
        tl = self.project.current
        out = []
        for info in infos:
            clip = info["mediaPoolItem"]
            kind = "video" if info.get("mediaType", 1) == 1 else "audio"
            index = int(info.get("trackIndex", 1))
            if index > len(tl.tracks[kind]):
                return []
            if "startFrame" in info:
                start, end = int(info["startFrame"]), int(info["endFrame"])
                duration = end - start + (1 if self.end_inclusive else 0)
            else:
                start, duration = 0, clip.frames
            record = int(info.get("recordFrame", tl.start_frame)) + (tl.start_frame if self.record_relative and "recordFrame" in info else 0)
            item = FakeItem(tl, clip, kind, index, record, duration, start)
            tl.tracks[kind][index - 1].append(item)
            out.append(item)
        return out


class FakeProject:
    def __init__(self, frames_of, end_inclusive=False, record_relative=False):
        self.timelines: list[FakeTimeline] = []
        self.current = None
        self.rendering = False
        self.jobs: list[dict] = []
        self.pool = FakePool(self, frames_of, end_inclusive, record_relative)

    def GetName(self):
        return "nomiss"

    def GetMediaPool(self):
        return self.pool

    def GetTimelineCount(self):
        return len(self.timelines)

    def GetTimelineByIndex(self, index):
        return self.timelines[index - 1]

    def SetCurrentTimeline(self, tl):
        self.current = tl
        return True

    def GetCurrentTimeline(self):
        return self.current

    def IsRenderingInProgress(self):
        return self.rendering

    def GetRenderJobList(self):
        return list(self.jobs)

    def DeleteRenderJob(self, job_id):
        self.jobs = [j for j in self.jobs if j["JobId"] != job_id]
        return True


class FakeResolve:
    EXPORT_DRT, EXPORT_NONE, COMPOSITE_NORMAL = "drt", "none", 0

    def __init__(self, project):
        self.project, self.saved = project, 0

    def GetProjectManager(self):
        return self

    def GetCurrentProject(self):
        return self.project

    def SaveProject(self):
        self.saved += 1
        return True

    def Fusion(self):
        class FontManager:
            @staticmethod
            def GetFontList():
                return {"Manrope": {}, "JetBrains Mono": {}}

            @staticmethod
            def AddFont(path):
                return True

        class Fusion:
            pass
        fusion = Fusion()
        fusion.FontManager = FontManager
        return fusion


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    monkeypatch.setattr(ra, "open_dialogs", lambda: [])


def fake_repo(root: Path, missing_shots: tuple[str, ...] = ()) -> b.Pfade:
    shutil.copy(REPO / "timeline.json", root / "timeline.json")
    shutil.copy(REPO / "script.json", root / "script.json")
    pfade = b.Pfade(root)
    pfade.renders.mkdir(parents=True)
    frames = sk.soll_frames(TL)
    for szene in TL["szenen"]:
        if szene["id"] in missing_shots:
            continue
        (pfade.renders / f"{szene['id']}.mov").write_bytes(b"mov")
        (pfade.renders / f"{szene['id']}.json").write_text(json.dumps(sk.meta(szene["id"], TL, frames[szene["id"]])))
    for fmt in b.FORMATE:
        (pfade.fusion / fmt).mkdir(parents=True)
        laengen = {s["id"]: s["ende_f"] - s["start_f"] for s in TL["szenen"]}
        laengen["hud"] = TL["dauer_f"]
        laengen["abspann"] = TL["abspann"]["ende_f"] - TL["abspann"]["start_f"]
        for cid, laenge in laengen.items():
            (pfade.fusion / fmt / f"{cid}.comp").write_text(
                f"Composition {{\n\tGlobalRange = {{ 0, {laenge - 1} }},\n\tTools = {{\n"
                f"\t\tMediaOut1 = MediaOut {{ }},\n\t}}\n}}\n")
        pfade.arbeit.mkdir(parents=True, exist_ok=True)
        pfade.traeger(fmt).write_bytes(b"mov")
    pfade.stems.mkdir(parents=True)
    for stem in b.STEMS:
        (pfade.stems / f"{stem}.wav").write_bytes(b"wav")
    return pfade


def frames_of_factory(pfade: b.Pfade):
    frames = sk.soll_frames(TL)

    def frames_of(path: str) -> int:
        p = Path(path)
        if p.parent == pfade.renders:
            return frames[p.stem]
        if p.name.startswith("traeger_"):
            return b.TRAEGER_F
        if p.suffix == ".wav":
            return int(TL["dauer_f"])
        return 100
    return frames_of


def probe_factory(pfade: b.Pfade, overrides: dict | None = None):
    frames = frames_of_factory(pfade)

    def probe(path: Path) -> dict:
        path = Path(path)
        if path.name in (overrides or {}):
            return overrides[path.name]
        if path.suffix == ".wav":
            return {"frames": 0, "breite": None, "hoehe": None, "dauer_s": TL["dauer_f"] / FPS}
        if path.name.startswith("traeger_"):
            w, h = b.FORMATE[path.stem.split("_")[1]]
            return {"frames": b.TRAEGER_F, "breite": w, "hoehe": h, "dauer_s": b.TRAEGER_F / FPS}
        return {"frames": frames(str(path)), "breite": 1440, "hoehe": 1440, "dauer_s": 0.0}
    return probe


def build(pfade, project, formate=("16x9", "9x16"), teilweise=False, probe=None):
    plaene = b.plaene_laden(list(formate), pfade, teilweise)
    return b.Bauer(FakeResolve(project), project, pfade, log=lambda *_: None,
                   medien_probe=probe or probe_factory(pfade)).bauen(plaene)


def by_name(project, name):
    return next(tl for tl in project.timelines if tl.name == name)


def snapshot(project):
    """Clip count and per-track item counts of every timeline."""
    return {
        "clips": len(project.pool.all_clips()),
        "timelines": sorted(tl.name for tl in project.timelines),
        "tracks": {tl.name: {(k, n + 1): len(t) for k, tracks in tl.tracks.items() for n, t in enumerate(tracks)}
                   for tl in project.timelines},
    }


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["16x9", "9x16"])
def test_plan_positions_frame_exact_from_timeline(tmp_path, fmt):
    pfade = fake_repo(tmp_path)
    p = b.plaene_laden([fmt], pfade)[0]
    v1 = {c.name: c for c in p.clips if c.spur == "video" and c.track == 1}
    v2 = {c.comp.stem: c for c in p.clips if c.spur == "video" and c.track == 2}
    assert len(v1) == len(v2) == len(TL["szenen"])
    for szene in TL["szenen"]:
        laenge = szene["ende_f"] - szene["start_f"]
        shot, comp = v1[szene["id"]], v2[szene["id"]]
        assert (shot.record_f, shot.laenge, shot.quelle_start) == (szene["start_f"], laenge, 30)
        assert shot.medium == pfade.renders / f"{szene['id']}.mov"
        # the shot keeps 30 frames of handle on both sides
        meta = json.loads((pfade.renders / f"{szene['id']}.json").read_text())
        assert meta["frames"] - (shot.quelle_start + shot.laenge) == 30 == meta["handle_f"]
        assert (comp.record_f, comp.laenge, comp.quelle_start) == (szene["start_f"], laenge, 0)
        assert comp.comp == pfade.fusion / fmt / f"{szene['id']}.comp"
        assert comp.medium == pfade.traeger(fmt)
    hud = [c for c in p.clips if c.track == 3 and c.spur == "video"]
    abspann = [c for c in p.clips if c.track == 4 and c.spur == "video"]
    assert [(c.record_f, c.laenge, c.comp.name) for c in hud] == [(0, TL["dauer_f"], "hud.comp")]
    assert [(c.record_f, c.laenge, c.comp.name) for c in abspann] == [
        (TL["abspann"]["start_f"], TL["abspann"]["ende_f"] - TL["abspann"]["start_f"], "abspann.comp")]
    assert TL["abspann"]["start_f"] == round((TL["dauer_s"] - 2.0) * FPS)
    audio = sorted((c.track, c.medium.stem, c.record_f, c.ganz) for c in p.clips if c.spur == "audio")
    assert audio == [(i, s, 0, True) for i, s in enumerate(b.STEMS, start=1)]
    cuts = [s["start_f"] for s in TL["szenen"][1:]]
    assert [(x.schnitt_f, x.dauer) for x in p.blenden] == [(f, 12) for f in cuts]
    titles = {s["id"]: s["titel"] for s in SCRIPT["szenen"]}
    assert [(m.f, m.name) for m in p.marker[:-1]] == [(s["start_f"], titles[s["id"]]) for s in TL["szenen"]]
    assert p.marker[-1].f == TL["abspann"]["start_f"]


def test_plan_rejects_wrong_handles(tmp_path):
    pfade = fake_repo(tmp_path)
    info = pfade.renders / "s3_netz.json"
    meta = json.loads(info.read_text())
    meta["frames"] -= 1
    info.write_text(json.dumps(meta))
    with pytest.raises(b.BauFehler, match="s3_netz"):
        b.plaene_laden(["16x9"], pfade)


def test_plan_rejects_shifted_shot_start(tmp_path):
    pfade = fake_repo(tmp_path)
    info = pfade.renders / "s5_betrieb.json"
    meta = json.loads(info.read_text())
    meta["start_s"] += 1 / FPS
    info.write_text(json.dumps(meta))
    with pytest.raises(b.BauFehler, match="s5_betrieb"):
        b.plaene_laden(["16x9"], pfade)


def test_plan_missing_shot_fails_unless_partial(tmp_path):
    pfade = fake_repo(tmp_path, missing_shots=("s4_ernstfall",))
    with pytest.raises(b.BauFehler, match="s4_ernstfall"):
        b.plaene_laden(["16x9"], pfade)
    p = b.plaene_laden(["16x9"], pfade, teilweise=True)[0]
    assert any("s4_ernstfall" in a for a in p.ausgelassen)
    # no dissolve across the hole
    assert {(x.links, x.rechts) for x in p.blenden}.isdisjoint(
        {("s3_netz", "s4_ernstfall"), ("s4_ernstfall", "s5_betrieb"), ("s3_netz", "s5_betrieb")})


# --------------------------------------------------------------------------------------
# Builder against the fake Resolve
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("end_inclusive,record_relative", [(True, False), (False, False), (True, True)])
def test_build_places_everything_frame_exact(tmp_path, end_inclusive, record_relative):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade), end_inclusive, record_relative)
    results = build(pfade, project)
    assert [r["blenden"] for r in results] == [7, 7]
    for fmt, (w, h) in b.FORMATE.items():
        tl = by_name(project, f"nomissuccess {fmt}")
        assert (tl.settings["timelineResolutionWidth"], tl.settings["timelineResolutionHeight"]) == (str(w), str(h))
        assert tl.settings["timelineInputResMismatchBehavior"] == "scaleToCrop"
        assert tl.start_frame == TC_START and ":" not in tl.name
        shots = {i.clip.GetName(): i for i in tl.tracks["video"][0] if i.kind == "video"}
        comps = sorted((i for i in tl.tracks["video"][1]), key=lambda i: i.start)
        for szene, comp in zip(TL["szenen"], comps):
            item = shots[f"{szene['id']}.mov"]
            laenge = szene["ende_f"] - szene["start_f"]
            assert (item.start, item.duration, item.source_start) == (TC_START + szene["start_f"], laenge, 30)
            assert (comp.start, comp.duration, comp.source_start) == (TC_START + szene["start_f"], laenge, 0)
            assert Path(comp.comp_path).name == f"{szene['id']}.comp" and f"/{fmt}/" in comp.comp_path
            assert comp.props["CompositeMode"] == FakeResolve.COMPOSITE_NORMAL
        transitions = [i for i in tl.tracks["video"][0] if i.kind == "transition"]
        assert sorted(t.start + t.duration // 2 for t in transitions) == [TC_START + s["start_f"] for s in TL["szenen"][1:]]
        (hud,) = tl.tracks["video"][2]
        (abspann,) = tl.tracks["video"][3]
        assert (hud.start, hud.duration) == (TC_START, TL["dauer_f"])
        assert (abspann.start, abspann.GetEnd()) == (TC_START + TL["abspann"]["start_f"], TC_START + TL["dauer_f"])
        assert [len(t) for t in tl.tracks["audio"]] == [1] * 8
        assert all(t[0].start == TC_START for t in tl.tracks["audio"])
        assert len(tl.markers) == len(TL["szenen"]) + 1
        assert tl.markers[TL["szenen"][1]["start_f"]]["name"] == "Schaufenster"
    names = {f.GetName(): f for f in project.pool.root.subs}
    assert set(names) == {"01 Blender", "02 Fusion", "03 Audio"}
    assert len(names["01 Blender"].clips) == 8 and len(names["02 Fusion"].clips) == 2
    assert len(names["03 Audio"].clips) == 8


def test_second_run_duplicates_nothing_and_keeps_foreign_media(tmp_path):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    pool = project.pool
    foreign = pool.ImportMedia(["/home/someone/other/footage.mov"])[0]
    foreign_repo = pool.ImportMedia([str(pfade.repo / "work" / "fusion" / "traeger_12000.mov")])[0]
    foreign_tl = pool.CreateEmptyTimeline("zz typo-test")
    project.SetCurrentTimeline(foreign_tl)
    pool.AppendToTimeline([{"mediaPoolItem": foreign_repo, "startFrame": 0, "endFrame": 9, "trackIndex": 1}])

    build(pfade, project)
    first = snapshot(project)
    build(pfade, project)
    second = snapshot(project)

    assert first == second
    assert first["clips"] == 2 + 8 + 2 + 8
    assert first["timelines"] == ["nomissuccess 16x9", "nomissuccess 9x16", "zz typo-test"]
    assert foreign in pool.all_clips() and foreign_repo in pool.all_clips()
    assert len(foreign_tl.items()) == 1 and foreign_tl.items()[0].clip is foreign_repo
    backups = sorted(p.name for p in pfade.arbeit.glob("*.drt"))
    assert len(backups) >= 1 and all(n.startswith("nomissuccess_") for n in backups)


def test_rebuilding_one_format_keeps_the_other_timeline_intact(tmp_path):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    build(pfade, project)
    before = snapshot(project)
    build(pfade, project, formate=("16x9",))
    assert snapshot(project) == before
    other = by_name(project, "nomissuccess 9x16")
    assert all(i.clip in project.pool.all_clips() for i in other.items() if i.kind != "transition")


def test_rebuild_reuses_live_clips_and_drops_unused_duplicates(tmp_path):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    build(pfade, project)
    count = len(project.pool.all_clips())
    old = {c.path: c for c in project.pool.all_clips()}
    project.pool.SetCurrentFolder(project.pool.root)
    project.pool.ImportMedia([str(pfade.stems / "vo.wav")])  # an unused duplicate of our own file
    build(pfade, project)
    new = [c for c in project.pool.all_clips()]
    assert len(new) == count and [c.path for c in new].count(str(pfade.stems / "vo.wav")) == 1
    assert all(old[c.path] is c for c in new)  # clips of the live timelines were reused


def test_unused_own_clip_is_reimported(tmp_path):
    """No timeline uses it any more: it is replaced (a re-encoded movie must not stay stale)."""
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    project.pool.SetCurrentFolder(project.pool.root)
    stale = project.pool.ImportMedia([str(pfade.renders / "s2_website.mov")])[0]
    build(pfade, project)
    clips = [c for c in project.pool.all_clips() if c.path == stale.path]
    assert len(clips) == 1 and clips[0] is not stale


def test_failure_while_building_keeps_the_old_timelines(tmp_path):
    """Review H1: a failing build must not leave the project without timelines."""
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    build(pfade, project)
    before = snapshot(project)
    original = FakePool.AppendToTimeline

    def fail_on_9x16(self, infos):
        if self.project.current.name.startswith("nomissuccess 9x16"):
            return []
        return original(self, infos)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakePool, "AppendToTimeline", fail_on_9x16)
        with pytest.raises(b.BauFehler, match="AppendToTimeline failed"):
            build(pfade, project)
    assert snapshot(project) == before
    assert not any(tl.name.endswith(" neu") for tl in project.timelines)
    build(pfade, project)  # and the next run works again
    assert snapshot(project) == before


def test_preflight_failure_touches_nothing(tmp_path):
    """Review H1: a movie one frame short is caught before anything is deleted or imported."""
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    build(pfade, project)
    before, imports = snapshot(project), project.pool.imports
    short = {"s3_netz.mov": {"frames": 1510, "breite": 1440, "hoehe": 1440, "dauer_s": 0.0}}
    with pytest.raises(b.BauFehler, match="(?s)preflight.*s3_netz.mov"):
        build(pfade, project, probe=probe_factory(pfade, short))
    assert snapshot(project) == before and project.pool.imports == imports
    (pfade.fusion / "9x16" / "hud.comp").write_text("Composition { GlobalRange = { 0, 99 }, }")
    with pytest.raises(b.BauFehler, match="hud.comp: no MediaOut1"):
        build(pfade, project)
    assert snapshot(project) == before


def test_build_refuses_while_rendering(tmp_path):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    build(pfade, project)
    before = snapshot(project)
    project.rendering = True
    with pytest.raises(b.BauFehler, match="rendering"):
        build(pfade, project)
    assert snapshot(project) == before


def test_own_render_jobs_of_a_deleted_timeline_are_removed(tmp_path):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    build(pfade, project)
    project.jobs = [{"JobId": "a", "TimelineName": "nomissuccess 16x9"}, {"JobId": "b", "TimelineName": "fremd"}]
    build(pfade, project)
    assert [j["JobId"] for j in project.jobs] == ["b"]


def test_input_sizing_fallback_zooms_each_shot(tmp_path, monkeypatch):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    original = FakeTimeline.SetSetting

    def refuse_sizing(self, key, value):
        if key == "timelineInputResMismatchBehavior":
            return False
        return original(self, key, value)

    monkeypatch.setattr(FakeTimeline, "SetSetting", refuse_sizing)
    results = build(pfade, project)
    assert all(r["zoom_pro_clip"] for r in results)
    tl = by_name(project, "nomissuccess 9x16")
    shots = [i for i in tl.tracks["video"][0] if i.kind == "video"]
    assert all(abs(i.props["ZoomX"] - 1920 / 1080) < 1e-9 for i in shots)


# --------------------------------------------------------------------------------------
# shots_kodieren: when is a shot finished?
# --------------------------------------------------------------------------------------

def _png(path: Path, size=1440):
    from PIL import Image
    Image.new("RGB", (size, size), (10, 20, 30)).save(path)


def test_shot_finished_only_with_all_frames_readable_and_no_blender(tmp_path):
    folder = tmp_path / "s9_test"
    folder.mkdir()
    for index in range(1, 6):
        _png(folder / f"{index:04d}.png")
    assert sk.status("s9_test", 5, tmp_path, running=set()).fertig
    assert not sk.status("s9_test", 5, tmp_path, running={"s9_test"}).fertig
    assert not sk.status("s9_test", 6, tmp_path, running=set()).fertig
    (folder / "0005.png").write_bytes((folder / "0004.png").read_bytes()[:40])  # half-written
    st = sk.status("s9_test", 5, tmp_path, running=set())
    assert not st.fertig and not st.letztes_lesbar


def test_shot_meta_matches_timeline_contract():
    frames = sk.soll_frames(TL)
    assert frames == {"s1_nacht": 1050, "s2_website": 1823, "s3_netz": 1511, "s4_ernstfall": 910,
                      "s5_betrieb": 978, "s6_beweis": 772, "s7_team": 1233, "s8_morgen": 1004}
    meta = sk.meta("s2_website", TL, frames["s2_website"])
    assert meta["start_s"] == pytest.approx(16.5 - 0.5)
    assert (meta["handle_f"], meta["frames"], meta["shot"], meta["szene"]) == (30, 1823, "s2_website", "s2_website")


# --------------------------------------------------------------------------------------
# rendern: loudness decision (measure -> re-mux reference -> trim AAC overshoot)
# --------------------------------------------------------------------------------------

def _loudness_sequence(monkeypatch, measurements):
    from resolve import rendern as r
    seq = iter(measurements)
    calls: list[float] = []
    monkeypatch.setattr(r, "lautheit", lambda path: next(seq))
    monkeypatch.setattr(r, "ton_neu_muxen", lambda mp4, gain_db=0.0: calls.append(gain_db))
    return r, calls


def test_loudness_in_tolerance_is_left_alone(monkeypatch):
    r, calls = _loudness_sequence(monkeypatch, [{"lufs": -14.3, "dbtp": -1.2}])
    report = r.lautheit_sichern(Path("x.mp4"))
    assert calls == [] and report["massnahme"] == "keine" and r.lautheit_ok(report["ergebnis"])


def test_loudness_drift_remuxes_the_reference(monkeypatch):
    r, calls = _loudness_sequence(monkeypatch, [{"lufs": -16.0, "dbtp": -3.0}, {"lufs": -14.3, "dbtp": -1.2}])
    report = r.lautheit_sichern(Path("x.mp4"))
    assert calls == [0.0] and "neu gemuxt" in report["massnahme"] and r.lautheit_ok(report["ergebnis"])


def test_aac_overshoot_gets_minus_0_3_db_on_the_reference(monkeypatch):
    """Review L4: -1.0 dBTP is not enough (one decimal, AAC) — re-mux with -0.3 dB."""
    r, calls = _loudness_sequence(monkeypatch, [{"lufs": -14.1, "dbtp": -1.0}, {"lufs": -14.1, "dbtp": -1.0},
                                                {"lufs": -14.4, "dbtp": -1.3}])
    report = r.lautheit_sichern(Path("x.mp4"))
    assert calls == [0.0, -0.3] and r.lautheit_ok(report["ergebnis"]) and "-0.3 dB" in report["massnahme"]
    # a re-muxed reference that is fine needs no gain
    r, calls = _loudness_sequence(monkeypatch, [{"lufs": -14.1, "dbtp": -1.0}, {"lufs": -14.3, "dbtp": -1.2}])
    assert r.lautheit_sichern(Path("x.mp4"))["ergebnis"]["dbtp"] == -1.2 and calls == [0.0]
    # and a miss stays a miss (no success without success)
    r, calls = _loudness_sequence(monkeypatch, [{"lufs": -16.0, "dbtp": -0.5}, {"lufs": -16.0, "dbtp": -0.5},
                                                {"lufs": -16.3, "dbtp": -0.8}])
    assert not r.lautheit_ok(r.lautheit_sichern(Path("x.mp4"))["ergebnis"])


def test_true_peak_needs_reserve():
    from resolve import rendern as r
    assert not r.lautheit_ok({"lufs": -14.0, "dbtp": -1.0})
    assert r.lautheit_ok({"lufs": -14.0, "dbtp": -1.1})


def test_master_mix_must_match_the_reference():
    from resolve import rendern as r
    ref = {"lufs": -14.3, "dbtp": -1.2}
    assert r.mix_abweichung({"lufs": -14.2, "dbtp": -1.1}, ref) is None
    assert "weicht" in r.mix_abweichung({"lufs": -15.1, "dbtp": -1.2}, ref)  # e.g. a muted SFX track


def test_alpha_check_maps_spot_frame_to_the_scene_shot():
    """Review L9: source frame = spot frame - scene start + handle (not just + handle)."""
    from resolve import rendern as r
    szene, frame, quelle = r.alpha_frames(TL)
    assert szene["id"] == "s1_nacht" and quelle == frame + 30
    tl = json.loads(json.dumps(TL))
    tl["events"].append({"id": "probe", "szene": "s3_netz", "f": 3000})
    szene, frame, quelle = r.alpha_frames(tl, "probe")
    assert (szene["id"], frame, quelle) == ("s3_netz", 3000, 3000 - 2753 + 30)


def test_fallback_filter_is_frame_exact():
    """Shots trimmed with 6 frames around each cut, dissolves centred on the cut,
    black tail to the end: the graph yields exactly dauer_f frames."""
    import re as _re
    from resolve import rendern as r
    graph = r.fallback_filter("9x16", TL, overlay_input=8)
    trims = [tuple(map(int, m)) for m in _re.findall(r"trim=start_frame=(\d+):end_frame=(\d+)", graph)]
    szenen = TL["szenen"]
    assert trims[0] == (30, 30 + szenen[0]["ende_f"] - szenen[0]["start_f"] + 6)
    assert trims[-1] == (24, 30 + szenen[-1]["ende_f"] - szenen[-1]["start_f"])
    offsets = [round(float(o) * 60) for o in _re.findall(r"offset=([\d.]+)", graph)]
    assert offsets == [s["start_f"] - 6 for s in szenen[1:]]
    length = trims[0][1] - trims[0][0]
    for (a, b_), off in zip(trims[1:], offsets):
        length = off + (b_ - a)
    tail = int(_re.search(r"tpad=stop=(\d+)", graph).group(1))
    assert length + tail == TL["dauer_f"]
    assert "crop=1080:1920" in graph and "out_color_matrix=bt709" in graph


def test_blender_own_encode_is_not_accepted(tmp_path):
    """Review H2: renders/<shot>.json written by Blender's encode (no 1440², no BT.709)."""
    pfade = fake_repo(tmp_path)
    info = pfade.renders / "s7_team.json"
    meta = json.loads(info.read_text())
    del meta["kodierer"]
    info.write_text(json.dumps(meta))
    with pytest.raises(b.BauFehler, match="s7_team.*not a shots_kodieren encode"):
        b.plaene_laden(["16x9"], pfade)
    p = b.plaene_laden(["16x9"], pfade, teilweise=True)[0]
    assert "s7_team" not in {c.name for c in p.clips}


def test_missing_dissolve_fails_the_final_build(tmp_path, monkeypatch):
    """Review M5: 0 of 7 dissolves must not end in success."""
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    monkeypatch.setattr(FakeItem, "AddTransition", lambda self, options: None)
    with pytest.raises(b.BauFehler, match="Cross Dissolve"):
        build(pfade, project)
    assert not project.timelines  # nothing half-built is left


def test_frisch_reimports_every_clip(tmp_path):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    build(pfade, project)
    before = snapshot(project)
    old = {c.path: c for c in project.pool.all_clips()}
    plaene = b.plaene_laden(["16x9", "9x16"], pfade)
    b.Bauer(FakeResolve(project), project, pfade, log=lambda *_: None,
            medien_probe=probe_factory(pfade)).bauen(plaene, frisch=True)
    assert snapshot(project) == before
    assert all(old[c.path] is not c for c in project.pool.all_clips())
    assert len(list(pfade.arbeit.glob("*.drt"))) >= 2


def test_encode_meta_marks_its_producer_and_source(tmp_path):
    meta = sk.meta("s1_nacht", TL, 1050, quelle_mtime=123.5)
    assert meta["kodierer"] == "shots_kodieren" and meta["quelle_mtime"] == 123.5


def test_small_proxy_frame_blocks_the_shot(tmp_path):
    folder = tmp_path / "s9_test"
    folder.mkdir()
    for index in range(1, 4):
        _png(folder / f"{index:04d}.png", size=1440)
    assert sk.status("s9_test", 3, tmp_path, running=set()).fertig
    _png(folder / "0002.png", size=480)
    st = sk.status("s9_test", 3, tmp_path, running=set())
    assert not st.fertig and st.kaputt == [2]


def test_comp_or_stem_older_than_timeline_blocks_the_final_build(tmp_path):
    import os
    pfade = fake_repo(tmp_path)
    t = (pfade.repo / "timeline.json").stat().st_mtime
    stale = pfade.fusion / "9x16" / "s4_ernstfall.comp"
    os.utime(stale, (t - 60, t - 60))
    b.plaene_laden(["16x9"], pfade)  # the 16x9 comps are fresh
    with pytest.raises(b.BauFehler, match="s4_ernstfall.comp is older"):
        b.plaene_laden(["9x16"], pfade)
    os.utime(pfade.stems / "sfx_hit.wav", (t - 60, t - 60))
    with pytest.raises(b.BauFehler, match="sfx_hit.wav is older"):
        b.plaene_laden(["16x9"], pfade)


def test_build_without_comps_and_save_before_every_comp_import(tmp_path):
    pfade = fake_repo(tmp_path)
    project = FakeProject(frames_of_factory(pfade))
    plaene = b.plaene_laden(["16x9"], pfade, mit_comps=False)
    assert not any(c.comp for c in plaene[0].clips) and plaene[0].ausgelassen
    resolve = FakeResolve(project)
    b.Bauer(resolve, project, pfade, log=lambda *_: None, medien_probe=probe_factory(pfade)).bauen(plaene)
    tl = by_name(project, "nomissuccess 16x9")
    assert [len(t) for t in tl.tracks["video"][1:]] == [0, 0, 0]
    assert len([i for i in tl.tracks["video"][0] if i.kind == "video"]) == 8
    # with comps: one save before each of the 10 imports (+ the final save)
    project2 = FakeProject(frames_of_factory(pfade))
    resolve2 = FakeResolve(project2)
    b.Bauer(resolve2, project2, pfade, log=lambda *_: None,
            medien_probe=probe_factory(pfade)).bauen(b.plaene_laden(["16x9"], pfade))
    assert resolve2.saved == 10 + 1


def test_truncated_png_in_the_middle_blocks_the_shot(tmp_path):
    """Review M1: a PNG cut off mid-sequence must not become a silent freeze frame."""
    folder = tmp_path / "s9_test"
    folder.mkdir()
    for index in range(1, 6):
        _png(folder / f"{index:04d}.png")
    data = (folder / "0003.png").read_bytes()
    (folder / "0003.png").write_bytes(data[: len(data) // 2])
    st = sk.status("s9_test", 5, tmp_path, running=set())
    assert st.letztes_lesbar and st.kaputt == [3] and not st.fertig


def test_encoder_refuses_when_the_disk_is_nearly_full(tmp_path, monkeypatch):
    """06.10.: the system disk ran full and Blender wrote truncated PNGs — never start a
    multi-GB encode without room."""
    import shutil as _shutil
    folder = tmp_path / "s1_nacht"
    folder.mkdir()
    tl = json.loads(json.dumps(TL))
    tl["szenen"] = [dict(TL["szenen"][0], ende_f=2)]  # 2 + 60 frames
    for index in range(1, 63):
        _png(folder / f"{index:04d}.png")
    monkeypatch.setattr(sk, "blender_shots_running", lambda proc=None: set())
    monkeypatch.setattr(_shutil, "disk_usage", lambda path: _shutil._ntuple_diskusage(100, 99, 10 ** 9))
    monkeypatch.setattr(sk, "png_decodes", lambda path: True)
    calls = []
    monkeypatch.setattr(sk.subprocess, "run", lambda *a, **k: calls.append(a))
    result = sk.kodieren("s1_nacht", tl, tmp_path)
    assert result.startswith("offen: zu wenig Platz") and calls == []


def test_render_refuses_without_room_for_the_masters(tmp_path, monkeypatch):
    import shutil as _shutil
    from resolve import rendern as r
    monkeypatch.setattr(_shutil, "disk_usage", lambda path: _shutil._ntuple_diskusage(100, 99, 10 * 10 ** 9))
    with pytest.raises(RuntimeError, match="zu wenig Platz"):
        r.platz_pruefen(2, tmp_path)
    r.platz_pruefen(0, tmp_path)  # 3 GB reserve fits



def test_symlinked_renders_are_still_our_own_clips(tmp_path):
    """06.10.: renders/ became a symlink to another disk. If Resolve stores the resolved
    path, the builder must still recognise (not duplicate, not orphan) its clips."""
    (tmp_path / "repo").mkdir()
    pfade = fake_repo(tmp_path / "repo")
    elsewhere = tmp_path / "andere_platte" / "renders"
    elsewhere.parent.mkdir()
    shutil.move(str(pfade.renders), str(elsewhere))
    pfade.renders.symlink_to(elsewhere)
    def frames_of(path):
        return frames_of_factory(pfade)(str(pfade.renders / Path(path).name) if str(elsewhere) in path else path)
    project = FakeProject(frames_of)
    project.pool.resolve_symlinks = True
    build(pfade, project)
    first = snapshot(project)
    assert all(str(elsewhere) in c.path for c in project.pool.all_clips() if c.path.endswith(".mov")
               and "traeger" not in c.path)
    assert all(pfade.eigen(c.path) for c in project.pool.all_clips())
    build(pfade, project)
    assert snapshot(project) == first



def test_alpha_verdict_needs_transparency_and_visible_type():
    """Measured 06.10. on the real master at clock_roll: 98.6 % equal, 1.2 % type."""
    from resolve import rendern as r
    assert r.alpha_urteil(0.986, 0.0117)      # real master: transparent layers + counter + HUD
    assert not r.alpha_urteil(0.999, 0.0001)  # no comps at all
    assert not r.alpha_urteil(0.20, 0.30)     # opaque layers cover the shot
