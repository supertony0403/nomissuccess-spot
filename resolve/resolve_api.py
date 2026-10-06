"""Small connection helper for the DaVinci Resolve scripting API (external Python).

Usage::

    from resolve.resolve_api import connect
    resolve = connect()            # raises ResolveNotReachable when Resolve is down
    project = resolve.GetProjectManager().GetCurrentProject()

The helper only sets the three environment variables Blackmagic documents for
external interpreters and imports ``DaVinciResolveScript``. It never switches
projects or pages; callers decide what to touch.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

RESOLVE_SCRIPT_API = "/opt/resolve/Developer/Scripting"
RESOLVE_SCRIPT_LIB = "/opt/resolve/libs/Fusion/fusionscript.so"
PROJECT_NAME = "nomiss"


class ResolveNotReachable(RuntimeError):
    """Resolve is not running, or external scripting is unavailable."""


def _prepare_environment() -> None:
    os.environ.setdefault("RESOLVE_SCRIPT_API", RESOLVE_SCRIPT_API)
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", RESOLVE_SCRIPT_LIB)
    modules = os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules")
    if modules not in sys.path:
        sys.path.append(modules)


def connect() -> Any:
    """Return the ``Resolve`` application object or raise ResolveNotReachable."""
    _prepare_environment()
    try:
        dvr = importlib.import_module("DaVinciResolveScript")
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ResolveNotReachable(f"DaVinciResolveScript not importable: {exc}") from exc
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise ResolveNotReachable("scriptapp('Resolve') returned None — is Resolve running?")
    return resolve


def current_project(resolve: Any, expected: str | None = PROJECT_NAME) -> Any:
    """Return the open project; refuse to work on an unexpected one.

    We never call ``LoadProject``: switching projects would switch Anthony's UI.
    """
    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        raise ResolveNotReachable("no project is open in Resolve")
    if expected is not None and project.GetName() != expected:
        raise ResolveNotReachable(
            f"open project is {project.GetName()!r}, expected {expected!r} — not switching"
        )
    return project


def find_timeline(project: Any, name: str) -> Any | None:
    """Return the timeline called ``name`` or None."""
    for index in range(1, int(project.GetTimelineCount() or 0) + 1):
        timeline = project.GetTimelineByIndex(index)
        if timeline is not None and timeline.GetName() == name:
            return timeline
    return None


def open_dialogs() -> list[dict]:
    """Resolve windows other than the main window (message boxes). A modal box blocks
    the scripting API (``GetCurrentPage()`` → None, ImportMedia/AppendToTimeline fail)
    and its text is not logged. Hyprland only; returns [] where hyprctl is missing.
    Seen on 21.1.1: AddRenderJob with a TargetDir outside Media Storage (auto-closes),
    AppendToTimeline of a *still* with startFrame/endFrame (stays open, stacks)."""
    import json
    import subprocess

    try:
        out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=5).stdout
        clients = json.loads(out or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return [{"address": c["address"], "title": c["title"], "size": c["size"]}
            for c in clients
            if c.get("class") == "resolve" and c.get("mapped") and not c["title"].startswith("DaVinci Resolve")]


class ResolveDialogOpen(RuntimeError):
    """A Resolve message box appeared; a human (or the caller, deliberately) must close it."""


def _guard(step: str) -> None:
    dialogs = open_dialogs()
    if dialogs:
        raise ResolveDialogOpen(f"after {step}: Resolve shows a dialog {dialogs} — stop, do not retry blindly")


def ensure_fonts(resolve: Any, font_dir: Path, family: str) -> bool:
    """Make Fusion's font list contain ``family`` without restarting Resolve.

    A font installed after Resolve started is invisible to Text+ ("Could not find font")
    until restart; ``FontManager.ScanDir`` did not help on 21.1.1, ``AddFont`` per file
    does (session only). Returns True if the family is available afterwards."""
    manager = resolve.Fusion().FontManager
    if family in (manager.GetFontList() or {}):
        return True
    for path in sorted(Path(font_dir).expanduser().glob("*.[ot]tf")):
        manager.AddFont(str(path))
    return family in (manager.GetFontList() or {})


@contextmanager
def scratch_timeline(project: Any, name: str) -> Iterator[Any]:
    """Create an empty timeline for a test, make it current, and always remove it again
    (restoring the previously current timeline). Refuses names with a colon."""
    if ":" in name:
        raise ValueError("Resolve rejects ':' in timeline names")
    media_pool = project.GetMediaPool()
    previous = project.GetCurrentTimeline()
    stale = find_timeline(project, name)
    if stale is not None:
        media_pool.DeleteTimelines([stale])
    timeline = media_pool.CreateEmptyTimeline(name)
    if timeline is None:
        raise ResolveNotReachable(f"CreateEmptyTimeline({name!r}) failed")
    project.SetCurrentTimeline(timeline)
    try:
        yield timeline
    finally:
        media_pool.DeleteTimelines([timeline])
        if previous is not None:
            project.SetCurrentTimeline(previous)


CARRIER_FRAMES = 12000  # 200 s at 60 fps — longer than the spot


def carrier_clip(project: Any, path: Path) -> Any:
    """Media-pool item of a long, tiny black ProRes movie used as a timeline carrier for
    Fusion comps of arbitrary length (created with ffmpeg if missing). Video clips honour
    startFrame/endFrame in AppendToTimeline; stills do not (and raise a modal dialog)."""
    import subprocess

    path = Path(path).resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                        "-i", f"color=c=black:s=256x144:r=60:d={CARRIER_FRAMES / 60}",
                        "-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le", str(path)],
                       check=True, timeout=300)
    media_pool = project.GetMediaPool()
    for clip in media_pool.GetRootFolder().GetClipList() or []:
        if clip.GetClipProperty("File Path") == str(path):
            return clip
    items = media_pool.ImportMedia([str(path)])
    _guard("ImportMedia")
    if not items:
        raise ResolveNotReachable(f"ImportMedia failed for {path}")
    return items[0]


MEDIA_STORAGE_RENDER_DIR = Path.home() / "Videos" / "zz-nomiss-comp-test"


def render_comp_frames(project: Any, carrier: Any, comp_path: Path, frames: list[int], out_dir: Path,
                       prefix: str, width: int = 1920, height: int = 1080) -> list[Path]:
    """Control render of a ``.comp`` through the Deliver pipeline (what the final master uses).

    Builds a scratch timeline at ``width``×``height``, puts ``carrier`` (see
    ``carrier_clip``) on it for as many frames as needed, imports the comp onto that clip
    and renders each requested frame as a 1-frame PNG job *with alpha*. Jobs write into
    the Media Storage (Resolve 21 silently refuses other targets, see diagnose_render.py);
    the files are then moved to ``out_dir`` as ``<prefix>_<frame:04d>.png``. Jobs,
    clip and timeline are removed afterwards.

    Not used: a Saver + ``comp.Render()`` — it works, but Resolve 21.1.1 then opens a
    modal box that blocks the scripting API until someone closes it.
    """
    import shutil
    import time

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    length = max(frames) + 2
    if length > CARRIER_FRAMES:
        raise ValueError(f"frame {max(frames)} is beyond the carrier ({CARRIER_FRAMES} frames)")
    stage = MEDIA_STORAGE_RENDER_DIR / prefix
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    _guard("start")
    written: list[Path] = []
    jobs: list[str] = []
    with scratch_timeline(project, f"zz comp-test {prefix}") as timeline:
        timeline.SetSetting("useCustomSettings", "1")
        timeline.SetSetting("timelineResolutionWidth", str(width))
        timeline.SetSetting("timelineResolutionHeight", str(height))
        appended = project.GetMediaPool().AppendToTimeline(
            [{"mediaPoolItem": carrier, "startFrame": 0, "endFrame": length, "trackIndex": 1, "mediaType": 1}])
        _guard("AppendToTimeline")
        if not appended:
            raise ResolveNotReachable("AppendToTimeline failed for the carrier clip")
        item = appended[0]
        comp = item.ImportFusionComp(str(Path(comp_path).resolve()))
        _guard("ImportFusionComp")
        if comp is None:
            raise ResolveNotReachable(f"ImportFusionComp failed for {comp_path}")
        item.LoadFusionCompByName(item.GetFusionCompNameList()[-1])
        start = int(item.GetStart())
        try:
            project.SetCurrentRenderFormatAndCodec("png", "RGB8")
            for frame in sorted(set(frames)):
                ok = project.SetRenderSettings({
                    "SelectAllFrames": False, "MarkIn": start + frame, "MarkOut": start + frame,
                    "TargetDir": str(stage), "CustomName": f"{prefix}_{frame:04d}_",
                    "ExportVideo": True, "ExportAudio": False, "ExportAlpha": True,
                    "FormatWidth": width, "FormatHeight": height,
                })
                job = project.AddRenderJob() if ok else ""
                if not job:
                    raise ResolveNotReachable(f"AddRenderJob refused frame {frame}")
                jobs.append(job)
            if not project.StartRendering(jobs, False):
                raise ResolveNotReachable("StartRendering returned False")
            deadline = time.time() + 60 + 20 * len(jobs)
            while project.IsRenderingInProgress() and time.time() < deadline:
                time.sleep(0.5)
            failed = {j: project.GetRenderJobStatus(j) for j in jobs}
            failed = {j: st for j, st in failed.items() if st.get("JobStatus") != "Complete"}
            if failed:
                raise ResolveNotReachable(f"render jobs not complete: {failed}")
            for frame in sorted(set(frames)):
                hits = sorted(stage.glob(f"{prefix}_{frame:04d}_*.png"))
                if hits:
                    dest = out_dir / f"{prefix}_{frame:04d}.png"
                    shutil.move(str(hits[0]), dest)
                    written.append(dest)
        finally:
            for job in jobs:
                project.DeleteRenderJob(job)
            timeline.DeleteClips([item], False)
    shutil.rmtree(stage, ignore_errors=True)
    _guard("cleanup")
    return written


@contextmanager
def keep_page(resolve: Any) -> Iterator[None]:
    """Restore Anthony's current page afterwards (StartRendering switches to Deliver)."""
    page = resolve.GetCurrentPage()
    try:
        yield
    finally:
        if page and resolve.GetCurrentPage() != page:
            resolve.OpenPage(page)
