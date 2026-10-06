"""Diagnose why ``Project.AddRenderJob()`` returns ``''`` (DaVinci Resolve Studio 21.1.1, Linux).

Finding (05.10.2026, reproduced and verified):

    Resolve 21 only accepts render targets inside its *Media Storage* locations
    (Preferences → System → Media Storage; config.dat ``Site.1.FS.N.Root``). Here that is
    only ``/home/tony/Videos``. For any other TargetDir, AddRenderJob shows a message
    box „Render Path Inaccessible — Please select a render path from within the media
    storage.“ for about a second (window title „Nachricht“, nothing in the log) and
    returns ''. Format, codec, page, saved state, external vs. internal scripting and
    free disk space are irrelevant (all tested). TargetDir ``~/Videos/<anything>`` works
    at once; a symlink inside ~/Videos that points elsewhere is accepted as well (the
    check is on the path string), so renders can land in the repo without a UI change.

Other traps found on the way (keep scripts away from them):

* ``Composition.Render()`` with a Saver works, but afterwards Resolve opens a modal box
  that blocks the whole scripting API (``GetCurrentPage()`` → None) until it is closed.
* ``MediaPool.AppendToTimeline`` of a *still* with startFrame/endFrame ignores the range
  and leaves a modal box open; use a video clip as carrier instead.
* A render must not be interrupted by system suspend while a script deletes the job's
  timeline: the queue then stays "rendering" forever (StopRendering has no effect,
  jobs cannot be deleted) until someone stops it in the Deliver page.

Usage::

    python resolve/diagnose_render.py              # read-only report
    python resolve/diagnose_render.py --probe      # + AddRenderJob inside vs. outside media storage
    python resolve/diagnose_render.py --aufraeumen # remove this repo's test leftovers
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolve import resolve_api as ra  # noqa: E402

CONFIG = Path.home() / ".local/share/DaVinciResolve/configs/config.dat"
REPO = Path(__file__).resolve().parents[1]
TEST_PREFIX = "zz comp-test"


def media_storage_roots(config: Path = CONFIG) -> list[Path]:
    """Media Storage locations from Resolve's config.dat (read-only)."""
    if not config.exists():
        return []
    roots = []
    for line in config.read_text(errors="replace").splitlines():
        m = re.match(r"\s*Site\.\d+\.FS\.\d+\.Root\s*=\s*(.+?)\s*$", line)
        if m:
            roots.append(Path(m.group(1)))
    return roots


def target_allowed(target: Path, roots: list[Path]) -> bool:
    """Same rule as Resolve: the TargetDir *string* must lie under a Media Storage root."""
    t = Path(str(target)).expanduser().absolute()
    return any(t == r or r in t.parents for r in roots)


def report(resolve, project) -> dict:
    jobs = project.GetRenderJobList() or []
    return {
        "page": resolve.GetCurrentPage(),
        "dialogs": ra.open_dialogs(),
        "rendering": project.IsRenderingInProgress(),
        "timeline": project.GetCurrentTimeline().GetName() if project.GetCurrentTimeline() else None,
        "format": project.GetCurrentRenderFormatAndCodec(),
        "jobs": [(j["JobId"][:8], j.get("TimelineName"), project.GetRenderJobStatus(j["JobId"]).get("JobStatus"))
                 for j in jobs],
        "media_storage": [str(r) for r in media_storage_roots()],
    }


def probe(project) -> dict:
    """AddRenderJob with a target outside vs. inside Media Storage on a scratch timeline.
    Only creates jobs (never renders) and deletes them again."""
    if project.IsRenderingInProgress():
        return {"skipped": "a render is in progress — not touching the queue"}
    roots = media_storage_roots()
    inside = (roots[0] if roots else Path.home() / "Videos") / "zz-nomiss-diagnose"
    outside = REPO / "work" / "diagnose"
    inside.mkdir(parents=True, exist_ok=True)
    outside.mkdir(parents=True, exist_ok=True)
    result = {}
    with ra.scratch_timeline(project, f"{TEST_PREFIX} diagnose") as timeline:
        if timeline.InsertFusionCompositionIntoTimeline() is None:
            return {"error": "could not insert a Fusion composition"}
        project.SetCurrentRenderFormatAndCodec("png", "RGB8")
        for label, target in (("ausserhalb", outside), ("innerhalb", inside)):
            project.SetRenderSettings({"SelectAllFrames": True, "TargetDir": str(target), "CustomName": "diag"})
            job = project.AddRenderJob()
            result[label] = {"target": str(target), "allowed_by_rule": target_allowed(target, roots),
                             "job_id": job or ""}
            if job:
                project.DeleteRenderJob(job)
    inside.rmdir() if inside.exists() and not any(inside.iterdir()) else None
    return result


def aufraeumen(resolve, project) -> dict:
    """Remove this repo's test leftovers: render jobs and timelines named 'zz comp-test…',
    media-pool clips whose file lies in this repo, ~/Videos/zz-nomiss-comp-test."""
    import shutil

    if project.IsRenderingInProgress():
        return {"error": "render still in progress — stop it in the Deliver page first"}
    done = {"jobs": 0, "timelines": [], "clips": []}
    for job in project.GetRenderJobList() or []:
        if str(job.get("TimelineName", "")).startswith(TEST_PREFIX) and project.DeleteRenderJob(job["JobId"]):
            done["jobs"] += 1
    pool = project.GetMediaPool()
    names = [project.GetTimelineByIndex(i).GetName() for i in range(1, project.GetTimelineCount() + 1)]
    for name in names:
        if name.startswith(TEST_PREFIX):
            tl = ra.find_timeline(project, name)
            if tl is not None and pool.DeleteTimelines([tl]):
                done["timelines"].append(name)
    own = [c for c in pool.GetRootFolder().GetClipList() or []
           if str(c.GetClipProperty("File Path") or "").startswith(str(REPO))]
    if own and pool.DeleteClips(own):
        done["clips"] = [c.GetName() for c in own]
    shutil.rmtree(Path.home() / "Videos" / "zz-nomiss-comp-test", ignore_errors=True)
    done["saved"] = bool(resolve.GetProjectManager().SaveProject())
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true", help="AddRenderJob inside vs. outside Media Storage")
    ap.add_argument("--aufraeumen", action="store_true", help="remove this repo's Resolve test leftovers")
    args = ap.parse_args(argv)
    resolve = ra.connect()
    project = ra.current_project(resolve)
    for key, value in report(resolve, project).items():
        print(f"{key:14} {value}")
    if args.probe:
        print("probe         ", probe(project))
    if args.aufraeumen:
        print("aufraeumen    ", aufraeumen(resolve, project))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
