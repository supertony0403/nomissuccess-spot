"""Read ``timeline.json`` (the project's timing contract) for Blender shots.

Resolution order for the timeline file:

1. an explicit ``path`` argument,
2. the environment variable ``NOMISS_TIMELINE``,
3. ``<repo>/timeline.json`` (written by ``audio/plan.py``),
4. ``blender/fixtures/timeline_provisorisch.json`` (provisional timing).

Shot frames are 1-based.  Frame 1 of a shot is the scene start minus the
handle (0.5 s), the last frame is the scene end plus the handle.  Every shot
reads its times at render time, so a re-timed voice only needs a re-render.

Pure Python (no bpy) so it can be imported by tests as well.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "blender" / "fixtures" / "timeline_provisorisch.json"
HANDLE_S = 0.5

_cache: dict[tuple[str, float], dict] = {}


def resolve_path(path: str | os.PathLike | None = None) -> Path:
    """Return the timeline file that will be used."""
    if path:
        return Path(path)
    env = os.environ.get("NOMISS_TIMELINE")
    if env:
        return Path(env)
    real = REPO / "timeline.json"
    if real.exists():
        return real
    return FIXTURE


def load(path: str | os.PathLike | None = None) -> dict:
    """Load and validate the timeline; the dict carries ``_quelle``."""
    p = resolve_path(path)
    key = (str(p), p.stat().st_mtime)
    if key not in _cache:
        tl = json.loads(p.read_text(encoding="utf-8"))
        for field in ("fps", "szenen", "events"):
            if field not in tl:
                raise ValueError(f"{p}: Feld '{field}' fehlt")
        tl["_quelle"] = str(p)
        tl["_provisorisch"] = p.resolve() == FIXTURE.resolve()
        if tl["_provisorisch"]:
            print(f"[nomiss_timeline] WARNUNG: provisorische Zeiten aus {p}", file=sys.stderr)
        _cache[key] = tl
    return _cache[key]


def fps(tl: dict | None = None) -> int:
    return int((tl or load())["fps"])


def _frame(t_s: float, rate: int) -> int:
    return int(round(t_s * rate))


def szene(scene_id: str, tl: dict | None = None) -> dict:
    """Scene record with ``start_f``/``ende_f`` filled in if missing."""
    tl = tl or load()
    for sc in tl["szenen"]:
        if sc["id"] == scene_id:
            rate = fps(tl)
            out = dict(sc)
            out.setdefault("start_f", _frame(sc["start_s"], rate))
            out.setdefault("ende_f", _frame(sc["ende_s"], rate))
            return out
    raise KeyError(f"Szene '{scene_id}' nicht in {tl.get('_quelle')}")


def event(event_id: str, tl: dict | None = None) -> dict:
    tl = tl or load()
    for ev in tl["events"]:
        if ev["id"] == event_id:
            out = dict(ev)
            out.setdefault("f", _frame(ev["t_s"], fps(tl)))
            return out
    raise KeyError(f"Event '{event_id}' nicht in {tl.get('_quelle')}")


@dataclass(frozen=True)
class Shot:
    shot: str          # shot id == scene id, e.g. "s1_nacht"
    szene: str
    fps: int
    handle_f: int
    start_f_abs: int   # absolute timeline frame shown in shot frame 1
    frames: int        # number of frames incl. both handles
    start_s: float     # time of shot frame 1 (may be negative for s1)
    quelle: str
    provisorisch: bool

    def to_json(self) -> dict:
        return {
            "shot": self.shot,
            "szene": self.szene,
            "start_s": round(self.start_s, 4),
            "frames": self.frames,
            "handle_f": self.handle_f,
            "fps": self.fps,
            "timeline": self.quelle,
            "provisorisch": self.provisorisch,
        }


def shot(shot_id: str, tl: dict | None = None) -> Shot:
    tl = tl or load()
    rate = fps(tl)
    sc = szene(shot_id, tl)
    handle_f = int(round(HANDLE_S * rate))
    start_abs = sc["start_f"] - handle_f
    frames = (sc["ende_f"] - sc["start_f"]) + 2 * handle_f
    return Shot(
        shot=shot_id,
        szene=sc["id"],
        fps=rate,
        handle_f=handle_f,
        start_f_abs=start_abs,
        frames=frames,
        start_s=start_abs / rate,
        quelle=tl["_quelle"],
        provisorisch=bool(tl.get("_provisorisch")),
    )


def event_frame(event_id: str, shot_id: str, tl: dict | None = None) -> int:
    """1-based frame of ``event_id`` inside shot ``shot_id`` (incl. handle)."""
    tl = tl or load()
    return event(event_id, tl)["f"] - shot(shot_id, tl).start_f_abs + 1


def time_to_frame(t_s: float, shot_id: str, tl: dict | None = None) -> float:
    """Fractional 1-based shot frame for an absolute timeline time."""
    tl = tl or load()
    sh = shot(shot_id, tl)
    return (t_s - sh.start_s) * sh.fps + 1.0


def scene_frames(shot_id: str, tl: dict | None = None) -> tuple[int, int]:
    """First and last shot frame that belong to the scene itself."""
    sh = shot(shot_id, tl)
    return sh.handle_f + 1, sh.frames - sh.handle_f


def write_shot_json(sh: Shot, out_path: str | os.PathLike) -> None:
    Path(out_path).write_text(json.dumps(sh.to_json(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "HANDLE_S",
    "Shot",
    "event",
    "event_frame",
    "fps",
    "load",
    "resolve_path",
    "scene_frames",
    "shot",
    "szene",
    "time_to_frame",
    "write_shot_json",
    "asdict",
]
