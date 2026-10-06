"""Encode finished Blender PNG sequences to ProRes 422 HQ movies for Resolve.

For every shot ``renders/<shot>/####.png`` (image 1 = scene start - 0.5 s, 30 handle
frames before and after the scene) this writes

* ``renders/<shot>.mov``  - ProRes 422 HQ, 1440x1440, 60 fps, yuv422p10le, no audio
* ``renders/<shot>.json`` - ``{"shot", "szene", "start_s", "frames", "handle_f": 30, ...}``

``start_s`` is the spot time of image 1 (scene start - handle), so source frame
``handle_f`` (0-based) of the movie is the first frame of the scene.

A shot counts as finished when all images 1..N exist (N = scene length + 2 handles,
from ``timeline.json``), the last image decodes, and no Blender process is rendering
that shot right now. ``rc=0`` in the strand log is *not* a criterion (a ``--resume``
job can end at once with rc=0 while frames are still missing).

The sequences mix 1920x1920 (older frames) and 1440x1440 (newer frames); ffmpeg
re-initialises the scale filter on every size change, so the movie is uniform 1440x1440
with the same framing (checked on the s1 switch at image 367/368).

Idempotent: a shot whose ``.mov`` is newer than every PNG, has the right frame count
and size and a matching ``.json`` is skipped.

Usage::

    .venv/bin/python resolve/shots_kodieren.py --status
    .venv/bin/python resolve/shots_kodieren.py            # encode every finished shot once
    .venv/bin/python resolve/shots_kodieren.py --warten   # poll until all shots are encoded
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RENDERS = REPO / "renders"
TIMELINE = REPO / "timeline.json"
HANDLE_F = 30
SIZE = 1440
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"  # a complete PNG ends with this chunk

# RGB -> Y'CbCr with the BT.709 matrix and tagged as such (ffmpeg's untagged default is
# BT.601, which Resolve would read as Rec.709: a slight hue/saturation shift).
SCALE_FILTER = (f"scale={SIZE}:{SIZE}:flags=lanczos:out_color_matrix=bt709:out_range=tv,"
                "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv")
# (setparams: the PNGs carry an sRGB transfer tag that would otherwise win over -color_trc)
FFMPEG_VIDEO = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0",
                "-pix_fmt", "yuv422p10le", "-r", "60",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def load_timeline(path: Path = TIMELINE) -> dict:
    return json.loads(Path(path).read_text())


def soll_frames(timeline: dict) -> dict[str, int]:
    """Images per shot: scene length + head and tail handles."""
    return {s["id"]: int(s["ende_f"]) - int(s["start_f"]) + 2 * HANDLE_F for s in timeline["szenen"]}


@dataclass
class ShotStatus:
    shot: str
    soll: int
    vorhanden: int = 0
    fehlend: list[int] = field(default_factory=list)
    kaputt: list[int] = field(default_factory=list)
    letztes_lesbar: bool = False
    blender_laeuft: bool = False
    neueste_mtime: float = 0.0
    kodiert: bool = False

    @property
    def fertig(self) -> bool:
        return (self.vorhanden == self.soll and not self.fehlend and not self.kaputt
                and self.letztes_lesbar and not self.blender_laeuft)

    def zeile(self) -> str:
        if self.kodiert:
            zustand = "kodiert"
        elif self.fertig:
            zustand = "fertig, nicht kodiert"
        else:
            gruende = []
            if self.fehlend:
                gruende.append(f"{len(self.fehlend)} fehlen (ab {self.fehlend[0]})")
            if self.kaputt:
                gruende.append(f"{len(self.kaputt)} kaputt")
            if self.blender_laeuft:
                gruende.append("Blender rendert")
            if not self.letztes_lesbar and not self.fehlend:
                gruende.append("letztes Bild unlesbar")
            zustand = "offen: " + ", ".join(gruende)
        return f"{self.shot:14} {self.vorhanden:5}/{self.soll:<5} {zustand}"


def blender_shots_running(proc: Path = Path("/proc")) -> set[str]:
    """Shot ids that a running ``blender ... -P blender/shots/<shot>.py`` is rendering."""
    running: set[str] = set()
    for entry in proc.iterdir() if proc.exists() else []:
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if not args or b"blender" not in Path(args[0].decode(errors="replace")).name.encode():
            continue
        for arg in args:
            text = arg.decode(errors="replace")
            if text.endswith(".py") and "/shots/" in text:
                running.add(Path(text).stem)
    return running


def png_decodes(path: Path) -> bool:
    """True if ffmpeg decodes the image without error (a half-written PNG fails)."""
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                            capture_output=True, text=True, timeout=60)
    return result.returncode == 0 and not result.stderr.strip()


def status(shot: str, soll: int, renders: Path = RENDERS, running: set[str] | None = None) -> ShotStatus:
    st = ShotStatus(shot, soll)
    folder = renders / shot
    st.blender_laeuft = shot in (running if running is not None else blender_shots_running())
    for index in range(1, soll + 1):
        path = folder / f"{index:04d}.png"
        try:
            stat = path.stat()
        except FileNotFoundError:
            st.fehlend.append(index)
            continue
        st.vorhanden += 1
        st.neueste_mtime = max(st.neueste_mtime, stat.st_mtime)
        if stat.st_size < 64:
            st.kaputt.append(index)
            continue
        with path.open("rb") as handle:
            head = handle.read(24)
            handle.seek(-len(PNG_IEND), os.SEEK_END)
            tail = handle.read()
        # a truncated PNG (no IEND) would decode as a silent freeze frame; a smaller frame
        # (proxy/smoke render) must never be upscaled into the master
        width, height = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        if head[:8] != PNG_SIGNATURE or tail != PNG_IEND or width < SIZE or width != height:
            st.kaputt.append(index)
    last = folder / f"{soll:04d}.png"
    st.letztes_lesbar = last.exists() and png_decodes(last)
    st.kodiert = ist_aktuell(shot, st)
    return st


def ffprobe_video(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,nb_read_packets,codec_name,profile,r_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, timeout=120)
    streams = json.loads(out.stdout or "{}").get("streams") or [{}]
    return streams[0]


KODIERER = "shots_kodieren"  # bauen.py only accepts movies written here (not Blender's own encode)


def meta(shot: str, timeline: dict, frames: int, quelle_mtime: float = 0.0, mov: Path | None = None) -> dict:
    szene = next(s for s in timeline["szenen"] if s["id"] == shot)
    fps = int(timeline["fps"])
    stat = mov.stat() if mov is not None and mov.exists() else None
    return {
        "kodierer": KODIERER,
        "quelle_mtime": quelle_mtime,
        "mov_groesse": stat.st_size if stat else 0,
        "mov_mtime": stat.st_mtime if stat else 0.0,
        "shot": shot,
        "szene": szene["id"],
        "start_s": round(float(szene["start_s"]) - HANDLE_F / fps, 6),
        "frames": frames,
        "handle_f": HANDLE_F,
        "fps": fps,
        "breite": SIZE,
        "hoehe": SIZE,
        "szene_start_f": int(szene["start_f"]),
        "szene_ende_f": int(szene["ende_f"]),
    }


def ist_aktuell(shot: str, st: ShotStatus, renders: Path = RENDERS) -> bool:
    mov, info = renders / f"{shot}.mov", renders / f"{shot}.json"
    if not (mov.exists() and info.exists()) or not st.vorhanden:
        return False
    try:
        data = json.loads(info.read_text())
    except ValueError:
        return False
    if data.get("kodierer") != KODIERER or data.get("frames") != st.soll or data.get("breite") != SIZE:
        return False  # e.g. Blender's own encode (other size, BT.601) overwrote ours
    if data.get("quelle_mtime") != st.neueste_mtime:
        return False  # frames changed after the encode
    stat = mov.stat()
    if (data.get("mov_groesse"), data.get("mov_mtime")) == (stat.st_size, stat.st_mtime):
        return True  # the movie is the one we wrote and checked (no full read every poll)
    probe = ffprobe_video(mov)
    return (int(probe.get("nb_read_packets") or 0) == st.soll and probe.get("width") == SIZE
            and probe.get("height") == SIZE)


def kodieren(shot: str, timeline: dict, renders: Path = RENDERS, force: bool = False) -> str:
    """Encode one shot if it is finished. Returns 'kodiert', 'aktuell' or 'offen: …'."""
    soll = soll_frames(timeline)[shot]
    st = status(shot, soll, renders)
    if st.kodiert and not force:
        return "aktuell"
    if not st.fertig:
        return "offen: " + st.zeile().split("offen: ", 1)[-1]
    mov = renders / f"{shot}.mov"
    tmp = renders / f".{shot}.tmp.mov"
    # -xerror: any decode error aborts with rc != 0 instead of duplicating the previous frame
    cmd = ["nice", "-n", "10", "ffmpeg", "-v", "error", "-xerror", "-y", "-framerate", "60", "-start_number", "1",
           "-i", str(renders / shot / "%04d.png"), "-frames:v", str(soll),
           "-vf", SCALE_FILTER, *FFMPEG_VIDEO, "-an", str(tmp)]
    started = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=3 * 3600, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{shot}: ffmpeg failed: {exc.stderr.strip()[-400:]}") from exc
    probe = ffprobe_video(tmp)
    count = int(probe.get("nb_read_packets") or 0)
    if count != soll or probe.get("width") != SIZE or probe.get("height") != SIZE:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{shot}: encoded {count} frames {probe.get('width')}x{probe.get('height')}, "
                           f"expected {soll} at {SIZE}x{SIZE}")
    after = status(shot, soll, renders)
    if after.neueste_mtime != st.neueste_mtime or not after.fertig:
        tmp.unlink(missing_ok=True)  # frames were rewritten while ffmpeg read them
        return "offen: Quelle hat sich während des Kodierens geändert"
    os.replace(tmp, mov)
    info = renders / f"{shot}.json"
    tmp_info = renders / f".{shot}.tmp.json"
    tmp_info.write_text(json.dumps(meta(shot, timeline, soll, st.neueste_mtime, mov), indent=2) + "\n")
    os.replace(tmp_info, info)
    return f"kodiert in {time.time() - started:.0f} s"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true", help="only report the state of every shot")
    ap.add_argument("--warten", action="store_true", help="poll until every shot is encoded")
    ap.add_argument("--intervall", type=int, default=60, help="poll interval in seconds (--warten)")
    ap.add_argument("--shot", action="append", help="limit to these shots")
    ap.add_argument("--force", action="store_true", help="re-encode even if up to date")
    args = ap.parse_args(argv)

    timeline = load_timeline()
    shots = args.shot or list(soll_frames(timeline))
    if args.status:
        running = blender_shots_running()
        for shot in shots:
            print(status(shot, soll_frames(timeline)[shot], running=running).zeile())
        return 0
    while True:
        timeline = load_timeline()  # the contract can be regenerated while we wait
        offen = []
        for shot in shots:
            try:
                result = kodieren(shot, timeline, force=args.force)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                # loud, but the watcher keeps serving the other shots
                result = f"offen: FEHLER {exc}"
            print(f"{time.strftime('%H:%M:%S')} {shot:14} {result}", flush=True)
            if result.startswith("offen"):
                offen.append(shot)
        if not args.warten or not offen:
            return 0 if not offen else 2
        args.force = False
        time.sleep(args.intervall)


if __name__ == "__main__":
    sys.exit(main())
