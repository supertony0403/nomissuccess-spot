#!/usr/bin/env bash
# Contact sheet from the final MP4: one frame every 2 s with its timestamp, tiled.
#   scripts/kontaktbogen.sh            # both formats
#   scripts/kontaktbogen.sh 9x16       # one format
# Writes out/kontakt-16x9.jpg / out/kontakt-9x16.jpg.
set -euo pipefail
cd "$(dirname "$0")/.."

bogen() {
  local fmt=$1 tile scale
  local mp4="out/nomissuccess-nachtschicht-${fmt}.mp4"
  [[ -f $mp4 ]] || { echo "fehlt: $mp4" >&2; return 1; }
  case $fmt in
    16x9) scale=384:216; tile=10x8 ;;   # 75 frames for 148.7 s -> 80 cells
    9x16) scale=216:384; tile=15x5 ;;
    *) echo "unbekanntes Format: $fmt" >&2; return 1 ;;
  esac
  ffmpeg -v error -y -i "$mp4" -an -vf \
    "fps=1/2:round=down,scale=${scale}:flags=lanczos,drawtext=font=JetBrains Mono:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=3:x=6:y=h-th-8:text='%{pts\:hms}',tile=${tile}:padding=4:margin=4:color=0x202020" \
    -frames:v 1 -q:v 3 "out/kontakt-${fmt}.jpg"
  echo "out/kontakt-${fmt}.jpg"
}

if [[ $# -eq 0 ]]; then set -- 16x9 9x16; fi
for fmt in "$@"; do bogen "$fmt"; done
