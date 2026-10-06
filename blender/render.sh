#!/usr/bin/env bash
# Render nomissuccess Blender shots headless and wrap them as ProRes 422 HQ.
#
#   blender/render.sh                 # all shots, final quality
#   blender/render.sh s8_morgen       # one shot
#   blender/render.sh s1_nacht --res 480 --samples 16   # quick preview
#
# Writes renders/<shot>/####.png (1920x1920 RGB 8-bit, 60 fps), renders/<shot>.mov
# (ProRes 422 HQ) and renders/<shot>.json ({shot, szene, start_s, frames, handle_f}).
# Times come from timeline.json at render time; SAMPLES overrides the EEVEE samples.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALL=(s1_nacht s8_morgen)
SAMPLES="${SAMPLES:-96}"

sel="${1:-all}"
[[ $# -gt 0 ]] && shift
if [[ "$sel" == "all" ]]; then shots=("${ALL[@]}"); else shots=("$sel"); fi

command -v blender >/dev/null || { echo "blender fehlt" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg fehlt" >&2; exit 1; }
mkdir -p "$REPO/work/blender" "$REPO/renders"

for shot in "${shots[@]}"; do
    script="$REPO/blender/shots/$shot.py"
    [[ -f "$script" ]] || { echo "unbekannter Shot: $shot" >&2; exit 1; }
    out="$REPO/renders/$shot"
    rm -rf "$out"
    mkdir -p "$out"
    log="$REPO/work/blender/render_$shot.log"
    echo "== $shot -> $out (Samples $SAMPLES, Log $log)"
    start=$(date +%s)
    blender -b --factory-startup -P "$script" -- --samples "$SAMPLES" "$@" 2>&1 | tee "$log" \
        | grep --line-buffered -E '^\[(s1|s8|nomiss_render)\]|Error|Traceback' || true
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "Blender-Fehler, siehe $log" >&2; exit 1; }
    n=$(find "$out" -maxdepth 1 -name '*.png' | wc -l)
    echo "   $n Bilder in $(( $(date +%s) - start )) s"
    ffmpeg -y -loglevel error -framerate 60 -i "$out/%04d.png" \
        -c:v prores_ks -profile:v 3 -vendor apl0 -pix_fmt yuv422p10le -r 60 \
        -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
        "$REPO/renders/$shot.mov"
    echo "   -> renders/$shot.mov"
    if [[ -f "$REPO/renders/$shot.json" ]]; then
        expected=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['frames'])" "$REPO/renders/$shot.json")
        [[ "$n" -eq "$expected" ]] || { echo "Bildanzahl $n != $expected" >&2; exit 1; }
    fi
done
