#!/usr/bin/env bash
# Records the studio window (with its title bar) while the lantern of
# examples/lantern-turntable.ktgraph.json turns, and cuts exactly one turn into
# seamlessly looping animations:
#   OUT.mp4         960 px video (e.g. for the klartraum.ai landing page)
#   OUT-poster.jpg  its first frame
#   OUT.gif         600 px GIF for the documentation
# macOS only; run from the repository root after
# building. The terminal needs the Screen Recording permission, the main
# display must fit the 1600x965 window, and the window must stay uncovered for
# the few seconds of the recording (the mouse cursor is moved out of the way).
#
#   tools/record_studio_animation.sh [OUT]      (default: docs/_static/studio)
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${1:-docs/_static/studio}"
STUDIO="${STUDIO:-build/klartraum_studio}"
WORK="$(mktemp -d)"
trap 'kill "${STUDIO_PID:-}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

# The graph's Time node runs at 120 degrees per second: one turn takes 3 s,
# i.e. 90 frames of a 30 fps recording. The video keeps all of them, the GIF
# every second one.
RECORD_FPS=30
TURN_FRAMES=90
VIDEO_WIDTH=960
GIF_FPS=15
# Width and 64 colours keep the GIF below 500 KB.
GIF_WIDTH=600

cat > "$WORK/window.swift" <<'EOF'
import AppKit
import CoreGraphics
// Prints "x y width height scale" of the studio window, in screen pixels.
let scale = NSScreen.main?.backingScaleFactor ?? 1.0
let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] ?? []
for w in list {
    let name = w[kCGWindowName as String] as? String ?? ""
    guard name.hasPrefix("Klartraum Studio"), let b = w[kCGWindowBounds as String] as? [String: Double] else { continue }
    print(Int(b["X"]! * scale), Int(b["Y"]! * scale), Int(b["Width"]! * scale), Int(b["Height"]! * scale))
    exit(0)
}
exit(1)
EOF

cat > "$WORK/cursor.swift" <<'EOF'
import AppKit
import CoreGraphics
// Moves the mouse cursor to the bottom-right corner of the main display.
let frame = NSScreen.main!.frame
CGWarpMouseCursorPosition(CGPoint(x: frame.width - 1, y: frame.height - 1))
EOF

"$STUDIO" --window-position 160 40 examples/lantern-turntable.ktgraph.json > "$WORK/studio.log" 2>&1 &
STUDIO_PID=$!

BOUNDS=""
for _ in $(seq 1 60); do
    if BOUNDS="$(swift "$WORK/window.swift" 2>/dev/null)"; then
        break
    fi
    sleep 1
done
if [ -z "$BOUNDS" ]; then
    echo "the studio window did not appear" >&2
    exit 1
fi
# Let the graph compile and the window settle.
sleep 3
read -r X Y W H < <(swift "$WORK/window.swift")
swift "$WORK/cursor.swift"

# Listing devices always ends with an error status, so it is ignored.
SCREEN=$( (ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 || true) |
         sed -n 's/.*\[\([0-9]*\)\] Capture screen 0.*/\1/p')
ffmpeg -loglevel error -y -f avfoundation -capture_cursor 0 -framerate "$RECORD_FPS" \
    -pixel_format uyvy422 -i "$SCREEN:none" -t 5 \
    -vf "crop=$W:$H:$X:$Y" -c:v libx264 -preset ultrafast -crf 10 -pix_fmt yuv420p "$WORK/window.mov"

# Pick the start frame whose view matches the view one turn later best, so
# the cut loops seamlessly despite small timing jitter in the recording.
ffmpeg -loglevel error -y -i "$WORK/window.mov" \
    -vf "scale=160:96,format=gray" -f rawvideo "$WORK/frames.gray"
START=$(python3 - "$WORK/frames.gray" "$TURN_FRAMES" <<'EOF'
import sys
data = open(sys.argv[1], "rb").read()
turn = int(sys.argv[2])
size = 160 * 96
frames = [data[i:i + size] for i in range(0, len(data) - size + 1, size)]
def diff(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)
steps = sorted(diff(frames[i], frames[i + 1]) for i in range(len(frames) - 1))
score, start = min((diff(frames[s], frames[s + turn]), s) for s in range(len(frames) - turn))
# A seamless cut differs far less than two consecutive frames do.
if score > steps[len(steps) // 2] / 4:
    sys.exit(f"no seamless turn found (best difference {score:.2f})")
print(start)
EOF
)

LAST=$((START + TURN_FRAMES - 1))
ffmpeg -loglevel error -y -i "$WORK/window.mov" \
    -vf "select='between(n\,$START\,$LAST)',setpts=N/$RECORD_FPS/TB,scale=$VIDEO_WIDTH:-2:flags=lanczos" \
    -r "$RECORD_FPS" -c:v libx264 -preset veryslow -crf 23 -pix_fmt yuv420p -movflags +faststart -an "$OUT.mp4"

ffmpeg -loglevel error -y -i "$WORK/window.mov" \
    -vf "select='eq(n\,$START)',scale=$VIDEO_WIDTH:-2:flags=lanczos" -frames:v 1 -q:v 3 "$OUT-poster.jpg"

ffmpeg -loglevel error -y -i "$WORK/window.mov" \
    -vf "select='between(n\,$START\,$LAST)*not(mod(n-$START\,2))',setpts=N/$GIF_FPS/TB,scale=$GIF_WIDTH:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
    -r "$GIF_FPS" -loop 0 "$OUT.gif"

for f in "$OUT.mp4" "$OUT-poster.jpg" "$OUT.gif"; do
    echo "wrote $f ($(du -h "$f" | cut -f1))"
done
echo "one turn from recorded frame $START"
