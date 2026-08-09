#!/usr/bin/env bash
# Rasterise the generated SVGs to the bitmaps crawlers and older browsers
# actually fetch. Run scripts/build_og_image.py first.
#
# Firefox is the rasteriser because it is the only one on this machine:
# there is no cairosvg, no rsvg-convert (librsvg2-bin is not installed,
# only the runtime library) and no Inkscape. Any of those would do the
# same job; none of them is required by this repository.
#
# Headless Firefox needs a profile directory that already exists, or it
# dies with SIGKILL and no useful message.
set -euo pipefail

cd "$(dirname "$0")/.."
docs="$PWD/docs"
profile="$(mktemp -d)"
trap 'rm -rf "$profile"' EXIT

shot() {
  local out="$1" size="$2" src="$3"
  firefox --headless --profile "$profile" --no-remote \
      --window-size="$size" --screenshot "$out" "file://$src" >/dev/null 2>&1
  echo "wrote ${out#"$PWD/"} ($size)"
}

shot "$docs/og-image.png"       1200,630 "$docs/og-image.svg"
shot "$docs/favicon.png"        32,32    "$docs/favicon.svg"
shot "$docs/apple-touch-icon.png" 180,180 "$docs/favicon.svg"

# favicon.ico, so a crawler that requests /favicon.ico by convention and
# does not read the markup still gets the mark. An ICO may carry a PNG
# payload verbatim, so this wraps the 32x32 rather than re-encoding it.
python3 - "$docs/favicon.png" "$docs/favicon.ico" <<'PY'
import struct
import sys

png = open(sys.argv[1], "rb").read()
if png[:8] != b"\x89PNG\r\n\x1a\n":
    raise SystemExit("not a PNG")

width, height = struct.unpack(">II", png[16:24])
if not (0 < width <= 256 and 0 < height <= 256):
    raise SystemExit(f"{width}x{height} does not fit an ICO directory entry")

header = struct.pack("<HHH", 0, 1, 1)                       # reserved, type, count
entry = struct.pack(
    "<BBBBHHII",
    width % 256, height % 256,                              # 0 means 256
    0, 0,                                                   # palette, reserved
    1, 32,                                                  # planes, bits per pixel
    len(png), len(header) + 16,
)
open(sys.argv[2], "wb").write(header + entry + png)
print(f"wrote docs/favicon.ico ({width}x{height}, PNG payload)")
PY
