"""Rasterise the AudioFlow mark into a Windows .ico, with no image libraries.

The SVG in ``assets/audioflow.svg`` is the design source. Windows needs an icon
file, and pulling in Pillow or cairosvg for five shapes is not worth it, so the
same geometry is drawn here with signed distance fields, supersampled, then
written as PNG (preview) and ICO (tray).

Run:
  python tools/make_icon.py            # writes assets/audioflow.ico
"""

from __future__ import annotations

import math
import os
import struct
import sys
import zlib

# Geometry in a 128x128 design space; matches assets/audioflow.svg.
COLOR = (0x2D, 0x7F, 0xF9)
STROKE = 3.5           # half of the 7-unit stroke
CAPSULE = (64.0, 46.0, 18.0, 32.0, 18.0)   # cx, cy, half_width, half_height, radius
CRADLE = (64.0, 56.0, 32.0)
STAND = ((64.0, 88.0), (64.0, 104.0))
BASE = ((48.0, 110.0), (80.0, 110.0))
BARS = (  # x, y, w, h
    (54.5, 37.0, 5.0, 18.0),
    (61.5, 31.0, 5.0, 30.0),
    (68.5, 36.0, 5.0, 20.0),
)


def _segment_distance(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    length_sq = vx * vx + vy * vy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    dx, dy = wx - t * vx, wy - t * vy
    return math.hypot(dx, dy)


def _stadium_distance(px, py, cx, cy, hw, hh, r):
    qx = abs(px - cx) - (hw - r)
    qy = abs(py - cy) - (hh - r)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside - r


def _round_rect_distance(px, py, x, y, w, h, r):
    return _stadium_distance(px, py, x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0, r)


def inside(px: float, py: float) -> bool:
    # Microphone body, drawn as an outline.
    if abs(_stadium_distance(px, py, *CAPSULE)) <= STROKE:
        return True

    # Waveform bars.
    for x, y, w, h in BARS:
        if _round_rect_distance(px, py, x, y, w, h, 2.5) <= 0.0:
            return True

    # Cradle: the lower half of a circle, with round ends.
    cx, cy, radius = CRADLE
    if py >= cy and abs(math.hypot(px - cx, py - cy) - radius) <= STROKE:
        return True
    for ex in (cx - radius, cx + radius):
        if math.hypot(px - ex, py - cy) <= STROKE:
            return True

    # Stand and base, with round caps.
    for (ax, ay), (bx, by) in (STAND, BASE):
        if _segment_distance(px, py, ax, ay, bx, by) <= STROKE:
            return True
    return False


def rasterise(size: int, supersample: int = 4) -> bytes:
    """Return RGBA bytes for a ``size`` x ``size`` icon."""
    out = bytearray()
    step = 128.0 / size
    samples = supersample * supersample
    for row in range(size):
        for col in range(size):
            hits = 0
            for sy in range(supersample):
                for sx in range(supersample):
                    px = (col + (sx + 0.5) / supersample) * step
                    py = (row + (sy + 0.5) / supersample) * step
                    if inside(px, py):
                        hits += 1
            alpha = int(round(255 * hits / samples))
            out += bytes((COLOR[0], COLOR[1], COLOR[2], alpha))
    return bytes(out)


def write_png(path: str, size: int, rgba: bytes) -> None:
    raw = bytearray()
    stride = size * 4
    for row in range(size):
        raw.append(0)                                  # filter: none
        raw += rgba[row * stride:(row + 1) * stride]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as handle:
        handle.write(png)


def write_ico(path: str, images: list[tuple[int, bytes]]) -> None:
    header = struct.pack("<HHH", 0, 1, len(images))
    entries = b""
    payload = b""
    offset = 6 + 16 * len(images)

    for size, rgba in images:
        stride = size * 4
        pixels = bytearray()
        for row in range(size - 1, -1, -1):            # bottom-up
            for col in range(size):
                index = row * stride + col * 4
                r, g, b, a = rgba[index:index + 4]
                pixels += bytes((b, g, r, a))          # BGRA
        mask_row = ((size + 31) // 32) * 4
        mask = b"\x00" * (mask_row * size)             # all opaque, alpha carries it
        bitmap = (struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0,
                              len(pixels) + len(mask), 0, 0, 0, 0)
                  + bytes(pixels) + mask)

        entries += struct.pack("<BBBBHHII", size if size < 256 else 0,
                               size if size < 256 else 0, 0, 0, 1, 32,
                               len(bitmap), offset)
        offset += len(bitmap)
        payload += bitmap

    with open(path, "wb") as handle:
        handle.write(header + entries + payload)


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets = os.path.join(root, "assets")
    os.makedirs(assets, exist_ok=True)

    sizes = (16, 32, 48)
    images = [(size, rasterise(size)) for size in sizes]
    ico = os.path.join(assets, "audioflow.ico")
    write_ico(ico, images)

    preview = os.path.join(root, "build", ".tmp", "audioflow-preview.png")
    os.makedirs(os.path.dirname(preview), exist_ok=True)
    write_png(preview, 256, rasterise(256, supersample=3))

    print("ico: %s (%d bytes, sizes %s)" % (ico, os.path.getsize(ico),
                                            ", ".join(map(str, sizes))))
    print("preview: %s" % preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
