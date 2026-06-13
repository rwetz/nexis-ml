# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""Project scaffolding.

Each template is a directory of files copied verbatim into the
destination, plus optional generated extras (example data) so that
`nexis-ml train` works immediately after `nexis-ml new`.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import struct
import zlib
from importlib import resources
from importlib.resources.abc import Traversable

TEMPLATES = {"tabular", "textgen", "image"}


def scaffold(template: str, dest: str, force: bool = False) -> str:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template: {template}")
    dest = os.path.abspath(dest)
    if os.path.isdir(dest) and os.listdir(dest) and not force:
        raise FileExistsError(
            f"{dest} exists and is not empty (use --force to scaffold anyway)"
        )
    src = resources.files(__package__) / template
    _copy_tree(src, dest)
    extra = _EXTRAS.get(template)
    if extra is not None:
        extra(dest)
    return dest


def _copy_tree(src: Traversable, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    for entry in src.iterdir():
        if entry.name == "__pycache__":
            continue
        target = os.path.join(dest, entry.name)
        if entry.is_dir():
            _copy_tree(entry, target)
        else:
            with resources.as_file(entry) as p:
                shutil.copy(p, target)


def _tabular_example_data(dest: str) -> None:
    """Two interleaved half-moons (plus a pure-noise column the model
    should learn to ignore) — deterministic, stdlib-only, and small
    enough that training takes seconds on CPU."""
    rng = random.Random(42)
    rows: list[tuple[float, float, float, int]] = []
    for label in (0, 1):
        for _ in range(120):
            t = rng.uniform(0.0, math.pi)
            if label == 0:
                x1, x2 = math.cos(t), math.sin(t)
            else:
                x1, x2 = 1.0 - math.cos(t), 0.5 - math.sin(t)
            x1 += rng.gauss(0.0, 0.12)
            x2 += rng.gauss(0.0, 0.12)
            rows.append((x1, x2, rng.gauss(0.0, 1.0), label))
    rng.shuffle(rows)
    data_dir = os.path.join(dest, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(
        os.path.join(data_dir, "example.csv"), "w", encoding="utf-8", newline=""
    ) as f:
        f.write("x1,x2,noise,label\n")
        for x1, x2, nz, label in rows:
            f.write(f"{x1:.5f},{x2:.5f},{nz:.5f},{label}\n")


def _png_gray(width: int, height: int, pixels: bytes) -> bytes:
    """Encode an 8-bit grayscale PNG with the stdlib only (zlib + struct).

    Lets `nexis-ml new image` generate example images without pulling in
    Pillow — scaffolding stays dependency-free; only training (train.py)
    needs the image library.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (none) per scanline
        raw.extend(pixels[y * width : (y + 1) * width])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _image_example_data(dest: str) -> None:
    """Four visually distinct pattern classes (horizontal / vertical /
    diagonal stripes + checkerboard), as folders of small grayscale PNGs.
    Deterministic, stdlib-only, and easy for a tiny CNN to separate so the
    sample grid looks right within a few epochs."""
    rng = random.Random(7)
    size = 24
    classes = ["horizontal", "vertical", "diagonal", "checker"]
    data_dir = os.path.join(dest, "data")
    for cls in classes:
        cdir = os.path.join(data_dir, cls)
        os.makedirs(cdir, exist_ok=True)
        for i in range(36):
            period = rng.choice([3, 4, 5])
            phase = rng.randint(0, period - 1)
            cell = rng.choice([2, 3, 4])
            on_w = max(1, period // 2)
            px = bytearray(size * size)
            for y in range(size):
                for x in range(size):
                    if cls == "horizontal":
                        on = (y + phase) % period < on_w
                    elif cls == "vertical":
                        on = (x + phase) % period < on_w
                    elif cls == "diagonal":
                        on = (x + y + phase) % period < on_w
                    else:  # checker
                        on = ((x // cell) + (y // cell)) % 2 == 0
                    base = 220 if on else 30
                    val = int(base + rng.gauss(0.0, 18.0))
                    px[y * size + x] = max(0, min(255, val))
            with open(os.path.join(cdir, f"img_{i:03d}.png"), "wb") as f:
                f.write(_png_gray(size, size, bytes(px)))


_EXTRAS = {"tabular": _tabular_example_data, "image": _image_example_data}
