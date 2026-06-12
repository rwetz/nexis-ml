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
from importlib import resources
from importlib.resources.abc import Traversable

TEMPLATES = {"tabular"}


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


_EXTRAS = {"tabular": _tabular_example_data}
