"""On-disk run store: <project>/.nexis-ml/runs/<run-id>/

  config.json    — config snapshot, written at run start
  metrics.jsonl  — append-only protocol event log (the full run, replayable)
  summary.json   — final stats, written atomically at run end
  checkpoints/   — model weights
  artifacts/     — generated files (confusion matrices, image grids, ...)

Non-append writes go through tmp+rename so a crash mid-write can never
leave a half-written file (same rationale as Nexis's write_if_changed).
Nexis renders finished runs by reading these files directly — no engine
process required.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import Any, TextIO

RUNS_SUBDIR = os.path.join(".nexis-ml", "runs")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "run"


def atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp, path)


def runs_root(project_dir: str) -> str:
    return os.path.join(project_dir, RUNS_SUBDIR)


class RunDir:
    """Filesystem handle for a single run directory."""

    def __init__(self, path: str):
        self.path = path
        self.config_path = os.path.join(path, "config.json")
        self.metrics_path = os.path.join(path, "metrics.jsonl")
        self.summary_path = os.path.join(path, "summary.json")
        self.checkpoints_dir = os.path.join(path, "checkpoints")
        self.artifacts_dir = os.path.join(path, "artifacts")
        self._metrics_file: TextIO | None = None

    @property
    def run_id(self) -> str:
        return os.path.basename(self.path)

    def create(self) -> None:
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def write_config(self, config: dict[str, Any]) -> None:
        atomic_write_json(self.config_path, config)

    def append_event(self, event: dict[str, Any]) -> None:
        if self._metrics_file is None:
            self._metrics_file = open(self.metrics_path, "a", encoding="utf-8")
        line = json.dumps(event, separators=(",", ":"), default=str)
        self._metrics_file.write(line + "\n")
        self._metrics_file.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        atomic_write_json(self.summary_path, summary)

    def close(self) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None


def new_run_dir(project_dir: str, name: str) -> RunDir:
    """Allocate a unique run directory: YYYY-MM-DD-HHMM-<slug>[-N]."""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    base = f"{stamp}-{slugify(name)}"
    root = runs_root(project_dir)
    os.makedirs(root, exist_ok=True)
    candidate = os.path.join(root, base)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(root, f"{base}-{n}")
        n += 1
    run = RunDir(candidate)
    run.create()
    return run


def list_runs(project_dir: str) -> list[dict[str, Any]]:
    """Newest-first list of runs with whatever metadata is on disk.

    The timestamp prefix makes dir names sort lexicographically, so
    reverse name order is newest-first. A run with no summary.json is
    still in progress (or crashed before finishing) — status "unknown".
    """
    root = runs_root(project_dir)
    if not os.path.isdir(root):
        return []
    runs: list[dict[str, Any]] = []
    for entry in sorted(os.listdir(root), reverse=True):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        info: dict[str, Any] = {"run": entry, "dir": path, "status": "unknown"}
        config = _read_json(os.path.join(path, "config.json"))
        if config is not None:
            info["config"] = config
        summary = _read_json(os.path.join(path, "summary.json"))
        if isinstance(summary, dict):
            info.update(summary)
        runs.append(info)
    return runs


def _read_json(path: str) -> Any | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
