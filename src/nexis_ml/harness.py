"""Training harness — the `nexis_ml.track()` API used inside train.py.

Design rule from the spec: train.py is the user's file. The harness
wraps a normal training loop with metric emission, run storage, and
cancel handling — it never hides the model definition.

    import nexis_ml

    with nexis_ml.track("mnist", config=cfg, total_epochs=10) as run:
        for epoch in range(1, 11):
            for batch in loader:
                ...
                run.log({"loss/train": loss.item()}, epoch=epoch)
                if run.cancelled:
                    break
            run.log({"acc/val": acc}, epoch=epoch)
            run.epoch(epoch)

Every event is both emitted on stdout (protocol mode) and appended to
the run's metrics.jsonl, so finished runs render without the engine.

Cancellation: in protocol mode a daemon thread watches stdin for
{"cmd": "cancel"} and sets `run.cancelled`; loops are expected to check
it and break cleanly (checkpoints still get written). In a terminal,
Ctrl+C does the same via KeyboardInterrupt — the run is finalized as
"cancelled" before the interrupt propagates.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from . import run_store
from .protocol import PROTOCOL_VERSION, ProtocolEmitter


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Run:
    """Handle for one training run. Created by `track()`, not directly."""

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None,
        run_dir: run_store.RunDir,
        total_epochs: int | None,
        emitter: ProtocolEmitter,
    ):
        self.name = name
        self.config = config or {}
        self.dir = run_dir
        self.total_epochs = total_epochs
        self._emitter = emitter
        self._step = 0
        self._epoch: int | None = None
        self._cancel = threading.Event()
        self._stats: dict[str, dict[str, Any]] = {}
        self._artifacts: list[dict[str, str]] = []
        self._last_values: dict[str, float] = {}
        self._started_at = _now()

    # ── identity / paths ──────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.dir.run_id

    @property
    def checkpoints_dir(self) -> str:
        return self.dir.checkpoints_dir

    @property
    def artifacts_dir(self) -> str:
        return self.dir.artifacts_dir

    # ── cancellation ──────────────────────────────────────────────────

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _handle_command(self, obj: Any) -> None:
        if isinstance(obj, dict) and obj.get("cmd") == "cancel":
            self._cancel.set()

    # ── logging ───────────────────────────────────────────────────────

    def log(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Log one or more scalar metrics at a step.

        Omitting `step` auto-increments an internal counter, so a plain
        `run.log({...})` per batch does the right thing.
        """
        if step is None:
            self._step += 1
            step = self._step
        else:
            self._step = max(self._step, step)
        if epoch is None:
            epoch = self._epoch
        for name, value in metrics.items():
            value = float(value)
            self._track_stat(name, value)
            event = self._emitter.emit(
                "metric", run=self.id, step=step, epoch=epoch, name=name, value=value
            )
            self.dir.append_event(event)

    def metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        self.log({name: value}, step=step, epoch=epoch)

    def epoch(self, i: int) -> None:
        """Mark the end of epoch `i` (1-based)."""
        self._epoch = i
        event = self._emitter.emit("epoch", run=self.id, epoch=i, of=self.total_epochs)
        self.dir.append_event(event)
        of = f"/{self.total_epochs}" if self.total_epochs else ""
        latest = "  ".join(f"{k}={v:.4g}" for k, v in sorted(self._last_values.items()))
        self._console(f"epoch {i}{of}  {latest}")

    def artifact(self, kind: str, path: str | os.PathLike[str]) -> None:
        # Absolute so Nexis never has to guess the base directory
        path = os.path.abspath(str(path))
        self._artifacts.append({"kind": kind, "path": path})
        event = self._emitter.emit("artifact", run=self.id, kind=kind, path=path)
        self.dir.append_event(event)

    def sample(self, input: Any, output: Any) -> None:
        """Preview an input/output pair (e.g. generated text per epoch)."""
        event = self._emitter.emit("sample", run=self.id, input=input, output=output)
        self.dir.append_event(event)

    def info(self, msg: str) -> None:
        self._emitter.emit("log", run=self.id, level="info", msg=msg)
        self._console(msg)

    # ── internals ─────────────────────────────────────────────────────

    def _track_stat(self, name: str, value: float) -> None:
        self._last_values[name] = value
        s = self._stats.get(name)
        if s is None:
            self._stats[name] = {"last": value, "min": value, "max": value, "count": 1}
        else:
            s["last"] = value
            s["min"] = min(s["min"], value)
            s["max"] = max(s["max"], value)
            s["count"] += 1

    def _console(self, msg: str) -> None:
        # stdout is protocol-only in protocol mode; humans read stderr.
        stream = sys.stderr if self._emitter.enabled else sys.stdout
        print(msg, file=stream, flush=True)

    def _start(self) -> None:
        self.dir.write_config(self.config)
        event = self._emitter.emit(
            "run.started",
            run=self.id,
            name=self.name,
            dir=os.path.abspath(self.dir.path),
            config=self.config,
            totalEpochs=self.total_epochs,
            protocol=PROTOCOL_VERSION,
            startedAt=self._started_at,
        )
        self.dir.append_event(event)
        self._console(f"run {self.id} started")

    def _finish(self, status: str) -> None:
        summary = {
            "status": status,
            "name": self.name,
            "startedAt": self._started_at,
            "finishedAt": _now(),
            "totalEpochs": self.total_epochs,
            "lastEpoch": self._epoch,
            "metrics": self._stats,
            "artifacts": self._artifacts,
        }
        self.dir.write_summary(summary)
        event = self._emitter.emit(
            "run.finished", run=self.id, status=status, summary=summary
        )
        self.dir.append_event(event)
        self.dir.close()
        self._console(f"run {self.id} finished: {status}")


def _start_stdin_watcher(run: Run) -> None:
    """Watch stdin for control commands ({"cmd": "cancel"}).

    Daemon thread: exits with the process, and on stdin EOF (Nexis
    closing the pipe) the loop just ends. Malformed lines are ignored
    per the protocol's forward-compatibility rule.
    """

    def watch() -> None:
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    run._handle_command(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except (OSError, ValueError):
            pass

    threading.Thread(target=watch, name="nexis-ml-stdin-watcher", daemon=True).start()


@contextmanager
def track(
    name: str,
    config: dict[str, Any] | None = None,
    project_dir: str = ".",
    total_epochs: int | None = None,
    emitter: ProtocolEmitter | None = None,
) -> Iterator[Run]:
    """Open a tracked run: allocates the run dir, emits run.started,
    and guarantees a run.finished + summary.json on every exit path
    (ok / cancelled / error)."""
    emitter = emitter if emitter is not None else ProtocolEmitter()
    run_dir = run_store.new_run_dir(project_dir, name)
    run = Run(name, config, run_dir, total_epochs, emitter)
    if emitter.enabled:
        _start_stdin_watcher(run)
    run._start()
    try:
        yield run
    except KeyboardInterrupt:
        run._finish("cancelled")
        raise
    except BaseException:
        run._finish("error")
        raise
    else:
        run._finish("cancelled" if run.cancelled else "ok")
