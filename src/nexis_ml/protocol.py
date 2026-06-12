# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""Protocol v1 — NDJSON event stream over stdout.

The contract with Nexis (canonical spec: ML_SUITE.md in the Nexis repo,
vendored here as PROTOCOL.md):

  - One JSON object per line on stdout, only when protocol mode is on.
  - Human-readable output goes to stderr while protocol mode is on.
  - Unknown event types and unknown fields must be ignored by consumers.

Protocol mode is enabled by the `--nexis-protocol` CLI flag, which sets
NEXIS_ML_PROTOCOL=1 in the environment so the harness inside train.py
picks it up too.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, TextIO

PROTOCOL_VERSION = 1

ENV_FLAG = "NEXIS_ML_PROTOCOL"


def protocol_mode_enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


class ProtocolEmitter:
    """Writes protocol events as NDJSON lines. Thread-safe."""

    def __init__(self, enabled: bool | None = None, out: TextIO | None = None):
        self.enabled = protocol_mode_enabled() if enabled is None else enabled
        self._out = out if out is not None else sys.stdout
        self._lock = threading.Lock()

    def emit(self, ev: str, /, **fields: Any) -> dict[str, Any]:
        """Build the event object and write it to the stream if enabled.

        Returns the event dict either way so callers can persist it to
        the run's on-disk event log.
        """
        event: dict[str, Any] = {"ev": ev, **fields}
        if self.enabled:
            line = json.dumps(event, separators=(",", ":"), default=str)
            with self._lock:
                self._out.write(line + "\n")
                self._out.flush()
        return event
