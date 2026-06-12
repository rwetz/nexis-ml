# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""Config loading shared by templates.

Reads TOML tolerantly: Windows editors (Notepad, PowerShell's
Set-Content) often write a UTF-8 BOM, which `tomllib.load` rejects with
a confusing parse error. Decoding via utf-8-sig strips it when present
and is a no-op otherwise.
"""

from __future__ import annotations

import tomllib
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = f.read()
    return tomllib.loads(data.decode("utf-8-sig"))
