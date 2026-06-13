# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""nexis-ml — hobby-grade ML engine for the Nexis terminal.

Public API for train.py scripts:

    import nexis_ml

    with nexis_ml.track("my-run", config=cfg, total_epochs=10) as run:
        run.log({"loss/train": 0.5})
        run.epoch(1)
"""

from .config import load_config
from .device import estimate_mlp_params, resolve_device
from .harness import Run, track

__version__ = "0.5.0"

__all__ = [
    "track",
    "Run",
    "load_config",
    "resolve_device",
    "estimate_mlp_params",
    "__version__",
]
