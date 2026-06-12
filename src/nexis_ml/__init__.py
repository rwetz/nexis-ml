"""nexis-ml — hobby-grade ML engine for the Nexis terminal.

Public API for train.py scripts:

    import nexis_ml

    with nexis_ml.track("my-run", config=cfg, total_epochs=10) as run:
        run.log({"loss/train": 0.5})
        run.epoch(1)
"""

from .harness import Run, track

__version__ = "0.1.0"

__all__ = ["track", "Run", "__version__"]
