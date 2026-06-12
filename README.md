# nexis-ml

Hobby-grade ML engine for [Nexis](https://github.com/rwetz/Nexis) — create,
train, and inspect **small models on small data**, with metrics streamed
live to the Nexis terminal (or plain text in any other terminal).

This is the engine half of the Nexis ML Suite: Nexis ships the UI
(panels, live charts, run browser) and spawns this tool to do the actual
work. It is deliberately useful standalone too — every command works in
a plain shell.

**Not** an MLOps platform, not distributed training, not an LLM serving
stack. Train a tiny MLP on your spreadsheet, watch the loss curve drop,
poke at the checkpoint.

## Install

```sh
pip install nexis-ml[torch]
```

The core package is **stdlib-only** (detection, run listing, and replay
stay instant); PyTorch is only needed to actually train, hence the
extra.

## Quick start

```sh
nexis-ml new tabular my-experiment
cd my-experiment
nexis-ml train
nexis-ml runs
```

`new` scaffolds a project with example data (two half-moons), a
`train.toml` for hyperparameters, and a `train.py` that is **yours to
edit** — change the architecture, rerun, compare.

## Commands

| Command | What it does |
|---|---|
| `nexis-ml new <template> <dir>` | Scaffold a project (templates: `tabular`; planned: `image`, `textgen`) |
| `nexis-ml train [dir] [--config train.toml]` | Run the project's `train.py` |
| `nexis-ml runs [dir] [--json]` | List runs with final metrics |
| `nexis-ml replay <run-dir> [--delay ms]` | Re-stream a finished run's event log (frontend dev tool) |

Global flag `--nexis-protocol` (or `NEXIS_ML_PROTOCOL=1`) switches
stdout to the NDJSON event stream Nexis consumes — see
[PROTOCOL.md](PROTOCOL.md). Without it you get human-readable progress.

## Project layout (scaffolded)

```
my-experiment/
  train.py          # model + loop — the file you edit
  train.toml        # hyperparameters
  data/             # your CSVs / files
  .nexis-ml/runs/   # one dir per run: config.json, metrics.jsonl,
                    # summary.json, checkpoints/, artifacts/
```

## The harness API

```python
import nexis_ml

with nexis_ml.track("my-run", config=cfg, total_epochs=10) as run:
    for epoch in range(1, 11):
        ...
        run.log({"loss/train": loss.item()}, epoch=epoch)   # per batch
        run.epoch(epoch)                                    # epoch boundary
        run.artifact("confusion-matrix", path)              # generated files
        if run.cancelled:                                   # Nexis "Cancel" / Ctrl+C
            break
```

A `run.finished` event and `summary.json` are guaranteed on every exit
path (ok / cancelled / error).

## Development

```sh
python -m venv .venv
.venv\Scripts\pip install -e .[dev]      # Windows
.venv/bin/pip install -e .[dev]          # elsewhere
pytest
```

The test suite needs no torch — core stays framework-free by design.

## License

Apache-2.0, same as Nexis.
