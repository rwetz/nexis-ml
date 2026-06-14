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
pip install nexis-ml[torch]          # into the active venv
pipx install nexis-ml[torch]         # or: on PATH in every shell
```

The core package is **stdlib-only** (detection, run listing, and replay
stay instant); PyTorch is only needed to actually train, hence the
extra. (Inside Nexis you don't need any of this — the ML Lab panel
installs the engine for you.)

### GPU (NVIDIA)

The default torch wheel is CPU-only. For CUDA, install torch from the
PyTorch index first — `--force-reinstall` matters, pip won't otherwise
swap a `+cpu` build for a `+cuXXX` build of the same version:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
pip install nexis-ml[torch]
```

Then set `device = "gpu"` in `train.toml` — or leave the default
`"auto"`, which uses the GPU only when the job is big enough to
benefit. `nexis-ml env` shows what your install can do.

## Quick start

```sh
nexis-ml new tabular my-experiment
cd my-experiment
nexis-ml train
nexis-ml runs
nexis-ml infer --run <run-id> --input '{"x1": 0.5, "x2": -0.2}'   # predict from the checkpoint
```

Or train a tiny text model and watch it write:

```sh
nexis-ml new textgen tiny-writer && cd tiny-writer
nexis-ml train
nexis-ml infer --run <run-id> --input "Once upon a time"
```

`new` scaffolds a project with example data (two half-moons), a
`train.toml` for hyperparameters, and a `train.py` that is **yours to
edit** — change the architecture, rerun, compare.

## Commands

| Command | What it does |
|---|---|
| `nexis-ml new <template> [dir]` | Scaffold a project (templates: `tabular`, `textgen`, `image`) |
| `nexis-ml train [dir] [--config train.toml]` | Run the project's `train.py` |
| `nexis-ml infer --run <id> [--input …]` | One-shot prediction from a checkpoint (text for `textgen`, class for `tabular`) |
| `nexis-ml serve --run <id>` | Inference loop: one JSON request per stdin line → one NDJSON response (drives the ML Lab playground) |
| `nexis-ml runs [dir] [--json]` | List runs with final metrics |
| `nexis-ml env` | JSON capability report (torch version, CUDA, GPU name) |
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
        if run.should_stop(patience=5):                     # early stopping
            break
        if run.cancelled:                                   # Nexis "Cancel" / Ctrl+C
            break
```

On a CUDA run the harness also emits a `mem/gpu_mb` metric each epoch so
the GPU footprint shows up alongside your curves.

A `run.finished` event and `summary.json` are guaranteed on every exit
path (ok / cancelled / error).

## Development

```sh
python -m venv .venv
.venv\Scripts\pip install -e .[dev]      # Windows
.venv/bin/pip install -e .[dev]          # elsewhere
pytest
ruff check src tests && ruff format --check src tests
```

The test suite needs no torch — core stays framework-free by design.

## License

Apache-2.0, same as Nexis.
