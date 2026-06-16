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
| `nexis-ml new <template> [dir]` | Scaffold a project (templates: `tabular`, `textgen`, `image`, `blank`) |
| `nexis-ml train [dir] [--config train.toml]` | Run the project's `train.py` |
| `nexis-ml infer --run <id> [--input …]` | One-shot prediction from a checkpoint (text for `textgen`, class for `tabular`) |
| `nexis-ml serve --run <id>` | Inference loop: one JSON request per stdin line → one NDJSON response (drives the ML Lab playground) |
| `nexis-ml runs [dir] [--json]` | List runs with final metrics |
| `nexis-ml export --run <id> [--out f.html]` | Write a self-contained HTML report (charts + summary + samples) |
| `nexis-ml env` | JSON capability report (torch version, CUDA, GPU name) |
| `nexis-ml replay <run-dir> [--delay ms]` | Re-stream a finished run's event log (frontend dev tool) |

Global flag `--nexis-protocol` (or `NEXIS_ML_PROTOCOL=1`) switches
stdout to the NDJSON event stream Nexis consumes — see
[PROTOCOL.md](PROTOCOL.md). Without it you get human-readable progress.

## Templates

`nexis-ml new <template>` scaffolds one of:

| Template | What you get |
|---|---|
| `tabular` | A small MLP over a CSV — classification (few distinct target values) or regression. Ships example data (two interleaved half-moons), so it trains immediately. Writes a confusion matrix per epoch. |
| `textgen` | A tiny character-level GPT over a `.txt` file. Ships a small bundled corpus and streams a generated-text sample each pass; checkpoints embed the vocab so `infer`/`serve` can decode. |
| `image` | A small CNN over a folder-per-class image directory. Ships four generated pattern classes (stdlib-only PNG writer) and writes a per-epoch sample-prediction grid plus a confusion matrix. |
| `blank` | A minimal `train.py` with the harness wired up but no model — start a network from scratch. |

Every scaffolded `train.py` is **yours to edit**: change the architecture,
rerun `nexis-ml train`, and compare the curves. The Rust engine
([`nexis-ml-rs`](https://github.com/rwetz/nexis-ml-rs)) is config-only, so
`textgen` and `blank` (which need an editable `train.py`) are Python-only.

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

## Publishing

Releases go to PyPI via **trusted publishing** (OIDC) — no API token is
stored. `publish.yml` runs on a published GitHub Release; CI's `package`
job builds + `twine check`s on every push so packaging never breaks by
surprise.

One-time setup (manual, on the web):

1. **PyPI** → *Publishing* → add a *pending publisher*: PyPI project
   `nexis-ml`, owner `rwetz`, repo `nexis-ml`, workflow `publish.yml`,
   environment `pypi`.
2. **GitHub** → repo *Settings → Environments* → create an environment
   named `pypi`.

Cutting a release:

1. Bump `__version__` in `src/nexis_ml/__init__.py` (single source) and
   add a `CHANGELOG.md` entry.
2. Tag and push: `git tag vX.Y.Z && git push --tags`.
3. Create a GitHub Release for the tag → `publish.yml` builds and
   uploads to PyPI automatically.

Until the pending publisher exists on PyPI, the panel's install button
only works where the engine is already reachable (e.g. `pip install -e`).

## License

Apache-2.0, same as Nexis.
