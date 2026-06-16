# Changelog

All notable changes to nexis-ml. Versions follow [SemVer](https://semver.org/);
pre-1.0, minor bumps may change the CLI or harness API.

## [0.7.2] — 2026-06-16

### Added
- **`blank` template** (`nexis-ml new blank`) — a from-scratch starting
  point: a minimal `train.py` that trains a tiny net on a synthetic XOR toy
  problem so it runs immediately, but is meant to be rewritten (design the
  `Net`, replace `make_data`). For when you'd rather build the architecture
  yourself than start from a worked example.

## [0.7.1] — 2026-06-15

### Fixed
- The `tabular` template now accepts a **scalar `hidden`** (e.g.
  `hidden = 16`) in `train.toml`, not only a list — matching the panel's
  hyperparameter form and the Rust engine's `[model] hidden` key.

## [0.7.0] — 2026-06-13

### Added
- **`nexis-ml export --run <id>`** — writes a self-contained HTML report
  for a run (inline SVG metric charts, a metrics summary, the confusion
  matrix, the latest sample-prediction grid embedded as base64, generated
  text samples, and the config). No torch/network/external assets, so it
  opens and shares anywhere. Defaults to `<run>/report.html`.

### Changed
- Release/packaging hardening: `publish.yml` now runs `twine check`, CI
  gained a `package` job (build + `twine check` + a guard that the
  textgen corpus ships in the wheel), and `pytest` is pinned to `tests/`.
  See README "Publishing" for the one-time PyPI trusted-publisher setup.

## [0.6.0] — 2026-06-13

### Added
- **`run.should_stop(...)`** — an early-stopping helper in the harness.
  Call it once per eval; it returns True once the monitored metric hasn't
  improved for `patience` evals (`mode="min"` for losses, `"max"` for
  accuracy).
- **GPU memory telemetry** — on a CUDA run the harness now emits a
  `mem/gpu_mb` metric each epoch (current allocated MB), so the panel can
  plot the GPU footprint. CPU runs are unaffected and stay torch-free.

## [0.5.0] — 2026-06-13

### Added
- **`image` template** — a small CNN over folders of images
  (`nexis-ml new image`, one folder per class). Streams loss + accuracy,
  writes a per-epoch confusion matrix and a **sample-prediction grid**
  PNG (green border = correct, red = wrong). Ships four bundled pattern
  classes generated with a stdlib-only PNG writer, so `new image` stays
  dependency-free; training adds `Pillow` (now in the `torch` extra) for
  image loading/saving.

### Note
- Image-model inference isn't in the playground yet; `infer`/`serve`
  report this clearly rather than mis-loading the checkpoint.

## [0.4.0] — 2026-06-13

### Added
- **Inference** — `nexis-ml infer` (one-shot) and `nexis-ml serve` (a
  stdin/stdout request→response loop, NDJSON). Both load a run's
  `checkpoints/best.pt` (or `last`) and predict: text continuation for
  `textgen`, class + probabilities for `tabular`. `serve` opens with a
  `ready` event (template + device), then answers one JSON request per
  line — this is what the Nexis ML Lab playground drives. `--run` accepts
  a run id, a run dir, or a checkpoint path.
- The built-in predictors rebuild a template's model from the **sizes
  saved in the checkpoint**, so tweaking sizes in `train.toml` still
  works; editing the model *code* is reported clearly instead of
  crashing. See `inference.py` and PROTOCOL.md.

## [0.3.0] — 2026-06-13

### Added
- **`textgen` template** — a char-level tiny GPT (`nexis-ml new textgen`).
  Trains a small hand-rolled transformer on any UTF-8 `.txt` file and,
  after every pass, emits a `sample` event with freshly generated text —
  the "watch gibberish become words" demo. Ships with a short bundled
  corpus (`data/input.txt`) and streams `loss/train`, `loss/val`, and
  `perplexity/val`. Best/last checkpoints embed the vocabulary so a
  future `infer`/`serve` can decode without the training text.

## [0.2.0] — 2026-06-12

### Added
- **Device selection** — `device = "auto" | "cpu" | "gpu"` in `train.toml`.
  `auto` picks the GPU only when the job is big enough to benefit (rows ×
  parameters heuristic) and explains its choice via a `run.info` line.
  The chosen device rides the protocol as a `device` field on
  `run.started` and in the run summary.
- **`nexis-ml env`** — one-line JSON capability report (python, torch
  version, CUDA availability, GPU name). Works without torch installed.
- **`nexis-ml new <template>`** — the directory argument is now optional
  and defaults to `./<template>`.
- `nexis_ml.load_config()` — BOM-tolerant TOML loading shared by
  templates (Notepad and PowerShell's `Set-Content` write UTF-8 BOMs
  that `tomllib` alone rejects).

### Changed
- In `--nexis-protocol` mode, `new` prints its human-readable output to
  stderr so stdout stays pure NDJSON.
- Version is single-sourced from `nexis_ml.__version__`.

## [0.1.0] — 2026-06-12

Initial release: protocol v1 (NDJSON over stdio), per-run store
(`.nexis-ml/runs/` with append-only `metrics.jsonl` and atomic
summaries), `track()` training harness with stdin cancel handling, and
the `tabular` template (MLP over any CSV, auto classification /
regression, confusion-matrix artifacts, best/last checkpoints).
CLI: `new` / `train` / `runs` / `replay`.
