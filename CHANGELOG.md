# Changelog

All notable changes to nexis-ml. Versions follow [SemVer](https://semver.org/);
pre-1.0, minor bumps may change the CLI or harness API.

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
