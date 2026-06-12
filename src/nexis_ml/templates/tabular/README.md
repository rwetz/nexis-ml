# Tabular MLP project

Scaffolded by `nexis-ml new tabular`. Train a small multi-layer
perceptron on any CSV.

## Run it

```sh
pip install nexis-ml[torch]   # once
nexis-ml train
```

The example data (`data/example.csv`) is two interleaved half-moons
plus a noise column — the model should reach ~95%+ validation accuracy
in a few seconds on CPU.

## Use your own data

1. Drop a CSV with a header row into `data/`.
2. Point `[data] path` and `target` in `train.toml` at it.
3. `nexis-ml train`

Numeric columns become features; non-numeric feature columns are
skipped. A target with ≤20 distinct values is treated as
classification, otherwise regression.

## Files

- `train.py` — the model and training loop. **Yours to edit.**
- `train.toml` — hyperparameters.
- `.nexis-ml/runs/<run-id>/` — created per run: `metrics.jsonl` (event
  log), `summary.json`, `checkpoints/best.pt`, `artifacts/` (confusion
  matrices).

## Inspect runs

```sh
nexis-ml runs            # list runs with final metrics
nexis-ml replay <run>    # re-stream a finished run's events
```
