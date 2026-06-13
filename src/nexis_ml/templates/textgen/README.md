# Char-level text generator

Scaffolded by `nexis-ml new textgen`. A small GPT-style transformer that
learns to write one character at a time from a plain text file. After
every training pass it generates a fresh snippet — watch it turn from
gibberish into words.

## Run it

```sh
pip install nexis-ml[torch]   # once
nexis-ml train
```

The bundled `data/input.txt` is a short story (~7 KB). The model
overfits it on purpose — that's how a tiny char-level model on a tiny
corpus produces real words. On a GPU a pass takes a second or two; on
CPU, a little longer.

## Use your own text

1. Drop a `.txt` file into `data/` (a book from Project Gutenberg, your
   journal, song lyrics, code — anything UTF-8).
2. Point `[data] path` in `train.toml` at it.
3. `nexis-ml train`

More text and more `epochs` give better, less repetitive output. Bigger
`context` / `embed` / `layers` make a stronger (slower) model.

## What to tune

- `[sample] temperature` — the dial between sense and surprise. `0.8` is
  a good start; push toward `1.2` for wilder text, toward `0.5` for safer.
- `[model] context` — how far back the model can "see." Larger = more
  coherent, more memory and compute.
- `[train] epochs` / `steps_per_epoch` — total training. Watch the
  validation-loss curve flatten to know when more won't help.

## Files

- `train.py` — the model (a hand-rolled tiny GPT) and the loop. **Yours
  to edit.**
- `train.toml` — hyperparameters.
- `data/input.txt` — the training text.
- `.nexis-ml/runs/<run-id>/` — created per run: `metrics.jsonl` (event
  log), `summary.json`, `checkpoints/best.pt` (includes the vocabulary).

## Inspect runs

```sh
nexis-ml runs            # list runs with final metrics
nexis-ml replay <run>    # re-stream a finished run's events (and samples)
```
