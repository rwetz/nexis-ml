# Image classifier project

Scaffolded by `nexis-ml new image`. Train a small CNN on folders of
images — one folder per class.

## Run it

```sh
pip install nexis-ml[torch]   # once (includes Pillow for image loading)
nexis-ml train
```

The bundled `data/` has four pattern classes (horizontal / vertical /
diagonal stripes + checkerboard) of small grayscale PNGs. After every
pass the run writes a **sample-prediction grid** (green border = correct,
red = wrong) and a confusion matrix — both show up in the Nexis ML Lab.

## Use your own images

```
data/
  cats/   img1.png img2.png …
  dogs/   …
```

1. Make one folder per class under `data/` and drop images in.
2. `nexis-ml train`

Images are converted to grayscale and resized to the first image's size.
Mixed sizes are fine (they're resized to match).

## What to tune

- `[model] conv1` / `conv2` / `hidden` — model capacity.
- `[train] epochs` / `lr` / `batch_size` — training length and speed.
- `[sample] grid` — how many validation images appear in the grid.

## Files

- `train.py` — the CNN and training loop. **Yours to edit.**
- `train.toml` — hyperparameters.
- `data/<class>/` — your images.
- `.nexis-ml/runs/<run-id>/` — per run: `metrics.jsonl`, `summary.json`,
  `checkpoints/best.pt`, `artifacts/` (confusion matrices + sample grids).
