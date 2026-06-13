"""Small image classifier — scaffolded by `nexis-ml new image`.

This file is YOURS. It trains a little CNN on folders of images (one
folder per class) and, after every pass, saves a grid of sample
predictions (green border = right, red = wrong) so you can watch it
learn to see. Edit the model, swap in your own images, rerun.

Out of the box it trains on data/<class>/*.png — four bundled pattern
classes (horizontal / vertical / diagonal stripes + checkerboard). Drop
your own folders of images into data/ and point [data] path at it.
"""

import json
import math
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

import nexis_ml

# ── config (train.toml) ───────────────────────────────────────────────

CONFIG_PATH = os.environ.get("NEXIS_ML_CONFIG", "train.toml")
cfg = nexis_ml.load_config(CONFIG_PATH)

data_cfg = cfg.get("data", {})
model_cfg = cfg.get("model", {})
train_cfg = cfg.get("train", {})
sample_cfg = cfg.get("sample", {})

DATA_DIR = data_cfg.get("path", "data")
CONV1 = int(model_cfg.get("conv1", 16))
CONV2 = int(model_cfg.get("conv2", 32))
HIDDEN = int(model_cfg.get("hidden", 64))

SEED = int(train_cfg.get("seed", 42))
EPOCHS = int(train_cfg.get("epochs", 12))
BATCH_SIZE = int(train_cfg.get("batch_size", 16))
LR = float(train_cfg.get("lr", 0.002))
VAL_SPLIT = float(train_cfg.get("val_split", 0.2))
GRID_N = int(sample_cfg.get("grid", 16))

random.seed(SEED)
torch.manual_seed(SEED)

# ── data: a folder per class ──────────────────────────────────────────

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

classes = sorted(
    d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
)
if len(classes) < 2:
    raise SystemExit(f"need at least 2 class folders in {DATA_DIR}/ — found {classes}")
class_to_idx = {c: i for i, c in enumerate(classes)}

samples = []  # (path, label)
for c in classes:
    cdir = os.path.join(DATA_DIR, c)
    for name in sorted(os.listdir(cdir)):
        if name.lower().endswith(IMG_EXTS):
            samples.append((os.path.join(cdir, name), class_to_idx[c]))
if not samples:
    raise SystemExit(f"no images found under {DATA_DIR}/<class>/")

# Image size comes from the first image; everything is resized to match
# and converted to single-channel grayscale.
with Image.open(samples[0][0]) as im0:
    WIDTH, HEIGHT = im0.size


def load_image(path):
    with Image.open(path) as im:
        im = im.convert("L").resize((WIDTH, HEIGHT))
        data = torch.tensor(list(im.getdata()), dtype=torch.float32)
    return (data / 255.0).view(1, HEIGHT, WIDTH)


X = torch.stack([load_image(p) for p, _ in samples])
y = torch.tensor([lbl for _, lbl in samples])

perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
X, y = X[perm], y[perm]
paths = [samples[i][0] for i in perm.tolist()]
n_val = max(1, int(len(X) * VAL_SPLIT))
X_val, y_val, paths_val = X[:n_val], y[:n_val], paths[:n_val]
X_train, y_train = X[n_val:], y[n_val:]

# ── model (a small CNN — edit freely) ─────────────────────────────────


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, CONV1, 3, padding=1)
        self.c2 = nn.Conv2d(CONV1, CONV2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # Figure out the flattened size with a dry run so any image size
        # works without hand-computing the dimensions.
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, HEIGHT, WIDTH))
        self.flat = d.flatten(1).shape[1]
        self.fc1 = nn.Linear(self.flat, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, len(classes))

    def _features(self, x):
        x = self.pool(F.relu(self.c1(x)))
        return self.pool(F.relu(self.c2(x)))

    def forward(self, x):
        x = self._features(x).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = CNN()
n_params = sum(p.numel() for p in model.parameters())

# ── device (cpu / gpu / auto) ─────────────────────────────────────────

device, device_reason = nexis_ml.resolve_device(
    train_cfg.get("device", "auto"), n_rows=len(X_train), n_params=n_params
)
model = model.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


@torch.no_grad()
def evaluate():
    model.eval()
    logits = model(X_val)
    loss = loss_fn(logits, y_val).item()
    preds = logits.argmax(dim=1)
    acc = (preds == y_val).float().mean().item()
    return loss, acc, preds


def confusion_matrix(preds):
    k = len(classes)
    m = [[0] * k for _ in range(k)]
    for t, p in zip(y_val.tolist(), preds.tolist()):
        m[int(t)][int(p)] += 1
    return {"labels": classes, "matrix": m}


def save_sample_grid(preds, path, n=GRID_N):
    """Compose a grid of validation images with their predicted labels —
    green border if correct, red if wrong."""
    n = min(n, len(paths_val))
    cols = min(8, n)
    rows = math.ceil(n / cols)
    tile, pad, label_h = 40, 4, 12
    cell_w, cell_h = tile + pad * 2, tile + pad * 2 + label_h
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 28))
    draw = ImageDraw.Draw(grid)
    for i in range(n):
        with Image.open(paths_val[i]) as im:
            thumb = im.convert("L").resize((tile, tile)).convert("RGB")
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w + pad, r * cell_h + pad
        correct = int(preds[i]) == int(y_val[i])
        color = (52, 168, 108) if correct else (220, 76, 76)
        grid.paste(thumb, (x0, y0))
        draw.rectangle([x0 - 1, y0 - 1, x0 + tile, y0 + tile], outline=color, width=2)
        draw.text((x0, y0 + tile + pad), classes[int(preds[i])][:7], fill=color)
    grid.save(path)


def save_checkpoint(path):
    torch.save(
        {
            "model": model.state_dict(),
            "classes": classes,
            "img_size": [HEIGHT, WIDTH],
            "channels": 1,
            "arch": {"conv1": CONV1, "conv2": CONV2, "hidden": HIDDEN},
        },
        path,
    )


# ── training loop ─────────────────────────────────────────────────────

run_config = {
    **cfg,
    "derived": {
        "classes": classes,
        "img_size": [HEIGHT, WIDTH],
        "n_train": len(X_train),
    },
}

with nexis_ml.track(
    "image", config=run_config, total_epochs=EPOCHS, device=str(device)
) as run:
    run.info(device_reason)
    run.info(
        f"{len(classes)} classes, {len(X_train)} train / {len(X_val)} val images "
        f"({WIDTH}x{HEIGHT}), {n_params:,} params"
    )
    best_val = math.inf
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_train))
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = order[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            run.log({"loss/train": loss.item()}, epoch=epoch)
            if run.cancelled:
                break

        val_loss, acc, preds = evaluate()
        run.log({"loss/val": val_loss, "acc/val": acc}, epoch=epoch)

        cm_path = os.path.join(run.artifacts_dir, f"cm-epoch{epoch}.json")
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(confusion_matrix(preds), f)
        run.artifact("confusion-matrix", cm_path)

        grid_path = os.path.join(run.artifacts_dir, f"samples-epoch{epoch}.png")
        save_sample_grid(preds, grid_path)
        run.artifact("image-grid", grid_path)

        save_checkpoint(os.path.join(run.checkpoints_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(run.checkpoints_dir, "best.pt"))

        run.epoch(epoch)
        if run.cancelled:
            run.info("cancelled — stopped after checkpoint")
            break

    run.info(f"best val loss: {best_val:.4f}")
