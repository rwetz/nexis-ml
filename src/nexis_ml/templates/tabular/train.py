"""Tabular MLP — scaffolded by `nexis-ml new tabular`.

This file is YOURS. Edit the model, the loss, the loop — then run
`nexis-ml train` again and watch the new curves.

What it does out of the box: loads the CSV named in train.toml, trains
a small MLP (classification if the target column has few distinct
values, regression otherwise), streams metrics live, writes a
confusion matrix per epoch, and checkpoints the best model.
"""

import csv
import json
import math
import os
import random

import torch
from torch import nn

import nexis_ml

# ── config (train.toml) ───────────────────────────────────────────────

CONFIG_PATH = os.environ.get("NEXIS_ML_CONFIG", "train.toml")
cfg = nexis_ml.load_config(CONFIG_PATH)

data_cfg = cfg.get("data", {})
train_cfg = cfg.get("train", {})

SEED = int(train_cfg.get("seed", 42))
EPOCHS = int(train_cfg.get("epochs", 15))
BATCH_SIZE = int(train_cfg.get("batch_size", 32))
LR = float(train_cfg.get("lr", 0.01))
VAL_SPLIT = float(train_cfg.get("val_split", 0.2))
HIDDEN = [int(h) for h in cfg.get("model", {}).get("hidden", [32, 16])]

random.seed(SEED)
torch.manual_seed(SEED)

# ── data ──────────────────────────────────────────────────────────────


def target_kind(values):
    """Classification if the target has few distinct values (or any
    non-numeric ones); regression only for numeric targets with many
    distinct values. Returns (is_classification, class_labels)."""
    unique = sorted(set(values))
    try:
        [float(v) for v in unique]
    except ValueError:
        return True, unique
    if len(unique) > 20:
        return False, []
    return True, sorted(unique, key=float)


def load_csv(path, target):
    """Numeric feature columns + target column → tensors. Non-numeric
    feature columns are skipped with a note."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    if target not in rows[0]:
        raise SystemExit(f"target column '{target}' not in {list(rows[0])}")

    feature_names = []
    for name in rows[0]:
        if name == target:
            continue
        try:
            float(rows[0][name])
            feature_names.append(name)
        except ValueError:
            print(f"note: skipping non-numeric column '{name}'")

    features = [[float(r[name]) for name in feature_names] for r in rows]
    raw_targets = [r[target] for r in rows]
    classification, classes = target_kind(raw_targets)
    if classification:
        index = {c: i for i, c in enumerate(classes)}
        targets = torch.tensor([index[v] for v in raw_targets])
    else:
        targets = torch.tensor([float(v) for v in raw_targets])
    return torch.tensor(features), targets, feature_names, classification, classes


DATA_PATH = data_cfg.get("path", "data/example.csv")
TARGET = data_cfg.get("target", "label")
X, y, feature_names, classification, classes = load_csv(DATA_PATH, TARGET)

# shuffle, split, standardize (stats from the train split only)
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
X, y = X[perm], y[perm]
n_val = max(1, int(len(X) * VAL_SPLIT))
X_val, y_val = X[:n_val], y[:n_val]
X_train, y_train = X[n_val:], y[n_val:]

mean = X_train.mean(dim=0)
std = X_train.std(dim=0).clamp_min(1e-8)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# ── device (cpu / gpu / auto) ─────────────────────────────────────────

out_dim = len(classes) if classification else 1
device, device_reason = nexis_ml.resolve_device(
    train_cfg.get("device", "auto"),
    n_rows=len(X_train),
    n_params=nexis_ml.estimate_mlp_params(X.shape[1], HIDDEN, out_dim),
)
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

# ── model ─────────────────────────────────────────────────────────────

layers, in_dim = [], X.shape[1]
for h in HIDDEN:
    layers += [nn.Linear(in_dim, h), nn.ReLU()]
    in_dim = h
layers.append(nn.Linear(in_dim, out_dim))
model = nn.Sequential(*layers).to(device)

loss_fn = nn.CrossEntropyLoss() if classification else nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def batch_loss(xb, yb):
    out = model(xb)
    if classification:
        return loss_fn(out, yb)
    return loss_fn(out.squeeze(-1), yb.float())


@torch.no_grad()
def evaluate():
    model.eval()
    logits = model(X_val)
    if classification:
        loss = loss_fn(logits, y_val).item()
        preds = logits.argmax(dim=1)
        acc = (preds == y_val).float().mean().item()
        return loss, {"acc/val": acc}, preds
    preds = logits.squeeze(-1)
    loss = loss_fn(preds, y_val.float()).item()
    return loss, {"rmse/val": math.sqrt(loss)}, preds


def confusion_matrix(preds):
    k = len(classes)
    m = [[0] * k for _ in range(k)]
    for t, p in zip(y_val.tolist(), preds.tolist()):
        m[int(t)][int(p)] += 1
    return {"labels": classes, "matrix": m}


def save_checkpoint(path):
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "features": feature_names,
            "classes": classes,
            "hidden": HIDDEN,
        },
        path,
    )


# ── training loop ─────────────────────────────────────────────────────

task = "classification" if classification else "regression"
run_config = {
    **cfg,
    "derived": {"features": feature_names, "classes": classes, "task": task},
}

with nexis_ml.track(
    "tabular", config=run_config, total_epochs=EPOCHS, device=str(device)
) as run:
    run.info(device_reason)
    run.info(
        f"{task}: {len(X_train)} train / {len(X_val)} val rows, "
        f"{len(feature_names)} features"
    )
    best_val = math.inf
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_train))
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = order[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = batch_loss(X_train[idx], y_train[idx])
            loss.backward()
            optimizer.step()
            run.log({"loss/train": loss.item()}, epoch=epoch)
            if run.cancelled:
                break

        val_loss, extra, preds = evaluate()
        run.log({"loss/val": val_loss, **extra}, epoch=epoch)

        if classification:
            cm_path = os.path.join(run.artifacts_dir, f"cm-epoch{epoch}.json")
            with open(cm_path, "w", encoding="utf-8") as f:
                json.dump(confusion_matrix(preds), f)
            run.artifact("confusion-matrix", cm_path)

        save_checkpoint(os.path.join(run.checkpoints_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(run.checkpoints_dir, "best.pt"))

        run.epoch(epoch)
        if run.cancelled:
            run.info("cancelled — stopped after checkpoint")
            break

    run.info(f"best val loss: {best_val:.4f}")
