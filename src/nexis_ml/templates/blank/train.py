"""A blank starting point — design your own network.

Unlike the other templates, this one assumes nothing about your data or
model. It trains a tiny net on a synthetic toy problem so it runs the
moment you hit Train — but the whole point is to make it yours:

  • design the network: edit the `Net` class below (add layers, change
    widths, swap activations), then run `nexis-ml train` again, and
  • bring your own data: replace `make_data` when you're ready.

Tip: the toy problem is an XOR pattern, which a single linear layer
*can't* solve — delete the hidden layer in `Net` and watch accuracy
stall, then add it back. Architecture matters; that's the lesson.
"""

import os

import torch
from torch import nn

import nexis_ml

# ── config (train.toml) ───────────────────────────────────────────────

CONFIG_PATH = os.environ.get("NEXIS_ML_CONFIG", "train.toml")
cfg = nexis_ml.load_config(CONFIG_PATH)
train_cfg = cfg.get("train", {})

SEED = int(train_cfg.get("seed", 42))
EPOCHS = int(train_cfg.get("epochs", 100))
LR = float(train_cfg.get("lr", 0.01))
torch.manual_seed(SEED)

# ── data — replace with your own ──────────────────────────────────────


def make_data(n: int = 600):
    """Two classes in an XOR pattern (same-sign vs opposite-sign points)."""
    x = torch.randn(n, 2)
    y = (x[:, 0] * x[:, 1] > 0).long()
    return x, y


X, Y = make_data()
n_val = max(1, len(X) // 5)
X_train, Y_train = X[n_val:], Y[n_val:]
X_val, Y_val = X[:n_val], Y[:n_val]

IN_DIM = X.shape[1]
OUT_DIM = int(Y.max().item()) + 1

# ── model — THIS is yours to design ───────────────────────────────────


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Edit freely: add nn.Linear layers, activations, dropout, …
        self.layers = nn.Sequential(
            nn.Linear(IN_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, OUT_DIM),
        )

    def forward(self, x):
        return self.layers(x)


model = Net()
n_params = sum(p.numel() for p in model.parameters())

# ── device + optimizer ────────────────────────────────────────────────

device, device_reason = nexis_ml.resolve_device(
    train_cfg.get("device", "auto"), n_rows=len(X_train), n_params=n_params
)
model = model.to(device)
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_val, Y_val = X_val.to(device), Y_val.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── train ─────────────────────────────────────────────────────────────

with nexis_ml.track(
    "blank", config=cfg, total_epochs=EPOCHS, device=str(device)
) as run:
    run.info(device_reason)
    run.info(f"{n_params:,} parameters — edit `Net` in train.py to redesign it")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), Y_train)
        loss.backward()
        optimizer.step()
        run.log({"loss/train": loss.item()}, epoch=epoch)

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            val_loss = loss_fn(logits, Y_val).item()
            acc = (logits.argmax(dim=1) == Y_val).float().mean().item()
        run.log({"loss/val": val_loss, "acc/val": acc}, epoch=epoch)

        run.epoch(epoch)
        if run.cancelled:
            break
