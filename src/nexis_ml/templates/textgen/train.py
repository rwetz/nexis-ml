"""Char-level tiny transformer — scaffolded by `nexis-ml new textgen`.

This file is YOURS. It's a small GPT (nanoGPT-flavored) that learns to
write one character at a time from a plain .txt file. Edit the model,
the loop, the sampling — then run `nexis-ml train` again and watch the
generated text turn from gibberish into words.

Out of the box it trains on data/input.txt (a short bundled story),
streams the loss live, and after every pass prints a fresh sample of
text it has dreamed up — that snapshot is the whole point of the
template. Swap in your own .txt (a book, your journal, song lyrics) and
re-run.
"""

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

import nexis_ml

# ── config (train.toml) ───────────────────────────────────────────────

CONFIG_PATH = os.environ.get("NEXIS_ML_CONFIG", "train.toml")
cfg = nexis_ml.load_config(CONFIG_PATH)

data_cfg = cfg.get("data", {})
model_cfg = cfg.get("model", {})
train_cfg = cfg.get("train", {})
sample_cfg = cfg.get("sample", {})

DATA_PATH = data_cfg.get("path", "data/input.txt")

CONTEXT = int(model_cfg.get("context", 128))  # chars of history the model sees
EMBED = int(model_cfg.get("embed", 128))  # width of the model
HEADS = int(model_cfg.get("heads", 4))  # attention heads
LAYERS = int(model_cfg.get("layers", 4))  # transformer blocks
DROPOUT = float(model_cfg.get("dropout", 0.1))

SEED = int(train_cfg.get("seed", 42))
EPOCHS = int(train_cfg.get("epochs", 20))
STEPS_PER_EPOCH = int(train_cfg.get("steps_per_epoch", 200))
BATCH_SIZE = int(train_cfg.get("batch_size", 32))
LR = float(train_cfg.get("lr", 0.003))
VAL_SPLIT = float(train_cfg.get("val_split", 0.1))

SAMPLE_LEN = int(sample_cfg.get("length", 240))  # chars to dream up each pass
TEMPERATURE = float(sample_cfg.get("temperature", 0.8))  # >1 wilder, <1 safer
PRIME = sample_cfg.get("prime", "\n")  # seed text for each sample

if EMBED % HEADS != 0:
    raise SystemExit(
        f"model.embed ({EMBED}) must be divisible by model.heads ({HEADS})"
    )

torch.manual_seed(SEED)

# ── data ──────────────────────────────────────────────────────────────

if not os.path.isfile(DATA_PATH):
    raise SystemExit(
        f"no text at {DATA_PATH} — point [data] path in train.toml at a .txt file"
    )

with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()
if len(text) < CONTEXT + 2:
    raise SystemExit(
        f"{DATA_PATH} is too short ({len(text)} chars) for context={CONTEXT}"
    )

# Char-level vocabulary: every distinct character is one token.
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n_val = max(CONTEXT + 1, int(len(data) * VAL_SPLIT))
train_data = data[: len(data) - n_val]
val_data = data[len(data) - n_val :]


def get_batch(split, device):
    """A batch of (input, target) windows. The target is the input
    shifted one character to the right — predict the next char."""
    source = train_data if split == "train" else val_data
    high = len(source) - CONTEXT - 1
    ix = torch.randint(0, max(high, 1), (BATCH_SIZE,))
    x = torch.stack([source[i : i + CONTEXT] for i in ix])
    y = torch.stack([source[i + 1 : i + 1 + CONTEXT] for i in ix])
    return x.to(device), y.to(device)


# ── model (a small GPT — edit freely) ─────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask, so each position can
    only attend to itself and the positions before it."""

    def __init__(self):
        super().__init__()
        self.heads = HEADS
        self.key = nn.Linear(EMBED, EMBED)
        self.query = nn.Linear(EMBED, EMBED)
        self.value = nn.Linear(EMBED, EMBED)
        self.proj = nn.Linear(EMBED, EMBED)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(CONTEXT, CONTEXT)).view(1, 1, CONTEXT, CONTEXT)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, t, c = x.shape
        hs = c // self.heads
        k = self.key(x).view(b, t, self.heads, hs).transpose(1, 2)
        q = self.query(x).view(b, t, self.heads, hs).transpose(1, 2)
        v = self.value(x).view(b, t, self.heads, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = self.attn_drop(torch.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    """Attention + a small MLP, each with a residual connection."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(EMBED)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED, 4 * EMBED),
            nn.GELU(),
            nn.Linear(4 * EMBED, EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, EMBED)
        self.pos = nn.Embedding(CONTEXT, EMBED)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature):
        """Sample characters one at a time, feeding each back in."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -CONTEXT:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


model = TinyGPT()
n_params = sum(p.numel() for p in model.parameters())

# ── device (cpu / gpu / auto) ─────────────────────────────────────────

device, device_reason = nexis_ml.resolve_device(
    train_cfg.get("device", "auto"), n_rows=len(train_data), n_params=n_params
)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


@torch.no_grad()
def estimate_loss(eval_batches=20):
    """Average loss over a few batches of each split (smooths the noise
    of any single batch)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            xb, yb = get_batch(split, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample_text():
    """Generate a fresh snippet, primed with PRIME (default a newline)."""
    start = encode(PRIME) or [0]
    idx = torch.tensor([start], dtype=torch.long, device=device)
    out = model.generate(idx, SAMPLE_LEN, TEMPERATURE)
    return decode(out[0].tolist())


def save_checkpoint(path):
    torch.save(
        {
            "model": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "context": CONTEXT,
            "embed": EMBED,
            "heads": HEADS,
            "layers": LAYERS,
        },
        path,
    )


# ── training loop ─────────────────────────────────────────────────────

run_config = {**cfg, "derived": {"vocab_size": vocab_size, "params": n_params}}

with nexis_ml.track(
    "textgen", config=run_config, total_epochs=EPOCHS, device=str(device)
) as run:
    run.info(device_reason)
    run.info(
        f"char-level GPT: {n_params:,} params, vocab {vocab_size}, "
        f"{len(train_data):,} train / {len(val_data):,} val chars"
    )
    best_val = math.inf
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for _ in range(STEPS_PER_EPOCH):
            xb, yb = get_batch("train", device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            run.log({"loss/train": loss.item()}, epoch=epoch)
            if run.cancelled:
                break

        losses = estimate_loss()
        val_loss = losses["val"]
        # Perplexity = e^loss: "how many characters is the model effectively
        # choosing between." 1.0 is perfect; it falls toward the vocab size
        # for a random model. Clamp the exponent so an early spike can't inf.
        perplexity = math.exp(min(val_loss, 20.0))
        run.log({"loss/val": val_loss, "perplexity/val": perplexity}, epoch=epoch)

        # The payoff: a snapshot of what the model writes right now.
        snippet = sample_text()
        run.sample(input=PRIME, output=snippet)

        save_checkpoint(os.path.join(run.checkpoints_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(run.checkpoints_dir, "best.pt"))

        run.epoch(epoch)
        if run.cancelled:
            run.info("cancelled — stopped after checkpoint")
            break

    run.info(f"best val loss: {best_val:.4f}")
