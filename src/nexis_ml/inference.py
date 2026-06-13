# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""Inference from a trained checkpoint — the engine half of the Nexis ML
Lab's playground (and `nexis-ml infer` / `serve` in a plain terminal).

Design note: a template's model lives in *its* train.py (yours to edit),
so the engine can't import it for inference. Instead it rebuilds the
model here from the **sizes saved in the checkpoint** — `context`/`embed`/
`heads`/`layers` for textgen, `hidden`/`features`/`classes` for tabular.

The practical contract that follows: changing a size in train.toml is
fully supported (the new size rides in the checkpoint), but changing the
model *code* in train.py can make these built-in predictors mismatch the
weights — training still works, you'd just write your own inference. A
size mismatch is reported clearly rather than crashing.

torch is imported lazily so the stdlib core (run/checkpoint resolution)
stays import-light and testable without it.
"""

from __future__ import annotations

import math
import os
from typing import Any

from . import run_store


# ── locating a run + its checkpoint (no torch needed) ─────────────────


def resolve_run_dir(project_dir: str, run: str) -> str:
    """`run` may be a run id, a path to a run directory, or a path to a
    checkpoint file. Return the run directory it refers to."""
    if not run:
        raise ValueError("no run given")
    # A direct path to the run dir (has a checkpoints/ child) or to a
    # checkpoint file inside one.
    cand = os.path.abspath(run)
    if os.path.isdir(os.path.join(cand, "checkpoints")):
        return cand
    if (
        os.path.isfile(cand)
        and os.path.basename(os.path.dirname(cand)) == "checkpoints"
    ):
        return os.path.dirname(os.path.dirname(cand))
    # Otherwise treat it as a run id under <project>/.nexis-ml/runs/.
    by_id = os.path.join(run_store.runs_root(project_dir), run)
    if os.path.isdir(by_id):
        return by_id
    raise FileNotFoundError(
        f"no run '{run}' (looked for a run dir, a checkpoint, and "
        f"{os.path.join(run_store.RUNS_SUBDIR, run)})"
    )


def find_checkpoint(run_dir: str, which: str = "best") -> str:
    """Path to the run's checkpoint, preferring `which` (best|last) and
    falling back to the other if only one was written."""
    ckpts = os.path.join(run_dir, "checkpoints")
    primary = os.path.join(ckpts, f"{which}.pt")
    if os.path.isfile(primary):
        return primary
    other = "last" if which == "best" else "best"
    fallback = os.path.join(ckpts, f"{other}.pt")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"no checkpoint in {ckpts} (expected best.pt or last.pt)")


# ── model rebuilds (mirror the shipped templates; torch-only) ─────────


def _build_tinygpt(vocab: int, context: int, embed: int, heads: int, layers: int):
    """A TinyGPT matching templates/textgen/train.py, parameterized by the
    dims saved in the checkpoint. Dropout is 0 (eval-only, and a no-op for
    state_dict shape)."""
    import torch
    from torch import nn

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.key = nn.Linear(embed, embed)
            self.query = nn.Linear(embed, embed)
            self.value = nn.Linear(embed, embed)
            self.proj = nn.Linear(embed, embed)
            self.attn_drop = nn.Dropout(0.0)
            self.resid_drop = nn.Dropout(0.0)
            mask = torch.tril(torch.ones(context, context)).view(1, 1, context, context)
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
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(embed)
            self.mlp = nn.Sequential(
                nn.Linear(embed, 4 * embed),
                nn.GELU(),
                nn.Linear(4 * embed, embed),
                nn.Dropout(0.0),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, embed)
            self.pos = nn.Embedding(context, embed)
            self.drop = nn.Dropout(0.0)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.ln_f = nn.LayerNorm(embed)
            self.head = nn.Linear(embed, vocab, bias=False)
            self.context = context

        def forward(self, idx):
            t = idx.shape[1]
            pos = torch.arange(t, device=idx.device)
            x = self.drop(self.tok(idx) + self.pos(pos))
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

    return TinyGPT()


def _build_mlp(in_dim: int, hidden: list[int], out_dim: int):
    """The tabular MLP from templates/tabular/train.py."""
    from torch import nn

    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── predictors ────────────────────────────────────────────────────────


class _MismatchError(RuntimeError):
    """The saved weights don't fit the rebuilt model — almost always
    because the model *code* in train.py was edited."""


def _load_state(model, state: dict, template: str) -> None:
    import torch

    try:
        model.load_state_dict(state)
    except (RuntimeError, KeyError) as e:
        raise _MismatchError(
            f"this {template} checkpoint doesn't match the built-in model — "
            "the architecture in train.py looks edited, so built-in inference "
            "can't load it (training and your own inference still work)"
        ) from e
    _ = torch  # keep torch import local + obvious


class TextgenPredictor:
    template = "textgen"

    def __init__(self, ckpt: dict[str, Any]):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stoi = {str(k): int(v) for k, v in ckpt["stoi"].items()}
        self.itos = {int(k): str(v) for k, v in ckpt["itos"].items()}
        self.context = int(ckpt["context"])
        self.model = _build_tinygpt(
            vocab=len(self.stoi),
            context=self.context,
            embed=int(ckpt["embed"]),
            heads=int(ckpt["heads"]),
            layers=int(ckpt["layers"]),
        )
        _load_state(self.model, ckpt["model"], self.template)
        self.model.to(self.device).eval()
        # Surfaced in the serve `ready` event so the playground can label
        # the model.
        self.meta = {"vocab": len(self.stoi), "context": self.context}

    def _encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        import torch

        prompt = str(request.get("input", "") or "")
        max_new = max(1, min(int(request.get("maxNew", 200)), 4000))
        temperature = float(request.get("temperature", 0.8))
        start = self._encode(prompt) or [0]
        idx = torch.tensor([start], dtype=torch.long, device=self.device)
        with torch.no_grad():
            for _ in range(max_new):
                logits = self.model(idx[:, -self.context :])
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, nxt], dim=1)
        ids = idx[0].tolist()
        text = "".join(self.itos[int(i)] for i in ids)
        # Slice the continuation by generated *tokens* (not prompt string
        # length) so it stays aligned even if the prompt had out-of-vocab
        # characters that were dropped during encoding.
        continuation = "".join(self.itos[int(i)] for i in ids[len(start) :])
        return {"input": prompt, "output": text, "continuation": continuation}


class TabularPredictor:
    template = "tabular"

    def __init__(self, ckpt: dict[str, Any]):
        import torch

        self.device = "cpu"  # tabular models are tiny; CPU is instant
        self.features: list[str] = [str(f) for f in ckpt["features"]]
        self.classes: list[str] = [str(c) for c in ckpt["classes"]]
        self.classification = len(self.classes) > 0
        self.mean = ckpt["mean"].to(self.device)
        self.std = ckpt["std"].to(self.device)
        out_dim = len(self.classes) if self.classification else 1
        self.model = _build_mlp(len(self.features), list(ckpt["hidden"]), out_dim)
        _load_state(self.model, ckpt["model"], self.template)
        self.model.to(self.device).eval()
        # The playground renders an input form from these.
        self.meta = {
            "features": self.features,
            "classes": self.classes,
            "task": "classification" if self.classification else "regression",
        }
        _ = torch

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        import torch

        raw = request.get("input", {})
        if not isinstance(raw, dict):
            raise ValueError("tabular input must be an object of feature: value")
        # Missing features fall back to the training mean (≈ 0 after
        # standardizing), so a partial row still predicts.
        row = []
        for i, name in enumerate(self.features):
            if name in raw and raw[name] is not None:
                row.append(float(raw[name]))
            else:
                row.append(float(self.mean[i]))
        x = (torch.tensor([row], dtype=torch.float32) - self.mean) / self.std
        with torch.no_grad():
            out = self.model(x)
        if self.classification:
            probs = torch.softmax(out, dim=1)[0]
            idx = int(probs.argmax())
            return {
                "input": raw,
                "output": {
                    "label": self.classes[idx],
                    "probs": {c: float(probs[i]) for i, c in enumerate(self.classes)},
                },
            }
        return {"input": raw, "output": {"value": float(out.squeeze())}}


def load_predictor(run_dir: str, which: str = "best"):
    """Load a checkpoint and return the matching predictor. Template is
    detected from the checkpoint's keys (textgen carries a vocab; tabular
    carries feature/class metadata)."""
    import torch

    ckpt_path = find_checkpoint(run_dir, which)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"unrecognized checkpoint at {ckpt_path}")
    if "stoi" in ckpt and "itos" in ckpt:
        return TextgenPredictor(ckpt)
    if "features" in ckpt and "classes" in ckpt:
        return TabularPredictor(ckpt)
    raise ValueError(
        f"unrecognized checkpoint at {ckpt_path} (not a textgen or tabular model)"
    )
