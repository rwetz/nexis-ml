# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""Device selection for training templates.

"auto" picks the GPU only when the job is big enough to benefit: for a
tiny model on a few hundred rows, per-batch transfer overhead makes the
GPU *slower* than the CPU — a confusing first experience. The threshold
is deliberately rough; this is hobby scale, not a scheduler.

Imports torch lazily so the stdlib-only core never depends on it.
"""

from __future__ import annotations

# Approximate work per epoch (rows x parameters) above which the GPU
# reliably wins. Tuned to say "GPU" only when it's obviously right.
AUTO_GPU_THRESHOLD = 50_000_000


def auto_prefers_gpu(n_rows: int, n_params: int) -> bool:
    return n_rows * max(n_params, 1) >= AUTO_GPU_THRESHOLD


def estimate_mlp_params(in_dim: int, hidden: list[int], out_dim: int) -> int:
    """Weights + biases of a plain MLP — enough signal for `auto`."""
    dims = [in_dim, *hidden, out_dim]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def resolve_device(
    requested: str | None = "auto",
    *,
    n_rows: int = 0,
    n_params: int = 0,
):
    """Map a train.toml `device` setting to a torch.device.

    Returns (torch.device, human-readable reason). The reason is meant
    to be shown to the user verbatim via run.info().
    """
    import torch

    req = (requested or "auto").strip().lower()
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None

    if req in ("cuda", "gpu"):
        if cuda:
            return torch.device("cuda"), f"training on GPU ({gpu_name}) — requested"
        return (
            torch.device("cpu"),
            "GPU requested, but this torch install has no CUDA support — "
            "using CPU (install the CUDA build of torch to fix this)",
        )
    if req == "cpu":
        return torch.device("cpu"), "training on CPU — requested"

    # auto
    if cuda and auto_prefers_gpu(n_rows, n_params):
        return (
            torch.device("cuda"),
            f"training on GPU ({gpu_name}) — picked automatically for this job size",
        )
    if cuda:
        return (
            torch.device("cpu"),
            "training on CPU — this job is small enough that the GPU would be "
            'slower (set device = "gpu" in train.toml to force it)',
        )
    return torch.device("cpu"), "training on CPU"
