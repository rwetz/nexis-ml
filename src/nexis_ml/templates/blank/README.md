# Blank model

A from-scratch starting point — for when you'd rather build the network
yourself than start from a worked example.

`train.py` trains a tiny net on a synthetic toy problem so it runs
immediately, but it's yours to rewrite:

- **Design the network** — edit the `Net` class in `train.py` (add layers,
  change widths, swap activations).
- **Bring your own data** — replace `make_data`.

Run `nexis-ml train` (or hit **Train** in the panel) and watch the curves.
Streamed metrics: `loss/train`, `loss/val`, `acc/val`.

> The toy problem is an XOR pattern, which a single linear layer can't
> separate — delete the hidden layer and watch accuracy stall, then add it
> back. That's the point: the architecture is the interesting part.
