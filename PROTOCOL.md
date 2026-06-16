# Nexis ML protocol — v1

> Vendored copy. The canonical spec lives in the Nexis repo
> (`ML_SUITE.md`); if they disagree, the Nexis copy wins.

Transport: **NDJSON over stdio**. When the engine runs with
`--nexis-protocol` (or `NEXIS_ML_PROTOCOL=1`), every line on stdout is
one JSON object; human-readable output goes to stderr. Control
messages arrive as JSON lines on stdin.

## Events (engine → Nexis, stdout)

```jsonc
{ "ev": "run.started",  "run": "2026-06-12-0931-tabular", "name": "tabular",
  "dir": "...", "config": { /* hyperparams */ }, "totalEpochs": 15,
  "device": "cuda:0",  // or "cpu"; null when the template doesn't report it
  "protocol": 1, "startedAt": "..." }
{ "ev": "metric",       "run": "…", "step": 120, "epoch": 1, "name": "loss/train", "value": 0.482 }
{ "ev": "epoch",        "run": "…", "epoch": 1, "of": 15 }
{ "ev": "artifact",     "run": "…", "kind": "confusion-matrix", "path": "…/artifacts/cm-epoch1.json" }
{ "ev": "sample",       "run": "…", "input": "…", "output": "…" }
{ "ev": "log",          "run": "…", "level": "info", "msg": "…" }
{ "ev": "run.finished", "run": "…", "status": "ok",  // or "cancelled" | "error"
  "summary": { "metrics": { "loss/train": { "last": 0.1, "min": 0.1, "max": 1.9, "count": 980 } },
               "artifacts": [ … ], "lastEpoch": 15, … } }
```

## Commands (Nexis → engine, stdin)

```jsonc
{ "cmd": "cancel" }            // graceful stop; loop breaks after the checkpoint
{ "cmd": "pause" }             // block at the next epoch boundary
{ "cmd": "resume" }            // release a paused loop
```

`pause` / `resume` are honored at the next epoch boundary (a daemon stdin
watcher delivers them); both the Python and Rust engines implement all three.

## Inference: `serve` (request/response over stdio)

`nexis-ml serve --run <id>` loads a checkpoint and runs an
inference loop: **one JSON request per stdin line, one NDJSON event per
stdout line.** Unlike training, serve speaks NDJSON unconditionally (it's
a machine interface). It opens with a `ready` event, then answers each
request until stdin closes.

```jsonc
// engine → Nexis (stdout)
{ "ev": "ready", "run": "…", "template": "textgen", "device": "cuda", "protocol": 1,
  "meta": { "vocab": 42, "context": 128 } }   // tabular: { "features": [...], "classes": [...], "task": "classification" }

// Nexis → engine (stdin), then engine → Nexis (stdout):
//  textgen — request: { "input": "Once upon a time", "maxNew": 200, "temperature": 0.8 }
{ "ev": "prediction", "input": "Once upon a time", "output": "Once upon a time …",
  "continuation": " …" }              // output minus the echoed prompt

//  tabular — request: { "input": { "x1": 0.5, "x2": -0.2 } }   (missing features → training mean)
{ "ev": "prediction", "input": { … },
  "output": { "label": "1", "probs": { "0": 0.24, "1": 0.76 } } }  // regression: { "value": … }

{ "ev": "error", "msg": "request is not valid JSON" }   // bad request / bad checkpoint; loop continues
```

`nexis-ml infer --run <id> --input …` is the one-shot form: it prints a
human-readable prediction, or (under `--nexis-protocol`) emits a single
`prediction` event and exits.

The built-in predictors rebuild the shipped template architecture from
the **sizes saved in the checkpoint** — so changing a size in train.toml
is supported, but editing the model *code* in train.py may no longer
match (reported as an `error`, not a crash).

## Rules

- Unknown `ev` types are ignored by the client; unknown fields are
  ignored by both sides (forward compatibility).
- Every event is also appended to the run's `metrics.jsonl`, so Nexis
  renders finished runs by reading files — no engine process needed.
  `nexis-ml replay` re-streams that file for frontend development.
- Artifacts are files on disk referenced by **absolute** path (as is
  the run `dir`); the protocol never inlines binary data.
- `run.finished` + `summary.json` are guaranteed on every exit path.
- Exit codes: 0 ok, 1 error, 130 cancelled via Ctrl+C.
