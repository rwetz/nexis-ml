# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""nexis-ml CLI: new / train / runs / replay.

Exit codes: 0 ok, 1 error, 130 cancelled (Ctrl+C). When --nexis-protocol
is set, stdout carries NDJSON protocol events only.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time

from . import __version__, run_store
from .protocol import ENV_FLAG, ProtocolEmitter
from .templates import TEMPLATES, scaffold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nexis-ml",
        description="Hobby-grade ML engine for the Nexis terminal: "
        "scaffold, train, and inspect small-model experiments.",
    )
    parser.add_argument(
        "--version", action="version", version=f"nexis-ml {__version__}"
    )
    parser.add_argument(
        "--nexis-protocol",
        action="store_true",
        help="emit NDJSON protocol events on stdout (set automatically by Nexis)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a new project from a template")
    p_new.add_argument("template", choices=sorted(TEMPLATES))
    p_new.add_argument(
        "dir",
        nargs="?",
        default=None,
        help="project directory (default: ./<template>)",
    )
    p_new.add_argument(
        "--force", action="store_true", help="scaffold into a non-empty directory"
    )
    p_new.add_argument(
        "--nexis-protocol", action="store_true", default=argparse.SUPPRESS
    )

    p_train = sub.add_parser("train", help="run the project's train.py")
    p_train.add_argument("dir", nargs="?", default=".")
    p_train.add_argument(
        "--config", default="train.toml", help="config file, relative to the project"
    )
    # SUPPRESS so the subparser doesn't clobber the top-level flag's value
    # when the flag is given before the subcommand (argparse writes subparser
    # defaults into the same namespace after the parent has parsed).
    p_train.add_argument(
        "--nexis-protocol", action="store_true", default=argparse.SUPPRESS
    )

    p_runs = sub.add_parser("runs", help="list runs in a project")
    p_runs.add_argument("dir", nargs="?", default=".")
    p_runs.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser(
        "env",
        help="report python / torch / CUDA capabilities as one JSON line",
    )

    p_replay = sub.add_parser(
        "replay",
        help="re-emit a finished run's event log as live protocol output "
        "(frontend dev tool)",
    )
    p_replay.add_argument("path", help="run directory or metrics.jsonl file")
    p_replay.add_argument(
        "--delay", type=float, default=10.0, help="milliseconds between events"
    )

    args = parser.parse_args(argv)

    if getattr(args, "nexis_protocol", False):
        os.environ[ENV_FLAG] = "1"

    try:
        if args.command == "new":
            return _cmd_new(args)
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "runs":
            return _cmd_runs(args)
        if args.command == "replay":
            return _cmd_replay(args)
        if args.command == "env":
            return _cmd_env()
    except KeyboardInterrupt:
        return 130
    return 2


def _cmd_new(args: argparse.Namespace) -> int:
    target = args.dir if args.dir is not None else args.template
    try:
        dest = scaffold(args.template, target, force=args.force)
    except (FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # In protocol mode stdout belongs to NDJSON events; humans read stderr.
    out = sys.stderr if os.environ.get(ENV_FLAG) == "1" else sys.stdout
    print(f"created {args.template} project at {dest}", file=out)
    print("next:", file=out)
    print(f"  cd {target}", file=out)
    print(
        "  nexis-ml train        # needs torch: pip install nexis-ml[torch]", file=out
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    project = os.path.abspath(args.dir)
    script = os.path.join(project, "train.py")
    if not os.path.isfile(script):
        print(
            f"error: no train.py in {project} "
            "(scaffold one with `nexis-ml new <template> <dir>`)",
            file=sys.stderr,
        )
        return 1
    # train.py reads its config path from the environment so --config
    # works without the template needing argv parsing.
    os.environ["NEXIS_ML_CONFIG"] = args.config
    prev_cwd = os.getcwd()
    os.chdir(project)
    sys.path.insert(0, project)
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    finally:
        os.chdir(prev_cwd)
        try:
            sys.path.remove(project)
        except ValueError:
            pass
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    runs = run_store.list_runs(os.path.abspath(args.dir))
    if args.as_json:
        print(json.dumps(runs, indent=2, default=str))
        return 0
    if not runs:
        print("no runs found")
        return 0
    for r in runs:
        line = f"{r['run']}  [{r.get('status', 'unknown')}]"
        metrics = r.get("metrics") or {}
        parts = [
            f"{name}={stats['last']:.4g}"
            for name, stats in sorted(metrics.items())
            if isinstance(stats, dict) and isinstance(stats.get("last"), (int, float))
        ]
        if parts:
            line += "  " + "  ".join(parts)
        print(line)
    return 0


def _cmd_env() -> int:
    """Machine-readable capability report (Nexis shows a GPU chip off
    this). Works without torch installed — fields just come back null."""
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "nexisMl": __version__,
        "torch": None,
        "cudaAvailable": False,
        "gpuName": None,
    }
    try:
        import torch  # noqa: PLC0415 — deliberate: torch is optional

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cudaAvailable"] = True
            info["gpuName"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — any torch failure means "no torch"
        pass
    print(json.dumps(info))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    path = args.path
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.jsonl")
    if not os.path.isfile(path):
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    # Replay's whole purpose is producing protocol output, so it's
    # always enabled regardless of the flag.
    emitter = ProtocolEmitter(enabled=True)
    delay = max(args.delay, 0.0) / 1000.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = event.pop("ev", None)
            if not isinstance(ev, str):
                continue
            emitter.emit(ev, **event)
            if delay:
                time.sleep(delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
