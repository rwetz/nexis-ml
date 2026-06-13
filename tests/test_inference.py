# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""Inference run/checkpoint resolution + CLI error paths.

The actual prediction needs torch (model rebuild), so it's verified
manually; the suite stays framework-free per the project's design. These
tests cover everything that runs before torch is touched — which is also
exactly the part most likely to break on a bad path.
"""

import json
import os

import pytest

from nexis_ml import cli, inference


def _make_run(tmp_path, run_id, checkpoints=("best.pt", "last.pt")):
    run_dir = tmp_path / ".nexis-ml" / "runs" / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    for name in checkpoints:
        (run_dir / "checkpoints" / name).write_bytes(b"stub")
    return run_dir


def test_resolve_run_dir_by_id(tmp_path):
    run_dir = _make_run(tmp_path, "2026-01-01-0000-x")
    resolved = inference.resolve_run_dir(str(tmp_path), "2026-01-01-0000-x")
    assert os.path.samefile(resolved, run_dir)


def test_resolve_run_dir_by_dir_path(tmp_path):
    run_dir = _make_run(tmp_path, "r")
    assert os.path.samefile(
        inference.resolve_run_dir(str(tmp_path), str(run_dir)), run_dir
    )


def test_resolve_run_dir_by_checkpoint_path(tmp_path):
    run_dir = _make_run(tmp_path, "r")
    ckpt = run_dir / "checkpoints" / "best.pt"
    assert os.path.samefile(
        inference.resolve_run_dir(str(tmp_path), str(ckpt)), run_dir
    )


def test_resolve_run_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.resolve_run_dir(str(tmp_path), "nope")
    with pytest.raises(ValueError):
        inference.resolve_run_dir(str(tmp_path), "")


def test_find_checkpoint_prefers_requested_then_falls_back(tmp_path):
    run_dir = _make_run(tmp_path, "r")
    assert (
        os.path.basename(inference.find_checkpoint(str(run_dir), "best")) == "best.pt"
    )
    assert (
        os.path.basename(inference.find_checkpoint(str(run_dir), "last")) == "last.pt"
    )

    only_last = _make_run(tmp_path, "r2", checkpoints=("last.pt",))
    # asked for best, but only last exists → fall back rather than fail
    assert (
        os.path.basename(inference.find_checkpoint(str(only_last), "best")) == "last.pt"
    )


def test_find_checkpoint_none_raises(tmp_path):
    empty = tmp_path / ".nexis-ml" / "runs" / "r"
    (empty / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        inference.find_checkpoint(str(empty))


def test_infer_missing_run_errors_before_torch(tmp_path, capsys):
    assert cli.main(["infer", str(tmp_path), "--run", "nope", "--input", "hi"]) == 1
    assert "no run" in capsys.readouterr().err


def test_serve_missing_run_emits_ndjson_error(tmp_path, capsys):
    assert cli.main(["serve", str(tmp_path), "--run", "nope"]) == 1
    out = capsys.readouterr().out.strip()
    event = json.loads(out)
    assert event["ev"] == "error"
    assert "no run" in event["msg"]
