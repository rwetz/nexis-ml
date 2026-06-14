# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

import io
import json
import os

import pytest

from nexis_ml.harness import track
from nexis_ml.protocol import ProtocolEmitter


def emitter():
    buf = io.StringIO()
    return ProtocolEmitter(enabled=True, out=buf), buf


def events_of(buf):
    return [json.loads(line) for line in buf.getvalue().strip().splitlines() if line]


def test_full_run_lifecycle(tmp_path):
    em, buf = emitter()
    with track(
        "demo",
        config={"lr": 0.1},
        project_dir=str(tmp_path),
        total_epochs=2,
        emitter=em,
    ) as run:
        for epoch in (1, 2):
            run.log({"loss/train": 1.0 / epoch}, epoch=epoch)
            run.epoch(epoch)

    events = events_of(buf)
    assert events[0]["ev"] == "run.started"
    assert events[0]["totalEpochs"] == 2
    assert events[0]["config"] == {"lr": 0.1}
    assert events[-1]["ev"] == "run.finished"
    assert events[-1]["status"] == "ok"

    # on-disk event log mirrors the stream exactly
    run_dirs = list((tmp_path / ".nexis-ml" / "runs").iterdir())
    assert len(run_dirs) == 1
    logged = [
        json.loads(line)
        for line in (run_dirs[0] / "metrics.jsonl").read_text().strip().splitlines()
    ]
    assert logged == events

    summary = json.loads((run_dirs[0] / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["lastEpoch"] == 2
    assert summary["metrics"]["loss/train"] == {
        "last": 0.5,
        "min": 0.5,
        "max": 1.0,
        "count": 2,
    }


def test_auto_step_increments(tmp_path):
    em, buf = emitter()
    with track("demo", project_dir=str(tmp_path), emitter=em) as run:
        run.log({"a": 1.0})
        run.log({"a": 2.0})
        run.log({"a": 3.0}, step=10)
        run.log({"a": 4.0})  # resumes after the explicit step
    steps = [e["step"] for e in events_of(buf) if e["ev"] == "metric"]
    assert steps == [1, 2, 10, 11]


def test_cancel_command_marks_run_cancelled(tmp_path):
    em, buf = emitter()
    with track("demo", project_dir=str(tmp_path), emitter=em) as run:
        assert not run.cancelled
        run._handle_command({"cmd": "cancel"})
        assert run.cancelled
        run._handle_command({"cmd": "definitely-not-a-command"})  # ignored
    assert events_of(buf)[-1]["status"] == "cancelled"


def test_exception_marks_run_error_and_reraises(tmp_path):
    em, buf = emitter()
    with pytest.raises(RuntimeError, match="boom"):
        with track("demo", project_dir=str(tmp_path), emitter=em):
            raise RuntimeError("boom")
    events = events_of(buf)
    assert events[-1]["ev"] == "run.finished"
    assert events[-1]["status"] == "error"
    # summary still written on the error path
    run_dir = next((tmp_path / ".nexis-ml" / "runs").iterdir())
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "error"


def test_device_recorded_in_started_event_and_summary(tmp_path):
    em, buf = emitter()
    with track("demo", project_dir=str(tmp_path), emitter=em, device="cuda:0"):
        pass
    events = events_of(buf)
    assert events[0]["device"] == "cuda:0"
    assert events[-1]["summary"]["device"] == "cuda:0"


def test_should_stop_after_patience_without_improvement(tmp_path):
    em, _ = emitter()
    with track("demo", project_dir=str(tmp_path), emitter=em) as run:
        # Improving losses → never stop.
        for v in (1.0, 0.8, 0.6):
            run.log({"loss/val": v})
            assert run.should_stop(patience=2) is False
        # Now plateau: two non-improving evals trip patience=2.
        run.log({"loss/val": 0.61})
        assert run.should_stop(patience=2) is False  # wait = 1
        run.log({"loss/val": 0.62})
        assert run.should_stop(patience=2) is True  # wait = 2


def test_should_stop_max_mode_for_accuracy(tmp_path):
    em, _ = emitter()
    with track("demo", project_dir=str(tmp_path), emitter=em) as run:
        run.log({"acc/val": 0.5})
        assert run.should_stop("acc/val", patience=1, mode="max") is False
        run.log({"acc/val": 0.49})  # no improvement
        assert run.should_stop("acc/val", patience=1, mode="max") is True


def test_should_stop_is_false_until_metric_seen(tmp_path):
    em, _ = emitter()
    with track("demo", project_dir=str(tmp_path), emitter=em) as run:
        assert run.should_stop("loss/val", patience=1) is False


def test_artifact_and_sample_events(tmp_path):
    em, buf = emitter()
    with track("demo", project_dir=str(tmp_path), emitter=em) as run:
        run.artifact("confusion-matrix", "artifacts/cm.json")
        run.sample("once upon", "a time")
    events = events_of(buf)
    kinds = {e["ev"] for e in events}
    assert {"artifact", "sample"} <= kinds
    summary = [e for e in events if e["ev"] == "run.finished"][0]["summary"]
    assert summary["artifacts"] == [
        {"kind": "confusion-matrix", "path": os.path.abspath("artifacts/cm.json")}
    ]
