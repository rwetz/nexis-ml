# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

import json
import os

from nexis_ml import run_store


def test_new_run_dir_creates_structure(tmp_path):
    run = run_store.new_run_dir(str(tmp_path), "My Run!")
    assert os.path.isdir(run.checkpoints_dir)
    assert os.path.isdir(run.artifacts_dir)
    assert "my-run" in run.run_id


def test_new_run_dir_unique_on_collision(tmp_path):
    a = run_store.new_run_dir(str(tmp_path), "demo")
    b = run_store.new_run_dir(str(tmp_path), "demo")
    assert a.path != b.path


def test_append_event_round_trips(tmp_path):
    run = run_store.new_run_dir(str(tmp_path), "demo")
    events = [
        {"ev": "run.started", "run": run.run_id},
        {"ev": "metric", "run": run.run_id, "name": "loss/train", "value": 0.5},
    ]
    for e in events:
        run.append_event(e)
    run.close()
    with open(run.metrics_path, encoding="utf-8") as f:
        read_back = [json.loads(line) for line in f]
    assert read_back == events


def test_summary_write_is_atomic(tmp_path):
    run = run_store.new_run_dir(str(tmp_path), "demo")
    run.write_summary({"status": "ok"})
    assert json.load(open(run.summary_path, encoding="utf-8")) == {"status": "ok"}
    assert not os.path.exists(run.summary_path + ".tmp")


def test_list_runs_newest_first_with_status(tmp_path):
    older = run_store.RunDir(str(tmp_path / ".nexis-ml" / "runs" / "2026-01-01-0900-a"))
    newer = run_store.RunDir(str(tmp_path / ".nexis-ml" / "runs" / "2026-06-12-0900-b"))
    for rd in (older, newer):
        rd.create()
        rd.write_config({"lr": 0.1})
    older.write_summary({"status": "ok", "metrics": {}})
    # `newer` has no summary -> still running or crashed -> "unknown"

    runs = run_store.list_runs(str(tmp_path))
    assert [r["run"] for r in runs] == ["2026-06-12-0900-b", "2026-01-01-0900-a"]
    assert runs[0]["status"] == "unknown"
    assert runs[1]["status"] == "ok"
    assert runs[1]["config"] == {"lr": 0.1}


def test_list_runs_empty_project(tmp_path):
    assert run_store.list_runs(str(tmp_path)) == []
