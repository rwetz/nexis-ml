# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

import csv
import json

from nexis_ml import cli
from nexis_ml.protocol import ENV_FLAG


def test_new_scaffolds_tabular(tmp_path, capsys):
    dest = tmp_path / "proj"
    assert cli.main(["new", "tabular", str(dest)]) == 0
    assert (dest / "train.py").is_file()
    assert (dest / "train.toml").is_file()
    assert (dest / "README.md").is_file()
    with open(dest / "data" / "example.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 240
    assert set(rows[0]) == {"x1", "x2", "noise", "label"}
    assert {r["label"] for r in rows} == {"0", "1"}


def test_new_defaults_dir_to_template_name(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["new", "tabular"]) == 0
    assert (tmp_path / "tabular" / "train.py").is_file()


def test_new_scaffolds_textgen(tmp_path, capsys):
    dest = tmp_path / "proj"
    assert cli.main(["new", "textgen", str(dest)]) == 0
    assert (dest / "train.py").is_file()
    assert (dest / "train.toml").is_file()
    assert (dest / "README.md").is_file()
    # The corpus ships with the template (copied verbatim, not generated).
    corpus = dest / "data" / "input.txt"
    assert corpus.is_file()
    assert len(corpus.read_text(encoding="utf-8")) > 1000


def test_new_scaffolds_image_with_class_folders(tmp_path, capsys):
    dest = tmp_path / "proj"
    assert cli.main(["new", "image", str(dest)]) == 0
    assert (dest / "train.py").is_file()
    assert (dest / "train.toml").is_file()
    data = dest / "data"
    class_dirs = sorted(p.name for p in data.iterdir() if p.is_dir())
    assert class_dirs == ["checker", "diagonal", "horizontal", "vertical"]
    pngs = list((data / "horizontal").glob("*.png"))
    assert len(pngs) == 36
    # Generated with the stdlib PNG writer — valid 8-bit grayscale signature.
    assert pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_new_in_protocol_mode_keeps_stdout_clean(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NEXIS_ML_PROTOCOL", "1")
    dest = tmp_path / "proj"
    assert cli.main(["new", "tabular", str(dest)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "created tabular project" in captured.err


def test_new_refuses_nonempty_dir(tmp_path, capsys):
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / "precious.txt").write_text("do not clobber")
    assert cli.main(["new", "tabular", str(dest)]) == 1
    assert (dest / "precious.txt").read_text() == "do not clobber"
    assert "not empty" in capsys.readouterr().err


def test_train_without_project_errors(tmp_path, capsys):
    assert cli.main(["train", str(tmp_path)]) == 1
    assert "no train.py" in capsys.readouterr().err


def test_runs_json_empty(tmp_path, capsys):
    assert cli.main(["runs", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_replay_re_emits_events(tmp_path, capsys):
    events = [
        {"ev": "run.started", "run": "r", "totalEpochs": 1},
        {"ev": "metric", "run": "r", "step": 1, "name": "loss/train", "value": 1.0},
        {"ev": "run.finished", "run": "r", "status": "ok"},
    ]
    f = tmp_path / "metrics.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    assert cli.main(["replay", str(f), "--delay", "0"]) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert out == events


def test_replay_accepts_run_dir(tmp_path, capsys):
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"ev": "log", "msg": "hi"}) + "\n", encoding="utf-8"
    )
    assert cli.main(["replay", str(tmp_path), "--delay", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["msg"] == "hi"


def test_replay_missing_path_errors(tmp_path, capsys):
    assert cli.main(["replay", str(tmp_path / "nope.jsonl")]) == 1


def test_protocol_flag_before_subcommand(tmp_path, capsys, monkeypatch):
    # regression: the train subparser's own --nexis-protocol default used
    # to clobber the top-level flag's parsed value
    monkeypatch.delenv(ENV_FLAG, raising=False)
    cli.main(["--nexis-protocol", "train", str(tmp_path)])  # fails (no train.py)
    import os

    assert os.environ.get(ENV_FLAG) == "1"


def test_protocol_flag_after_subcommand(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    cli.main(["train", str(tmp_path), "--nexis-protocol"])  # fails (no train.py)
    import os

    assert os.environ.get(ENV_FLAG) == "1"
