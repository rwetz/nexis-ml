# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

import json
import os

from nexis_ml import cli, report


def _write_run(tmp_path):
    run = tmp_path / ".nexis-ml" / "runs" / "2026-01-01-0000-demo"
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "cm.json").write_text(
        json.dumps({"labels": ["0", "1"], "matrix": [[5, 1], [0, 4]]}),
        encoding="utf-8",
    )
    events = [
        {"ev": "run.started", "run": "r", "name": "tabular"},
        {
            "ev": "metric",
            "run": "r",
            "step": 1,
            "epoch": 1,
            "name": "loss/train",
            "value": 1.0,
        },
        {
            "ev": "metric",
            "run": "r",
            "step": 2,
            "epoch": 1,
            "name": "loss/train",
            "value": 0.5,
        },
        {"ev": "sample", "run": "r", "output": "hello world"},
        {
            "ev": "artifact",
            "run": "r",
            "kind": "confusion-matrix",
            "path": str(run / "artifacts" / "cm.json"),
        },
        {"ev": "run.finished", "run": "r", "status": "ok"},
    ]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "name": "tabular",
                "device": "cpu",
                "lastEpoch": 1,
                "metrics": {
                    "loss/train": {"last": 0.5, "min": 0.5, "max": 1.0, "count": 2}
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "config.json").write_text(
        json.dumps({"train": {"epochs": 1}}), encoding="utf-8"
    )
    return run


def test_build_html_report_includes_key_sections(tmp_path):
    run = _write_run(tmp_path)
    out = report.build_html_report(str(run))
    assert "<html" in out and "</html>" in out
    assert "loss/train" in out
    assert "polyline" in out  # a chart was drawn
    assert "Confusion matrix" in out
    assert "hello world" in out  # generated sample
    assert "Summary" in out
    assert "nexis-ml" in out


def test_write_html_report_writes_file(tmp_path):
    run = _write_run(tmp_path)
    out = report.write_html_report(str(run))
    assert os.path.isfile(out) and out.endswith("report.html")
    assert "<html" in open(out, encoding="utf-8").read()


def test_export_cli_writes_report(tmp_path, capsys, monkeypatch):
    # Another test may have left NEXIS_ML_PROTOCOL set; force human mode so
    # the "wrote" line lands on stdout deterministically.
    monkeypatch.delenv("NEXIS_ML_PROTOCOL", raising=False)
    _write_run(tmp_path)
    code = cli.main(["export", str(tmp_path), "--run", "2026-01-01-0000-demo"])
    assert code == 0
    assert "wrote" in capsys.readouterr().out
    assert (
        tmp_path / ".nexis-ml" / "runs" / "2026-01-01-0000-demo" / "report.html"
    ).is_file()


def test_export_cli_missing_run_errors(tmp_path, capsys):
    assert cli.main(["export", str(tmp_path), "--run", "nope"]) == 1
    assert "no run" in capsys.readouterr().err
