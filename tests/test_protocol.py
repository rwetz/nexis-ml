import io
import json

from nexis_ml.protocol import ProtocolEmitter


def test_emits_ndjson_when_enabled():
    buf = io.StringIO()
    em = ProtocolEmitter(enabled=True, out=buf)
    em.emit("metric", run="r", step=1, name="loss/train", value=0.5)
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "ev": "metric",
        "run": "r",
        "step": 1,
        "name": "loss/train",
        "value": 0.5,
    }


def test_silent_when_disabled_but_still_returns_event():
    buf = io.StringIO()
    em = ProtocolEmitter(enabled=False, out=buf)
    event = em.emit("log", level="info", msg="hi")
    assert buf.getvalue() == ""
    assert event == {"ev": "log", "level": "info", "msg": "hi"}


def test_one_line_per_event():
    buf = io.StringIO()
    em = ProtocolEmitter(enabled=True, out=buf)
    for i in range(5):
        em.emit("metric", run="r", step=i, name="x", value=float(i))
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # every line is valid standalone JSON


def test_non_serializable_values_fall_back_to_str():
    buf = io.StringIO()
    em = ProtocolEmitter(enabled=True, out=buf)

    class Weird:
        def __str__(self):
            return "weird"

    em.emit("log", msg=Weird())
    assert json.loads(buf.getvalue())["msg"] == "weird"
