import json

import pytest

from nexis_ml import cli
from nexis_ml.device import auto_prefers_gpu, estimate_mlp_params, resolve_device

torch = pytest.importorskip("torch")

CUDA = torch.cuda.is_available()


def test_auto_threshold():
    # demo project scale: tiny -> CPU
    assert not auto_prefers_gpu(n_rows=240, n_params=700)
    # 50k rows x 15k params -> GPU territory
    assert auto_prefers_gpu(n_rows=50_000, n_params=15_000)


def test_estimate_mlp_params():
    # 3 -> 32 -> 16 -> 2: (3*32+32) + (32*16+16) + (16*2+2) = 690
    assert estimate_mlp_params(3, [32, 16], 2) == 690


def test_requested_cpu_always_cpu():
    device, reason = resolve_device("cpu", n_rows=10**9, n_params=10**6)
    assert device.type == "cpu"
    assert "requested" in reason


def test_auto_small_job_prefers_cpu_even_with_cuda():
    device, _ = resolve_device("auto", n_rows=240, n_params=700)
    assert device.type == "cpu"


@pytest.mark.skipif(CUDA, reason="asserts the no-CUDA fallback path")
def test_gpu_request_without_cuda_falls_back_with_explanation():
    device, reason = resolve_device("gpu", n_rows=1, n_params=1)
    assert device.type == "cpu"
    assert "CUDA" in reason


@pytest.mark.skipif(not CUDA, reason="needs a CUDA torch build")
def test_gpu_request_with_cuda_uses_cuda():
    device, reason = resolve_device("gpu", n_rows=1, n_params=1)
    assert device.type == "cuda"
    assert "GPU" in reason


@pytest.mark.skipif(not CUDA, reason="needs a CUDA torch build")
def test_auto_big_job_with_cuda_uses_cuda():
    device, _ = resolve_device("auto", n_rows=10**6, n_params=10**5)
    assert device.type == "cuda"


def test_env_command_reports_capabilities(capsys):
    assert cli.main(["env"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["nexisMl"]
    assert isinstance(info["torch"], str)  # torch installed in the dev venv
    assert info["cudaAvailable"] == CUDA
