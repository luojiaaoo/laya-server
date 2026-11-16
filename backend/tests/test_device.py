import sys
from types import SimpleNamespace

import pytest

from server.config import Settings
from server.device import validate_device


def mock_cuda(monkeypatch, *, available=True, count=2):
    cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: count,
                           current_device=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))


def test_default_device_is_cuda(monkeypatch):
    monkeypatch.setenv("LAYA_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LAYA_ADMIN_PASSWORD_HASH", "test-hash")
    monkeypatch.delenv("LAYA_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("LAYA_DEVICE", raising=False)
    assert Settings.from_env().device == "cuda"
    mock_cuda(monkeypatch)
    assert validate_device(None) == "cuda"


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_visible_gpu_is_accepted(monkeypatch, device):
    mock_cuda(monkeypatch)
    assert validate_device(device) == device


def test_missing_cuda_fails_instead_of_selecting_cpu(monkeypatch):
    mock_cuda(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        validate_device("cuda")


def test_missing_gpu_index_fails(monkeypatch):
    mock_cuda(monkeypatch, count=1)
    with pytest.raises(RuntimeError, match="unavailable CUDA device"):
        validate_device("cuda:1")


def test_explicit_cpu_does_not_require_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert validate_device("cpu") == "cpu"


@pytest.mark.parametrize("device", ["auto", "cuda:-1", "cuda:invalid", "gpu"])
def test_invalid_device_is_rejected(device):
    with pytest.raises(ValueError, match="LAYA_DEVICE"):
        validate_device(device)
