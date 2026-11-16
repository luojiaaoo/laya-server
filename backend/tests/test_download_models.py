import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests.exceptions import Timeout


@pytest.fixture
def downloader():
    script = Path(__file__).resolve().parents[2] / "scripts" / "download-models.py"
    return runpy.run_path(str(script))


def test_default_command_downloads_only_multilingual(downloader, monkeypatch, tmp_path):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        for filename in kwargs["allow_patterns"]:
            path = Path(kwargs["local_dir"]) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("model-file", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setenv("LAYA_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["download-models.py"])
    downloader["main"]()
    assert len(calls) == 1
    assert calls[0]["model_id"] == "convaiinnovations/laya-multilingual"
    assert calls[0]["revision"] == "master"
    assert (tmp_path / "multilingual" / "model.safetensors").is_file()
    assert (tmp_path / "multilingual" / "encoder" / "config.json").is_file()
    assert {path.name for path in tmp_path.iterdir()} == {"multilingual"}


def test_incomplete_snapshot_fails(downloader, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=lambda **kwargs: None))
    with pytest.raises(RuntimeError, match="missing required model file"):
        downloader["download_model"](tmp_path)


def test_network_errors_retry_and_eventually_fail(downloader, monkeypatch, tmp_path):
    calls = []

    def timeout(**kwargs):
        calls.append(kwargs)
        raise Timeout("network unavailable")

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=timeout))
    monkeypatch.setattr(downloader["time"], "sleep", lambda delay: None)
    with pytest.raises(Timeout):
        downloader["download_model"](tmp_path)
    assert len(calls) == 5
