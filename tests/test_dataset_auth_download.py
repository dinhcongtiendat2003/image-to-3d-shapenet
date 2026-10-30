from pathlib import Path
from types import SimpleNamespace

import pytest

from image_to_3d_shapenet.config import DatasetConfig
from image_to_3d_shapenet.dataset import auth
from image_to_3d_shapenet.dataset.auth import MissingTokenError, get_hf_token
from image_to_3d_shapenet.dataset.download import download_category


def test_token_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    monkeypatch.setattr(auth, "_read_colab_token", lambda: "colab-token")
    assert get_hf_token() == "environment-token"


def test_token_falls_back_to_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(auth, "_read_colab_token", lambda: "colab-token")
    assert get_hf_token() == "colab-token"


def test_missing_token_has_actionable_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(auth, "_read_colab_token", lambda: None)
    with pytest.raises(MissingTokenError) as caught:
        get_hf_token()
    message = str(caught.value)
    assert "ShapeNet/ShapeNetCore" in message
    assert "giảng viên" in message
    assert "trường" in message
    assert "Read" in message


def test_download_requests_only_configured_category(tmp_path: Path) -> None:
    archive = tmp_path / "03001627.zip"
    archive.write_bytes(b"zip-placeholder")
    calls: list[dict[str, object]] = []

    class FakeApi:
        def dataset_info(self, **kwargs: object) -> SimpleNamespace:
            calls.append(dict(kwargs))
            sibling = SimpleNamespace(rfilename="03001627.zip", size=archive.stat().st_size)
            return SimpleNamespace(siblings=[sibling])

    def fake_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(archive)

    result = download_category(
        DatasetConfig(min_free_disk_gb=0.0),
        tmp_path / "cache",
        token="test-only-token",
        api=FakeApi(),
        download_fn=fake_download,
    )
    assert result == archive
    download_call = calls[1]
    assert download_call["repo_id"] == "ShapeNet/ShapeNetCore"
    assert download_call["filename"] == "03001627.zip"
    assert download_call["repo_type"] == "dataset"

