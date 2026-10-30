"""Download exactly one configured ShapeNetCore category archive."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from image_to_3d_shapenet.config import DatasetConfig

from .auth import ACCESS_INSTRUCTIONS, get_hf_token


class DatasetAccessError(RuntimeError):
    """Raised for missing or rejected access without exposing credentials."""


class DatasetResourceError(RuntimeError):
    """Raised when configured disk limits would be exceeded."""


def _remote_file_size(config: DatasetConfig, token: str, api: Any) -> int:
    info = api.dataset_info(
        repo_id=config.repo_id,
        files_metadata=True,
        token=token,
    )
    for sibling in info.siblings or []:
        if sibling.rfilename == config.zip_filename and sibling.size is not None:
            return int(sibling.size)
    raise DatasetResourceError(
        f"Không đọc được kích thước của {config.zip_filename}; dừng để tránh vượt giới hạn đĩa."
    )


def download_category(
    config: DatasetConfig,
    cache_dir: str | Path,
    *,
    token: str | None = None,
    api: Any | None = None,
    download_fn: Callable[..., str] = hf_hub_download,
) -> Path:
    """Download one ZIP via the Hub cache, which provides resumable transfers."""
    if Path(config.zip_filename).name != config.zip_filename:
        raise ValueError("dataset.zip_filename must be a basename")
    credential = token or get_hf_token()
    target_cache = Path(cache_dir).expanduser().resolve()
    target_cache.mkdir(parents=True, exist_ok=True)
    hub_api = api or HfApi()

    try:
        remote_bytes = _remote_file_size(config, credential, hub_api)
        max_bytes = int(config.max_archive_size_gb * 1024**3)
        if remote_bytes > max_bytes:
            raise DatasetResourceError(
                f"Archive {config.zip_filename} có {remote_bytes / 1024**3:.2f} GiB, "
                f"vượt giới hạn {config.max_archive_size_gb:.2f} GiB."
            )
        required = remote_bytes + int(config.min_free_disk_gb * 1024**3)
        free = shutil.disk_usage(target_cache).free
        if free < required:
            raise DatasetResourceError(
                f"Không đủ đĩa: cần tối thiểu {required / 1024**3:.2f} GiB, "
                f"còn {free / 1024**3:.2f} GiB."
            )
        downloaded = download_fn(
            repo_id=config.repo_id,
            filename=config.zip_filename,
            repo_type="dataset",
            token=credential,
            cache_dir=str(target_cache),
        )
    except DatasetResourceError:
        raise
    except HfHubHTTPError:
        raise DatasetAccessError(
            "Không thể truy cập ShapeNetCore bằng token hiện tại.\n" + ACCESS_INSTRUCTIONS
        ) from None
    except Exception as error:
        raise RuntimeError(
            f"Tải ShapeNetCore thất bại ({type(error).__name__}); token không được ghi log."
        ) from None

    result = Path(downloaded)
    if not result.is_file():
        raise RuntimeError("Hugging Face không trả về archive hợp lệ")
    return result

