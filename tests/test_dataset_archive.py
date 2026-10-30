import zipfile
from pathlib import Path

import pytest

from image_to_3d_shapenet.dataset.archive import UnsafeArchiveError, safe_extract_zip


def _extract(archive: Path, destination: Path) -> Path:
    return safe_extract_zip(
        archive,
        destination,
        max_extracted_bytes=1024 * 1024,
        max_members=20,
    )


def test_safe_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "blocked")
    with pytest.raises(UnsafeArchiveError, match="unsafe ZIP member"):
        _extract(archive, tmp_path / "dataset")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extraction_resumes_and_marks_complete(tmp_path: Path) -> None:
    archive = tmp_path / "category.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("03001627/model/models/model_normalized.obj", "v 0 0 0\n")
    destination = _extract(archive, tmp_path / "dataset")
    extracted = destination / "03001627/model/models/model_normalized.obj"
    original_mtime = extracted.stat().st_mtime_ns
    _extract(archive, destination)
    assert extracted.stat().st_mtime_ns == original_mtime
    assert (destination / ".extract_complete.json").is_file()


def test_safe_extraction_enforces_uncompressed_limit(tmp_path: Path) -> None:
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("large.bin", b"0" * 128)
    with pytest.raises(UnsafeArchiveError, match="expands"):
        safe_extract_zip(archive, tmp_path / "dataset", max_extracted_bytes=64, max_members=2)

