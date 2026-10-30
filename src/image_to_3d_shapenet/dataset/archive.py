"""ZIP extraction with traversal, symlink, size and resume protection."""

from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


class UnsafeArchiveError(RuntimeError):
    """Raised when an archive can escape or exhaust the extraction target."""


def _safe_relative(name: str) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise UnsafeArchiveError(f"unsafe ZIP member: {name!r}")
    if pure.parts[0].endswith(":"):
        raise UnsafeArchiveError(f"unsafe ZIP member: {name!r}")
    return Path(*pure.parts)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def safe_extract_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    max_extracted_bytes: int,
    max_members: int,
    min_free_bytes: int = 0,
) -> Path:
    """Extract sequentially and resume matching files after interruption."""
    archive_path = Path(archive).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".extract_complete.json"
    archive_stat = archive_path.stat()
    fingerprint = {
        "archive_size": archive_stat.st_size,
        "archive_mtime_ns": archive_stat.st_mtime_ns,
        "archive_name": archive_path.name,
    }
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == fingerprint:
                return target
        except (OSError, json.JSONDecodeError):
            pass

    with zipfile.ZipFile(archive_path) as source:
        members = source.infolist()
        if len(members) > max_members:
            raise UnsafeArchiveError(f"ZIP has {len(members)} members; limit is {max_members}")
        total_size = sum(item.file_size for item in members if not item.is_dir())
        if total_size > max_extracted_bytes:
            raise UnsafeArchiveError(
                f"ZIP expands to {total_size} bytes; limit is {max_extracted_bytes}"
            )
        free = shutil.disk_usage(target).free
        if free < total_size + min_free_bytes:
            raise UnsafeArchiveError("insufficient free disk for safe extraction")

        for info in members:
            relative = _safe_relative(info.filename)
            if _is_symlink(info):
                raise UnsafeArchiveError(f"ZIP symlink is not allowed: {info.filename!r}")
            output = (target / relative).resolve()
            if not output.is_relative_to(target):
                raise UnsafeArchiveError(f"ZIP member escapes destination: {info.filename!r}")
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.is_file() and output.stat().st_size == info.file_size:
                continue
            temporary = output.with_name(output.name + ".part")
            with source.open(info) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            os.replace(temporary, output)

    marker.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return target

