"""Restore a figure's flat local source store into its temporary virtual tree.

This module uses only the standard library. The original renderer and data bytes
stay unchanged; paths.json records their former paths below render_support/fs.
Nothing is rendered and no original external source directory is accessed here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil


SCHEMA = "thesis-flat-render-support-v1"


def windows_io_path(path):
    """Use extended absolute Windows paths only at the filesystem boundary."""
    value = Path(path)
    if os.name != "nt":
        return value
    absolute = os.path.abspath(os.fspath(value))
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _relative_name(value, field):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid {field}: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid {field}: {value!r}")
    name = PurePosixPath(value)
    if name.is_absolute():
        raise ValueError(f"Absolute {field} is not permitted: {value!r}")
    return name


def _validated_files(support):
    manifest = support / "paths.json"
    storage = support / "files"
    if (support.is_symlink() or manifest.is_symlink() or storage.is_symlink() or
            not manifest.is_file() or not storage.is_dir()):
        raise ValueError(f"Render support requires ordinary local storage: {support}")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or not isinstance(record.get("files"), list):
        raise ValueError(f"Unsupported render support manifest: {manifest}")
    if not record["files"]:
        raise ValueError(f"Empty render support manifest: {manifest}")
    virtual_names, stored_names, plans = set(), set(), []
    for item in record["files"]:
        if not isinstance(item, dict):
            raise ValueError("Render support file records must be objects")
        virtual = _relative_name(item.get("virtual"), "virtual path")
        stored = _relative_name(item.get("stored"), "stored path")
        if len(stored.parts) != 2 or stored.parts[0] != "files":
            raise ValueError(f"Stored file must be directly inside files/: {stored}")
        if virtual.as_posix().casefold() in virtual_names:
            raise ValueError(f"Duplicate or Windows-colliding virtual path: {virtual}")
        if stored.as_posix().casefold() in stored_names:
            raise ValueError(f"Duplicate or Windows-colliding stored path: {stored}")
        virtual_names.add(virtual.as_posix().casefold())
        stored_names.add(stored.as_posix().casefold())
        source = support.joinpath(*stored.parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing ordinary render support file: {source}")
        data = source.read_bytes()
        if (len(data) != item.get("bytes") or
                hashlib.sha256(data).hexdigest() != item.get("sha256")):
            raise ValueError(f"Render support size or SHA256 mismatch: {source}")
        plans.append((virtual, source, item["bytes"], item["sha256"]))
    actual = {
        "files/" + path.name
        for path in (support / "files").iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual != {item["stored"] for item in record["files"]}:
        raise ValueError(f"Render support files/ differs from its manifest: {support}")
    if any(path.is_dir() or path.is_symlink() for path in (support / "files").iterdir()):
        raise ValueError(f"Render support files/ must contain ordinary files only: {support}")
    return plans


def restore_render_support(support_dir, destination):
    """Verify all stored bytes and restore the original tree into a new directory.

    Return its filesystem path. On Windows this is an extended-length absolute
    path, so renderer I/O can use the restored tree even under a long temp root.
    """
    support = windows_io_path(support_dir)
    target = windows_io_path(destination)
    if target.exists():
        raise FileExistsError(f"Render support destination must be new: {target}")
    plans = _validated_files(support)
    target.mkdir(parents=True)
    try:
        for virtual, source, expected_bytes, expected_sha256 in plans:
            data = source.read_bytes()
            if (len(data) != expected_bytes or
                    hashlib.sha256(data).hexdigest() != expected_sha256):
                raise ValueError(f"Render support file changed during restoration: {source}")
            output = target.joinpath(*virtual.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
    except BaseException:
        shutil.rmtree(target)
        raise
    return target
