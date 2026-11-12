#!/usr/bin/env python3
"""Inspect, locate, verify and restore a thesis release using the standard library.

These commands read saved files only. They never import experiment packages or
load models. Artifact archives must be downloaded separately when not attached.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile


SCHEMA = "thesis-publication-release-v1"
BUFFER_SIZE = 1024 * 1024


class ReleaseError(Exception):
    """A release file, catalogue or archive failed an explicit integrity rule."""


def relative_path(value: str) -> str:
    """Accept canonical POSIX relative paths, including non-ASCII file names."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReleaseError(f"Unsafe relative path: {value!r}")
    if any(ord(character) < 32 for character in value):
        raise ReleaseError(f"Control character in relative path: {value!r}")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ReleaseError(f"Absolute or drive-relative path rejected: {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ReleaseError(f"Non-canonical or escaping path rejected: {value!r}")
    if PurePosixPath(value).is_absolute():
        raise ReleaseError(f"Absolute path rejected: {value!r}")
    return value


def checked_path(root: Path, name: str, *, allow_missing: bool = False) -> Path:
    """Reject links and non-directory ancestors before using a release path."""
    name = relative_path(name)
    target = root
    parts = name.split("/")
    missing = False
    for index, part in enumerate(parts):
        target = target / part
        if missing:
            continue
        try:
            info = target.lstat()
        except FileNotFoundError:
            if not allow_missing:
                raise ReleaseError(f"Missing file: {name}") from None
            missing = True
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseError(f"Symbolic link rejected: {name}")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(info.st_mode):
                raise ReleaseError(f"Non-directory ancestor rejected: {name}")
        elif not stat.S_ISREG(info.st_mode):
            raise ReleaseError(f"Non-regular file rejected: {name}")
        elif info.st_nlink != 1:
            raise ReleaseError(f"Hard-linked file rejected: {name}")
    return target


def read_json(root: Path, name: str) -> dict:
    path = checked_path(root, name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ReleaseError(f"Cannot read {name}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"Expected a JSON object in {name}")
    return value


def load_manifest(root: Path) -> dict:
    manifest = read_json(root, "release_manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ReleaseError(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    if manifest.get("mode") not in ("shared", "public", "anonymous"):
        raise ReleaseError("Manifest mode must be shared, public or anonymous")
    if manifest.get("scientific_execution") != 0:
        raise ReleaseError("Manifest must declare scientific_execution: 0")
    paths = set()
    for group in ("source_files", "artifact_files"):
        entries = manifest.get(group)
        if not isinstance(entries, list):
            raise ReleaseError(f"Manifest {group} must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ReleaseError(f"Invalid entry in {group}")
            name = relative_path(entry.get("path"))
            if name == "release_manifest.json" or name in paths:
                raise ReleaseError(f"Duplicate, overlapping or self-hashed manifest path: {name}")
            if type(entry.get("size")) is not int or entry["size"] < 0:
                raise ReleaseError(f"Invalid file size: {name}")
            if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise ReleaseError(f"Invalid SHA256: {name}")
            paths.add(name)
    for name in paths:
        for parent in PurePosixPath(name).parents:
            if str(parent) in paths:
                raise ReleaseError(f"File/directory collision in manifest: {name}")
    return manifest


def digest_file(root: Path, entry: dict) -> None:
    name = entry["path"]
    path = checked_path(root, name)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ReleaseError(f"File ceased to be an independent regular file: {name}")
        if info.st_size != entry["size"]:
            raise ReleaseError(f"Size mismatch: {name} ({info.st_size} != {entry['size']})")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise ReleaseError(f"SHA256 mismatch: {name}")


def inventory(root: Path, entries: list) -> tuple[list, list]:
    present, missing = [], []
    for entry in entries:
        path = checked_path(root, entry["path"], allow_missing=True)
        (present if path.exists() else missing).append(entry)
    return present, missing


def print_scientific_boundary() -> None:
    print("Scientific execution: 0 (no model loading, environment, inference, training or evaluation)")


def inspect_release(root: Path, manifest: dict) -> int:
    source_present, source_missing = inventory(root, manifest["source_files"])
    present, missing = inventory(root, manifest["artifact_files"])
    print(f"Mode: {manifest['mode']}")
    print(f"Packaged sources: {len(manifest['source_files'])} listed; {len(source_present)} present; {len(source_missing)} missing")
    print(f"External artifacts: {len(manifest['artifact_files'])} listed; {len(present)} present; {len(missing)} missing")
    print("SHA256 is not checked by inspect; use check or check --require-data.")
    if missing:
        print("External materials require separate download and extract before full-data verification.")
    print_scientific_boundary()
    return int(bool(source_missing))


def check_release(root: Path, manifest: dict, require_data: bool) -> int:
    errors = []
    source_verified = 0
    for entry in manifest["source_files"]:
        try:
            digest_file(root, entry)
            source_verified += 1
        except (ReleaseError, OSError) as error:
            errors.append(str(error))
    print(f"Mode: {manifest['mode']}")
    print(f"Source integrity: {'PASS' if not errors else 'FAIL'} ({source_verified}/{len(manifest['source_files'])} files verified)")
    artifact_verified = 0
    try:
        present, missing = inventory(root, manifest["artifact_files"])
    except (ReleaseError, OSError) as error:
        errors.append(str(error))
        present, missing = [], []
    else:
        print(f"External data: {len(present)}/{len(manifest['artifact_files'])} files present; {len(missing)} missing")
        if require_data:
            for entry in present:
                try:
                    digest_file(root, entry)
                    artifact_verified += 1
                except (ReleaseError, OSError) as error:
                    errors.append(str(error))
            if missing:
                errors.append(f"External data incomplete: {len(missing)} artifact files missing")
        if missing:
            print("External data is NOT fully attached. Download and extract the artifact archives.")
            for entry in missing[:5]:
                print(f"  Missing: {entry['path']}")
    if require_data:
        complete = artifact_verified == len(manifest["artifact_files"]) and not errors
        print(f"External data SHA256: {'PASS' if complete else 'FAIL'} ({artifact_verified} files verified)")
    else:
        print("External data SHA256: NOT CHECKED (use check --require-data)")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print_scientific_boundary()
    return int(bool(errors))


def chapter_name(value: str) -> str:
    match = re.fullmatch(r"(?:chapter|c)?0*(\d+)", value.casefold())
    return f"chapter{int(match.group(1)):02d}" if match else value


def lookup_release(root: Path, args: argparse.Namespace) -> int:
    catalogue = read_json(root, "registry/paper_navigation.json")
    chapters = catalogue.get("chapters", [])
    if not isinstance(chapters, list):
        raise ReleaseError("Navigation catalogue has no chapters list")
    if args.chapter:
        chapters = [chapter for chapter in chapters if chapter.get("chapter") == chapter_name(args.chapter)]
    sections = [dict(section, chapter=chapter["chapter"]) for chapter in chapters for section in chapter.get("sections", [])]
    assets = [dict(asset, chapter=chapter["chapter"]) for chapter in chapters for asset in chapter.get("assets", [])]
    models = [dict(model, chapter=chapter["chapter"]) for chapter in chapters for model in chapter.get("models", [])]
    if args.section:
        def in_section(number):
            return number == args.section or str(number).startswith(args.section + ".")
        sections = [section for section in sections if in_section(section["number"])]
        models = [model for model in models if any(in_section(number) for number in model.get("paper_sections", []))]
        assets = [asset for asset in assets if any(in_section(number) for number in asset.get("paper_sections", []))]
    for kind, number in (("figure", args.figure), ("table", args.table)):
        if number:
            number = number.removeprefix("图").removeprefix("表")
            assets = [asset for asset in assets if asset.get("kind") == kind and asset.get("number") == number]
            ids = {model_id for asset in assets for model_id in asset.get("model_ids", [])}
            models = [model for model in models if model.get("id") in ids]
            numbers = {section for asset in assets for section in asset.get("paper_sections", [])}
            sections = [section for section in sections if section["number"] in numbers]
    if args.method:
        query = args.method.casefold()
        models = [model for model in models if any(query in str(model.get(field, "")).casefold() for field in ("paper_method", "method", "variant"))]
        ids = {model["id"] for model in models}
        assets = [asset for asset in assets if ids.intersection(asset.get("model_ids", []))]
        numbers = {number for model in models for number in model.get("paper_sections", [])}
        sections = [section for section in sections if any(number == section["number"] or number.startswith(section["number"] + ".") for number in numbers)]
    result = {
        "notice": "Catalogue references are relative paths. Models, data and other materials may require separate download; lookup does not verify their presence or load them.",
        "scientific_execution": 0,
        "model_loading": 0,
        "record_semantics": {
            "assets.actual_inputs": "Actual inputs to the selected figure or table, as registered.",
            "models": "Related retained models and their complete attached records; not all records are actual inputs to the selected figure or table.",
        },
        "sections_found": len(sections), "assets_found": len(assets), "models_found": len(models),
        "sections": sections, "assets": assets, "models": models,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def ensure_parent(root: Path, name: str) -> Path:
    """Create only ordinary destination directories, checking every component."""
    current = root
    for part in relative_path(name).split("/")[:-1]:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ReleaseError(f"Unsafe destination directory: {name}")
    return checked_path(root, name, allow_missing=True)


class ConcatenatedParts(io.RawIOBase):
    """Read a split gzip stream without writing a merged copy to disk."""

    def __init__(self, paths: list[Path]):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.closed:
            raise ValueError("read of closed split archive")
        while True:
            if self.current is None:
                try:
                    path = next(self.paths)
                except StopIteration:
                    return 0
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                self.current = os.fdopen(descriptor, "rb")
                info = os.fstat(self.current.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    self.current.close()
                    self.current = None
                    raise ReleaseError(f"Archive part is not an independent regular file: {path.name}")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def verified_parts(parts_manifest: Path) -> list[Path]:
    directory = parts_manifest.parent.resolve()
    description = read_json(directory, parts_manifest.name)
    parts = description.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ReleaseError("Parts manifest must contain a nonempty parts list")
    paths = []
    prefix = None
    width = None
    for index, entry in enumerate(parts, 1):
        if not isinstance(entry, dict):
            raise ReleaseError("Invalid parts manifest entry")
        name = relative_path(entry.get("name"))
        if "/" in name:
            raise ReleaseError(f"Archive part name must be a basename: {name}")
        match = re.fullmatch(r"(.+\.tar\.gz)\.part(\d{3,})", name)
        if not match or int(match.group(2)) != index:
            raise ReleaseError(f"Parts must be in continuous .part001, .part002 order: {name}")
        if prefix is None:
            prefix, width = match.group(1), len(match.group(2))
        elif prefix != match.group(1) or width != len(match.group(2)):
            raise ReleaseError(f"Archive parts have inconsistent names: {name}")
        if type(entry.get("size")) is not int or entry["size"] <= 0:
            raise ReleaseError(f"Invalid archive part size: {name}")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ReleaseError(f"Invalid archive part SHA256: {name}")
        digest_file(directory, dict(entry, path=name))
        paths.append(directory / name)
    return paths


def extract_archives(root: Path, manifest: dict, archives: list[Path], parts_manifest: Path | None = None) -> dict:
    allowed = {entry["path"]: entry for entry in manifest["artifact_files"]}
    sources = {entry["path"] for entry in manifest["source_files"]}
    parent_directories = {str(parent) for name in allowed for parent in PurePosixPath(name).parents if str(parent) != "."}
    seen = set()
    restored = 0
    unchanged = 0
    if not archives and parts_manifest is None:
        raise ReleaseError("Provide complete .tar.gz archives or --parts-manifest PATH")
    # All split parts are validated before opening the tar stream or writing files.
    parts = verified_parts(parts_manifest) if parts_manifest is not None else None
    for archive in archives:
        if not str(archive).endswith(".tar.gz"):
            raise ReleaseError("Use complete .tar.gz archives. Rejoin .tar.gz.partNNN fragments before extract.")
        info = archive.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ReleaseError(f"Archive must be an independent regular file: {archive}")
    inputs = list(archives) + ([parts] if parts is not None else [])
    for archive in inputs:
        if isinstance(archive, list):
            stream = io.BufferedReader(ConcatenatedParts(archive), buffer_size=BUFFER_SIZE)
        else:
            stream = archive.open("rb")
        with stream, tarfile.open(fileobj=stream, mode="r|gz") as bundle:
            for member in bundle:
                # Never pass member names to extract()/extractall().
                raw_name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
                name = relative_path(raw_name)
                if member.isdir():
                    if name not in parent_directories:
                        raise ReleaseError(f"Unregistered archive directory rejected: {name}")
                    continue
                if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or getattr(member, "sparse", None):
                    raise ReleaseError(f"Archive link or special entry rejected: {name}")
                if name in sources or name not in allowed:
                    raise ReleaseError(f"Archive file is not an allowed artifact (source overwrite prohibited): {name}")
                if name in seen:
                    raise ReleaseError(f"Duplicate archive file rejected: {name}")
                seen.add(name)
                expected = allowed[name]
                if member.size != expected["size"]:
                    raise ReleaseError(f"Archive size mismatch: {name}")
                target = ensure_parent(root, name)
                temporary = None
                try:
                    with bundle.extractfile(member) as incoming:
                        with tempfile.NamedTemporaryFile(prefix=".release-restore-", dir=target.parent, delete=False) as outgoing:
                            temporary = Path(outgoing.name)
                            digest = hashlib.sha256()
                            count = 0
                            for block in iter(lambda: incoming.read(BUFFER_SIZE), b""):
                                count += len(block)
                                if count > expected["size"]:
                                    raise ReleaseError(f"Archive content exceeds registered size: {name}")
                                digest.update(block)
                                outgoing.write(block)
                            outgoing.flush()
                            os.fsync(outgoing.fileno())
                    if count != expected["size"] or digest.hexdigest() != expected["sha256"]:
                        raise ReleaseError(f"Archive SHA256 or size mismatch: {name}")
                    # Repeat destination checks after streaming and before committing.
                    target = checked_path(root, name, allow_missing=True)
                    if target.exists():
                        try:
                            digest_file(root, expected)
                        except ReleaseError:
                            pass
                        else:
                            temporary.unlink()
                            temporary = None
                            unchanged += 1
                            continue
                    os.chmod(temporary, 0o755 if member.mode & 0o111 else 0o644)
                    os.replace(temporary, target)
                    temporary = None
                    restored += 1
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
    if not seen:
        raise ReleaseError("Archive contained no registered artifact files")
    return {"restored": restored, "already_verified": unchanged, "scientific_execution": 0}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Release directory (defaults to this script's directory)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="List release mode, source files and missing external materials")
    check = commands.add_parser("check", help="Verify source hashes; optionally require every external artifact")
    check.add_argument("--require-data", action="store_true")
    lookup = commands.add_parser("lookup", help="Read saved chapter/figure/table/model associations")
    for name in ("figure", "table", "section", "chapter", "method"):
        lookup.add_argument("--" + name)
    extract = commands.add_parser("extract", help="Restore allowed artifacts from downloaded complete .tar.gz archives")
    extract.add_argument("archives", nargs="*", type=Path)
    extract.add_argument("--parts-manifest", type=Path,
                         help="parts.json beside continuous .tar.gz.part001..NNN files; validates and streams them without merging")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = args.root.resolve()
    try:
        manifest = load_manifest(root)
        if args.command == "inspect":
            return inspect_release(root, manifest)
        if args.command == "check":
            return check_release(root, manifest, args.require_data)
        if args.command == "lookup":
            return lookup_release(root, args)
        restored = extract_archives(root, manifest, args.archives, args.parts_manifest)
        print(json.dumps(restored, ensure_ascii=False, indent=2))
        print("Each restored file was size/SHA256 verified before atomic replacement. Run check --require-data for completeness.")
        return 0
    except (ReleaseError, OSError, tarfile.TarError, EOFError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
