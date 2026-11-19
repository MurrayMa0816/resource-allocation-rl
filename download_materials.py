#!/usr/bin/env python3
"""Download and verify the complete thesis materials from this repository's release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


API_VERSION = "2022-11-28"
BLOCK = 1024 * 1024
RESERVE = 64 * 1024 * 1024
DOWNLOAD_HOSTS = {
    "api.github.com", "github.com", "release-assets.githubusercontent.com",
    "objects.githubusercontent.com", "github-releases.githubusercontent.com",
}
SCHEMA = "complete-thesis-materials-release-v1"


class MaterialsError(Exception):
    """An input, integrity, or download failure safe to show without credentials."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value, field, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise MaterialsError("Invalid integer: " + field)
    return value


def relative_name(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(character) < 32 for character in value)):
        raise MaterialsError("Invalid package path")
    components = value.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise MaterialsError("Package path must stay inside the output directory: " + value)
    path = PurePosixPath(value)
    if path.is_absolute():
        raise MaterialsError("Absolute package path is forbidden: " + value)
    return path


def valid_digest(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise MaterialsError("Invalid SHA256: " + field)
    return value


def load_manifest(path):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MaterialsError("Cannot read a valid MATERIALS.json beside this script") from error
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise MaterialsError("Unsupported MATERIALS.json schema")
    repository = document.get("repository")
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise MaterialsError("Invalid GitHub repository in MATERIALS.json")
    tag = document.get("release_tag")
    if not isinstance(tag, str) or not tag or any(ord(char) < 32 for char in tag):
        raise MaterialsError("Invalid release tag in MATERIALS.json")
    package = document.get("package")
    parts = document.get("parts")
    if not isinstance(package, dict) or not isinstance(parts, list) or not parts:
        raise MaterialsError("Package and release parts are required")
    declared_files = integer(package.get("files"), "package.files", positive=True)
    declared_bytes = integer(package.get("bytes"), "package.bytes")
    entries, part_names, folded = {}, set(), set()
    for part in parts:
        if not isinstance(part, dict):
            raise MaterialsError("Invalid release part")
        name = part.get("name")
        parsed = relative_name(name)
        if len(parsed.parts) != 1 or name in part_names or not name.endswith(".zip"):
            raise MaterialsError("Release ZIP names must be unique single filenames")
        part_names.add(name)
        integer(part.get("bytes"), name + ".bytes", positive=True)
        valid_digest(part.get("sha256"), name)
        files = part.get("files")
        if not isinstance(files, list) or not files:
            raise MaterialsError("A release part must contain registered files: " + name)
        for item in files:
            if not isinstance(item, dict):
                raise MaterialsError("Invalid package file record")
            relative = item.get("path")
            relative_name(relative)
            if relative in entries:
                raise MaterialsError("A package file belongs to more than one part: " + relative)
            if os.name == "nt" and relative.casefold() in folded:
                raise MaterialsError("Package filenames collide on Windows: " + relative)
            folded.add(relative.casefold())
            integer(item.get("bytes"), relative + ".bytes")
            valid_digest(item.get("sha256"), relative)
            entries[relative] = item
    if len(entries) != declared_files or sum(item["bytes"] for item in entries.values()) != declared_bytes:
        raise MaterialsError("Package totals do not match the registered file list")
    # A registered file cannot also serve as another registered file's parent directory.
    for relative in entries:
        for parent in PurePosixPath(relative).parents:
            if parent.as_posix() in entries:
                raise MaterialsError("A file is also used as a directory: " + parent.as_posix())
    return document, entries


def output_path(root, relative):
    components = relative_name(relative).parts
    candidate = root.joinpath(*components)
    for path in (candidate, *candidate.parents):
        if path == root.parent:
            break
        if path.is_symlink():
            raise MaterialsError("Output must not contain symbolic links: " + str(path))
        if path == root:
            break
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise MaterialsError("Output path leaves the selected directory")
    return candidate


def matching_file(path, record):
    if path.is_symlink() or not path.is_file():
        raise MaterialsError("Expected an ordinary file: " + str(path))
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise MaterialsError("Existing output differs; it will not be overwritten: " + str(path))
    return True


def inspect_output(root, entries, *, require_complete=False):
    if root.is_symlink():
        raise MaterialsError("--output must not be a symbolic link")
    if root.exists() and not root.is_dir():
        raise MaterialsError("--output must be a directory")
    present = set()
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise MaterialsError("Output must not contain symbolic links: " + str(path))
            if path.is_dir():
                continue
            relative = path.relative_to(root).as_posix()
            if not path.is_file() or relative not in entries:
                raise MaterialsError("Unregistered file in output: " + relative)
            matching_file(path, entries[relative])
            present.add(relative)
    missing = set(entries) - present
    if require_complete and missing:
        raise MaterialsError("Output is incomplete: " + str(len(missing)) + " registered files missing")
    return present, missing


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def check_space(output, cache, remaining_bytes, largest_part):
    output_device = nearest_existing(output)
    cache_device = nearest_existing(cache)
    same_device = output_device.stat().st_dev == cache_device.stat().st_dev
    output_required = remaining_bytes + RESERVE + (largest_part if same_device else 0)
    if shutil.disk_usage(output_device).free < output_required:
        raise MaterialsError("Insufficient output disk space: need at least " + str(output_required) + " free bytes")
    if not same_device and shutil.disk_usage(cache_device).free < largest_part + RESERVE:
        raise MaterialsError("Insufficient cache disk space for one release part and reserve")


def checked_url(url):
    try:
        parsed = urllib.parse.urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.hostname in DOWNLOAD_HOSTS
                 and parsed.username is None and parsed.password is None and parsed.port in (None, 443))
    except ValueError:
        valid = False
    if not valid:
        raise MaterialsError("Rejected an unexpected GitHub download destination")
    return parsed


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        destination = urllib.parse.urljoin(request.full_url, newurl)
        parsed = checked_url(destination)
        redirected = super().redirect_request(request, fp, code, message, headers, destination)
        if redirected is not None and parsed.hostname != "api.github.com":
            for collection in (redirected.headers, redirected.unredirected_hdrs):
                for key in list(collection):
                    if key.lower() == "authorization":
                        del collection[key]
        return redirected


def get_token():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            process = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"],
                                     capture_output=True, text=True, timeout=20, check=False,
                                     env=gh_environment())
            token = process.stdout.strip() if process.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            token = None
    if token:
        token = token.strip()
        if not token or not token.isascii() or any(not 33 <= ord(character) <= 126 for character in token):
            raise MaterialsError("GitHub token must contain only the token, without whitespace")
    return token


def gh_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith("GIT_TRACE") and key not in {"GH_DEBUG", "GIT_CURL_VERBOSE"}}


class GhResponse:
    """A bounded-memory gh api stdout stream; credentials stay inside gh."""
    def __init__(self, url, *, binary=False):
        parsed = checked_url(url)
        if parsed.hostname != "api.github.com":
            raise MaterialsError("GitHub CLI transport only accepts the GitHub API host")
        accept = "application/octet-stream" if binary else "application/vnd.github+json"
        try:
            self.process = subprocess.Popen(
                ["gh", "api", "--method", "GET", "-H", "Accept: " + accept,
                 "-H", "X-GitHub-Api-Version: " + API_VERSION, url],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=gh_environment())
        except OSError:
            raise MaterialsError("Cannot start the authenticated GitHub CLI transport") from None
        self.headers = {}
        self.eof = False

    def __enter__(self):
        return self

    def read(self, size=-1):
        try:
            value = self.process.stdout.read(size)
        except OSError:
            self.close(terminate=True)
            raise MaterialsError("GitHub CLI download stream was interrupted") from None
        if not value or size < 0 or len(value) < size:
            self.eof = True
            try:
                result = self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.close(terminate=True)
                raise MaterialsError("GitHub CLI did not finish its download") from None
            if result:
                raise MaterialsError("GitHub CLI could not read the release; check gh authentication and repository access")
        return value

    def close(self, *, terminate=False):
        process = self.process
        try:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None and terminate:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            # No process stderr or environment is ever echoed into diagnostics.
            pass

    def __exit__(self, kind, error, traceback):
        self.close(terminate=kind is not None or not self.eof)
        return False


def request(url, token, *, binary=False):
    parsed = checked_url(url)
    headers = {"User-Agent": "thesis-materials-downloader", "Accept": "application/octet-stream" if binary else "application/vnd.github+json"}
    if parsed.hostname == "api.github.com":
        headers["X-GitHub-Api-Version"] = API_VERSION
        if token:
            headers["Authorization"] = "Bearer " + token
    opener = urllib.request.build_opener(SafeRedirect())
    try:
        return opener.open(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as error:
        if (not token and parsed.hostname == "api.github.com"
                and error.code in (401, 403, 404) and shutil.which("gh")):
            error.close()
            return GhResponse(url, binary=binary)
        raise


def api_json(url, token):
    try:
        with request(url, token) as response:
            content = response.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise MaterialsError("GitHub metadata response is unexpectedly large")
        return json.loads(content)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            raise MaterialsError("GitHub could not read this release (HTTP " + str(error.code)
                                 + "). For a private repository, set GH_TOKEN/GITHUB_TOKEN or sign in using gh auth login.") from None
        raise MaterialsError("GitHub metadata request failed (HTTP " + str(error.code) + ")") from None
    except (urllib.error.URLError, OSError, ValueError):
        raise MaterialsError("Cannot read GitHub release metadata; check network access and try again") from None


def release_assets(document, token):
    base = "https://api.github.com/repos/" + document["repository"]
    tag = urllib.parse.quote(document["release_tag"], safe="")
    release = api_json(base + "/releases/tags/" + tag, token)
    if not isinstance(release, dict) or type(release.get("id")) is not int:
        raise MaterialsError("GitHub did not return the registered release")
    assets = {}
    for page in range(1, 1001):
        batch = api_json(base + "/releases/" + str(release["id"]) + "/assets?per_page=100&page=" + str(page), token)
        if not isinstance(batch, list):
            raise MaterialsError("GitHub did not return release assets")
        for asset in batch:
            if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
                raise MaterialsError("GitHub returned invalid release asset metadata")
            name = asset["name"]
            if name in assets:
                raise MaterialsError("Duplicate GitHub release asset: " + name)
            assets[name] = asset
        if len(batch) < 100:
            break
    else:
        raise MaterialsError("Too many release asset pages")
    for part in document["parts"]:
        asset = assets.get(part["name"])
        if asset is None or type(asset.get("id")) is not int or asset["id"] <= 0 or asset.get("size") != part["bytes"]:
            raise MaterialsError("Release asset is missing or has the wrong size: " + part["name"])
        if asset.get("digest") not in (None, "sha256:" + part["sha256"]):
            raise MaterialsError("GitHub release asset digest differs: " + part["name"])
    return assets


def download_part(document, part, asset, cache, token):
    target = cache / part["name"]
    if target.is_file() and not target.is_symlink() and target.stat().st_size == part["bytes"] and sha256(target) == part["sha256"]:
        return target
    url = "https://api.github.com/repos/" + document["repository"] + "/releases/assets/" + str(asset["id"])
    last = None
    for attempt in range(3):
        temporary = None
        try:
            handle = tempfile.NamedTemporaryFile(prefix="materials-", suffix=".download.part", dir=cache, delete=False)
            temporary = Path(handle.name)
            digest, count = hashlib.sha256(), 0
            with handle, request(url, token, binary=True) as response:
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        correct_length = int(length) == part["bytes"]
                    except ValueError:
                        correct_length = False
                    if not correct_length:
                        raise MaterialsError("Downloaded part length differs: " + part["name"])
                while True:
                    block = response.read(BLOCK)
                    if not block:
                        break
                    count += len(block)
                    if count > part["bytes"]:
                        raise MaterialsError("Downloaded part exceeds its registered size: " + part["name"])
                    digest.update(block)
                    handle.write(block)
            if count != part["bytes"] or digest.hexdigest() != part["sha256"]:
                raise MaterialsError("Downloaded part SHA256 or size differs: " + part["name"])
            temporary.replace(target)
            return target
        except urllib.error.HTTPError as error:
            last = MaterialsError("GitHub asset download failed (HTTP " + str(error.code) + "): " + part["name"])
            if error.code not in (429, 500, 502, 503, 504):
                raise last from None
        except (urllib.error.URLError, OSError):
            last = MaterialsError("Release part download interrupted: " + part["name"])
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        if attempt < 2:
            time.sleep(1 + attempt)
    raise last


def extract_part(archive, part, output):
    expected = {item["path"]: item for item in part["files"]}
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        names = [member.filename for member in members]
        if len(set(names)) != len(names):
            raise MaterialsError("Duplicate ZIP entries: " + part["name"])
        for member in members:
            relative_name(member.filename)
            relative_name(member.orig_filename)
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or member.flag_bits & 1:
                raise MaterialsError("ZIP must contain only unencrypted ordinary files: " + member.filename)
            if member.filename not in expected or member.file_size != expected[member.filename]["bytes"]:
                raise MaterialsError("Unregistered ZIP member or wrong size: " + member.filename)
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise MaterialsError("ZIP member must use STORE or DEFLATE compression: " + member.filename)
        if set(names) != set(expected):
            raise MaterialsError("ZIP file set differs from MATERIALS.json: " + part["name"])
        for member in members:
            record = expected[member.filename]
            target = output_path(output, member.filename)
            if target.exists():
                matching_file(target, record)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            output_path(output, member.filename)
            temporary = None
            try:
                handle = tempfile.NamedTemporaryFile(prefix=".materials-", suffix=".part", dir=target.parent, delete=False)
                temporary = Path(handle.name)
                digest, count = hashlib.sha256(), 0
                with handle, package.open(member) as source:
                    while True:
                        block = source.read(BLOCK)
                        if not block:
                            break
                        count += len(block)
                        if count > record["bytes"]:
                            raise MaterialsError("Extracted file exceeds registered size: " + member.filename)
                        digest.update(block)
                        handle.write(block)
                if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise MaterialsError("Extracted file SHA256 differs: " + member.filename)
                os.chmod(temporary, 0o644 | ((member.external_attr >> 16) & 0o111))
                if target.exists():
                    matching_file(target, record)
                    temporary.unlink()
                else:
                    temporary.replace(target)
                temporary = None
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()


def restore(document, entries, output, cache):
    present, missing = inspect_output(output, entries)
    if not missing:
        return len(present)
    pending = [part for part in document["parts"] if any(item["path"] in missing for item in part["files"])]
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise MaterialsError("Cache and output directories must be separate")
    check_space(output, cache, sum(entries[path]["bytes"] for path in missing), max(part["bytes"] for part in pending))
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    token = get_token()
    assets = release_assets(document, token)
    for index, part in enumerate(pending, 1):
        print("Downloading " + part["name"] + " (" + str(index) + "/" + str(len(pending)) + ")", flush=True)
        archive = download_part(document, part, assets[part["name"]], cache, token)
        try:
            extract_part(archive, part, output)
        finally:
            # Only one complete ZIP needs to occupy cache space at a time.
            archive.unlink(missing_ok=True)
        print("Verified " + part["name"], flush=True)
    complete, missing = inspect_output(output, entries, require_complete=True)
    return len(complete)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory, or an incomplete restore whose existing files match the manifest")
    parser.add_argument("--cache", type=Path, help="Optional temporary cache location; ZIPs are removed after extraction")
    parser.add_argument("--verify-only", action="store_true", help="Check an already complete directory without network or credentials")
    args = parser.parse_args(argv)
    document, entries = load_manifest(Path(__file__).resolve().with_name("MATERIALS.json"))
    if args.output.expanduser().is_symlink():
        raise MaterialsError("--output must not be a symbolic link")
    output = args.output.expanduser().resolve()
    if args.verify_only:
        present, missing = inspect_output(output, entries, require_complete=True)
        count = len(present)
    elif args.cache is not None:
        if args.cache.expanduser().is_symlink():
            raise MaterialsError("--cache must not be a symbolic link")
        fingerprint = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:16]
        cache = args.cache.expanduser().resolve() / ("thesis-materials-" + fingerprint)
        count = restore(document, entries, output, cache)
    else:
        with tempfile.TemporaryDirectory(prefix="thesis-materials-") as temporary:
            count = restore(document, entries, output, Path(temporary))
    print(json.dumps({"status": "PASS", "files": count, "bytes": document["package"]["bytes"],
                      "output": str(output), "model_loading": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MaterialsError, zipfile.BadZipFile) as error:
        print("Error: " + str(error), file=sys.stderr)
        raise SystemExit(1)
    except OSError as error:
        print("Error: file operation failed: " + (error.strerror or "check file permissions and available disk space"), file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Stopped. Verified completed files are preserved; run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
