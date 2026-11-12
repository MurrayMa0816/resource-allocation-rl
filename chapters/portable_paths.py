"""Project and artifact paths shared by the chapter entry points."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def final_data_root():
    first = os.environ.get("RESOURCE_ALLOCATION_DATA_ROOT")
    second = os.environ.get("FINAL_DATA_ROOT")
    if first and second and Path(first).expanduser().resolve() != Path(second).expanduser().resolve():
        raise ValueError("RESOURCE_ALLOCATION_DATA_ROOT and FINAL_DATA_ROOT must agree")
    return Path(first or second).expanduser().resolve() if first or second else project_root()


@lru_cache(maxsize=1)
def _paths():
    path = project_root() / "chapters/runtime_paths.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _mapped(value, entries):
    if value in entries:
        return Path(entries[value])
    for parent in Path(value).parents:
        prefix = str(parent)
        if prefix in entries:
            return Path(entries[prefix]) / Path(value).relative_to(parent)
    return None


def chapter_data_root(chapter):
    number = int(str(chapter).removeprefix("chapter"))
    return final_data_root() / "chapters" / f"chapter{number:02d}"


def data_path(value):
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return resolve_path(candidate)
    if len(candidate.parts) > 1 and candidate.parts[0] == "experiments":
        return chapter_data_root(candidate.parts[1]) / Path(*candidate.parts[2:])
    mapped = _mapped(str(candidate), _paths().get("data", {}))
    return final_data_root() / (mapped if mapped is not None else candidate)


def resolve_path(value):
    candidate = Path(value).expanduser()
    text = str(candidate)
    code = project_root()
    data = final_data_root()
    if candidate.is_absolute() and candidate.is_relative_to(code):
        relative = candidate.relative_to(code).as_posix()
        relocated = _mapped(relative, _paths().get("source", {}))
        if relocated is not None:
            return code / relocated
    source = _mapped(text, _paths().get("source", {}))
    if source is not None:
        return code / source
    artifact = _mapped(text, _paths().get("data", {}))
    if artifact is not None:
        return data / artifact
    old_project = "/root/gamewar/resource_allocation_rl"
    if text == old_project:
        return code
    if text.startswith(old_project + "/"):
        relative = Path(text[len(old_project) + 1:])
        source = _mapped(str(relative), _paths().get("source", {}))
        if source is not None:
            return code / source
        local = code / relative
        return local if local.exists() else data / relative
    if text in ("/hy-tmp", "/hy-tmp/resource_allocation_final"):
        return data
    if text.startswith("/hy-tmp/resource_allocation_final/"):
        return data / text.removeprefix("/hy-tmp/resource_allocation_final/")
    if not candidate.is_absolute() and candidate.parts and candidate.parts[0] == "experiments":
        return data_path(candidate)
    if candidate.is_absolute() and candidate.is_relative_to(code):
        return candidate
    if candidate.is_absolute() and candidate.is_relative_to(data):
        return candidate
    if text.startswith(("/hy-tmp/", "/root/resource_allocation_run_storage_20261101/")):
        raise FileNotFoundError("No artifact path is registered for: " + text)
    return candidate


def eurosat_root():
    data = final_data_root()
    modern = data / "chapters/chapter02/data/eurosat_rgb"
    legacy = data / "datasets/eurosat_rgb"
    return modern if modern.exists() or not legacy.exists() else legacy


def chapter_source_paths(chapter):
    number = int(chapter)
    root = project_root()
    chapter_root = root / "chapters" / f"chapter{number:02d}"
    paths = [root / "chapters", chapter_root, chapter_root / "source"]
    if number == 3:
        paths.append(root / "chapters/chapter02/source")
    if number == 5:
        internal = chapter_root / "source/_internal/src"
        paths += [internal, internal / "chapter4_credit_assignment",
                  internal / "chapter4_sample_efficiency_v5", internal / "chapter5_multiobjective"]
    for relative in _paths().get("import_paths", {}).get(f"chapter{number:02d}", []):
        paths.append(root / relative)
    return list(dict.fromkeys(p for p in paths if p.is_dir()))


def chapter_environment(chapter):
    env = dict(os.environ)
    env["RESOURCE_ALLOCATION_PROJECT_ROOT"] = str(project_root())
    env["RESOURCE_ALLOCATION_DATA_ROOT"] = str(final_data_root())
    env["FINAL_DATA_ROOT"] = str(final_data_root())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(map(str, chapter_source_paths(chapter)))
    env["CH5_FULL_SPEND_DATA_ROOT"] = str(eurosat_root())
    if int(chapter) == 5:
        env["CH5_WORKSPACE"] = str(project_root() / "chapters/chapter05/source/_internal/src")
    return env
