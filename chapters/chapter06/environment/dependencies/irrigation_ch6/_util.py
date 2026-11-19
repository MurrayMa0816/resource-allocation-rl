"""Serialization and identities shared by the application, never policy inputs."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import hashlib
from importlib import resources
import json
from numbers import Real

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def digest(value):
    raw = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def real(value, name, *, minimum=0., maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its finite allowed range")
    return result


def package_files():
    """SHA of actual runtime sources/data, independent of install directory."""
    result = {}
    def walk(root, prefix=""):
        for p in sorted(root.iterdir(), key=lambda x: x.name):
            name = prefix + p.name
            if p.is_dir():
                if p.name != "__pycache__":
                    walk(p, name + "/")
            elif (p.name != "release_manifest.json" and
                  (p.name == "LICENSE" or p.name.endswith((".py", ".csv", ".json", ".rst", ".cff", ".txt")))):
                result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    walk(resources.files("irrigation_ch6"))
    return result

