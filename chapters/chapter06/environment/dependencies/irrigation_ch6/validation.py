"""Offline release/input verification; no downloads, policy fitting or training."""
from __future__ import annotations

from importlib import resources
import json
import platform

import gymnasium
import numpy as np

from ._util import digest, package_files
from ._core.validation import verify_packaged_inputs


def verify_release():
    release = resources.files("irrigation_ch6").joinpath("release_manifest.json")
    if not release.is_file():
        raise ValueError("A final release requires its packaged release manifest")
    manifest = json.loads(release.read_text(encoding="utf-8"))
    actual = package_files()
    if manifest.get("schema") != "irrigation-ch6-release-v1":
        raise ValueError("unsupported release manifest")
    if manifest.get("files_sha256") != actual or manifest.get("implementation_sha256") != digest(actual):
        raise ValueError("release source/data/attribution identity mismatch")
    physical = verify_packaged_inputs()
    return {"verified": True, "application_version": "1.0.0",
            "physical_case": "Hongmiao93-v2", "files": len(actual),
            "implementation_sha256": digest(actual), "physical_inputs": physical,
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "gymnasium": gymnasium.__version__},
            "formal_training_started": False}

