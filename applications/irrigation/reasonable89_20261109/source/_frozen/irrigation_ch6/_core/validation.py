"""Offline integrity checks for the packaged Hongmiao93 inputs."""
from __future__ import annotations

import hashlib
from importlib import resources
import json


def verify_packaged_inputs():
    """Verify every published input digest and load the complete default case."""
    base = resources.files("irrigation_ch6._core").joinpath("data", "hongmiao93")
    manifest = json.loads(base.joinpath("manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "hongmiao93_packaged_v4":
        raise ValueError("invalid Hongmiao93 input manifest")
    hashes = manifest.get("files_sha256", {})
    required = {"regions_93.csv", "region_parameters_frozen.csv", "weather.csv",
                "weather_prior.json", "quota.csv", "case_model.json",
                "budget_contract.json", "source_registry.csv",
                "reference_irrigation_demand.csv"}
    packaged = {p.name for p in base.iterdir() if p.is_file() and p.name != "manifest.json"}
    if not required <= set(hashes) or set(hashes) != packaged:
        raise ValueError("Hongmiao93 manifest file set is incomplete or unexpected")
    for path, expected in hashes.items():
        if "/" in path or "\\" in path or path in ("", ".", ".."):
            raise ValueError("invalid packaged input path")
        actual = hashlib.sha256(base.joinpath(path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"packaged input checksum mismatch: {path}")
    from .direct_scenarios import load_default_direct_scenario
    scene = load_default_direct_scenario()
    if scene.N != manifest["n"] or scene.T != manifest["physical_days"]:
        raise ValueError("Hongmiao93 dimensions disagree with manifest")
    if abs(float(scene.areas_ha.sum()) - manifest["area_ha"]) > 1e-9:
        raise ValueError("Hongmiao93 area disagrees with manifest")
    from . import __version__
    return {"input_files": len(hashes), "default_version": __version__,
            "regions": scene.N, "crops_per_region": 1,
            "physical_days": scene.T, "decision_steps": manifest["decision_days"],
            "default_profile": scene.metadata["parameter_profile"],
            "daily_quota_m3": scene.metadata["daily_quota_m3"],
            "seasonal_budget_m3": scene.metadata["seasonal_budget_m3"],
            "verified": True,
            "identity": "published Hongmiao service-unit rice areas; transferred physical model and research controls"}
