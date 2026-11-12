"""Execution controls only: explicit devices and complete sampling RNG state."""
from __future__ import annotations

import random
import zipfile
import numpy as np
import torch


def resolve_device(device="cpu"):
    value = torch.device(device)
    if value.type == "cpu" and value.index is None:
        return "cpu"
    if value.type != "cuda":
        raise ValueError("chapter-2 execution supports cpu or explicit cuda devices")
    index = 0 if value.index is None else value.index
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"requested CUDA device is unavailable: {value}")
    return f"cuda:{index}"


def capture_rng():
    result = dict(python=random.getstate(), numpy=np.random.get_state(),
                  torch=torch.get_rng_state())
    # Do not initialize CUDA for a CPU-only run merely to save a checkpoint.
    result["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return result


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value.get("cuda") is not None:
        if not torch.cuda.is_available() or len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("saved CUDA sampling streams require the same device count")
        torch.cuda.set_rng_state_all([s.cpu() for s in value["cuda"]])


def execution_metadata(device, cpu_threads):
    device = resolve_device(device)
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 128:
        raise ValueError("cpu_threads must be an integer between 1 and 128")
    return dict(device=device, cpu_threads=cpu_threads,
                arithmetic="float32; no AMP or reduced-mantissa TF32",
                torch_cuda=torch.version.cuda,
                accelerator=(torch.cuda.get_device_name(torch.device(device))
                             if device.startswith("cuda:") else None))


def apply_execution(device="cpu", cpu_threads=1):
    metadata = execution_metadata(device, cpu_threads)
    torch.set_num_threads(cpu_threads)
    # A hardware speed comparison must not silently change arithmetic precision.
    if metadata["device"].startswith("cuda:"):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return metadata


def select_device(requested, method, profile):
    """Resolve once before registration; never silently switch mid-run."""
    if requested != "auto":
        return resolve_device(requested)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def save_replay_archive(path, arrays):
    """Standard NPZ with level-1 DEFLATE: lossless, faster, modestly larger."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, array in arrays.items():
            if not name or "/" in name or "\\" in name:
                raise ValueError("invalid replay array name")
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
