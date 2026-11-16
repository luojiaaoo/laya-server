"""Validate the requested inference device without loading a model."""

import re


def validate_device(device: str | None) -> str:
    device = device or "cuda"
    if device in ("cpu", "mps"):
        return device
    if not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("LAYA_DEVICE must be cpu, mps, cuda, or cuda:<index>")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install CUDA-enabled PyTorch, check the NVIDIA "
            "driver and container GPU access (--gpus all), or set LAYA_DEVICE=cpu."
        )
    index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"LAYA_DEVICE={device} refers to an unavailable CUDA device")
    return device
