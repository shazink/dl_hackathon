"""Strict deterministic execution and serializable RNG state."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_loader_generator(seed: int, experience: int, epoch: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + experience * 100_000 + epoch)
    return generator


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint requires CUDA RNG state, but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
