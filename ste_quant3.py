from __future__ import annotations
import torch

def quantize_to_fixed_grid(w: torch.Tensor, n_bits: int = 4, step: float = 0.05 / 7) -> torch.Tensor:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    L = (2 ** (n_bits - 1)) - 1
    clip = L * step

    q = torch.round(w / step) * step
    q = torch.clamp(q, -clip, clip)
    return q

def ste_quantize_weight_with_mask(
    w: torch.Tensor,
    mask: torch.Tensor,
    n_bits: int = 4,
    step: float = 0.05 / 7,
) -> torch.Tensor:
    q = quantize_to_fixed_grid(w, n_bits=n_bits, step=step)

    mask = mask.to(device=w.device, dtype=w.dtype)

    return w + (mask * (q - w)).detach()
