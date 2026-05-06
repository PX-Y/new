from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class QATConfig:
    # master switch(False = no quant)
    enabled: bool = True

    n_bits_w: int = 4
    step_w: float = 0.035/7
    atol: float = 0.001

    # which params to quantize
    selector_mode: str = "gpt2_custom"
    include_substrings: Optional[Tuple[str, ...]] = None
    exclude_substrings: Tuple[str, ...] = ("bias", "norm", "ln_", "wte", "wpe", "lm_head")


    dist_scale: float = 56

    # ---- Lagrange constraint: lambda*(f - beta) ----
    use_lagrange: bool = True
    beta: float = 3
    dual_lr: float = 7e-5
    lambda_init: float = 0.5
    lambda_max: Optional[float] = 30

    # PI controller for lambda
#    lambda_use_pi: bool = True
#    lambda_kp: float = 1.0
#    lambda_ki: float = 0.1
#    lambda_i_clamp: float = 10.0

    # ---- gamma controller ----
#    q_target: float = 0.95
#    gamma_init: float = 0.0
#    gamma_lr: float = 0.1
#    gamma_max: float = 3.0
#    gamma_start_step: int = 200
#    qrate_every: int = 50
#    qrate_ema_momentum: float = 0.7

    # ---- sensitivity EMA ----
#    sens_enable: bool = True
#    sens_ema: float = 0.95

    #--- hess ---
    hess_enable: bool = True
    hess_inverse: bool = True
    hess_rel_min: float = 0.2
    hess_rel_max: float = 5.0
