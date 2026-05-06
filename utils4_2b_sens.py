import math
import torch
from typing import Optional, Dict
from .config import QATConfig
from .param_filter import QuantParamSelector, iter_named_quant_params
from .quant_ops_2b import quantize_to_levels
from .sensitivity import SensitivityEMA


@torch.no_grad()
def _collect_sensitivity_weights(
    model,
    *,
    qat: QATConfig,
    selector: QuantParamSelector,
    sens_ema: Optional[SensitivityEMA],
) -> Dict[str, torch.Tensor]:
    """Return one scalar sensitivity weight per selected parameter.

    Larger gradient-square EMA means the parameter is more sensitive, so we
    assign a smaller quantization pull. We normalize by the mean across selected
    tensors and clamp for stability.
    """
    if sens_ema is None or (not getattr(qat, "sens_enable", True)):
        return {}

    names = []
    vals = []
    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=qat.include_substrings,
        exclude_substrings=qat.exclude_substrings,
    ):
        s = sens_ema.get(name, p.device)
        if s is None:
            continue
        names.append(name)
        vals.append(s.float().clamp_min(getattr(qat, "sens_eps", 1e-8)))

    if not vals:
        return {}

    S = torch.stack(vals)
    mean_s = S.mean().clamp_min(getattr(qat, "sens_eps", 1e-8))

    # rel > 1 means more sensitive than average.
    # weight = rel^(-power), so sensitive tensors get smaller quant pull.
    power = float(getattr(qat, "sens_power", 0.5))
    w_min = float(getattr(qat, "sens_w_min", 0.2))
    w_max = float(getattr(qat, "sens_w_max", 5.0))

    weights = {}
    for name, s in zip(names, vals):
        rel = (s / mean_s).clamp_min(getattr(qat, "sens_eps", 1e-8))
        w = rel.pow(-power).clamp(w_min, w_max).detach()
        weights[name] = w
    return weights


@torch.no_grad()
def prepare_theory_matched_quant_update_with_sensitivity(
    model,
    optimizer,
    *,
    qat: QATConfig,
    selector: QuantParamSelector,
    dual_lambda: float,
    sens_ema: Optional[SensitivityEMA] = None,
    warmup: bool = False,
) -> Optional[Dict[str, object]]:
    """Theory-matched AdamFX-style update with optional sensitivity weighting.

    Base update:
        x <- x - lr * (dist_scale * (x - q(x)) + lambda * Adam(g))

    Sensitivity update:
        x <- x - lr * (dist_scale * s_weight * (x - q(x)) + lambda * Adam(g))

    where s_weight is smaller for tensors with larger gradient-square EMA.
    During warmup, the quantization pull is disabled by setting dist_scale_eff=0.
    """
    if not qat.enabled:
        return None

    group_map = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    sens_weights = _collect_sensitivity_weights(
        model, qat=qat, selector=selector, sens_ema=sens_ema
    )

    prepared = []
    debug_weights = []

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=qat.include_substrings,
        exclude_substrings=qat.exclude_substrings,
    ):
        if not p.requires_grad:
            continue

        p_old = p.data.detach().clone()
        q_old = quantize_to_levels(p_old, qat.quant_levels_w)
        prox_term = (p_old - q_old).float()

        task_delta = torch.zeros_like(p_old, dtype=torch.float32)
        lr = 0.0

        if p.grad is not None:
            g = p.grad.detach().float()
            group = group_map[id(p)]

            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))

            state = optimizer.state[p]
            exp_avg = state.get("exp_avg", torch.zeros_like(g)).detach().float()
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(g)).detach().float()
            step_raw = state.get("step", 0)
            step_t = int(step_raw.item() if torch.is_tensor(step_raw) else step_raw) + 1

            exp_avg_next = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1)
            exp_avg_sq_next = exp_avg_sq.mul(beta2).addcmul(g, g, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step_t
            bias_correction2 = 1.0 - beta2 ** step_t
            denom = exp_avg_sq_next.sqrt().div(math.sqrt(bias_correction2)).add_(eps)

            adam_term = exp_avg_next.div(denom) / bias_correction1

            if wd != 0.0:
                adam_term = adam_term + wd * p_old.detach().float()

            task_delta = adam_term

        sens_weight = sens_weights.get(
            name,
            torch.ones((), device=p_old.device, dtype=torch.float32),
        ).to(device=p_old.device, dtype=torch.float32)

        dist_scale_eff = 0.0 if warmup else float(qat.dist_scale)

        desired = p_old - lr * (
            dist_scale_eff * sens_weight * prox_term
            + float(dual_lambda) * task_delta
        )

        debug_weights.append(float(sens_weight.detach().cpu().item()))
        prepared.append((p, desired.to(dtype=p.dtype), p_old, q_old))

    if debug_weights:
        dbg = {
            "sens_w_min": min(debug_weights),
            "sens_w_mean": sum(debug_weights) / len(debug_weights),
            "sens_w_max": max(debug_weights),
        }
    else:
        dbg = {}

    return {"prepared": prepared, "debug": dbg}


@torch.no_grad()
def apply_prepared_quant_update(pkg: Optional[Dict[str, object]]) -> Optional[Dict[str, float]]:
    if pkg is None:
        return None
    for p, desired, _, _ in pkg["prepared"]:
        p.data.copy_(desired)
    return pkg["debug"]
