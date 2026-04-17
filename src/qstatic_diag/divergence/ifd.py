from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import ActivationRecord, LayerDivergence, LayerKind, LayerMeta

log = get_logger(__name__)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Primitive divergence functions
# ---------------------------------------------------------------------------


def _safe_norm(t: Tensor) -> float:
    """Frobenius / L2 norm with NaN/Inf guard."""
    val = t.float().norm().item()
    if not math.isfinite(val):
        return 0.0
    return val


def compute_fc_divergence(
    pre_act: Tensor,   # z_l = W_l h_{l-1} + b_l     shape: (B, n_l)
    post_act: Tensor,  # h_l = sigma(z_l)             shape: (B, n_l)
    weight: Tensor,    # W_l                           shape: (n_l, n_{l-1})
) -> float:
    r"""
    Fully-connected divergence:

        D_FC^l = ||J(h_l)||_F * ||h_l||_2 * ||W_l||_F

    For ReLU: ||J(h_l)||_F = sqrt( #active neurons ).
    For other activations: approximate via finite differences or identity.
    """
    # Approximate Jacobian norm as fraction of active units
    active_fraction = (post_act.float().abs() > EPS).float().mean().item()
    j_norm = math.sqrt(max(active_fraction * post_act.shape[-1], EPS))

    h_norm = _safe_norm(post_act)
    w_norm = _safe_norm(weight)

    return j_norm * h_norm * w_norm


def compute_conv_divergence(
    activation: Tensor,  # A_l shape: (B, C_l, H_l, W_l)
    weight: Tensor,      # W_l shape: (C_l, C_{l-1}, k, k)
) -> float:
    r"""
    Convolutional divergence:
        D_conv^l = (1/|Omega_l|) * ||A_l||_F * ||W_l||_F
    """
    omega = activation.numel() / activation.shape[0]  # per-sample volume
    a_norm = _safe_norm(activation)
    w_norm = _safe_norm(weight)
    return (a_norm * w_norm) / (omega + EPS)


def compute_attention_divergence(
    head_outputs: Optional[Tensor],  # (H, B, S, d_v)  or fallback to output
    output: Tensor,                  # (B, S, E)  full output
    wq: Tensor, wk: Tensor, wv: Tensor,  # per-head projection weight norms
    num_heads: int,
    seq_len: int,
) -> float:
    r"""
    Self-attention divergence:

        D_attn = sum_h (1/n) * ||A^h||_F * (||W_Q^h||_F + ||W_K^h||_F + ||W_V^h||_F)
    """
    wq_norm = _safe_norm(wq)
    wk_norm = _safe_norm(wk)
    wv_norm = _safe_norm(wv)
    proj_sum = wq_norm + wk_norm + wv_norm

    if head_outputs is not None and head_outputs.ndim == 4:
        # head_outputs: (H, B, S, d_v)
        total = 0.0
        for h in range(head_outputs.shape[0]):
            a_norm = _safe_norm(head_outputs[h])
            total += (a_norm / (seq_len + EPS)) * (proj_sum / num_heads)
        return total
    else:
        # Fallback: use full output norm
        a_norm = _safe_norm(output)
        return (a_norm / (seq_len + EPS)) * proj_sum


def compute_residual_divergence(
    residual_output: Tensor,  # F(T_l)  — branch output only
    identity: Tensor,         # T_l     — shortcut
) -> float:
    r"""
    Residual divergence:

        D_res^l = ||F(T_l)||_2 / (||T_l||_2 + eps)
    """
    f_norm = _safe_norm(residual_output)
    id_norm = _safe_norm(identity)
    return f_norm / (id_norm + EPS)


def compute_general_divergence(
    t_l: Tensor,    # activation at layer l
    t_l1: Tensor,   # activation at layer l+1
    w_l: Tensor,    # weights at layer l
    w_l1: Tensor,   # weights at layer l+1
    t_lm1: Optional[Tensor] = None,  # activation at layer l-1 (optional)
) -> float:
    r"""
    General divergence formulation:

        D_l = (||T_{l+1} - T_l|| / (||T_l|| + eps))
              * ( ||W_{l+1} T_l|| - ||W_l T_{l-1}|| )
    """
    # --- relative activation change ---
    diff_norm = _safe_norm(t_l1.float() - t_l.float().reshape(t_l1.shape) if t_l.shape != t_l1.shape else t_l1.float() - t_l.float())
    base_norm = _safe_norm(t_l)
    rel_change = diff_norm / (base_norm + EPS)

    # --- weighted transformation difference ---
    if t_lm1 is not None:
        try:
            wl1_tl = torch.matmul(
                w_l1.float().view(w_l1.shape[0], -1),
                t_l.float().view(t_l.shape[0], -1).T,
            )
            wl_tlm1 = torch.matmul(
                w_l.float().view(w_l.shape[0], -1),
                t_lm1.float().view(t_lm1.shape[0], -1).T,
            )
            weight_diff = _safe_norm(wl1_tl) - _safe_norm(wl_tlm1)
        except Exception:
            weight_diff = _safe_norm(w_l1) - _safe_norm(w_l)
    else:
        weight_diff = _safe_norm(w_l1) - _safe_norm(w_l)

    return rel_change * weight_diff


# ---------------------------------------------------------------------------
# Per-filter/head divergence helpers
# ---------------------------------------------------------------------------


def per_filter_divergence(
    activation: Tensor,  # (B, C, ...) — post-activation
    weight: Tensor,      # (C_out, ...) — kernel or weight matrix
    kind: LayerKind,
) -> List[float]:
    """Return a divergence value per output filter / channel."""
    results: List[float] = []
    n_filters = activation.shape[1] if activation.ndim > 2 else activation.shape[-1]

    for f in range(n_filters):
        if activation.ndim > 2:
            act_f = activation[:, f]  # (B, H, W) or (B,)
        else:
            act_f = activation[:, f]  # (B,)

        w_f = weight[f] if f < len(weight) else weight[-1]

        a_norm = _safe_norm(act_f)
        w_norm = _safe_norm(w_f)
        omega = act_f.numel() / activation.shape[0]
        div = (a_norm * w_norm) / (omega + EPS)
        results.append(div)

    return results


# ---------------------------------------------------------------------------
# IFD Engine
# ---------------------------------------------------------------------------


class IFDEngine:
    """
    Orchestrates IFD computation across all layers and all diagnostic samples.

    Workflow::

        engine = IFDEngine(model, metas)
        layer_divs = engine.compute(activation_store)
    """

    def __init__(
        self,
        model: nn.Module,
        metas: List[LayerMeta],
    ) -> None:
        self.model = model
        self.metas = metas
        self._param_cache = self._build_param_cache()

    def _build_param_cache(self) -> Dict[str, Dict[str, Optional[Tensor]]]:
        """Cache weight/bias tensors per named module."""
        cache: Dict[str, Dict[str, Optional[Tensor]]] = {}
        name_to_mod = dict(self.model.named_modules())
        for meta in self.metas:
            mod = name_to_mod.get(meta.name)
            if mod is None:
                cache[meta.name] = {}
                continue
            params: Dict[str, Optional[Tensor]] = {}
            w = getattr(mod, "weight", None)
            b = getattr(mod, "bias", None)
            params["weight"] = w.detach().cpu() if w is not None else None
            params["bias"] = b.detach().cpu() if b is not None else None
            # MHA projection weights
            if isinstance(mod, nn.MultiheadAttention):
                params["in_proj_weight"] = (
                    mod.in_proj_weight.detach().cpu()
                    if mod.in_proj_weight is not None
                    else None
                )
            cache[meta.name] = params
        return cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        activation_store,  # ActivationStore
    ) -> List[LayerDivergence]:
        """
        Compute IFD for every layer.

        Parameters
        ----------
        activation_store:
            Populated by :func:`qstatic_diag.tracing.hooks.capture_activations`.

        Returns
        -------
        List of :class:`LayerDivergence` in the same order as ``self.metas``.
        """
        layer_divs: List[LayerDivergence] = []

        layer_names = [m.name for m in self.metas]

        for i, meta in enumerate(self.metas):
            records = activation_store.get(meta.name)
            if not records:
                log.debug("No activations for layer '%s', skipping.", meta.name)
                continue

            per_sample_divs: List[float] = []
            per_filter_divs_accum: Optional[List[List[float]]] = None

            # Previous layer activations
            prev_records = activation_store.get(layer_names[i - 1]) if i > 0 else []

            for s_idx, record in enumerate(records):
                act = record.output  # post-activation

                params = self._param_cache.get(meta.name, {})
                weight = params.get("weight")

                if weight is None:
                    # No weight → use activation norm as proxy
                    d = _safe_norm(act) / (act.numel() / max(act.shape[0], 1) + EPS)
                elif meta.kind == LayerKind.CONV:
                    d = compute_conv_divergence(act, weight)
                elif meta.kind == LayerKind.FC:
                    pre = record.pre_activation if record.pre_activation is not None else act
                    d = compute_fc_divergence(pre, act, weight)
                elif meta.kind == LayerKind.ATTENTION:
                    ip = params.get("in_proj_weight")
                    d_k = meta.head_dim or 64
                    num_h = meta.num_heads or 1
                    embed = meta.in_features or act.shape[-1]
                    if ip is not None:
                        wq = ip[:embed, :]
                        wk = ip[embed : 2 * embed, :]
                        wv = ip[2 * embed :, :]
                    else:
                        wq = wk = wv = weight
                    seq_len = act.shape[1] if act.ndim == 3 else 1
                    d = compute_attention_divergence(
                        record.head_outputs, act, wq, wk, wv, num_h, seq_len
                    )
                elif meta.kind == LayerKind.RESIDUAL:
                    residual_out = record.residual_branch_output
                    if residual_out is not None:
                        d = compute_residual_divergence(residual_out, act)
                    else:
                        d = _safe_norm(act) / (act.numel() / max(act.shape[0], 1) + EPS)
                else:
                    d = _safe_norm(act) / (act.numel() / max(act.shape[0], 1) + EPS)

                per_sample_divs.append(d)

                # Per-filter divergence (Conv + FC only)
                if meta.kind in (LayerKind.CONV, LayerKind.FC) and weight is not None:
                    pf = per_filter_divergence(act, weight, meta.kind)
                    if per_filter_divs_accum is None:
                        per_filter_divs_accum = [[] for _ in pf]
                    for fi, fval in enumerate(pf):
                        if fi < len(per_filter_divs_accum):
                            per_filter_divs_accum[fi].append(fval)

            if not per_sample_divs:
                continue

            mean_div = float(sum(per_sample_divs) / len(per_sample_divs))

            std_div = float(statistics.stdev(per_sample_divs)) if len(per_sample_divs) > 1 else 0.0

            pf_means: Optional[List[float]] = None
            if per_filter_divs_accum:
                pf_means = [
                    float(sum(vals) / len(vals)) if vals else 0.0
                    for vals in per_filter_divs_accum
                ]

            layer_divs.append(
                LayerDivergence(
                    layer_name=meta.name,
                    kind=meta.kind,
                    mean=mean_div,
                    std=std_div,
                    per_sample=per_sample_divs,
                    per_filter=pf_means,
                )
            )

        # Compute normalised contributions rho_l
        total = sum(ld.mean for ld in layer_divs) + EPS
        for ld in layer_divs:
            ld.rho = ld.mean / total

        return layer_divs
