from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from qstatic_diag.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared interface
# ---------------------------------------------------------------------------


class BaselineRanker:
    """Abstract interface for a baseline importance ranker."""

    name: str = "base"

    def rank_filters(
        self,
        model: nn.Module,
        diag_data: Tensor,
    ) -> Dict[str, List[float]]:
        """
        Returns
        -------
        dict mapping layer_name → list of per-filter importance scores
        (higher = more important, lower = more prunable).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomBaseline(BaselineRanker):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def rank_filters(self, model: nn.Module, diag_data: Tensor) -> Dict[str, List[float]]:
        scores: Dict[str, List[float]] = {}
        for name, module in model.named_modules():
            w = getattr(module, "weight", None)
            if w is None:
                continue
            n_filters = w.shape[0]
            scores[name] = torch.rand(n_filters, generator=self.rng).tolist()
        return scores


# ---------------------------------------------------------------------------
# Weight-norm baseline
# ---------------------------------------------------------------------------


class NormBaseline(BaselineRanker):
    """L2 norm of each filter's weight vector."""

    name = "weight_norm"

    def rank_filters(self, model: nn.Module, diag_data: Tensor) -> Dict[str, List[float]]:
        scores: Dict[str, List[float]] = {}
        for name, module in model.named_modules():
            w = getattr(module, "weight", None)
            if w is None:
                continue
            w_flat = w.detach().float().view(w.shape[0], -1)
            norms = w_flat.norm(dim=1).tolist()
            scores[name] = norms
        return scores


# ---------------------------------------------------------------------------
# Activation sparsity baseline
# ---------------------------------------------------------------------------


class ActivationSparsityBaseline(BaselineRanker):
    """
    Per-filter fraction of non-zero activations on the diagnostic set.
    """

    name = "activation_sparsity"

    def rank_filters(self, model: nn.Module, diag_data: Tensor) -> Dict[str, List[float]]:
        from qstatic_diag.tracing.hooks import capture_activations

        scores: Dict[str, List[float]] = {}
        model.eval()
        with torch.no_grad():
            with capture_activations(model) as store:
                model(diag_data)

        for layer_name in store.all_names():
            records = store.get(layer_name)
            if not records:
                continue
            acts = torch.stack([r.output.float() for r in records], dim=0)
            # Shape: (num_records, B, C, ...) or (num_records, B, ..., C)
            # Flatten to (num_records * B, C, ...)
            try:
                acts_flat = acts.view(-1, *acts.shape[2:])
                if acts_flat.ndim > 2:
                    # (N, C, ...) → per-channel sparsity
                    n_filters = acts_flat.shape[1]
                    sparsity = []
                    for f in range(n_filters):
                        active = (acts_flat[:, f].abs() > 1e-6).float().mean().item()
                        sparsity.append(active)
                    scores[layer_name] = sparsity
            except Exception:
                pass
        return scores


# ---------------------------------------------------------------------------
# Warm-up gradient baseline
# ---------------------------------------------------------------------------


class WarmUpGradientBaseline(BaselineRanker):
    """
    Run a few gradient steps.
    """

    name = "warmup_gradient"

    def __init__(self, n_steps: int = 5, lr: float = 1e-3) -> None:
        self.n_steps = n_steps
        self.lr = lr

    def rank_filters(self, model: nn.Module, diag_data: Tensor) -> Dict[str, List[float]]:
        model_copy = copy.deepcopy(model).train()
        opt = torch.optim.SGD(model_copy.parameters(), lr=self.lr)

        grad_accum: Dict[str, Tensor] = {}

        for step in range(self.n_steps):
            opt.zero_grad()
            try:
                out = model_copy(diag_data)
                loss = out.float().abs().mean()
                loss.backward()
                opt.step()
            except Exception as exc:
                log.warning("WarmUpGradient step %d failed: %s", step, exc)
                break

            for name, param in model_copy.named_parameters():
                if param.grad is not None:
                    g = param.grad.detach().float()
                    if name not in grad_accum:
                        grad_accum[name] = g.abs()
                    else:
                        grad_accum[name] += g.abs()

        scores: Dict[str, List[float]] = {}
        for pname, g_sum in grad_accum.items():
            layer_name = ".".join(pname.split(".")[:-1])
            g_flat = g_sum.view(g_sum.shape[0], -1)
            importance = g_flat.norm(dim=1).tolist()
            scores[layer_name] = importance

        return scores
