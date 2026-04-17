from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from qstatic_diag.diagnostics.pipeline import DiagnosticPipeline, PipelineConfig
from qstatic_diag.evaluation.baselines import (
    ActivationSparsityBaseline,
    NormBaseline,
    RandomBaseline,
    WarmUpGradientBaseline,
)
from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.seed import seed_everything

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic Accuracy
# ---------------------------------------------------------------------------


@dataclass
class AccuracyResult:
    """Spearman rank correlation between IFD rank and post-training rank."""

    spearman_rho: float
    p_value: float
    n_layers: int
    method: str


def run_diagnostic_accuracy(
    model_factory: Callable[[], nn.Module],
    proxy_data: Tensor,
    post_training_scores: Dict[str, float],  # layer_name → importance after training
    config: Optional[PipelineConfig] = None,
) -> AccuracyResult:
    """
    Measure how well the quasi-static IFD ranking correlates with
    post-training importance scores
    """
    from scipy.stats import spearmanr

    config = config or PipelineConfig()
    model = model_factory()

    pipeline = DiagnosticPipeline(config)
    report = pipeline.run(model, proxy_data)

    ifd_scores = {ld.layer_name: ld.mean for ld in report.layer_divergences}

    common = sorted(set(ifd_scores) & set(post_training_scores))
    if len(common) < 3:
        log.warning("Too few common layers (%d) for meaningful correlation.", len(common))
        return AccuracyResult(0.0, 1.0, len(common), "ifd")

    x = np.array([ifd_scores[n] for n in common])
    y = np.array([post_training_scores[n] for n in common])

    rho, pval = spearmanr(x, y)
    log.info("Diagnostic accuracy: Spearman rho=%.3f (p=%.4f)", rho, pval)
    return AccuracyResult(float(rho), float(pval), len(common), "ifd")


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyResult:
    """Wall-clock comparison between quasi-static and training baselines."""

    method: str
    wall_seconds: float
    num_forward_passes: int
    speedup_vs_warmup: Optional[float] = None


def run_efficiency_comparison(
    model: nn.Module,
    proxy_data: Tensor,
    config: Optional[PipelineConfig] = None,
    warmup_steps: int = 10,
) -> List[EfficiencyResult]:
    config = config or PipelineConfig()
    results: List[EfficiencyResult] = []

    # --- IFD ---
    t0 = time.perf_counter()
    pipeline = DiagnosticPipeline(config)
    report = pipeline.run(model, proxy_data)
    ifd_time = time.perf_counter() - t0
    results.append(EfficiencyResult("ifd", ifd_time, report.num_forward_passes))

    # --- Warm-up gradient ---
    baseline = WarmUpGradientBaseline(n_steps=warmup_steps)
    t0 = time.perf_counter()
    baseline.rank_filters(model, proxy_data)
    grad_time = time.perf_counter() - t0
    results.append(EfficiencyResult("warmup_gradient", grad_time, warmup_steps))

    # Compute speedups
    for r in results:
        r.speedup_vs_warmup = grad_time / max(r.wall_seconds, 1e-6)

    for r in results:
        log.info("[%s] %.2fs  (%.1fx vs warmup)", r.method, r.wall_seconds, r.speedup_vs_warmup)

    return results


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


@dataclass
class RobustnessResult:
    param: str
    value: Any
    mean_ifd_variance: float
    fraction_stable_rankings: float


def run_robustness_sweep(
    model_factory: Callable[[], nn.Module],
    proxy_data: Tensor,
    m_values: Optional[List[int]] = None,
    seed_values: Optional[List[int]] = None,
    base_config: Optional[PipelineConfig] = None,
) -> List[RobustnessResult]:
    """Sweep over sample sizes and seeds; measure stability of layer rankings."""
    from scipy.stats import spearmanr

    base_config = base_config or PipelineConfig()
    m_values = m_values or [10, 25, 50, 100, 200]
    seed_values = seed_values or list(range(5))
    results: List[RobustnessResult] = []

    # --- Sample-size sweep ---
    reference_cfg = PipelineConfig(**{**vars(base_config), "m": 200, "seed": 0})
    ref_model = model_factory()
    ref_report = DiagnosticPipeline(reference_cfg).run(ref_model, proxy_data)
    ref_scores = {ld.layer_name: ld.mean for ld in ref_report.layer_divergences}

    for m in m_values:
        cfg = PipelineConfig(**{**vars(base_config), "m": m, "seed": 0})
        model = model_factory()
        report = DiagnosticPipeline(cfg).run(model, proxy_data)
        scores = {ld.layer_name: ld.mean for ld in report.layer_divergences}
        common = sorted(set(scores) & set(ref_scores))
        if common:
            x = np.array([scores[n] for n in common])
            y = np.array([ref_scores[n] for n in common])
            rho, _ = spearmanr(x, y)
        else:
            rho = 0.0
        results.append(RobustnessResult("m", m, 0.0, float(rho)))

    # --- Seed sweep ---
    all_means: Dict[str, List[float]] = {}
    for s in seed_values:
        cfg = PipelineConfig(**{**vars(base_config), "seed": s})
        model = model_factory()
        report = DiagnosticPipeline(cfg).run(model, proxy_data)
        for ld in report.layer_divergences:
            all_means.setdefault(ld.layer_name, []).append(ld.mean)

    cv_vals = []
    for name, vals in all_means.items():
        if len(vals) > 1:
            mean = np.mean(vals)
            std = np.std(vals)
            cv_vals.append(std / (mean + 1e-9))
    mean_cv = float(np.mean(cv_vals)) if cv_vals else 0.0
    results.append(RobustnessResult("seed_cv", seed_values, mean_cv, 1.0 - mean_cv))

    return results


# ---------------------------------------------------------------------------
# Generalizability
# ---------------------------------------------------------------------------


@dataclass
class GeneralizabilityResult:
    architecture: str
    task: str
    n_vulnerable_filters: int
    n_vulnerable_layers: int
    wall_seconds: float


def run_generalizability(
    models: Dict[str, nn.Module],
    proxy_datasets: Dict[str, Tensor],
    config: Optional[PipelineConfig] = None,
) -> List[GeneralizabilityResult]:
    """Run diagnostics on multiple model/data combinations."""
    config = config or PipelineConfig()
    results: List[GeneralizabilityResult] = []

    for arch_name, model in models.items():
        for task_name, data in proxy_datasets.items():
            try:
                t0 = time.perf_counter()
                report = DiagnosticPipeline(config).run(model, data)
                elapsed = time.perf_counter() - t0
                results.append(
                    GeneralizabilityResult(
                        architecture=arch_name,
                        task=task_name,
                        n_vulnerable_filters=report.total_vulnerable_filters,
                        n_vulnerable_layers=report.total_vulnerable_layers,
                        wall_seconds=elapsed,
                    )
                )
                log.info(
                    "[%s/%s] %.2fs | %d vuln filters | %d vuln layers",
                    arch_name, task_name, elapsed,
                    report.total_vulnerable_filters, report.total_vulnerable_layers,
                )
            except Exception as exc:
                log.error("[%s/%s] Failed: %s", arch_name, task_name, exc)

    return results
