from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy import stats

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import LayerDivergence

log = get_logger(__name__)


def bootstrap_ci(
    samples: List[float],
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for the mean.

    Returns
    -------
    (ci_low, ci_high)
    """
    if len(samples) < 2:
        m = samples[0] if samples else 0.0
        return m, m

    rng = np.random.default_rng(seed)
    arr = np.array(samples, dtype=float)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


def add_bootstrap_ci(
    layer_divs: List[LayerDivergence],
    n_bootstrap: int = 200,
    alpha: float = 0.05,
) -> List[LayerDivergence]:
    """Annotate each LayerDivergence with bootstrap CI in-place."""
    for ld in layer_divs:
        if len(ld.per_sample) < 2:
            ld.ci_low = ld.mean
            ld.ci_high = ld.mean
        else:
            ld.ci_low, ld.ci_high = bootstrap_ci(ld.per_sample, n_bootstrap, alpha)
    return layer_divs


def filter_vulnerability_pvalue(
    per_sample_divs: List[float],
    tau_min: float,
) -> float:
    """
    p-value for H0: D_{l,f} >= tau_min  (filter not vulnerable).

    p = fraction of samples where D_{l,f} < tau_min.
    """
    if not per_sample_divs:
        return 1.0
    arr = np.array(per_sample_divs, dtype=float)
    return float((arr < tau_min).mean())


def z_confidence_interval(
    mean: float,
    std: float,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Gaussian approximation CI (Eq. in paper)."""
    z = stats.norm.ppf(1 - alpha / 2)
    margin = z * std / math.sqrt(max(n, 1))
    return mean - margin, mean + margin
