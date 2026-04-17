from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import LayerDivergence, VulnerabilityKind

log = get_logger(__name__)


def run_laydiag(
    layer_divs: List[LayerDivergence],
    gamma_min: float = 0.05,
    gamma_max: float = 0.40,
    outlier_sigma: float = 2.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run LAYDIAG on a list of layer divergences.

    Parameters
    ----------
    layer_divs:
        Ordered list with pre-computed ``rho`` (normalised contribution).
    gamma_min:
        Below this rho → bottleneck / removable.
    gamma_max:
        Above this rho → instability source.
    outlier_sigma:
        Standard-deviation multiplier for statistical outlier detection.

    Returns
    -------
    vulnerable_layers, removable_layers
    """
    rhos = np.array([ld.rho for ld in layer_divs], dtype=float)
    mu = rhos.mean()
    sigma = rhos.std() + 1e-9

    vulnerable: List[Dict] = []
    removable: List[Dict] = []

    for ld in layer_divs:
        rho = ld.rho
        reasons: List[str] = []

        if rho < gamma_min:
            reasons.append(VulnerabilityKind.BOTTLENECK_LAYER.value)
            removable.append(
                {
                    "layer_name": ld.layer_name,
                    "rho": rho,
                    "mean_divergence": ld.mean,
                    "kind": VulnerabilityKind.REDUNDANT_LAYER.value,
                }
            )

        if rho > gamma_max:
            reasons.append(VulnerabilityKind.INSTABILITY_SOURCE.value)

        if rho > mu + outlier_sigma * sigma:
            reasons.append(VulnerabilityKind.OUTLIER_HIGH.value)

        if rho < mu - outlier_sigma * sigma:
            reasons.append(VulnerabilityKind.OUTLIER_LOW.value)

        if reasons:
            vulnerable.append(
                {
                    "layer_name": ld.layer_name,
                    "rho": rho,
                    "mean_divergence": ld.mean,
                    "kinds": reasons,
                }
            )

    log.info(
        "LAYDIAG: %d vulnerable layers, %d removable layers.",
        len(vulnerable),
        len(removable),
    )
    return vulnerable, removable
