from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import LayerDivergence, LayerKind, VulnerabilityKind

log = get_logger(__name__)


def run_flame(
    layer_divs: List[LayerDivergence],
    p_min: float = 0.20,
    p_max: float = 0.95,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run FLAME

    Parameters
    ----------
    layer_divs:
        Output of :class:`~qstatic_diag.divergence.ifd.IFDEngine`.
    p_min:
        Bottom percentile threshold for dead / dormant filters.
    p_max:
        Top percentile threshold for chaotically active filters.

    Returns
    -------
    vulnerable_filters, unstable_filters
        Each is a list of dicts with keys:
        ``layer_name``, ``filter_idx``, ``divergence``, ``threshold``, ``kind``.
    """
    vulnerable: List[Dict] = []
    unstable: List[Dict] = []

    for ld in layer_divs:
        if ld.per_filter is None or len(ld.per_filter) == 0:
            continue

        pf = np.array(ld.per_filter, dtype=float)
        tau_min = float(np.percentile(pf, p_min * 100))
        tau_max = float(np.percentile(pf, p_max * 100))

        for f_idx, div in enumerate(pf):
            if div < tau_min:
                vulnerable.append(
                    {
                        "layer_name": ld.layer_name,
                        "filter_idx": f_idx,
                        "divergence": float(div),
                        "threshold": tau_min,
                        "kind": VulnerabilityKind.DEAD_FILTER.value,
                    }
                )
            if div > tau_max:
                unstable.append(
                    {
                        "layer_name": ld.layer_name,
                        "filter_idx": f_idx,
                        "divergence": float(div),
                        "threshold": tau_max,
                        "kind": VulnerabilityKind.UNSTABLE_FILTER.value,
                    }
                )

    log.info(
        "FLAME: %d vulnerable filters, %d unstable filters identified.",
        len(vulnerable),
        len(unstable),
    )
    return vulnerable, unstable
