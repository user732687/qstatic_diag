from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import LayerDivergence, LayerKind, VulnerabilityKind

log = get_logger(__name__)

EPS = 1e-9


def run_blockdiag(
    layer_divs: List[LayerDivergence],
    block_definitions: Dict[str, List[str]],
    efficiency_threshold: float = 0.10,
    residual_low: float = 0.05,
    residual_high: float = 0.80,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run BLOCKDIAG.

    Parameters
    ----------
    layer_divs:
        Layer divergences from IFDEngine.
    block_definitions:
        Mapping block_name → [layer_name, ...].
    efficiency_threshold:
        Below this block efficiency eta_B → inefficient block.
    residual_low:
        Below this residual dominance → inactive residual branch.
    residual_high:
        Above this → identity path is redundant.

    Returns
    -------
    vulnerable_blocks, block_importance_ranking
    """
    div_map: Dict[str, LayerDivergence] = {ld.layer_name: ld for ld in layer_divs}

    vulnerable: List[Dict] = []
    importance: List[Dict] = []

    for block_name, layer_names in block_definitions.items():
        block_layers = [div_map[n] for n in layer_names if n in div_map]
        if not block_layers:
            continue

        # Block-level aggregate divergence
        d_block = sum(ld.mean for ld in block_layers)

        # Input-output divergence approximation
        first_mean = block_layers[0].mean
        last_mean = block_layers[-1].mean
        d_io = abs(last_mean - first_mean) / (first_mean + EPS)

        eta_b = d_block / (d_io + EPS)

        # Residual dominance: fraction of layers that are residual-kind
        res_layers = [ld for ld in block_layers if ld.kind == LayerKind.RESIDUAL]
        delta_b: Optional[float] = None
        if res_layers:
            res_div = sum(ld.mean for ld in res_layers)
            total_div = d_block + EPS
            delta_b = res_div / total_div

        reasons: List[str] = []
        if eta_b < efficiency_threshold:
            reasons.append(VulnerabilityKind.INEFFICIENT_BLOCK.value)
        if delta_b is not None:
            if delta_b < residual_low:
                reasons.append(VulnerabilityKind.INACTIVE_RESIDUAL.value)
            elif delta_b > residual_high:
                reasons.append(VulnerabilityKind.DOMINANT_RESIDUAL.value)

        entry = {
            "block_name": block_name,
            "block_divergence": d_block,
            "efficiency": eta_b,
            "residual_dominance": delta_b,
            "num_layers": len(block_layers),
        }
        importance.append(entry)

        if reasons:
            vuln_entry = {**entry, "kinds": reasons}
            vulnerable.append(vuln_entry)

    # Sort importance descending by block divergence
    importance.sort(key=lambda x: x["block_divergence"], reverse=True)

    log.info(
        "BLOCKDIAG: %d blocks analysed, %d vulnerable.",
        len(importance),
        len(vulnerable),
    )
    return vulnerable, importance
