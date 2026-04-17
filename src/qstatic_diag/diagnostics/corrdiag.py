from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import LayerDivergence, VulnerabilityKind

log = get_logger(__name__)


def run_corrdiag(
    layer_divs: List[LayerDivergence],
    rho_thresh: float = 0.90,
    min_samples: int = 5,
    correct_multiple: bool = True,
) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
    """
    Run CORRDIAG.

    Parameters
    ----------
    layer_divs:
        Layer divergences with ``per_sample`` vectors of equal length.
    rho_thresh:
        Absolute Pearson |r| above which a pair is flagged.
    min_samples:
        Minimum samples required to compute a reliable correlation.
    correct_multiple:
        Apply Benjamini-Hochberg FDR correction on p-values.

    Returns
    -------
    V : np.ndarray of shape (L, L) — full correlation matrix.
    correlated_pairs : list of (layer_i, layer_j, r_ij) for |r| > rho_thresh.
    """
    # Build matrix D: shape (L, m)
    # Only include layers with per_sample data of length ≥ min_samples
    eligible = [ld for ld in layer_divs if len(ld.per_sample) >= min_samples]
    if len(eligible) < 2:
        log.warning("CORRDIAG: fewer than 2 eligible layers, skipping.")
        return np.zeros((len(layer_divs), len(layer_divs))), []

    L = len(eligible)
    names = [ld.layer_name for ld in eligible]

    # Trim all to the minimum available sample count
    min_m = min(len(ld.per_sample) for ld in eligible)
    D = np.array([ld.per_sample[:min_m] for ld in eligible], dtype=float)  # (L, m)

    # Pearson correlation matrix
    V = np.corrcoef(D)  # (L, L)
    np.fill_diagonal(V, 0.0)  # self-correlation not interesting

    # --- p-value matrix and optional BH correction ---
    pvals: List[float] = []
    pair_indices: List[Tuple[int, int]] = []
    for i in range(L):
        for j in range(i + 1, L):
            r = V[i, j]
            # t-statistic for Pearson r
            t = r * np.sqrt(min_m - 2) / (np.sqrt(1 - r**2 + 1e-12))
            p = 2 * stats.t.sf(abs(t), df=min_m - 2)
            pvals.append(p)
            pair_indices.append((i, j))

    if correct_multiple and pvals:
        from statsmodels.stats.multitest import multipletests
        reject, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
    else:
        pvals_corr = np.array(pvals)

    # Collect significant pairs above threshold
    correlated_pairs: List[Tuple[str, str, float]] = []
    for k, (i, j) in enumerate(pair_indices):
        r = float(V[i, j])
        if abs(r) > rho_thresh:
            correlated_pairs.append((names[i], names[j], r))

    log.info(
        "CORRDIAG: %d layer pairs analysed, %d flagged (|r| > %.2f).",
        len(pair_indices),
        len(correlated_pairs),
        rho_thresh,
    )

    # Expand back to full L×L matrix indexed by original layer_divs order
    name_to_idx_elig = {n: i for i, n in enumerate(names)}
    all_names = [ld.layer_name for ld in layer_divs]
    V_full = np.zeros((len(layer_divs), len(layer_divs)))
    for ii, ni in enumerate(all_names):
        for jj, nj in enumerate(all_names):
            ei = name_to_idx_elig.get(ni)
            ej = name_to_idx_elig.get(nj)
            if ei is not None and ej is not None:
                V_full[ii, jj] = V[ei, ej]

    return V_full, correlated_pairs
