from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import DiagnosticReport, LayerDivergence

log = get_logger(__name__)

_PALETTE = "viridis"
_FIGSIZE_WIDE = (14, 5)
_FIGSIZE_SQUARE = (10, 9)


def _short(name: str, max_len: int = 30) -> str:
    """Truncate long layer names for axis labels."""
    return name if len(name) <= max_len else "…" + name[-max_len + 1:]


# ---------------------------------------------------------------------------
# 1. Layer-by-sample divergence heatmap
# ---------------------------------------------------------------------------


def plot_divergence_heatmap(
    report: DiagnosticReport,
    output_path: Path,
    max_layers: int = 50,
) -> None:
    """
    Heatmap H ∈ R^{L×m} where H[l, k] = D_l(x_k).
    """
    divs = report.layer_divergences[:max_layers]
    if not divs:
        return

    min_m = min(len(ld.per_sample) for ld in divs)
    mat = np.array([ld.per_sample[:min_m] for ld in divs], dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, min_m // 3), max(6, len(divs) // 3)))
    sns.heatmap(
        mat,
        ax=ax,
        cmap=_PALETTE,
        xticklabels=False,
        yticklabels=[_short(ld.layer_name) for ld in divs],
        cbar_kws={"label": "IFD"},
    )
    ax.set_title("Layer × Sample Divergence Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Diagnostic Sample Index")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved divergence heatmap → %s", output_path)


# ---------------------------------------------------------------------------
# 2. Per-layer mean IFD ranking bar chart
# ---------------------------------------------------------------------------


def plot_layer_ranking(
    report: DiagnosticReport,
    output_path: Path,
    top_k: int = 40,
) -> None:
    """Horizontal bar chart of layers ranked by mean IFD."""
    divs = sorted(report.layer_divergences, key=lambda x: x.mean, reverse=True)[:top_k]
    if not divs:
        return

    names = [_short(ld.layer_name) for ld in divs]
    means = [ld.mean for ld in divs]
    ci_err = [
        [ld.mean - ld.ci_low for ld in divs],
        [ld.ci_high - ld.mean for ld in divs],
    ]

    fig, ax = plt.subplots(figsize=(9, max(5, len(divs) * 0.35)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(means)))
    bars = ax.barh(names[::-1], means[::-1], color=colors[::-1], xerr=[ci_err[0][::-1], ci_err[1][::-1]],
                   error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "grey"})
    ax.set_xlabel("Mean IFD")
    ax.set_title(f"Layer IFD Ranking (top {top_k})", fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved layer ranking → %s", output_path)


# ---------------------------------------------------------------------------
# 3. Filter vulnerability histogram
# ---------------------------------------------------------------------------


def plot_filter_histogram(
    report: DiagnosticReport,
    output_path: Path,
) -> None:
    """Distribution of per-filter divergence values across all layers."""
    all_vals: List[float] = []
    for ld in report.layer_divergences:
        if ld.per_filter:
            all_vals.extend(ld.per_filter)

    if not all_vals:
        log.warning("No per-filter divergences — skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.hist(all_vals, bins=60, color="#2196F3", edgecolor="white", linewidth=0.4)
    n_vuln = report.total_vulnerable_filters
    ax.set_title(
        f"Per-Filter IFD Distribution  ({n_vuln} flagged vulnerable)", fontweight="bold"
    )
    ax.set_xlabel("Filter IFD")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved filter histogram → %s", output_path)


# ---------------------------------------------------------------------------
# 4. Block efficiency chart
# ---------------------------------------------------------------------------


def plot_block_efficiency(
    report: DiagnosticReport,
    output_path: Path,
) -> None:
    """Scatter plot of block divergence vs efficiency."""
    importance = report.block_importance
    if not importance:
        log.warning("No block importance data — skipping block efficiency chart.")
        return

    names = [b["block_name"] for b in importance]
    d_blocks = [b["block_divergence"] for b in importance]
    etas = [b["efficiency"] for b in importance]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(d_blocks, etas, c=range(len(names)), cmap="plasma", s=80, zorder=3)
    for i, name in enumerate(names):
        ax.annotate(
            _short(name, 20),
            (d_blocks[i], etas[i]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.axhline(0.10, color="red", linestyle="--", linewidth=0.8, label="efficiency threshold")
    ax.set_xlabel("Block Divergence (D_B)")
    ax.set_ylabel("Block Efficiency (η_B)")
    ax.set_title("Block-Level Efficiency vs Divergence", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved block efficiency chart → %s", output_path)


# ---------------------------------------------------------------------------
# 5. Correlation matrix heatmap
# ---------------------------------------------------------------------------


def plot_correlation_matrix(
    report: DiagnosticReport,
    output_path: Path,
    max_layers: int = 40,
) -> None:
    """CORRDIAG pairwise correlation matrix."""
    V = report.correlation_matrix
    if V is None or V.shape[0] == 0:
        log.warning("No correlation matrix — skipping.")
        return

    V_sub = V[:max_layers, :max_layers]
    names = [_short(ld.layer_name, 20) for ld in report.layer_divergences[:max_layers]]

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    mask = np.eye(len(names), dtype=bool)
    sns.heatmap(
        V_sub,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Cross-Layer Divergence Correlation Matrix", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved correlation matrix → %s", output_path)


# ---------------------------------------------------------------------------
# 6. Seed stability plot
# ---------------------------------------------------------------------------


def plot_seed_stability(
    report: DiagnosticReport,
    output_path: Path,
    max_layers: int = 40,
) -> None:
    """
    Error-bar plot of ensemble mean ± std across initialisations.
    """
    lds = [
        ld for ld in report.layer_divergences
        if ld.ensemble_mean is not None and ld.ensemble_std is not None
    ][:max_layers]

    if not lds:
        log.warning("No ensemble data — skipping seed stability plot.")
        return

    names = [_short(ld.layer_name) for ld in lds]
    means = [ld.ensemble_mean for ld in lds]
    stds = [ld.ensemble_std for ld in lds]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.errorbar(
        range(len(names)),
        means,
        yerr=stds,
        fmt="o-",
        color="#E91E63",
        ecolor="#BDBDBD",
        capsize=3,
        linewidth=1.2,
        markersize=5,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Ensemble Mean IFD ± Std")
    ax.set_title("Seed Stability of IFD Estimates", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved seed stability plot → %s", output_path)


# ---------------------------------------------------------------------------
# Convenience: save all plots at once
# ---------------------------------------------------------------------------


def save_all_plots(report: DiagnosticReport, output_dir: Path) -> None:
    """Render and save every diagnostic visualisation to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_divergence_heatmap(report, output_dir / "heatmap.png")
    plot_layer_ranking(report, output_dir / "layer_ranking.png")
    plot_filter_histogram(report, output_dir / "filter_histogram.png")
    plot_block_efficiency(report, output_dir / "block_efficiency.png")
    plot_correlation_matrix(report, output_dir / "correlation_matrix.png")
    plot_seed_stability(report, output_dir / "seed_stability.png")
