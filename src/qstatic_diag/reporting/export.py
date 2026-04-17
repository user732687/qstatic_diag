"""
Report export: JSON, CSV, and Markdown.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import DiagnosticReport

log = get_logger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def save_json(report: DiagnosticReport, path: Path) -> None:
    """Export full report as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    d = {
        "summary": {
            "total_layers": report.total_layers,
            "total_filters": report.total_filters,
            "total_vulnerable_filters": report.total_vulnerable_filters,
            "total_vulnerable_layers": report.total_vulnerable_layers,
            "wall_time_seconds": report.wall_time_seconds,
            "num_forward_passes": report.num_forward_passes,
        },
        "config": report.config,
        "layer_divergences": [
            {
                "name": ld.layer_name,
                "kind": ld.kind.name,
                "mean": ld.mean,
                "std": ld.std,
                "rho": ld.rho,
                "ci_low": ld.ci_low,
                "ci_high": ld.ci_high,
                "per_filter": ld.per_filter,
                "ensemble_mean": ld.ensemble_mean,
                "ensemble_std": ld.ensemble_std,
            }
            for ld in report.layer_divergences
        ],
        "vulnerable_filters": report.vulnerable_filters,
        "unstable_filters": report.unstable_filters,
        "vulnerable_layers": report.vulnerable_layers,
        "removable_layers": report.removable_layers,
        "vulnerable_blocks": report.vulnerable_blocks,
        "block_importance": report.block_importance,
        "correlated_pairs": [
            {"layer_i": a, "layer_j": b, "r": r} for a, b, r in report.correlated_pairs
        ],
    }
    with open(path, "w") as f:
        json.dump(d, f, indent=2, cls=_NumpyEncoder)
    log.info("JSON report saved → %s", path)


def save_csv(report: DiagnosticReport, path: Path) -> None:
    """Export per-layer divergence table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer_name", "kind", "mean", "std", "rho", "ci_low", "ci_high"],
        )
        writer.writeheader()
        for ld in report.layer_divergences:
            writer.writerow(
                {
                    "layer_name": ld.layer_name,
                    "kind": ld.kind.name,
                    "mean": round(ld.mean, 6),
                    "std": round(ld.std, 6),
                    "rho": round(ld.rho, 6),
                    "ci_low": round(ld.ci_low, 6),
                    "ci_high": round(ld.ci_high, 6),
                }
            )
    log.info("CSV report saved → %s", path)


def save_markdown(report: DiagnosticReport, path: Path) -> None:
    """Export a Markdown summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quasi-Static Diagnostic Report\n",
        "## Summary\n",
        f"- **Total layers analysed:** {report.total_layers}",
        f"- **Total filters/channels:** {report.total_filters}",
        f"- **Vulnerable filters:** {report.total_vulnerable_filters}",
        f"- **Vulnerable layers:** {report.total_vulnerable_layers}",
        f"- **Wall time:** {report.wall_time_seconds:.1f}s",
        f"- **Forward passes:** {report.num_forward_passes}",
        "",
        "## Per-Layer IFD (top 20 by divergence)\n",
        "| Layer | Kind | Mean IFD | Std | rho | CI (95%) |",
        "|---|---|---|---|---|---|",
    ]
    sorted_divs = sorted(report.layer_divergences, key=lambda x: x.mean, reverse=True)[:20]
    for ld in sorted_divs:
        lines.append(
            f"| `{ld.layer_name}` | {ld.kind.name} | {ld.mean:.4f} | "
            f"{ld.std:.4f} | {ld.rho:.4f} | [{ld.ci_low:.4f}, {ld.ci_high:.4f}] |"
        )

    lines += [
        "",
        "## Vulnerable Layers\n",
        "| Layer | rho | Reasons |",
        "|---|---|---|",
    ]
    for vl in report.vulnerable_layers:
        reasons = ", ".join(vl.get("kinds", [vl.get("kind", "?")]))
        lines.append(f"| `{vl['layer_name']}` | {vl['rho']:.4f} | {reasons} |")

    lines += [
        "",
        "## Vulnerable Blocks\n",
        "| Block | D_block | Efficiency | Residual dom. | Reasons |",
        "|---|---|---|---|---|",
    ]
    for vb in report.vulnerable_blocks:
        reasons = ", ".join(vb.get("kinds", []))
        delta = f"{vb['residual_dominance']:.4f}" if vb["residual_dominance"] is not None else "N/A"
        lines.append(
            f"| `{vb['block_name']}` | {vb['block_divergence']:.4f} | "
            f"{vb['efficiency']:.4f} | {delta} | {reasons} |"
        )

    lines += [
        "",
        "## Highly Correlated Layer Pairs\n",
        "| Layer A | Layer B | r |",
        "|---|---|---|",
    ]
    for a, b, r in report.correlated_pairs[:20]:
        lines.append(f"| `{a}` | `{b}` | {r:.4f} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Markdown report saved → %s", path)
