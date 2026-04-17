#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from rich.console import Console
from rich.table import Table

from qstatic_diag.diagnostics.pipeline import DiagnosticPipeline, PipelineConfig
from qstatic_diag.models.zoo import TinyCNN
from qstatic_diag.reporting.export import save_csv, save_json, save_markdown
from qstatic_diag.utils.seed import seed_everything
from qstatic_diag.visualization.plots import save_all_plots

console = Console()


def main(output_dir: str = "./results/cnn_demo") -> None:
    seed_everything(42)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Model + proxy data
    # -----------------------------------------------------------------------
    console.rule("Step 1: Model & Data")
    model = TinyCNN(in_channels=3, num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"Model   : TinyCNN  ({n_params:,} parameters)")

    proxy = torch.randn(1000, 3, 32, 32)
    console.print(f"Proxy   : {proxy.shape}")

    # -----------------------------------------------------------------------
    # 2. Pipeline configuration
    # -----------------------------------------------------------------------
    console.rule("Step 2: Pipeline Config")
    cfg = PipelineConfig(
        m=100,
        seed=42,
        p_min=0.20,
        p_max=0.95,
        gamma_min=0.05,
        gamma_max=0.40,
        bootstrap_n=200,
        num_seeds=2,
        device="cpu",
    )
    console.print(f"Diagnostic samples : m = {cfg.m}")
    console.print(f"Multi-seed runs    : {cfg.num_seeds}")

    # -----------------------------------------------------------------------
    # 3. Run diagnosis
    # -----------------------------------------------------------------------
    console.rule("Step 3: Running Diagnosis")
    report = DiagnosticPipeline(cfg).run(model, proxy)

    # -----------------------------------------------------------------------
    # 4. Print summary table
    # -----------------------------------------------------------------------
    console.rule("Step 4: Results")
    table = Table(title="Diagnostic Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Layers analysed", str(report.total_layers))
    table.add_row("Filters/channels", str(report.total_filters))
    table.add_row("Vulnerable filters", str(report.total_vulnerable_filters))
    table.add_row("Unstable filters", str(len(report.unstable_filters)))
    table.add_row("Vulnerable layers", str(report.total_vulnerable_layers))
    table.add_row("Removable layers", str(len(report.removable_layers)))
    table.add_row("Vulnerable blocks", str(len(report.vulnerable_blocks)))
    table.add_row("Correlated pairs", str(len(report.correlated_pairs)))
    table.add_row("Wall time", f"{report.wall_time_seconds:.2f}s")
    table.add_row("Forward passes", str(report.num_forward_passes))
    console.print(table)

    # Top 5 layers by IFD
    sorted_divs = sorted(report.layer_divergences, key=lambda x: x.mean, reverse=True)[:5]
    layer_table = Table(title="Top 5 Layers by Mean IFD")
    layer_table.add_column("Layer")
    layer_table.add_column("Kind")
    layer_table.add_column("Mean IFD")
    layer_table.add_column("ρ")
    layer_table.add_column("95% CI")
    for ld in sorted_divs:
        layer_table.add_row(
            ld.layer_name,
            ld.kind.name,
            f"{ld.mean:.5f}",
            f"{ld.rho:.4f}",
            f"[{ld.ci_low:.5f}, {ld.ci_high:.5f}]",
        )
    console.print(layer_table)

    # -----------------------------------------------------------------------
    # 5. Export
    # -----------------------------------------------------------------------
    console.rule("Step 5: Exporting")
    save_json(report, out / "report.json")
    save_csv(report, out / "layer_divergences.csv")
    save_markdown(report, out / "report.md")
    save_all_plots(report, out / "plots")
    console.print(f"All outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyCNN quasi-static diagnostic demo.")
    parser.add_argument("--output-dir", default="./results/cnn_demo")
    args = parser.parse_args()
    main(args.output_dir)
