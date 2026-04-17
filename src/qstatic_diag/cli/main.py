from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import click
import torch

from qstatic_diag.diagnostics.pipeline import DiagnosticPipeline, PipelineConfig
from qstatic_diag.reporting.export import save_csv, save_json, save_markdown
from qstatic_diag.utils.logging import get_logger, set_global_level
from qstatic_diag.utils.seed import seed_everything, get_device
from qstatic_diag.visualization.plots import save_all_plots

import logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(model_name: str, data_shape: Tuple[int, ...]) -> torch.nn.Module:
    """Load a model by name."""
    import importlib

    try:
        import torchvision.models as tvm
        model = getattr(tvm, model_name)(pretrained=False)
        return model
    except AttributeError:
        pass

    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """qstatic_diag — Quasi-Static Neural Network Diagnostics."""
    if verbose:
        set_global_level(logging.DEBUG)


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", "-m", default="tiny-cnn", show_default=True, help="Model name.")
@click.option(
    "--data-shape",
    default="3,32,32",
    show_default=True,
    help="Input shape as comma-separated ints, e.g. '3,32,32' or '32' for sequences.",
)
@click.option("--n-samples", default=500, show_default=True, help="Proxy set size N.")
@click.option("--diag-m", default=100, show_default=True, help="Diagnostic set size m.")
@click.option("--seed", default=42, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--output-dir", "-o", default="./qsdiag_output", show_default=True, help="Output directory."
)
@click.option("--no-plots", is_flag=True, help="Skip visualisation.")
@click.option("--num-seeds", default=1, show_default=True, help="Ensemble seeds (>1 = multi-seed).")
def diagnose(
    model: str,
    data_shape: str,
    n_samples: int,
    diag_m: int,
    seed: int,
    device: str,
    output_dir: str,
    no_plots: bool,
    num_seeds: int,
) -> None:
    """Run the full quasi-static diagnostic pipeline on a model."""
    shape = tuple(int(s) for s in data_shape.split(","))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading model: {model}")
    net = _load_model(model, shape)

    click.echo(f"Generating {n_samples} samples, shape={shape}")
    data = _load_data(n_samples, shape)

    cfg = PipelineConfig(
        m=diag_m,
        seed=seed,
        device=device,
        num_seeds=num_seeds,
    )

    click.echo("Running diagnostic pipeline…")
    report = DiagnosticPipeline(cfg).run(net, data)

    # Export
    save_json(report, out_dir / "report.json")
    save_csv(report, out_dir / "layer_divergences.csv")
    save_markdown(report, out_dir / "report.md")

    if not no_plots:
        click.echo("Generating visualisations…")
        save_all_plots(report, out_dir / "plots")

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo(f"  Layers analysed  : {report.total_layers}")
    click.echo(f"  Filters checked  : {report.total_filters}")
    click.echo(f"  Vulnerable filters: {report.total_vulnerable_filters}")
    click.echo(f"  Vulnerable layers : {report.total_vulnerable_layers}")
    click.echo(f"  Correlated pairs  : {len(report.correlated_pairs)}")
    click.echo(f"  Wall time         : {report.wall_time_seconds:.1f}s")
    click.echo(f"  Output dir        : {out_dir.resolve()}")
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", default="tiny-cnn", show_default=True)
@click.option("--data-shape", default="3,32,32", show_default=True)
@click.option("--n-samples", default=300, show_default=True)
@click.option("--warmup-steps", default=5, show_default=True)
def benchmark(model: str, data_shape: str, n_samples: int, warmup_steps: int) -> None:
    """Compare IFD diagnostic speed against warm-up gradient baseline."""
    from qstatic_diag.evaluation.experiments import run_efficiency_comparison

    shape = tuple(int(s) for s in data_shape.split(","))
    net = _load_model(model, shape)
    data = _load_data(n_samples, shape)

    results = run_efficiency_comparison(net, data, warmup_steps=warmup_steps)

    click.echo("\nEfficiency Comparison:")
    click.echo(f"{'Method':<25} {'Time (s)':<12} {'Fwd Passes':<15} {'Speedup'}")
    click.echo("-" * 65)
    for r in results:
        spd = f"{r.speedup_vs_warmup:.1f}x" if r.speedup_vs_warmup else "—"
        click.echo(f"{r.method:<25} {r.wall_seconds:<12.3f} {r.num_forward_passes:<15} {spd}")


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", default="tiny-cnn", show_default=True)
@click.option("--data-shape", default="3,32,32", show_default=True)
@click.option("--n-samples", default=500, show_default=True)
@click.option(
    "--output-dir", "-o", default="./qsdiag_sweep", show_default=True
)
def sweep(model: str, data_shape: str, n_samples: int, output_dir: str) -> None:
    """Run robustness sweep over sample sizes and seeds."""
    from qstatic_diag.evaluation.experiments import run_robustness_sweep
    from qstatic_diag.models.zoo import TinyCNN

    shape = tuple(int(s) for s in data_shape.split(","))
    data = _load_data(n_samples, shape)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def factory():
        return _load_model(model, shape)

    results = run_robustness_sweep(factory, data)

    click.echo("\nRobustness Sweep Results:")
    for r in results:
        click.echo(
            f"  [{r.param}={r.value}]  CV={r.mean_ifd_variance:.4f}  "
            f"rank_stability={r.fraction_stable_rankings:.4f}"
        )

    # Save
    data = [
        {
            "param": r.param,
            "value": str(r.value),
            "mean_ifd_variance": r.mean_ifd_variance,
            "fraction_stable_rankings": r.fraction_stable_rankings,
        }
        for r in results
    ]
    with open(out / "robustness.json", "w") as f:
        json.dump(data, f, indent=2)
    click.echo(f"\nResults saved → {out.resolve()}")


if __name__ == "__main__":
    cli()
