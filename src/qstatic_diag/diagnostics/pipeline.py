from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from qstatic_diag.data.sampling import diversity_sample, stratified_diversity_sample
from qstatic_diag.diagnostics.blockdiag import run_blockdiag
from qstatic_diag.diagnostics.corrdiag import run_corrdiag
from qstatic_diag.diagnostics.flame import run_flame
from qstatic_diag.diagnostics.laydiag import run_laydiag
from qstatic_diag.divergence.ifd import IFDEngine
from qstatic_diag.stats.confidence import add_bootstrap_ci
from qstatic_diag.tracing.hooks import capture_activations
from qstatic_diag.tracing.topology import extract_topology, get_block_definitions
from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.seed import seed_everything
from qstatic_diag.utils.types import DiagnosticReport, LayerDivergence

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All hyperparameters controlling the diagnostic pipeline."""

    # Sample selection
    m: int = 100
    seed: int = 42
    # FLAME
    p_min: float = 0.20
    p_max: float = 0.95
    # LAYDIAG
    gamma_min: float = 0.05
    gamma_max: float = 0.40
    outlier_sigma: float = 2.0
    # BLOCKDIAG
    efficiency_threshold: float = 0.10
    residual_low: float = 0.05
    residual_high: float = 0.80
    # CORRDIAG
    rho_thresh: float = 0.90
    # Confidence
    bootstrap_n: int = 200
    alpha: float = 0.05
    # Multi-seed stability
    num_seeds: int = 1  # >1 enables ensemble stability
    device: str = "cpu"
    # Inference batch size
    batch_size: int = 32


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DiagnosticPipeline:
    """End-to-end diagnostic pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        model: nn.Module,
        proxy_data: Tensor,
        proxy_labels: Optional[Tensor] = None,
    ) -> DiagnosticReport:
        """
        Run the full diagnostic pipeline.

        Parameters
        ----------
        model:
            PyTorch model.
        proxy_data:
            Unlabeled or labeled proxy dataset tensor (N, ...).
        proxy_labels:
            Optional class labels for stratified sampling.

        Returns
        -------
        :class:`~qstatic_diag.utils.types.DiagnosticReport`
        """
        cfg = self.config
        t0 = time.time()
        seed_everything(cfg.seed)

        device = torch.device(cfg.device)
        model = model.to(device).eval()

        # -----------------------------------------------------------
        # Phase 1: Sample selection
        # -----------------------------------------------------------
        log.info("Phase 1: Selecting %d diagnostic samples…", cfg.m)
        if proxy_labels is not None:
            diag_data, diag_labels, _ = stratified_diversity_sample(
                proxy_data, proxy_labels, cfg.m, seed=cfg.seed
            )
        else:
            diag_data, _ = diversity_sample(proxy_data, cfg.m, seed=cfg.seed)
            diag_labels = None

        diag_data = diag_data.to(device)

        # -----------------------------------------------------------
        # Phase 2: Topology extraction
        # -----------------------------------------------------------
        log.info("Phase 2: Extracting network topology…")
        metas = extract_topology(model)
        block_defs = get_block_definitions(metas)
        layer_names = [m.name for m in metas]

        # -----------------------------------------------------------
        # Phase 3: Forward pass(es) and IFD computation
        # -----------------------------------------------------------
        log.info("Phase 3: Computing IFD (m=%d samples)…", len(diag_data))
        engine = IFDEngine(model, metas)

        if cfg.num_seeds > 1:
            layer_divs = self._multi_seed_forward(
                model, metas, engine, diag_data, cfg, device
            )
        else:
            layer_divs = self._single_seed_forward(model, engine, diag_data, cfg, device)

        # Bootstrap CI
        layer_divs = add_bootstrap_ci(layer_divs, n_bootstrap=cfg.bootstrap_n, alpha=cfg.alpha)

        num_fwd = len(diag_data) * cfg.num_seeds

        # -----------------------------------------------------------
        # Phase 4: Multi-scale vulnerability detection
        # -----------------------------------------------------------
        log.info("Phase 4: Running diagnostic algorithms…")

        vuln_filters, unstable_filters = run_flame(
            layer_divs, p_min=cfg.p_min, p_max=cfg.p_max
        )
        vuln_layers, removable_layers = run_laydiag(
            layer_divs,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
            outlier_sigma=cfg.outlier_sigma,
        )
        vuln_blocks, block_importance = run_blockdiag(
            layer_divs,
            block_defs,
            efficiency_threshold=cfg.efficiency_threshold,
            residual_low=cfg.residual_low,
            residual_high=cfg.residual_high,
        )
        corr_matrix, corr_pairs = run_corrdiag(layer_divs, rho_thresh=cfg.rho_thresh)

        # -----------------------------------------------------------
        # Phase 5: Assemble report
        # -----------------------------------------------------------
        wall = time.time() - t0
        report = DiagnosticReport(
            layer_divergences=layer_divs,
            vulnerable_filters=vuln_filters,
            unstable_filters=unstable_filters,
            vulnerable_layers=vuln_layers,
            removable_layers=removable_layers,
            vulnerable_blocks=vuln_blocks,
            block_importance=block_importance,
            correlation_matrix=corr_matrix,
            correlated_pairs=corr_pairs,
            total_layers=len(metas),
            total_filters=sum(
                len(ld.per_filter) for ld in layer_divs if ld.per_filter
            ),
            total_vulnerable_filters=len(vuln_filters),
            total_vulnerable_layers=len(vuln_layers),
            wall_time_seconds=wall,
            num_forward_passes=num_fwd,
            config=vars(cfg),
        )

        log.info(
            "Diagnosis complete in %.1fs  |  %d vuln filters  |  %d vuln layers",
            wall,
            len(vuln_filters),
            len(vuln_layers),
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_seed_forward(
        self,
        model: nn.Module,
        engine: IFDEngine,
        diag_data: Tensor,
        cfg: PipelineConfig,
        device: torch.device,
    ) -> List[LayerDivergence]:
        with torch.no_grad():
            with capture_activations(model) as store:
                for i in range(0, len(diag_data), cfg.batch_size):
                    batch = diag_data[i : i + cfg.batch_size]
                    try:
                        model(batch)
                    except Exception as exc:
                        log.warning("Forward pass error: %s", exc)
        return engine.compute(store)

    def _multi_seed_forward(
        self,
        model: nn.Module,
        metas,
        engine: IFDEngine,
        diag_data: Tensor,
        cfg: PipelineConfig,
        device: torch.device,
    ) -> List[LayerDivergence]:
        """Run diagnostics."""
        import copy
        import statistics

        seed_results: List[List[LayerDivergence]] = []

        for r in range(cfg.num_seeds):
            seed_everything(cfg.seed + r)
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(device).eval()
            eng_copy = IFDEngine(model_copy, metas)

            with torch.no_grad():
                with capture_activations(model_copy) as store:
                    for i in range(0, len(diag_data), cfg.batch_size):
                        batch = diag_data[i : i + cfg.batch_size]
                        try:
                            model_copy(batch)
                        except Exception as exc:
                            log.warning("Multi-seed forward error (seed %d): %s", r, exc)
            seed_results.append(eng_copy.compute(store))

        # Merge: compute ensemble mean/std per layer
        merged: List[LayerDivergence] = []
        base = seed_results[0]
        for idx, ld_base in enumerate(base):
            run_means = [
                seed_results[r][idx].mean
                for r in range(cfg.num_seeds)
                if idx < len(seed_results[r])
            ]
            ens_mean = statistics.mean(run_means)
            ens_std = statistics.stdev(run_means) if len(run_means) > 1 else 0.0
            ld_base.ensemble_mean = ens_mean
            ld_base.ensemble_std = ens_std
            merged.append(ld_base)

        return merged
