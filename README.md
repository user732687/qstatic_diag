# Quasi-Static Topological Diagnosis Framework
---

## Overview

Training large neural networks is expensive.  Many architectural flaws (dead filters, unstable layers, redundant blocks) are detectable before a single gradient update, from the signal propagation statistics of a randomly-initialised network.

`qstatic_diag` implements the Information Flow Divergence (IFD) framework.

## Diagnostic Algorithms

### FLAME (Filter-Level)
Computes per-filter IFD across all diagnostic samples.  Flags filters below the `p_min` percentile as *dead* and above `p_max` as *unstable*.

### LAYDIAG (Layer-Level)
Normalises per-layer divergence as ρ_l = D_l / Σ D_l.  Flags layers below `gamma_min` (bottleneck), above `gamma_max` (instability source), or as statistical outliers (±2σ).

### BLOCKDIAG (Block-Level)
Groups layers into architectural blocks (residual blocks, transformer encoder layers).  Computes block efficiency η_B = D_B / D_B^{io} and residual dominance δ_B.

### CORRDIAG (Cross-Layer)
Pearson correlation between per-sample divergence vectors across all layer pairs.  Pairs with |r| > `rho_thresh` indicate functional redundancy (r ≈ +1) or destructive interference (r ≈ −1).  BH FDR correction applied by default.

---

## Quick Start

### Install

```bash
git clone https://github.com/.../qstatic_diag.git
cd qstatic_diag
pip install -e ".[dev]"
```

---

## Project Structure

```
src/qstatic_diag/
├── data/           # Diversity sampling 
├── models/         # Models folder
├── tracing/        # PyTorch forward-hook activation capture, topology extraction
├── divergence/     # IFD computation (FC, Conv, Attention, Residual)
├── diagnostics/    # FLAME, LAYDIAG, BLOCKDIAG, CORRDIAG, unified pipeline
├── stats/          # Stats
├── evaluation/     # Baselines, experiment pipelines
├── reporting/      # JSON / CSV / Markdown export
├── visualization/  # Heatmaps, rankings, histograms, correlation matrix
├── cli/            # Click CLI
└── utils/          # Types, logging, seeding
configs/            # YAML configs
scripts/            # Scripts
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{qstatic2026,
  title   = {},
  year    = {2026},
}
```

---

## License

MIT License.
