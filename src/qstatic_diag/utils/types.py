from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Tensor = torch.Tensor
LayerName = str
FilterIdx = int
HeadIdx = int
BlockName = str


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LayerKind(Enum):
    """Coarse classification of a neural-network layer."""

    FC = auto()          # Fully-connected / Linear
    CONV = auto()        # Convolutional (any dimensionality)
    ATTENTION = auto()   # (Multi-head) self-attention
    RESIDUAL = auto()    # Residual / skip-connection block wrapper
    NORM = auto()        # Batch/Layer/Group normalisation
    ACTIVATION = auto()  # Non-linearity (ReLU, GELU, …)
    POOL = auto()        # Pooling
    EMBEDDING = auto()   # Token / positional embedding
    OTHER = auto()       # Catch-all


class VulnerabilityKind(Enum):
    """Label attached to a flagged component."""

    DEAD_FILTER = "dead_filter"
    UNSTABLE_FILTER = "unstable_filter"
    BOTTLENECK_LAYER = "bottleneck_layer"
    INSTABILITY_SOURCE = "instability_source"
    REDUNDANT_LAYER = "redundant_layer"
    OUTLIER_HIGH = "outlier_high"
    OUTLIER_LOW = "outlier_low"
    INEFFICIENT_BLOCK = "inefficient_block"
    INACTIVE_RESIDUAL = "inactive_residual"
    DOMINANT_RESIDUAL = "dominant_residual"
    CORRELATED_PAIR = "correlated_pair"
    ANTICORRELATED_PAIR = "anticorrelated_pair"


# ---------------------------------------------------------------------------
# Layer metadata
# ---------------------------------------------------------------------------


@dataclass
class LayerMeta:
    """Static metadata extracted from a single layer."""

    name: LayerName
    kind: LayerKind
    module_class: str
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    # Convolution specifics
    kernel_size: Optional[Tuple[int, ...]] = None
    stride: Optional[Tuple[int, ...]] = None
    dilation: Optional[Tuple[int, ...]] = None
    groups: Optional[int] = None
    # Attention specifics
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    # Parameter counts
    num_params: int = 0
    # Block membership
    block_name: Optional[BlockName] = None


# ---------------------------------------------------------------------------
# Activation record
# ---------------------------------------------------------------------------


@dataclass
class ActivationRecord:
    """Captured activations and pre-activations for one layer / one forward pass."""

    layer_name: LayerName
    # Post-activation tensor (after sigma)
    output: Tensor
    # Pre-activation (before sigma), if available
    pre_activation: Optional[Tensor] = None
    # For residual blocks: output of the residual *branch* only
    residual_branch_output: Optional[Tensor] = None
    # For attention: per-head outputs stacked [H, B, S, D_v]
    head_outputs: Optional[Tensor] = None
    # Attention weight matrices [H, B, S, S]
    attention_weights: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# Divergence results
# ---------------------------------------------------------------------------


@dataclass
class LayerDivergence:
    """Aggregated IFD metrics for one layer."""

    layer_name: LayerName
    kind: LayerKind
    # Mean divergence across diagnostic samples
    mean: float
    std: float
    # Per-sample divergence vector (length = |D_diag|)
    per_sample: List[float] = field(default_factory=list)
    # Per-filter/head divergence (length = num_filters or num_heads)
    per_filter: Optional[List[float]] = None
    # Normalised contribution to total network divergence
    rho: float = 0.0
    # Bootstrap CI
    ci_low: float = 0.0
    ci_high: float = 0.0
    # Ensemble stats (multi-seed)
    ensemble_mean: Optional[float] = None
    ensemble_std: Optional[float] = None


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticReport:
    """Full vulnerability report produced by the unified pipeline."""

    # Ordered list of per-layer divergences
    layer_divergences: List[LayerDivergence] = field(default_factory=list)
    # Vulnerability sets
    vulnerable_filters: List[Dict[str, Any]] = field(default_factory=list)
    unstable_filters: List[Dict[str, Any]] = field(default_factory=list)
    vulnerable_layers: List[Dict[str, Any]] = field(default_factory=list)
    removable_layers: List[Dict[str, Any]] = field(default_factory=list)
    vulnerable_blocks: List[Dict[str, Any]] = field(default_factory=list)
    block_importance: List[Dict[str, Any]] = field(default_factory=list)
    # CORRDIAG output
    correlation_matrix: Optional[Any] = None  # np.ndarray at runtime
    correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    # Summary stats
    total_layers: int = 0
    total_filters: int = 0
    total_vulnerable_filters: int = 0
    total_vulnerable_layers: int = 0
    # Timing
    wall_time_seconds: float = 0.0
    num_forward_passes: int = 0
    # Config snapshot
    config: Dict[str, Any] = field(default_factory=dict)
