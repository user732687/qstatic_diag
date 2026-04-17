from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch.nn as nn

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import LayerKind, LayerMeta

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristic kind detection
# ---------------------------------------------------------------------------


def _detect_kind(module: nn.Module) -> LayerKind:
    cls = type(module)
    name = cls.__name__

    if isinstance(module, (nn.Linear,)):
        return LayerKind.FC
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return LayerKind.CONV
    if isinstance(module, nn.MultiheadAttention):
        return LayerKind.ATTENTION
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                             nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
        return LayerKind.NORM
    if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid,
                             nn.Tanh, nn.LeakyReLU, nn.ELU, nn.Mish)):
        return LayerKind.ACTIVATION
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool2d,
                             nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        return LayerKind.POOL
    if isinstance(module, nn.Embedding):
        return LayerKind.EMBEDDING

    lowname = name.lower()
    if "residual" in lowname or re.match(r"(basic|bottleneck)block", lowname):
        return LayerKind.RESIDUAL

    return LayerKind.OTHER


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


# ---------------------------------------------------------------------------
# Topology builder
# ---------------------------------------------------------------------------


def extract_topology(model: nn.Module) -> List[LayerMeta]:
    """
    Walk ``model.named_modules()`` and return a list of :class:`LayerMeta`
    for every leaf module.
    """
    metas: List[LayerMeta] = []
    seen: set = set()

    # Collect all modules with their paths
    all_modules: List[Tuple[str, nn.Module]] = list(model.named_modules())

    # Build a set of all non-leaf paths (to detect blocks)
    all_paths = {name for name, _ in all_modules}

    def _block_of(path: str) -> Optional[str]:
        """Return the immediate parent path that looks like a named block."""
        parts = path.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:depth])
            parent_mod = dict(all_modules).get(candidate)
            if parent_mod is not None:
                kind = _detect_kind(parent_mod)
                if kind == LayerKind.RESIDUAL:
                    return candidate
        return None

    for name, module in all_modules:
        if name in seen:
            continue

        is_leaf = len(list(module.children())) == 0
        is_attn = isinstance(module, nn.MultiheadAttention)
        if not (is_leaf or is_attn):
            continue

        seen.add(name)
        kind = _detect_kind(module)

        meta = LayerMeta(
            name=name,
            kind=kind,
            module_class=type(module).__name__,
            num_params=_count_params(module),
            block_name=_block_of(name),
        )

        if isinstance(module, nn.Linear):
            meta.in_features = module.in_features
            meta.out_features = module.out_features

        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            meta.in_features = module.in_channels
            meta.out_features = module.out_channels
            meta.kernel_size = module.kernel_size
            meta.stride = module.stride
            meta.dilation = module.dilation
            meta.groups = module.groups

        elif isinstance(module, nn.MultiheadAttention):
            meta.num_heads = module.num_heads
            meta.in_features = module.embed_dim
            meta.head_dim = module.head_dim if hasattr(module, "head_dim") else (
                module.embed_dim // module.num_heads
            )

        metas.append(meta)

    log.info(
        "Topology: %d layers extracted  (%d Conv, %d FC, %d Attn).",
        len(metas),
        sum(1 for m in metas if m.kind == LayerKind.CONV),
        sum(1 for m in metas if m.kind == LayerKind.FC),
        sum(1 for m in metas if m.kind == LayerKind.ATTENTION),
    )
    return metas


def get_block_definitions(metas: List[LayerMeta]) -> Dict[str, List[str]]:
    """
    Group layer names by their ``block_name`` field.

    Returns
    -------
    Dict mapping block_name → list of layer names in that block.
    """
    blocks: Dict[str, List[str]] = {}
    for m in metas:
        if m.block_name:
            blocks.setdefault(m.block_name, []).append(m.name)
    return blocks
