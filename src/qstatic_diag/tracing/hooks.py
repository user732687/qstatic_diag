from __future__ import annotations

import contextlib
from typing import Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn

from qstatic_diag.utils.logging import get_logger
from qstatic_diag.utils.types import ActivationRecord, LayerKind, LayerMeta

log = get_logger(__name__)

_RESIDUAL_PATTERNS = {"ResidualBlock", "BasicBlock", "Bottleneck", "ResLayer", "Block"}

_ATTENTION_CLASSES = (nn.MultiheadAttention,)


# ---------------------------------------------------------------------------
# Activation store
# ---------------------------------------------------------------------------


class ActivationStore:
    """Accumulates ActivationRecord objects across forward passes."""

    def __init__(self) -> None:
        self._store: Dict[str, List[ActivationRecord]] = {}

    def push(self, record: ActivationRecord) -> None:
        self._store.setdefault(record.layer_name, []).append(record)

    def get(self, layer_name: str) -> List[ActivationRecord]:
        return self._store.get(layer_name, [])

    def all_names(self) -> List[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def _make_standard_hook(
    store: ActivationStore,
    name: str,
) -> Callable:
    """Return a hook that captures module outputs."""

    def hook(
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor | Tuple,
    ) -> None:
        if isinstance(output, (tuple, list)):
            tensor = output[0]
        else:
            tensor = output

        record = ActivationRecord(
            layer_name=name,
            output=tensor.detach().cpu(),
        )
        store.push(record)

    return hook


def _make_attention_hook(
    store: ActivationStore,
    name: str,
) -> Callable:
    """Return a hook for nn.MultiheadAttention capturing per-head outputs."""

    def hook(
        module: nn.MultiheadAttention,
        inputs: Tuple,
        output: Tuple,
    ) -> None:
        # output = (attn_output [S, B, E], attn_weights [B, S, S] or None)
        attn_out, attn_weights = output
        record = ActivationRecord(
            layer_name=name,
            output=attn_out.detach().cpu(),
            attention_weights=attn_weights.detach().cpu() if attn_weights is not None else None,
        )
        store.push(record)

    return hook


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_activations(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
) -> Generator[ActivationStore, None, None]:
    """
    Context manager that registers forward hooks and yields an ActivationStore.

    Usage::

        with capture_activations(model) as store:
            model(x)
        records = store.get("layer_name")

    Parameters
    ----------
    model:
        The PyTorch module to instrument.
    layer_names:
        If provided, only instrument layers whose *named_modules* name is in
        this list.  If ``None``, instrument every leaf module.
    """
    store = ActivationStore()
    handles: List[torch.utils.hooks.RemovableHook] = []

    target_set = set(layer_names) if layer_names else None

    for name, module in model.named_modules():
        if not _is_leaf_or_attention(module):
            continue
        if target_set and name not in target_set:
            continue

        if isinstance(module, _ATTENTION_CLASSES):
            # Request attention weights
            _patch_attn_need_weights(module)
            h = module.register_forward_hook(_make_attention_hook(store, name))
        else:
            h = module.register_forward_hook(_make_standard_hook(store, name))
        handles.append(h)

    log.debug("Registered %d forward hooks.", len(handles))

    try:
        yield store
    finally:
        for h in handles:
            h.remove()
        log.debug("Removed %d forward hooks.", len(handles))


def _is_leaf_or_attention(module: nn.Module) -> bool:
    """True for leaf modules or attention modules we care about."""
    if isinstance(module, _ATTENTION_CLASSES):
        return True
    return len(list(module.children())) == 0


def _patch_attn_need_weights(module: nn.MultiheadAttention) -> None:
    """Monkey-patch forward to always return attention weights."""
    original_forward = module.forward

    def patched_forward(
        query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, **kw
    ):
        return original_forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            **kw,
        )

    module.forward = patched_forward
