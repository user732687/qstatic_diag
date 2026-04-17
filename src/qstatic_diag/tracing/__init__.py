"""Model tracing: forward-hook capture and topology extraction."""

from qstatic_diag.tracing.hooks import capture_activations, ActivationStore
from qstatic_diag.tracing.topology import extract_topology, get_block_definitions

__all__ = [
    "capture_activations",
    "ActivationStore",
    "extract_topology",
    "get_block_definitions",
]
