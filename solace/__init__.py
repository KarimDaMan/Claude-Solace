"""Claude Solace model package.

Exposes a Llama-family decoder-only transformer used by both
``claude-solace-local`` (16 GB RAM target) and ``claude-solace-hf``
(32 GB RAM target) variants. The two variants share this code and
differ only in their ``config.json``.
"""

from .configuration_solace import SolaceConfig
from .modeling_solace import (
    SolaceForCausalLM,
    SolaceModel,
    SolaceDecoderLayer,
    SolaceAttention,
    SolaceMLP,
    SolaceRMSNorm,
)

__all__ = [
    "SolaceConfig",
    "SolaceForCausalLM",
    "SolaceModel",
    "SolaceDecoderLayer",
    "SolaceAttention",
    "SolaceMLP",
    "SolaceRMSNorm",
]
