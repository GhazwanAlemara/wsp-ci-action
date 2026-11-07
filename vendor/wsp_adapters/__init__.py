"""WSP Hub Architecture Adapter Plugin Registry."""

from __future__ import annotations

from typing import Dict, Type

from wsp_adapters.base import (
    AdapterValidationError,
    BaseArchitectureAdapter,
    LayerDescriptor,
)
from wsp_adapters.gemma import GemmaAdapter
from wsp_adapters.llama import LlamaAdapter
from wsp_adapters.qwen3_5 import Qwen35Adapter

ADAPTER_REGISTRY: Dict[str, Type[BaseArchitectureAdapter]] = {
    "qwen3_5": Qwen35Adapter,
    "llama": LlamaAdapter,
    "gemma": GemmaAdapter,
}


def get_adapter(adapter_id: str, *, allow_experimental: bool = False) -> BaseArchitectureAdapter:
    cls = ADAPTER_REGISTRY.get(adapter_id)
    if cls is None:
        raise AdapterValidationError(
            f"Unknown architecture adapter '{adapter_id}'. Registered: {sorted(ADAPTER_REGISTRY)}"
        )
    instance = cls()
    instance.enforce_readiness_gate(allow_experimental=allow_experimental)
    return instance


__all__ = [
    "ADAPTER_REGISTRY",
    "AdapterValidationError",
    "BaseArchitectureAdapter",
    "GemmaAdapter",
    "LayerDescriptor",
    "LlamaAdapter",
    "Qwen35Adapter",
    "get_adapter",
]
