"""Base Interface & Validation Gate Contract for WSP Architecture Adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class AdapterValidationError(ValueError):
    """Raised when checkpoint headers, shapes, or plugin readiness gates fail."""


@dataclass(frozen=True)
class LayerDescriptor:
    layer_index: int
    attention_kind: str
    mlp_down_proj: str
    mlp_gate_proj: str
    mlp_up_proj: str


class BaseArchitectureAdapter(ABC):
    """Contract required for every model family onboarded to WSP Hub."""

    adapter_id: str = "base"
    supported_model_types: Tuple[str, ...] = ()
    production_verified: bool = False

    def enforce_readiness_gate(self, *, allow_experimental: bool = False) -> None:
        if not self.production_verified and not allow_experimental:
            raise AdapterValidationError(
                f"Adapter '{self.adapter_id}' is currently behind the Experimental Validation Gate "
                "and is not yet unlocked for public wsphub.dev publishing."
            )

    @abstractmethod
    def extract_text_config(self, config_json: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes text transformer dimensions from Hugging Face config.json."""

    @abstractmethod
    def map_layer_names(self, config_json: Dict[str, Any], tensor_headers: Dict[str, Any]) -> List[LayerDescriptor]:
        """Maps layer indices 0..L-1 to exact safetensors parameter names."""

    @abstractmethod
    def discover_tensors(self, tensor_headers: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorizes checkpoint tensors into surgical target groups."""

    @abstractmethod
    def validate_shapes(
        self,
        config_json: Dict[str, Any],
        tensor_headers: Dict[str, Any],
        target_modifications: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Validates architecture invariants and optional candidate tensor modifications."""
