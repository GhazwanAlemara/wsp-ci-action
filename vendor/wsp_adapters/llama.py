"""Gated Llama Family Architecture Adapter Plugin (`llama`).

Held behind `production_verified = False` until empirical multi-seed benchmark
validation passes on a reference Llama checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List

from wsp_adapters.base import (
    AdapterValidationError,
    BaseArchitectureAdapter,
    LayerDescriptor,
)


class LlamaAdapter(BaseArchitectureAdapter):
    adapter_id = "llama"
    supported_model_types = ("llama",)
    production_verified = False

    def extract_text_config(self, config_json: Dict[str, Any]) -> Dict[str, Any]:
        text_cfg = config_json.get("text_config") or config_json
        mtype = str(text_cfg.get("model_type") or config_json.get("model_type", ""))
        if mtype not in self.supported_model_types:
            raise AdapterValidationError(f"LlamaAdapter expected model_type='llama', got '{mtype}'")
        return {
            "model_type": mtype,
            "hidden_size": int(text_cfg["hidden_size"]),
            "intermediate_size": int(text_cfg["intermediate_size"]),
            "num_hidden_layers": int(text_cfg["num_hidden_layers"]),
            "num_attention_heads": int(text_cfg["num_attention_heads"]),
            "num_key_value_heads": int(
                text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"])
            ),
        }

    def map_layer_names(
        self, config_json: Dict[str, Any], tensor_headers: Dict[str, Any]
    ) -> List[LayerDescriptor]:
        tcfg = self.extract_text_config(config_json)
        prefix = (
            "language_model.model.layers"
            if any(k.startswith("language_model.model.layers.") for k in tensor_headers)
            else "model.layers"
        )
        return [
            LayerDescriptor(
                layer_index=i,
                attention_kind="grouped_query_attention",
                mlp_down_proj=f"{prefix}.{i}.mlp.down_proj.weight",
                mlp_gate_proj=f"{prefix}.{i}.mlp.gate_proj.weight",
                mlp_up_proj=f"{prefix}.{i}.mlp.up_proj.weight",
            )
            for i in range(tcfg["num_hidden_layers"])
        ]

    def discover_tensors(self, tensor_headers: Dict[str, Any]) -> Dict[str, List[str]]:
        return {
            "mlp_down_proj": sorted(
                k for k in tensor_headers if k.endswith(".mlp.down_proj.weight")
            ),
            "self_attn": sorted(k for k in tensor_headers if ".self_attn." in k),
        }

    def validate_shapes(
        self,
        config_json: Dict[str, Any],
        tensor_headers: Dict[str, Any],
        target_modifications: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        tcfg = self.extract_text_config(config_json)
        expected_down = [tcfg["hidden_size"], tcfg["intermediate_size"]]
        for desc in self.map_layer_names(config_json, tensor_headers):
            if desc.mlp_down_proj not in tensor_headers:
                raise AdapterValidationError(f"Missing Llama tensor: {desc.mlp_down_proj}")
            if list(tensor_headers[desc.mlp_down_proj]["shape"]) != expected_down:
                raise AdapterValidationError(f"Invalid shape on {desc.mlp_down_proj}")
        return {
            "adapter_id": self.adapter_id,
            "production_verified": self.production_verified,
            "mlp_down_proj_shape": expected_down,
        }
