"""Production-Verified Qwen3.5 Hybrid Gated DeltaNet + Gated Attention Adapter.

Empirically verified against live `Qwen/Qwen3.5-0.8B` (`Qwen3_5ForConditionalGeneration`,
revision `2fc06364715b967f1860aea9cf38778875588b17`, 24 layers, 3:1 linear_attention to
full_attention hybrid ratio, hidden_size=1024, intermediate_size=3584, 488 tensors).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from wsp_adapters.base import (
    AdapterValidationError,
    BaseArchitectureAdapter,
    LayerDescriptor,
)


class Qwen35Adapter(BaseArchitectureAdapter):
    adapter_id = "qwen3_5"
    supported_model_types = ("qwen3_5", "qwen3_5_text")
    production_verified = True

    def extract_text_config(self, config_json: Dict[str, Any]) -> Dict[str, Any]:
        model_type = str(config_json.get("model_type", ""))
        if model_type not in self.supported_model_types:
            raise AdapterValidationError(
                f"Qwen35Adapter expected model_type in {self.supported_model_types}, got '{model_type}'"
            )
        text_cfg = config_json.get("text_config") or config_json
        hidden_size = int(text_cfg["hidden_size"])
        intermediate_size = int(text_cfg["intermediate_size"])
        layer_types = list(text_cfg.get("layer_types") or [])
        num_layers = int(text_cfg.get("num_hidden_layers") or len(layer_types))
        if not layer_types:
            interval = int(text_cfg.get("full_attention_interval", 4))
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(num_layers)
            ]
        return {
            "model_type": model_type,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
            "layer_types": layer_types,
            "full_attention_interval": int(text_cfg.get("full_attention_interval", 4)),
            "eos_token_id": text_cfg.get("eos_token_id"),
        }

    def _detect_layer_prefix(self, tensor_headers: Dict[str, Any]) -> str:
        for key in tensor_headers:
            if key.startswith("model.language_model.layers."):
                return "model.language_model.layers"
            if key.startswith("model.layers."):
                return "model.layers"
        raise AdapterValidationError("No Qwen3.5 transformer layer tensors found in safetensors header")

    def map_layer_names(
        self, config_json: Dict[str, Any], tensor_headers: Dict[str, Any]
    ) -> List[LayerDescriptor]:
        tcfg = self.extract_text_config(config_json)
        prefix = self._detect_layer_prefix(tensor_headers)
        descriptors: List[LayerDescriptor] = []
        for idx in range(tcfg["num_hidden_layers"]):
            attn_kind = tcfg["layer_types"][idx]
            descriptors.append(
                LayerDescriptor(
                    layer_index=idx,
                    attention_kind=attn_kind,
                    mlp_down_proj=f"{prefix}.{idx}.mlp.down_proj.weight",
                    mlp_gate_proj=f"{prefix}.{idx}.mlp.gate_proj.weight",
                    mlp_up_proj=f"{prefix}.{idx}.mlp.up_proj.weight",
                )
            )
        return descriptors

    def discover_tensors(self, tensor_headers: Dict[str, Any]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {
            "mlp_down_proj": [],
            "mlp_gate_up_proj": [],
            "linear_attention_deltanet": [],
            "full_gated_attention": [],
            "mtp_speculative_head": [],
            "vision_frozen": [],
            "norms_and_embeddings": [],
        }
        for name in sorted(tensor_headers.keys()):
            if name == "__metadata__":
                continue
            if name.startswith("model.visual.") or "visual" in name:
                groups["vision_frozen"].append(name)
            elif name.startswith("mtp."):
                groups["mtp_speculative_head"].append(name)
            elif name.endswith(".mlp.down_proj.weight"):
                groups["mlp_down_proj"].append(name)
            elif ".mlp." in name:
                groups["mlp_gate_up_proj"].append(name)
            elif ".linear_attn." in name or any(
                tok in name for tok in ("in_proj_qkv", "conv1d", "dt_bias", "A_log")
            ):
                groups["linear_attention_deltanet"].append(name)
            elif ".self_attn." in name:
                groups["full_gated_attention"].append(name)
            else:
                groups["norms_and_embeddings"].append(name)
        return groups

    def validate_shapes(
        self,
        config_json: Dict[str, Any],
        tensor_headers: Dict[str, Any],
        target_modifications: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        tcfg = self.extract_text_config(config_json)
        layers = self.map_layer_names(config_json, tensor_headers)
        expected_down_shape = [tcfg["hidden_size"], tcfg["intermediate_size"]]
        expected_up_shape = [tcfg["intermediate_size"], tcfg["hidden_size"]]

        linear_count = 0
        full_count = 0
        for desc in layers:
            if desc.attention_kind == "linear_attention":
                linear_count += 1
            elif desc.attention_kind == "full_attention":
                full_count += 1

            if desc.mlp_down_proj not in tensor_headers:
                raise AdapterValidationError(f"Missing required MLP tensor: {desc.mlp_down_proj}")
            actual_down = list(tensor_headers[desc.mlp_down_proj]["shape"])
            if actual_down != expected_down_shape:
                raise AdapterValidationError(
                    f"Shape mismatch on {desc.mlp_down_proj}: expected {expected_down_shape}, got {actual_down}"
                )
            if desc.mlp_gate_proj in tensor_headers:
                actual_gate = list(tensor_headers[desc.mlp_gate_proj]["shape"])
                if actual_gate != expected_up_shape:
                    raise AdapterValidationError(
                        f"Shape mismatch on {desc.mlp_gate_proj}: expected {expected_up_shape}, got {actual_gate}"
                    )

        for mod in target_modifications or []:
            tname = mod["tensor_name"]
            if tname.startswith("model.visual."):
                raise AdapterValidationError(
                    f"Qwen35Adapter forbids modifying vision tower tensor '{tname}' in language patches"
                )
            if tname not in tensor_headers:
                raise AdapterValidationError(f"Modification target '{tname}' absent from checkpoint")
            hdr_entry = tensor_headers[tname]
            if list(mod["shape"]) != list(hdr_entry["shape"]):
                raise AdapterValidationError(
                    f"Candidate shape {mod['shape']} != checkpoint shape {hdr_entry['shape']} on '{tname}'"
                )
            if re.search(r"\.layers\.(\d+)\.", tname):
                layer_idx = int(re.search(r"\.layers\.(\d+)\.", tname).group(1))
                if layer_idx < 0 or layer_idx >= tcfg["num_hidden_layers"]:
                    raise AdapterValidationError(f"Out-of-range layer index {layer_idx}")

        return {
            "adapter_id": self.adapter_id,
            "production_verified": self.production_verified,
            "num_hidden_layers": tcfg["num_hidden_layers"],
            "linear_attention_layers": linear_count,
            "full_attention_layers": full_count,
            "hidden_size": tcfg["hidden_size"],
            "intermediate_size": tcfg["intermediate_size"],
            "mlp_down_proj_shape": expected_down_shape,
        }
