"""Standalone RFC 8785 JSON Canonicalization Scheme (JCS) for `wsphub` CLI."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict


def canonicalize_jcs(value: Any) -> str:
    """Serializes JSON values deterministically per RFC 8785."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("RFC 8785 forbids NaN and Infinity in canonical JSON")
        if value == 0.0:
            return "0"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return json.dumps(value, allow_nan=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonicalize_jcs(v) for v in value) + "]"
    if isinstance(value, dict):
        sorted_items = sorted(value.items(), key=lambda kv: str(kv[0]).encode("utf-16-be"))
        return (
            "{"
            + ",".join(
                f"{json.dumps(str(k), ensure_ascii=False)}:{canonicalize_jcs(v)}"
                for k, v in sorted_items
            )
            + "}"
        )
    raise TypeError(f"Unsupported value type for JCS canonicalization: {type(value)!r}")


def jcs_sha256(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(canonicalize_jcs(payload).encode("utf-8")).hexdigest()
