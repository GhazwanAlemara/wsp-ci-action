"""Strictly Opt-In Telemetry Client for `wsp` CLI.

Default behavior: ZERO network requests. Telemetry is transmitted ONLY if the user
explicitly passes `--opt-in-telemetry` or sets `WSP_TELEMETRY_OPT_IN=1`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import httpx


def is_telemetry_enabled(explicit_flag: bool = False) -> bool:
    if explicit_flag:
        return True
    return os.getenv("WSP_TELEMETRY_OPT_IN", "0").strip() in {"1", "true", "yes"}


def emit_opt_in_event(
    event_type: str,
    *,
    patch_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    explicit_flag: bool = False,
    api_base_url: Optional[str] = None,
) -> bool:
    if not is_telemetry_enabled(explicit_flag):
        return False

    base = (api_base_url or os.getenv("WSP_API_URL", "https://api.wsphub.dev")).rstrip("/")
    try:
        with httpx.Client(timeout=2.5, follow_redirects=False) as client:
            resp = client.post(
                f"{base}/api/v1/telemetry/opt-in",
                json={
                    "event_type": event_type,
                    "patch_id": patch_id,
                    "metadata": metadata or {},
                },
            )
            return resp.status_code < 300
    except Exception:
        return False
