"""Ed25519 Cryptographic Signing, Key Rotation & CRL Verification for `.wsp` v2.1."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any, Dict, Iterable, Optional, Tuple
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from wsp_cli.jcs import canonicalize_jcs

DOMAIN_SEPARATOR = "WSPHUB-PATCH-V2.1"


def generate_ed25519_keypair() -> Tuple[str, str]:
    """Returns (private_seed_hex, public_key_hex)."""
    seed = secrets.token_bytes(32)
    priv = Ed25519PrivateKey.from_private_bytes(seed)
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return seed.hex(), pub.hex()


def public_key_from_private_seed(private_seed_hex: str) -> str:
    seed = bytes.fromhex(private_seed_hex.strip())
    if len(seed) != 32:
        raise ValueError("Ed25519 private seed must be 32 bytes (64 hex chars)")
    priv = Ed25519PrivateKey.from_private_bytes(seed)
    return (
        priv.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )


def manifest_without_signature(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "signature_ed25519"}


def build_signing_message(manifest: Dict[str, Any], deltas_bytes: bytes) -> Tuple[bytes, str, str]:
    """Builds the domain-separated message covering RFC 8785 manifest + deltas.bin SHA-256."""
    deltas_sha256 = hashlib.sha256(deltas_bytes).hexdigest()
    unsigned = manifest_without_signature(manifest)
    if "deltas_sha256" in unsigned and unsigned["deltas_sha256"] != deltas_sha256:
        raise ValueError(
            f"Manifest deltas_sha256 ({unsigned['deltas_sha256']}) does not match deltas.bin ({deltas_sha256})"
        )
    canonical_manifest = canonicalize_jcs(unsigned)
    manifest_sha256 = hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest()
    msg = f"{DOMAIN_SEPARATOR}\n{canonical_manifest}\n{deltas_sha256}".encode("utf-8")
    return msg, manifest_sha256, deltas_sha256


def sign_wsp_payload(
    manifest: Dict[str, Any], deltas_bytes: bytes, private_seed_hex: str
) -> Tuple[str, str, str]:
    """Returns (signature_hex, manifest_canonical_sha256, deltas_sha256)."""
    seed = bytes.fromhex(private_seed_hex.strip())
    if len(seed) != 32:
        raise ValueError("Ed25519 private seed must be 32 bytes (64 hex chars)")
    priv = Ed25519PrivateKey.from_private_bytes(seed)
    msg, manifest_sha256, deltas_sha256 = build_signing_message(manifest, deltas_bytes)
    sig = priv.sign(msg)
    return sig.hex(), manifest_sha256, deltas_sha256


def verify_wsp_signature(
    manifest: Dict[str, Any],
    deltas_bytes: bytes,
    *,
    trusted_keys: Dict[str, str],
    revoked_ids: Optional[Iterable[str]] = None,
) -> Tuple[bool, str]:
    """Verifies Ed25519 signature, deltas_sha256, key trust, and CRL revocation status."""
    revoked_set = set(revoked_ids or ())
    patch_id = str(manifest.get("patch_id", ""))
    key_id = str(manifest.get("publisher_key_id", ""))
    sig_hex = str(manifest.get("signature_ed25519", ""))

    if patch_id in revoked_set:
        return False, f"Patch '{patch_id}' has been cryptographically REVOKED in CRL."
    if key_id in revoked_set:
        return False, f"Publisher key '{key_id}' has been REVOKED in CRL."
    if not key_id or key_id not in trusted_keys:
        return False, f"Untrusted or unknown publisher_key_id: '{key_id}'."
    if not sig_hex or len(sig_hex) != 128:
        return False, "Missing or malformed Ed25519 signature hex (expected 128 hex chars)."

    try:
        msg, _, deltas_sha256 = build_signing_message(manifest, deltas_bytes)
        pub_bytes = bytes.fromhex(trusted_keys[key_id])
        pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
        pub.verify(bytes.fromhex(sig_hex), msg)
    except InvalidSignature:
        return False, "Invalid Ed25519 signature over canonical manifest and deltas.bin."
    except Exception as exc:
        return False, f"Signature verification error: {exc}"

    return True, f"Verified Ed25519 signature via '{key_id}' (deltas_sha256={deltas_sha256[:16]}...)."
