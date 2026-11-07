"""`.wsp` v2.1 Deterministic Archive Builder, Reader & Signature Verifier."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from wsp_cli.crypto import sign_wsp_payload, verify_wsp_signature

FIXED_ZIP_DATETIME = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class WspArchive:
    manifest: Dict[str, Any]
    deltas_bytes: bytes
    config_blobs: Dict[str, bytes]
    archive_sha256: str


def _make_zip_info(arcname: str) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(filename=arcname, date_time=FIXED_ZIP_DATETIME)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    zi.create_system = 3
    return zi


def build_wsp_archive(
    *,
    manifest: Dict[str, Any],
    deltas_bytes: bytes,
    private_seed_hex: str,
    config_blobs: Optional[Dict[str, bytes]] = None,
    out_path: Optional[Path] = None,
) -> Tuple[bytes, Dict[str, Any], str]:
    """Signs manifest + deltas_bytes + config_blobs and builds a deterministic .wsp v2.1 archive.

    Returns (archive_bytes, signed_manifest, archive_sha256).
    """
    manifest_copy = json.loads(json.dumps(manifest))
    manifest_copy["schema_version"] = "2.1.0"

    # Fold any P0 config blobs into deterministic digest inside manifest before signing
    blobs = dict(sorted((config_blobs or {}).items()))
    if blobs:
        manifest_copy["config_blob_digests"] = {
            k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()
        }

    deltas_sha256 = hashlib.sha256(deltas_bytes).hexdigest()
    manifest_copy["deltas_sha256"] = deltas_sha256

    sig_hex, _, _ = sign_wsp_payload(manifest_copy, deltas_bytes, private_seed_hex)
    manifest_copy["signature_ed25519"] = sig_hex

    manifest_bytes = json.dumps(manifest_copy, indent=2, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr(_make_zip_info("manifest.json"), manifest_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        zf.writestr(_make_zip_info("deltas.bin"), deltas_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        for name, raw in blobs.items():
            safe_name = name.lstrip("/")
            zf.writestr(
                _make_zip_info(f"config_files/{safe_name}"),
                raw,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )

    archive_bytes = buf.getvalue()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(archive_bytes)

    return archive_bytes, manifest_copy, archive_sha256


MAX_ARCHIVE_ENTRIES = 64
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_CONFIG_BLOB_BYTES = 5 * 1024 * 1024
MAX_DELTA_BYTES = 512 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 550 * 1024 * 1024


def _bounded_zip_read(zf: zipfile.ZipFile, name: str, max_bytes: int) -> bytes:
    """Reads at most `max_bytes` from a zip entry, neutralizing spoofed zip header sizes / zip bombs."""
    with zf.open(name, mode="r") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Zip-bomb protection triggered: '{name}' exceeds {max_bytes} byte ceiling")
    return data


def read_wsp_archive(source: Path | bytes) -> WspArchive:
    raw_zip = source.read_bytes() if isinstance(source, Path) else source
    archive_sha256 = hashlib.sha256(raw_zip).hexdigest()

    with zipfile.ZipFile(io.BytesIO(raw_zip), mode="r") as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise ValueError(f"Archive contains too many entries ({len(infos)} > {MAX_ARCHIVE_ENTRIES})")
        declared_total = sum(max(0, info.file_size) for info in infos)
        if declared_total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"Decompressed archive size ({declared_total} B) exceeds safety ceiling ({MAX_TOTAL_UNCOMPRESSED_BYTES} B)"
            )

        names = set(zf.namelist())
        if "manifest.json" not in names or "deltas.bin" not in names:
            raise ValueError("Invalid .wsp archive: missing manifest.json or deltas.bin")

        manifest_raw = _bounded_zip_read(zf, "manifest.json", MAX_MANIFEST_BYTES)
        manifest = json.loads(manifest_raw.decode("utf-8"))
        deltas_bytes = _bounded_zip_read(zf, "deltas.bin", MAX_DELTA_BYTES)

        config_blobs: Dict[str, bytes] = {}
        total_read = len(manifest_raw) + len(deltas_bytes)
        for name in sorted(names):
            if name in ("manifest.json", "deltas.bin"):
                continue
            if name.endswith("/"):
                continue
            if not name.startswith("config_files/"):
                raise ValueError(f"Unexpected entry inside .wsp archive: {name!r}")
            rel = name[len("config_files/") :]
            if not rel or "/" in rel or "\\" in rel or ".." in rel:
                raise ValueError(f"Malicious path inside .wsp archive: {name!r}")
            blob = _bounded_zip_read(zf, name, MAX_CONFIG_BLOB_BYTES)
            total_read += len(blob)
            if total_read > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("Cumulative uncompressed archive bytes exceed safety ceiling")
            config_blobs[rel] = blob

    expected_digests = manifest.get("config_blob_digests", {})
    if not isinstance(expected_digests, dict):
        raise ValueError("manifest.config_blob_digests must be an object")
    if set(config_blobs.keys()) != set(expected_digests.keys()):
        raise ValueError("Archive contains unsigned or unexpected config blobs")
    for rel, expected_h in expected_digests.items():
        actual_h = hashlib.sha256(config_blobs[rel]).hexdigest()
        if actual_h != expected_h:
            raise ValueError(
                f"Tampered config blob '{rel}': expected {expected_h}, got {actual_h}"
            )

    return WspArchive(
        manifest=manifest,
        deltas_bytes=deltas_bytes,
        config_blobs=config_blobs,
        archive_sha256=archive_sha256,
    )


def inspect_wsp_archive(source: Path | bytes) -> Dict[str, Any]:
    arc = read_wsp_archive(source)
    m = arc.manifest
    return {
        "patch_id": m.get("patch_id"),
        "version": m.get("version"),
        "schema_version": m.get("schema_version"),
        "patch_tier": m.get("patch_tier", "P1_WEIGHTS"),
        "publisher_key_id": m.get("publisher_key_id"),
        "supernova_creator_id": m.get("supernova_creator_id"),
        "hf_repo_id": m.get("compatibility", {}).get("hf_repo_id"),
        "base_repo_revision": m.get("compatibility", {}).get("base_repo_revision"),
        "tensor_count": len(m.get("tensor_modifications", [])),
        "config_count": len(m.get("config_modifications", [])),
        "deltas_size_bytes": len(arc.deltas_bytes),
        "archive_sha256": arc.archive_sha256,
        "manifest": m,
    }


def verify_wsp_archive(
    source: Path | bytes,
    *,
    trusted_keys: Dict[str, str],
    revoked_ids: Optional[Iterable[str]] = None,
) -> Tuple[bool, str, WspArchive]:
    arc = read_wsp_archive(source)
    ok, msg = verify_wsp_signature(
        arc.manifest,
        arc.deltas_bytes,
        trusted_keys=trusted_keys,
        revoked_ids=revoked_ids,
    )
    return ok, msg, arc
