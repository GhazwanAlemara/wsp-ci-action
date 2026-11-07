"""WSP Hub Open-Source Client Library & CLI (`wsphub` / `wsp`)."""

from wsp_cli.crypto import generate_ed25519_keypair, sign_wsp_payload, verify_wsp_signature
from wsp_cli.engine import apply_patch, compute_directory_diff_patch, patch_model, revert_patch
from wsp_cli.package_format import (
    WspArchive,
    build_wsp_archive,
    inspect_wsp_archive,
    read_wsp_archive,
    verify_wsp_archive,
)

__version__ = "2.1.0"

__all__ = [
    "WspArchive",
    "apply_patch",
    "build_wsp_archive",
    "compute_directory_diff_patch",
    "generate_ed25519_keypair",
    "inspect_wsp_archive",
    "patch_model",
    "read_wsp_archive",
    "revert_patch",
    "sign_wsp_payload",
    "verify_wsp_archive",
    "verify_wsp_signature",
]
