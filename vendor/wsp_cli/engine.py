"""Zero-Loss Atomic Copy-on-Write (CoW), Strided/Sparse Tensor Patcher & Rollback Engine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

from wsp_cli.bf16 import (
    SAFETENSORS_TO_DTYPE,
    bytes_to_fp32,
    compute_delta_norms,
    fp32_to_bytes,
    raw_word_dtype,
)
from wsp_cli.package_format import (
    WspArchive,
    build_wsp_archive,
    read_wsp_archive,
    verify_wsp_archive,
)
from wsp_cli.safetensors_io import (
    TensorSpan,
    discover_model_tensors,
    is_hf_cache_path,
    read_tensor_bytes,
    sha256_file,
    write_tensor_bytes_at_span,
)


def safe_link_or_copy(src: Path, dst: Path) -> str:
    """Hard-links unchanged files on local POSIX/NTFS; falls back cleanly to copy across drives/virtual FS."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not is_hf_cache_path(src) and not src.is_symlink():
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def _resolve_contained_path(root_dir: Path, rel_path: str) -> Path:
    """Strictly enforces that `rel_path` stays inside `root_dir` (blocks absolute & `..` traversal)."""
    raw = str(rel_path or "").strip()
    if not raw or raw in (".", "..") or raw.startswith(("/", "\\")) or (len(raw) >= 2 and raw[1] == ":"):
        raise ValueError(f"Absolute or empty target path forbidden: {rel_path!r}")
    rel_obj = Path(raw)
    if rel_obj.is_absolute() or any(part in ("..", ".", "") for part in rel_obj.parts):
        raise ValueError(f"Path traversal detected in target path: {rel_path!r}")
    root_resolved = root_dir.resolve()
    candidate = root_resolved / rel_obj
    if candidate.is_symlink():
        raise PermissionError(f"Symlink target forbidden inside model directory: {rel_path!r}")
    resolved = candidate.resolve()
    if resolved == root_resolved or not resolved.is_relative_to(root_resolved):
        raise PermissionError(f"Target path escapes model root directory: {rel_path!r}")
    return resolved


def apply_tensor_modification_to_bytes(
    pre_bytes: bytes,
    mod: Dict[str, Any],
    delta_slice: bytes,
) -> Tuple[bytes, bytes]:
    """Applies a single v2.1 tensor modification and returns (post_bytes, minimal_rollback_bytes)."""
    shape = tuple(int(d) for d in mod["shape"])
    if not shape or any(d <= 0 for d in shape):
        raise ValueError(f"Invalid tensor shape: {shape!r}")
    dtype_name = str(mod["dtype"])
    encoding = str(mod["encoding"])
    wd = raw_word_dtype(dtype_name)

    pre_sha = hashlib.sha256(pre_bytes).hexdigest()
    if pre_sha != mod["pre_patch_tensor_sha256"]:
        raise ValueError(
            f"Pre-patch tensor SHA-256 mismatch on '{mod['tensor_name']}': "
            f"expected {mod['pre_patch_tensor_sha256']}, got {pre_sha}"
        )

    words = np.frombuffer(pre_bytes, dtype=wd).reshape(shape).copy()

    if encoding == "column_sparse":
        if len(shape) != 2:
            raise ValueError("column_sparse requires a 2D tensor")
        cols = [int(i) for i in mod["indices"]]
        if not cols or any(c < 0 or c >= shape[1] for c in cols):
            raise ValueError(f"Out-of-bounds column_sparse indices on '{mod['tensor_name']}'")
        rollback_bytes = np.ascontiguousarray(words[:, cols]).tobytes()
        replacement = np.frombuffer(delta_slice, dtype=wd).reshape(shape[0], len(cols))
        words[:, cols] = replacement
        post_bytes = words.tobytes()

    elif encoding == "row_sparse":
        if len(shape) != 2:
            raise ValueError("row_sparse requires a 2D tensor")
        rows = [int(i) for i in mod["indices"]]
        if not rows or any(r < 0 or r >= shape[0] for r in rows):
            raise ValueError(f"Out-of-bounds row_sparse indices on '{mod['tensor_name']}'")
        rollback_bytes = np.ascontiguousarray(words[rows, :]).tobytes()
        replacement = np.frombuffer(delta_slice, dtype=wd).reshape(len(rows), shape[1])
        words[rows, :] = replacement
        post_bytes = words.tobytes()

    elif encoding == "dense_slice":
        bounds = mod.get("slice_bounds") or [[0, d] for d in shape]
        if len(bounds) != len(shape):
            raise ValueError(f"dense_slice bounds rank mismatch on '{mod['tensor_name']}'")
        for dim_idx, b in enumerate(bounds):
            if len(b) != 2 or int(b[0]) < 0 or int(b[1]) <= int(b[0]) or int(b[1]) > shape[dim_idx]:
                raise ValueError(f"Out-of-bounds dense_slice range {b!r} for dim {dim_idx}")
        slices = tuple(slice(int(b[0]), int(b[1])) for b in bounds)
        sub_shape = tuple(int(b[1]) - int(b[0]) for b in bounds)
        rollback_bytes = np.ascontiguousarray(words[slices]).tobytes()
        replacement = np.frombuffer(delta_slice, dtype=wd).reshape(sub_shape)
        words[slices] = replacement
        post_bytes = words.tobytes()

    elif encoding == "low_rank":
        if len(shape) != 2:
            raise ValueError("low_rank requires a 2D tensor")
        rank = int(mod["rank"])
        if rank <= 0:
            raise ValueError(f"low_rank rank must be positive, got {rank}")
        rows_n, cols_n = shape
        u_bytes_len = rows_n * rank * 4
        v_bytes_len = rank * cols_n * 4
        if len(delta_slice) != u_bytes_len + v_bytes_len:
            raise ValueError(
                f"low_rank delta_slice byte length mismatch: expected {u_bytes_len + v_bytes_len}, got {len(delta_slice)}"
            )
        u_mat = np.frombuffer(delta_slice[:u_bytes_len], dtype="<f4").reshape(rows_n, rank)
        v_mat = np.frombuffer(delta_slice[u_bytes_len:], dtype="<f4").reshape(rank, cols_n)
        pre_f32 = bytes_to_fp32(pre_bytes, dtype_name, shape)
        post_f32 = pre_f32 + (u_mat @ v_mat)
        post_bytes = fp32_to_bytes(post_f32, dtype_name)
        rollback_bytes = pre_bytes

    else:
        raise ValueError(f"Unsupported .wsp v2.1 tensor encoding: {encoding!r}")

    post_sha = hashlib.sha256(post_bytes).hexdigest()
    if post_sha != mod["post_patch_tensor_sha256"]:
        raise ValueError(
            f"Post-patch tensor SHA-256 verification failed on '{mod['tensor_name']}': "
            f"expected {mod['post_patch_tensor_sha256']}, got {post_sha}"
        )

    return post_bytes, rollback_bytes


def restore_tensor_from_rollback_bytes(
    patched_bytes: bytes, mod: Dict[str, Any], rollback_slice: bytes
) -> bytes:
    """Restores exact pre-patch tensor bytes from minimal rollback slice and verifies SHA-256."""
    shape = tuple(int(d) for d in mod["shape"])
    dtype_name = str(mod["dtype"])
    encoding = str(mod["encoding"])
    wd = raw_word_dtype(dtype_name)

    if encoding == "low_rank":
        restored = rollback_slice
    else:
        words = np.frombuffer(patched_bytes, dtype=wd).reshape(shape).copy()
        if encoding == "column_sparse":
            cols = [int(i) for i in mod["indices"]]
            words[:, cols] = np.frombuffer(rollback_slice, dtype=wd).reshape(shape[0], len(cols))
        elif encoding == "row_sparse":
            rows = [int(i) for i in mod["indices"]]
            words[rows, :] = np.frombuffer(rollback_slice, dtype=wd).reshape(len(rows), shape[1])
        elif encoding == "dense_slice":
            bounds = mod.get("slice_bounds") or [[0, d] for d in shape]
            slices = tuple(slice(int(b[0]), int(b[1])) for b in bounds)
            sub_shape = tuple(int(b[1]) - int(b[0]) for b in bounds)
            words[slices] = np.frombuffer(rollback_slice, dtype=wd).reshape(sub_shape)
        else:
            raise ValueError(f"Unsupported encoding during rollback: {encoding!r}")
        restored = words.tobytes()

    restored_sha = hashlib.sha256(restored).hexdigest()
    if restored_sha != mod["pre_patch_tensor_sha256"]:
        raise ValueError(
            f"Rollback tensor SHA-256 mismatch on '{mod['tensor_name']}': "
            f"expected {mod['pre_patch_tensor_sha256']}, got {restored_sha}"
        )
    return restored


def encode_exact_tensor_diff(
    *,
    tensor_name: str,
    shard_file: str,
    pre_bytes: bytes,
    post_bytes: bytes,
    safetensors_dtype: str,
    shape: Tuple[int, ...],
    current_offset: int,
) -> Optional[Tuple[Dict[str, Any], bytes]]:
    """Computes the most compact exact v2.1 modification spec + raw post-patch delta bytes."""
    if pre_bytes == post_bytes:
        return None

    dtype_name = SAFETENSORS_TO_DTYPE[safetensors_dtype]
    wd = raw_word_dtype(dtype_name)
    pre_w = np.frombuffer(pre_bytes, dtype=wd).reshape(shape)
    post_w = np.frombuffer(post_bytes, dtype=wd).reshape(shape)
    diff_mask = pre_w != post_w
    if not np.any(diff_mask):
        return None

    frob_norm, max_abs = compute_delta_norms(pre_bytes, post_bytes, dtype_name, shape)
    pre_sha = hashlib.sha256(pre_bytes).hexdigest()
    post_sha = hashlib.sha256(post_bytes).hexdigest()

    candidates: List[Tuple[int, str, Optional[int], List[int], Optional[List[List[int]]], bytes, Dict[str, Any]]] = []

    if len(shape) == 2:
        rows_n, cols_n = shape
        changed_cols = [int(i) for i in np.where(np.any(diff_mask, axis=0))[0].tolist()]
        col_payload = np.ascontiguousarray(post_w[:, changed_cols]).tobytes()
        candidates.append(
            (
                len(col_payload),
                "column_sparse",
                1,
                changed_cols,
                None,
                col_payload,
                {
                    "method": "exact_post_patch_columns",
                    "modified_elements": rows_n * len(changed_cols),
                    "modified_columns": len(changed_cols),
                    "frobenius_norm_delta": frob_norm,
                    "max_abs_delta": max_abs,
                },
            )
        )

        changed_rows = [int(i) for i in np.where(np.any(diff_mask, axis=1))[0].tolist()]
        row_payload = np.ascontiguousarray(post_w[changed_rows, :]).tobytes()
        candidates.append(
            (
                len(row_payload),
                "row_sparse",
                0,
                changed_rows,
                None,
                row_payload,
                {
                    "method": "exact_post_patch_rows",
                    "modified_elements": len(changed_rows) * cols_n,
                    "modified_rows": len(changed_rows),
                    "frobenius_norm_delta": frob_norm,
                    "max_abs_delta": max_abs,
                },
            )
        )

    # Bounding-box dense_slice fallback (handles 1D/2D/ND tensors cleanly)
    coords = np.argwhere(diff_mask)
    mins = coords.min(axis=0).tolist()
    maxs = (coords.max(axis=0) + 1).tolist()
    bounds = [[int(s), int(e)] for s, e in zip(mins, maxs)]
    slices = tuple(slice(s, e) for s, e in bounds)
    slice_payload = np.ascontiguousarray(post_w[slices]).tobytes()
    slice_elems = int(np.prod([e - s for s, e in bounds]))
    candidates.append(
        (
            len(slice_payload),
            "dense_slice",
            None,
            [],
            bounds,
            slice_payload,
            {
                "method": "exact_post_patch_slice",
                "modified_elements": slice_elems,
                "frobenius_norm_delta": frob_norm,
                "max_abs_delta": max_abs,
            },
        )
    )

    candidates.sort(key=lambda c: c[0])
    _, best_enc, best_axis, best_indices, best_bounds, best_payload, prov = candidates[0]

    spec: Dict[str, Any] = {
        "tensor_name": tensor_name,
        "shard_file": shard_file,
        "shape": list(shape),
        "dtype": dtype_name,
        "encoding": best_enc,
        "axis": best_axis,
        "indices": best_indices,
        "slice_bounds": best_bounds,
        "rank": None,
        "pre_patch_tensor_sha256": pre_sha,
        "post_patch_tensor_sha256": post_sha,
        "binary_delta_offset": current_offset,
        "binary_delta_length": len(best_payload),
        "mathematical_provenance": prov,
    }
    return spec, best_payload


def encode_low_rank_modification(
    *,
    tensor_name: str,
    shard_file: str,
    pre_bytes: bytes,
    u_f32: np.ndarray,
    v_f32: np.ndarray,
    safetensors_dtype: str,
    current_offset: int,
) -> Tuple[Dict[str, Any], bytes, bytes]:
    """Builds a v2.1 low_rank tensor modification (W_new = round(fp32(W) + U @ V))."""
    dtype_name = SAFETENSORS_TO_DTYPE.get(safetensors_dtype, safetensors_dtype)
    rows_n, rank = u_f32.shape
    rank_v, cols_n = v_f32.shape
    if rank != rank_v:
        raise ValueError(f"Low-rank inner dimension mismatch: {rank} != {rank_v}")
    shape = (rows_n, cols_n)

    u_contig = np.ascontiguousarray(u_f32, dtype="<f4")
    v_contig = np.ascontiguousarray(v_f32, dtype="<f4")
    delta_payload = u_contig.tobytes() + v_contig.tobytes()

    pre_f32 = bytes_to_fp32(pre_bytes, dtype_name, shape)
    post_f32 = pre_f32 + (u_contig @ v_contig)
    post_bytes = fp32_to_bytes(post_f32, dtype_name)

    frob_norm, max_abs = compute_delta_norms(pre_bytes, post_bytes, dtype_name, shape)
    spec: Dict[str, Any] = {
        "tensor_name": tensor_name,
        "shard_file": shard_file,
        "shape": [rows_n, cols_n],
        "dtype": dtype_name,
        "encoding": "low_rank",
        "axis": None,
        "indices": [],
        "slice_bounds": None,
        "rank": int(rank),
        "pre_patch_tensor_sha256": hashlib.sha256(pre_bytes).hexdigest(),
        "post_patch_tensor_sha256": hashlib.sha256(post_bytes).hexdigest(),
        "binary_delta_offset": current_offset,
        "binary_delta_length": len(delta_payload),
        "mathematical_provenance": {
            "method": "svd_low_rank_fp32",
            "modified_elements": int(rows_n * rank + rank * cols_n),
            "rank": int(rank),
            "frobenius_norm_delta": frob_norm,
            "max_abs_delta": max_abs,
        },
    }
    return spec, delta_payload, post_bytes


def _deep_merge_json(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """RFC 7396 JSON Merge Patch implementation for P0 config patches."""
    result = dict(base)
    for k, v in patch.items():
        if v is None:
            result.pop(k, None)
        elif isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge_json(result[k], v)
        else:
            result[k] = v
    return result


def resolve_model_input_dir(
    model_dir_or_repo: str | Path,
    *,
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Path:
    """Resolves either a local directory or downloads a revision-pinned Hugging Face snapshot."""
    candidate = Path(model_dir_or_repo)
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    repo_id = str(model_dir_or_repo)
    if not revision:
        raise ValueError(
            f"Refusing unpinned Hugging Face download for '{repo_id}': base_repo_revision is mandatory."
        )
    local_snapshot = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=hf_token or os.getenv("HUGGINGFACE_TOKEN") or None,
    )
    return Path(local_snapshot).resolve()


def apply_patch(
    *,
    model_dir_or_repo: str | Path,
    patch_source: Path | bytes,
    out_dir: Path,
    trusted_keys: Dict[str, str],
    revoked_ids: Optional[Iterable[str]] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Verifies signature + base hashes, executes Copy-on-Write into out_dir, and writes rollback bundle."""
    ok, verify_msg, archive = verify_wsp_archive(
        patch_source, trusted_keys=trusted_keys, revoked_ids=revoked_ids
    )
    if not ok:
        raise ValueError(f"Cryptographic verification failed: {verify_msg}")

    manifest = archive.manifest
    compat = manifest.get("compatibility", {})
    revision = compat.get("base_repo_revision")

    src_dir = resolve_model_input_dir(model_dir_or_repo, revision=revision, hf_token=hf_token)
    out_resolved = out_dir.resolve()

    if src_dir == out_resolved:
        raise ValueError("Atomic CoW Law: out_dir must differ from input model_dir.")
    if is_hf_cache_path(out_resolved):
        raise PermissionError(f"Refusing to use Hugging Face cache directory as out_dir: {out_resolved}")
    if out_resolved.exists() and any(out_resolved.iterdir()):
        raise FileExistsError(f"Target output directory is non-empty: {out_resolved}")

    tensor_mods: List[Dict[str, Any]] = manifest.get("tensor_modifications", [])
    config_mods: List[Dict[str, Any]] = manifest.get("config_modifications", [])

    # 1. Verify base safetensors SHA-256 hashes BEFORE writing anything
    shard_sha_map: Dict[str, str] = dict(compat.get("shard_sha256_map") or {})
    if compat.get("base_safetensors_sha256") and "model.safetensors" not in shard_sha_map:
        if (src_dir / "model.safetensors").exists():
            shard_sha_map["model.safetensors"] = compat["base_safetensors_sha256"]

    pre_shard_hashes: Dict[str, str] = {}
    for shard_name, expected_sha in shard_sha_map.items():
        shard_path = src_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Expected base shard '{shard_name}' missing in {src_dir}")
        actual_sha = sha256_file(shard_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Base safetensors SHA-256 mismatch on '{shard_name}': "
                f"expected {expected_sha}, got {actual_sha}"
            )
        pre_shard_hashes[shard_name] = actual_sha

    spans_map: Dict[str, TensorSpan] = {}
    if tensor_mods:
        spans_map, _ = discover_model_tensors(src_dir)
        for mod in tensor_mods:
            tname = mod["tensor_name"]
            if tname not in spans_map:
                raise KeyError(f"Target tensor '{tname}' not found in checkpoint {src_dir}")
            span = spans_map[tname]
            if list(span.shape) != list(mod["shape"]):
                raise ValueError(
                    f"Shape mismatch on '{tname}': model has {list(span.shape)}, patch expects {mod['shape']}"
                )

    modified_shards = {spans_map[m["tensor_name"]].shard_file for m in tensor_mods}
    modified_configs = {c["target_file"] for c in config_mods}

    out_resolved.mkdir(parents=True, exist_ok=True)
    rollback_chunks: List[bytes] = []
    rollback_tensor_entries: List[Dict[str, Any]] = []
    rollback_config_entries: List[Dict[str, Any]] = []
    rollback_offset = 0

    try:
        # 2. Copy modified shards/configs; link-or-copy untouched files
        for item in src_dir.iterdir():
            if item.name == ".wsp_rollback" or item.is_dir():
                continue
            target_file = out_resolved / item.name
            if item.name in modified_shards or item.name in modified_configs:
                shutil.copy2(item, target_file)
            else:
                safe_link_or_copy(item, target_file)

        # 3. Apply P1 weight modifications directly at exact byte offsets
        total_delta_len = len(archive.deltas_bytes)
        for mod in tensor_mods:
            tname = mod["tensor_name"]
            span = spans_map[tname]
            target_shard = _resolve_contained_path(out_resolved, span.shard_file)
            src_shard = _resolve_contained_path(src_dir, span.shard_file)
            if span.shard_file not in pre_shard_hashes:
                pre_shard_hashes[span.shard_file] = sha256_file(src_shard)

            pre_bytes = read_tensor_bytes(target_shard, span)
            d_off = int(mod["binary_delta_offset"])
            d_len = int(mod["binary_delta_length"])
            if d_off < 0 or d_len <= 0 or (d_off + d_len) > total_delta_len:
                raise ValueError(
                    f"Out-of-bounds binary delta slice for '{tname}': offset={d_off}, length={d_len}, total={total_delta_len}"
                )
            delta_slice = archive.deltas_bytes[d_off : d_off + d_len]

            post_bytes, rb_slice = apply_tensor_modification_to_bytes(pre_bytes, mod, delta_slice)
            write_tensor_bytes_at_span(target_shard, span, post_bytes)

            # Re-read from disk to empirically verify post_patch_tensor_sha256 on disk
            disk_verify = read_tensor_bytes(target_shard, span)
            if hashlib.sha256(disk_verify).hexdigest() != mod["post_patch_tensor_sha256"]:
                raise IOError(f"Disk write verification failed for tensor '{tname}'")

            rollback_entry = dict(mod)
            rollback_entry["shard_file"] = span.shard_file
            rollback_entry["rollback_offset"] = rollback_offset
            rollback_entry["rollback_length"] = len(rb_slice)
            rollback_chunks.append(rb_slice)
            rollback_offset += len(rb_slice)
            rollback_tensor_entries.append(rollback_entry)

        # 4. Apply P0 configuration modifications
        for cmod in config_mods:
            fname = cmod["target_file"]
            target_cfg = _resolve_contained_path(out_resolved, fname)
            existed = target_cfg.exists()
            orig_bytes = target_cfg.read_bytes() if existed else b""
            if existed and cmod.get("pre_patch_sha256"):
                orig_sha = hashlib.sha256(orig_bytes).hexdigest()
                if orig_sha != cmod["pre_patch_sha256"]:
                    raise ValueError(
                        f"Pre-patch SHA-256 mismatch on config '{fname}': "
                        f"expected {cmod['pre_patch_sha256']}, got {orig_sha}"
                    )

            if fname in archive.config_blobs:
                new_cfg_bytes = archive.config_blobs[fname]
            elif cmod.get("replacement_text") is not None:
                new_cfg_bytes = cmod["replacement_text"].encode("utf-8")
            elif cmod.get("json_merge_patch") is not None:
                base_obj = json.loads(orig_bytes.decode("utf-8")) if existed else {}
                merged_obj = _deep_merge_json(base_obj, cmod["json_merge_patch"])
                new_cfg_bytes = (
                    json.dumps(merged_obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                ).encode("utf-8")
            else:
                raise ValueError(f"Config modification '{fname}' lacks payload")

            new_sha = hashlib.sha256(new_cfg_bytes).hexdigest()
            if new_sha != cmod["post_patch_sha256"]:
                raise ValueError(
                    f"Post-patch SHA-256 mismatch on config '{fname}': "
                    f"expected {cmod['post_patch_sha256']}, got {new_sha}"
                )
            target_cfg.write_bytes(new_cfg_bytes)

            rollback_config_entries.append(
                {
                    "target_file": fname,
                    "existed": existed,
                    "pre_patch_sha256": hashlib.sha256(orig_bytes).hexdigest() if existed else None,
                    "rollback_offset": rollback_offset,
                    "rollback_length": len(orig_bytes),
                }
            )
            rollback_chunks.append(orig_bytes)
            rollback_offset += len(orig_bytes)

        # 5. Write compressed rollback bundle (.wsp_rollback/backup.bin + rollback_manifest.json)
        rb_dir = out_resolved / ".wsp_rollback"
        rb_dir.mkdir(parents=True, exist_ok=True)
        raw_backup = b"".join(rollback_chunks)
        compressed_backup = zlib.compress(raw_backup, level=6)
        (rb_dir / "backup.bin").write_bytes(compressed_backup)
        rb_meta = {
            "patch_id": manifest.get("patch_id"),
            "version": manifest.get("version"),
            "pre_shard_hashes": pre_shard_hashes,
            "raw_backup_sha256": hashlib.sha256(raw_backup).hexdigest(),
            "tensor_entries": rollback_tensor_entries,
            "config_entries": rollback_config_entries,
        }
        (rb_dir / "rollback_manifest.json").write_text(
            json.dumps(rb_meta, indent=2, sort_keys=True), encoding="utf-8"
        )

    except Exception:
        shutil.rmtree(out_resolved, ignore_errors=True)
        raise

    return {
        "patch_id": manifest.get("patch_id"),
        "out_dir": str(out_resolved),
        "modified_tensors": len(tensor_mods),
        "modified_configs": len(config_mods),
        "rollback_bundle_bytes": len(compressed_backup),
    }


def revert_patch(patched_dir: Path) -> Dict[str, Any]:
    """Decompresses .wsp_rollback/backup.bin, restores exact original bytes, and verifies SHA-256."""
    target_dir = patched_dir.resolve()
    rb_dir = target_dir / ".wsp_rollback"
    meta_path = rb_dir / "rollback_manifest.json"
    backup_path = rb_dir / "backup.bin"

    if not meta_path.exists() or not backup_path.exists():
        raise FileNotFoundError(f"No .wsp_rollback bundle found in {target_dir}")

    rb_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_backup = zlib.decompress(backup_path.read_bytes())
    if hashlib.sha256(raw_backup).hexdigest() != rb_meta["raw_backup_sha256"]:
        raise ValueError("Corrupted .wsp_rollback/backup.bin: SHA-256 digest mismatch")

    tensor_entries: List[Dict[str, Any]] = rb_meta.get("tensor_entries", [])
    if tensor_entries:
        spans_map, _ = discover_model_tensors(target_dir)
        for entry in tensor_entries:
            tname = entry["tensor_name"]
            span = spans_map[tname]
            shard_path = _resolve_contained_path(target_dir, span.shard_file)
            current_bytes = read_tensor_bytes(shard_path, span)
            r_off = int(entry["rollback_offset"])
            r_len = int(entry["rollback_length"])
            if r_off < 0 or r_len <= 0 or (r_off + r_len) > len(raw_backup):
                raise ValueError(f"Out-of-bounds rollback tensor slice for '{tname}'")
            rb_slice = raw_backup[r_off : r_off + r_len]

            restored_bytes = restore_tensor_from_rollback_bytes(current_bytes, entry, rb_slice)
            write_tensor_bytes_at_span(shard_path, span, restored_bytes)

    for cfg_entry in rb_meta.get("config_entries", []):
        cfg_path = _resolve_contained_path(target_dir, cfg_entry["target_file"])
        if not cfg_entry["existed"]:
            if cfg_path.exists():
                cfg_path.unlink()
        else:
            r_off = int(cfg_entry["rollback_offset"])
            r_len = int(cfg_entry["rollback_length"])
            if r_off < 0 or r_len < 0 or (r_off + r_len) > len(raw_backup):
                raise ValueError(f"Out-of-bounds rollback config slice for {cfg_entry['target_file']}")
            orig_bytes = raw_backup[r_off : r_off + r_len]
            if hashlib.sha256(orig_bytes).hexdigest() != cfg_entry["pre_patch_sha256"]:
                raise ValueError(f"Config rollback hash mismatch for {cfg_entry['target_file']}")
            cfg_path.write_bytes(orig_bytes)

    # Verify full safetensors shard SHA-256 parity against original pre-patch shards
    for shard_name, expected_sha in rb_meta.get("pre_shard_hashes", {}).items():
        shard_verify_path = _resolve_contained_path(target_dir, shard_name)
        actual_sha = sha256_file(shard_verify_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Full-file rollback SHA-256 mismatch on '{shard_name}': "
                f"expected {expected_sha}, got {actual_sha}"
            )

    shutil.rmtree(rb_dir)
    return {
        "reverted_patch_id": rb_meta.get("patch_id"),
        "restored_tensors": len(tensor_entries),
        "restored_configs": len(rb_meta.get("config_entries", [])),
        "verified_shards": list(rb_meta.get("pre_shard_hashes", {}).keys()),
    }


def compute_directory_diff_patch(
    *,
    base_dir: Path,
    modified_dir: Path,
    patch_id: str,
    hf_repo_id: str,
    base_repo_revision: str,
    architecture_adapter: str,
    publisher_key_id: str,
    private_seed_hex: str,
    base_license: str = "apache-2.0",
    supernova_creator_id: str = "founder-datagenius",
    out_path: Optional[Path] = None,
) -> Tuple[bytes, Dict[str, Any], str]:
    """Diffs two local model directories and packages an exact signed .wsp v2.1 archive."""
    base_spans, base_shards = discover_model_tensors(base_dir)
    mod_spans, _ = discover_model_tensors(modified_dir)

    shard_sha256_map = {s: sha256_file(base_dir / s) for s in base_shards}
    delta_chunks: List[bytes] = []
    tensor_mods: List[Dict[str, Any]] = []
    offset = 0
    total_params = 0
    modified_params = 0

    for tname in sorted(base_spans.keys()):
        b_span = base_spans[tname]
        total_params += int(np.prod(b_span.shape))
        if tname not in mod_spans:
            raise KeyError(f"Tensor '{tname}' missing from modified_dir")
        m_span = mod_spans[tname]
        if b_span.shape != m_span.shape or b_span.dtype != m_span.dtype:
            raise ValueError(f"Incompatible tensor shape/dtype change on '{tname}'")

        pre_b = read_tensor_bytes(base_dir / b_span.shard_file, b_span)
        post_b = read_tensor_bytes(modified_dir / m_span.shard_file, m_span)
        encoded = encode_exact_tensor_diff(
            tensor_name=tname,
            shard_file=b_span.shard_file,
            pre_bytes=pre_b,
            post_bytes=post_b,
            safetensors_dtype=b_span.dtype,
            shape=b_span.shape,
            current_offset=offset,
        )
        if encoded is not None:
            spec, chunk = encoded
            tensor_mods.append(spec)
            delta_chunks.append(chunk)
            offset += len(chunk)
            modified_params += int(spec["mathematical_provenance"]["modified_elements"])

    tok_sha = (
        sha256_file(base_dir / "tokenizer.json")
        if (base_dir / "tokenizer.json").exists()
        else None
    )
    manifest: Dict[str, Any] = {
        "schema_version": "2.1.0",
        "patch_id": patch_id,
        "version": "1.0.0",
        "patch_tier": "P1_WEIGHTS" if tensor_mods else "P0_CONFIG",
        "publisher_key_id": publisher_key_id,
        "supernova_creator_id": supernova_creator_id,
        "registry_url": "https://wsphub.dev",
        "compatibility": {
            "hf_repo_id": hf_repo_id,
            "base_repo_revision": base_repo_revision,
            "base_license": base_license,
            "base_safetensors_sha256": shard_sha256_map.get("model.safetensors"),
            "shard_sha256_map": shard_sha256_map,
            "architecture_adapter": architecture_adapter,
            "tokenizer_sha256": tok_sha,
            "supported_runtimes": ["vllm>=0.33.0", "transformers>=5.20.2", "ollama"],
        },
        "tensor_modifications": tensor_mods,
        "config_modifications": [],
        "reproducibility_and_evaluation": {
            "modified_parameters": modified_params,
            "modified_fraction": round(modified_params / max(1, total_params), 8),
        },
    }
    return build_wsp_archive(
        manifest=manifest,
        deltas_bytes=b"".join(delta_chunks),
        private_seed_hex=private_seed_hex,
        out_path=out_path,
    )


def patch_model(
    model: Any,
    patch_source: Path | bytes,
    *,
    trusted_keys: Dict[str, str],
    revoked_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Patches an in-memory PyTorch / Transformers model (or dict state_dict) with cryptographic verification."""
    ok, msg, archive = verify_wsp_archive(
        patch_source, trusted_keys=trusted_keys, revoked_ids=revoked_ids
    )
    if not ok:
        raise ValueError(f"Refusing to patch in-memory model: {msg}")

    param_map: Dict[str, Any]
    if isinstance(model, dict):
        param_map = model
    elif hasattr(model, "named_parameters"):
        param_map = dict(model.named_parameters())
    else:
        raise TypeError("model must expose named_parameters() or be a parameter dict")

    patched_names: List[str] = []
    for mod in archive.manifest.get("tensor_modifications", []):
        tname = mod["tensor_name"]
        if tname not in param_map:
            raise KeyError(f"Parameter '{tname}' not found in loaded model")
        target = param_map[tname]
        d_off = int(mod["binary_delta_offset"])
        d_len = int(mod["binary_delta_length"])
        delta_slice = archive.deltas_bytes[d_off : d_off + d_len]

        if isinstance(target, np.ndarray):
            pre_bytes = np.ascontiguousarray(target).tobytes()
            post_bytes, _ = apply_tensor_modification_to_bytes(pre_bytes, mod, delta_slice)
            param_map[tname] = np.frombuffer(post_bytes, dtype=target.dtype).reshape(target.shape).copy()
        elif hasattr(target, "detach") and hasattr(target, "data"):
            # PyTorch Tensor path (handles bfloat16 via uint16 view without losing bits)
            import torch

            t_cpu = target.detach().cpu().contiguous()
            raw_u8 = t_cpu.view(torch.uint8).numpy().tobytes()
            post_bytes, _ = apply_tensor_modification_to_bytes(raw_u8, mod, delta_slice)
            new_u8 = torch.frombuffer(bytearray(post_bytes), dtype=torch.uint8).view(target.dtype).reshape(target.shape)
            with torch.no_grad():
                target.copy_(new_u8.to(device=target.device))
        else:
            raise TypeError(f"Unsupported parameter container for '{tname}': {type(target)!r}")
        patched_names.append(tname)

    return {
        "patch_id": archive.manifest.get("patch_id"),
        "patched_parameters": patched_names,
    }
