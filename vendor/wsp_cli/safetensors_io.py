"""Low-Level Safetensors Binary Header Parser, Shard Resolver & Cache Guard."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class TensorSpan:
    tensor_name: str
    shard_file: str
    dtype: str
    shape: Tuple[int, ...]
    file_byte_start: int
    file_byte_end: int

    @property
    def byte_length(self) -> int:
        return self.file_byte_end - self.file_byte_start


def is_hf_cache_path(path: Path) -> bool:
    """Detects Hugging Face shared blob cache paths or symlinked cache files."""
    parts_lower = [p.lower() for p in path.parts]
    for idx, part in enumerate(parts_lower):
        if part == "hub" and idx > 0 and "huggingface" in parts_lower[idx - 1]:
            return True
        if part.startswith("models--") and "blobs" in parts_lower:
            return True
    if path.exists():
        for child in ([path] if path.is_file() else list(path.glob("*.safetensors"))):
            if child.is_symlink():
                return True
    return False


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_safetensors_header(file_path: Path) -> Tuple[int, Dict[str, Any], Dict[str, TensorSpan]]:
    """Reads the 8-byte uint64 LE header size, JSON header, and exact file byte offsets."""
    file_size = file_path.stat().st_size
    if file_size < 8:
        raise ValueError(f"Invalid safetensors file (too small): {file_path}")

    with file_path.open("rb") as f:
        header_len_bytes = f.read(8)
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len <= 0 or header_len > 100 * 1024 * 1024 or 8 + header_len > file_size:
            raise ValueError(f"Corrupt safetensors header length ({header_len}) in {file_path}")
        raw_header = f.read(header_len)

    header_json: Dict[str, Any] = json.loads(raw_header.decode("utf-8"))
    data_base = 8 + header_len
    spans: Dict[str, TensorSpan] = {}

    for key, val in header_json.items():
        if key == "__metadata__":
            continue
        offsets = val["data_offsets"]
        start = data_base + int(offsets[0])
        end = data_base + int(offsets[1])
        if start < data_base or end > file_size or start > end:
            raise ValueError(
                f"Out-of-bounds tensor offsets [{start}, {end}] for '{key}' in {file_path}"
            )
        spans[key] = TensorSpan(
            tensor_name=key,
            shard_file=file_path.name,
            dtype=str(val["dtype"]),
            shape=tuple(int(d) for d in val["shape"]),
            file_byte_start=start,
            file_byte_end=end,
        )

    return header_len, header_json, spans


def discover_model_tensors(model_dir: Path) -> Tuple[Dict[str, TensorSpan], List[str]]:
    """Discovers all tensors across single-file or sharded safetensors checkpoints."""
    index_path = model_dir / "model.safetensors.index.json"
    shard_files: List[str] = []
    all_spans: Dict[str, TensorSpan] = {}

    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: Dict[str, str] = idx.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
    else:
        single = model_dir / "model.safetensors"
        if single.exists():
            shard_files = ["model.safetensors"]
        else:
            shard_files = sorted(p.name for p in model_dir.glob("*.safetensors"))

    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Referenced safetensors shard missing: {shard_path}")
        _, _, spans = parse_safetensors_header(shard_path)
        for tname, span in spans.items():
            all_spans[tname] = span

    return all_spans, shard_files


def read_tensor_bytes(shard_path: Path, span: TensorSpan) -> bytes:
    with shard_path.open("rb") as f:
        f.seek(span.file_byte_start)
        data = f.read(span.byte_length)
    if len(data) != span.byte_length:
        raise IOError(f"Short read for tensor {span.tensor_name} in {shard_path}")
    return data


def write_tensor_bytes_at_span(shard_path: Path, span: TensorSpan, new_bytes: bytes) -> None:
    if len(new_bytes) != span.byte_length:
        raise ValueError(
            f"Byte length mismatch writing '{span.tensor_name}': "
            f"expected {span.byte_length}, got {len(new_bytes)}"
        )
    if is_hf_cache_path(shard_path):
        raise PermissionError(
            f"Safety Guard: Refusing to write directly into Hugging Face cache path {shard_path}"
        )
    with shard_path.open("r+b") as f:
        f.seek(span.file_byte_start)
        f.write(new_bytes)
        f.flush()
