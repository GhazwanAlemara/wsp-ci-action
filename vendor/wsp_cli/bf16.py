"""Bit-Exact bfloat16 / float16 / float32 NumPy Conversion & Norm Utilities."""

from __future__ import annotations

from typing import Tuple
import numpy as np

DTYPE_TO_SAFETENSORS = {
    "bfloat16": "BF16",
    "float16": "F16",
    "float32": "F32",
}

SAFETENSORS_TO_DTYPE = {v: k for k, v in DTYPE_TO_SAFETENSORS.items()}

DTYPE_ITEMSIZE = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}


def raw_word_dtype(dtype_name: str) -> np.dtype:
    """Returns lossless unsigned integer word dtype preserving every bit of raw tensor storage."""
    size = DTYPE_ITEMSIZE[dtype_name]
    return np.dtype("<u2") if size == 2 else np.dtype("<u4")


def bytes_to_fp32(raw_bytes: bytes, dtype_name: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Converts raw little-endian safetensors bytes to float32 numpy array."""
    norm = SAFETENSORS_TO_DTYPE.get(dtype_name, dtype_name)
    if norm == "bfloat16":
        u16 = np.frombuffer(raw_bytes, dtype="<u2").reshape(shape)
        u32 = u16.astype(np.uint32) << 16
        return u32.view("<f4")
    if norm == "float16":
        return np.frombuffer(raw_bytes, dtype="<f2").reshape(shape).astype(np.float32)
    if norm == "float32":
        return np.frombuffer(raw_bytes, dtype="<f4").reshape(shape).copy()
    raise ValueError(f"Unsupported tensor dtype: {dtype_name}")


def fp32_to_bytes(arr_f32: np.ndarray, dtype_name: str) -> bytes:
    """Converts float32 array to target dtype bytes using IEEE-754 round-to-nearest-even."""
    norm = SAFETENSORS_TO_DTYPE.get(dtype_name, dtype_name)
    f32 = np.ascontiguousarray(arr_f32, dtype="<f4")
    if norm == "float32":
        return f32.tobytes()
    if norm == "float16":
        return f32.astype("<f2").tobytes()
    if norm == "bfloat16":
        bits = f32.view(np.uint32)
        is_nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
        lsb = (bits >> np.uint32(16)) & np.uint32(1)
        rounding_bias = np.uint32(0x7FFF) + lsb
        rounded = (bits + rounding_bias) >> np.uint32(16)
        # Canonical quiet NaN preservation
        bf16_u16 = np.where(is_nan, np.uint16(0x7FC0), rounded.astype(np.uint16))
        return np.ascontiguousarray(bf16_u16, dtype="<u2").tobytes()
    raise ValueError(f"Unsupported tensor dtype: {dtype_name}")


def compute_delta_norms(
    pre_bytes: bytes, post_bytes: bytes, dtype_name: str, shape: Tuple[int, ...]
) -> Tuple[float, float]:
    """Computes Frobenius norm and max absolute delta in float32 precision."""
    pre_f32 = bytes_to_fp32(pre_bytes, dtype_name, shape)
    post_f32 = bytes_to_fp32(post_bytes, dtype_name, shape)
    diff = post_f32 - pre_f32
    frob = float(np.linalg.norm(diff.ravel(), ord=2))
    max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    return round(frob, 6), round(max_abs, 6)
