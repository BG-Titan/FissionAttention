# SPDX-License-Identifier: Apache-2.0
"""ctypes loader + tensor marshalling for the FuseMoEMLA .so (no torch headers).

The library is built by ``setup.py`` with plain ``nvcc -shared`` (extern "C"
ABI, see ``csrc/mla.cu``) and searched for in this package directory, or via
the ``FISSIONATTN_SO`` environment variable.  torch is imported lazily inside
the call paths so this module stays importable on CPU-only boxes.

When the native library is missing (CPU-only box, build skipped) every call
falls back to the exact torch reference in :mod:`fissionattn.ref` — slow, but
it keeps the backend importable and testable anywhere.  Set
``FISSION_ATTN_REQUIRE_NATIVE=1`` to turn that fallback into an error instead.

Public API:
    native_available() -> bool
    decode(q, out, cache, block_table, seq_lens, ...) -> None
"""

from __future__ import annotations

import ctypes
import importlib.machinery
import os
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_cache: dict[str, object] = {}

_I64, _I32, _F32, _PTR = (
    ctypes.c_int64,
    ctypes.c_int32,
    ctypes.c_float,
    ctypes.c_void_p,
)


def _candidate_paths() -> list[str]:
    env = os.environ.get("FISSION_ATTN_SO")
    if env:
        return [env]
    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES) + [".so"]
    out: list[str] = []
    for suf in suffixes:
        out += sorted(str(p) for p in _PKG_DIR.glob(f"_mla*{suf}"))
    return out


class _Lib:
    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        lib.fissionattn_abi.restype = ctypes.c_int
        lib.fissionattn_abi.argtypes = []
        lib.fissionattn_has_gpu.restype = ctypes.c_int
        lib.fissionattn_has_gpu.argtypes = []
        lib.fissionattn_decode.restype = ctypes.c_int
        lib.fissionattn_decode.argtypes = [
            _PTR,  # q
            _PTR,  # out
            _PTR,  # lse
            _PTR,  # cache
            _I64,  # cache block stride
            _I64,  # cache token stride
            _PTR,  # block table
            _I64,  # block table row stride
            _PTR,  # seq_lens
            _I32,  # rows
            _I32,  # num_q_heads
            _I32,  # head_size (latent + rope)
            _I32,  # out_dim (latent)
            _I32,  # page_size
            _F32,  # scale
            _I32,  # dtype code
            _I32,  # index code
            _PTR,  # cuda stream
        ]


def native_available() -> bool:
    if "lib" in _cache:
        return bool(_cache["lib"])
    lib = None
    if sys.platform == "win32":
        os.environ.setdefault("PATH", "")
    for path in _candidate_paths():
        try:
            cand = ctypes.CDLL(path)
            if cand.fissionattn_abi() == 1:
                lib = _Lib(cand)
                break
        except OSError:
            continue
    _cache["lib"] = lib
    return lib is not None


def _dtype_code(t) -> int:
    import torch

    dt = getattr(t, "dtype", t)
    if dt == torch.float16:
        return 0
    if dt == torch.bfloat16:
        return 1
    raise ValueError(f"fissionattn: unsupported dtype {dt} (need float16/bfloat16)")


def _idx_code(t) -> int:
    import torch

    if t.dtype == torch.int32:
        return 0
    if t.dtype == torch.int64:
        return 1
    raise ValueError(f"fissionattn: block_table/seq_lens must be int32/int64, got {t.dtype}")


def _stream_ptr(t) -> int:
    import torch

    return torch.cuda.current_stream(t.device).cuda_stream


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(
            f"fissionattn {what} failed: code={rc} (cudaError_t, or -1000s arg error)"
        )


def cache_strides(cache, fn: str = "decode") -> tuple[int, int]:
    """Element (block, token) strides of an MLA paged cache.

    Accepts the vLLM MLA slab ``[num_blocks, 1, page, D]`` and the squeezed
    ``[num_blocks, page, D]`` view ``MLAAttention.bind_kv_cache`` hands the
    impl.  The last dim must be contiguous; nothing is ever made contiguous.
    """
    if cache.dim() == 4:
        if cache.size(1) != 1:
            raise ValueError(
                f"fissionattn.{fn}: 4-D MLA cache must have num_kv_heads == 1, "
                f"got shape {tuple(cache.shape)}"
            )
        inner = cache.reshape(cache.size(0), cache.size(2), cache.size(3))
    elif cache.dim() == 3:
        inner = cache
    else:
        raise ValueError(f"fissionattn.{fn}: expected a 3-D/4-D paged cache")
    if inner.stride(2) != 1:
        raise ValueError(f"fissionattn.{fn}: cache last dim must be contiguous")
    return inner.stride(0), inner.stride(1)


def decode(
    q,
    out,
    cache,
    block_table,
    seq_lens,
    num_q_heads: int,
    head_size: int,
    out_dim: int,
    page_size: int,
    scale: float,
    lse=None,
) -> None:
    """MLA absorbed decode: q [rows, H, head_size] -> out [rows, H, out_dim].

    ``cache`` is the paged latent(+rope) slab, ``block_table`` [rows, blocks]
    and ``seq_lens`` [rows] (one row per query row; multi-token decode rows
    must already be flattened by the caller).  ``lse`` is optional fp32
    [rows, H].
    """
    if q.device.type != "cuda":
        # The kernel needs a device pointer + CUDA stream; CPU tensors run the
        # identical math in torch (also the path the CPU tests use).
        if os.environ.get("FISSION_ATTN_REQUIRE_NATIVE") == "1":
            import warnings

            warnings.warn(
                "fissionattn: FISSION_ATTN_REQUIRE_NATIVE=1 ignored for CPU tensors"
            )
        lib = None
    else:
        lib = _cache.get("lib")
        if lib is None and not native_available():
            lib = None
    if lib is None:
        if os.environ.get("FISSION_ATTN_REQUIRE_NATIVE") == "1":
            raise RuntimeError("fissionattn: native library not loaded")
        from .ref import decode_ref_torch

        decode_ref_torch(
            q,
            out,
            cache,
            block_table,
            seq_lens,
            num_q_heads=num_q_heads,
            head_size=head_size,
            out_dim=out_dim,
            page_size=page_size,
            scale=scale,
            lse=lse,
        )
        return

    sb, st = cache_strides(cache)
    if block_table.stride(1) != 1:
        raise ValueError("fissionattn: block_table rows must be contiguous")
    if q.stride(2) != 1:
        raise ValueError("fissionattn: q last dim must be contiguous")
    rows = q.size(0)
    if seq_lens.size(0) != rows:
        raise ValueError(
            f"fissionattn: {rows} query rows but {seq_lens.size(0)} seq_lens "
            "(flatten multi-token decode rows first)"
        )
    rc = lib.lib.fissionattn_decode(  # type: ignore[union-attr]
        q.data_ptr(),
        out.data_ptr(),
        lse.data_ptr() if lse is not None else None,
        cache.data_ptr() if cache.dim() == 3 else cache.data_ptr(),
        sb,
        st,
        block_table.data_ptr(),
        block_table.stride(0),
        seq_lens.data_ptr(),
        rows,
        num_q_heads,
        head_size,
        out_dim,
        page_size,
        float(scale),
        _dtype_code(q),
        _idx_code(block_table),
        _stream_ptr(q),
    )
    _check(rc, "decode")
