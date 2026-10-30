# SPDX-License-Identifier: Apache-2.0
"""Pure references for the MLA absorbed-decode math (numpy + torch).

The kernel in ``csrc/mla.cu`` computes, for every query row ``r`` (one row per
request in vLLM's MLA decode split) and every q-head ``h``::

    k_i = cache[token_i]                       # D = kv_lora_rank + rope
    z   = softmax_i( (q[r,h] . k_i) * scale )
    out[r,h] = sum_i z_i * cache[token_i][:out_dim]      # latent part only
    lse[r,h] = log sum_i exp((q[r,h] . k_i) * scale)

Note the MQA shape: all q-heads share the single latent KV head, and the
*value* is the first ``out_dim`` (= kv_lora_rank) dims of the same row that
provides the key — the rope tail only participates in the dot product.

``naive_mla_attention`` is the numpy ground truth (no online softmax, no
paging) used by the CPU tests; ``decode_ref_torch`` is the same math with the
paged block table, used as the fallback path when the native library is absent
and as the parity target of the GPU test.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["naive_mla_attention", "decode_ref_torch", "gather_rows_torch"]


def naive_mla_attention(
    q: np.ndarray,
    kv: np.ndarray,
    scale: float,
    out_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One query row against an unpaged KV block.

    Args:
        q: ``[H, D]`` float (any of the fp types numpy holds; math in float64).
        kv: ``[S, D]`` latent||rope rows, in sequence order.
        scale: softmax scale applied to the dot products.
        out_dim: value width; defaults to ``kv.shape[1]``.

    Returns:
        out ``[H, out_dim]`` (dtype of ``q``) and lse ``[H]`` float64.
    """
    q64 = q.astype(np.float64)
    kv64 = kv.astype(np.float64)
    if out_dim is None:
        out_dim = kv64.shape[1]
    if kv64.shape[0] == 0:
        return (
            np.zeros((q64.shape[0], out_dim), dtype=q.dtype),
            np.full(q64.shape[0], -np.inf),
        )
    logits = (q64 @ kv64.T) * scale  # [H, S]
    m = logits.max(axis=-1, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    p = np.exp(logits - m_safe)
    l = p.sum(axis=-1, keepdims=True)
    l_safe = np.where(l > 0.0, l, 1.0)
    out = (p / l_safe) @ kv64[:, :out_dim]
    lse = np.where(l > 0.0, m_safe + np.log(l), -np.inf).squeeze(-1)
    return out.astype(q.dtype), lse


def gather_rows_torch(cache3, block_table_row, seq_len: int, page_size: int, chunk: int = 4096):
    """Yield ``[c, D]`` cache row slices for one request (torch, strided-safe)."""
    import torch

    pos = torch.arange(seq_len, device=cache3.device)
    for s in range(0, seq_len, chunk):
        pp = pos[s : s + chunk]
        blk = block_table_row[(pp // page_size).to(block_table_row.dtype)]
        off = pp % page_size
        yield cache3[blk, off]


def decode_ref_torch(
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
    chunk: int = 4096,
) -> None:
    """Paged MLA decode in plain torch (fp32 accumulation), native-or-CPU.

    Mirrors the kernel's contract exactly (see :mod:`fissionattn.ops`), so it
    doubles as the fallback when the ``.so`` is unavailable.
    """
    import torch

    cache3 = cache if cache.dim() == 3 else cache.reshape(cache.size(0), cache.size(2), cache.size(3))
    rows = q.size(0)
    out.view(rows, num_q_heads, out_dim)
    for r in range(rows):
        slen = int(seq_lens[r].item())
        o = torch.zeros(num_q_heads, out_dim, dtype=torch.float32, device=q.device)
        m = torch.full((num_q_heads, 1), -float("inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros(num_q_heads, 1, dtype=torch.float32, device=q.device)
        if slen > 0:
            qr = q[r].to(torch.float32)  # [H, D]
            for kv in gather_rows_torch(cache3, block_table[r], slen, page_size, chunk):
                logits = (qr @ kv.to(torch.float32).T) * scale  # [H, c]
                m_new = torch.maximum(m, logits.max(dim=-1, keepdim=True).values)
                m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
                corr = torch.exp(m - m_safe)
                p = torch.exp(logits - m_safe)
                l = l * corr + p.sum(dim=-1, keepdim=True)
                o = o * corr + p @ kv[:, :out_dim].to(torch.float32)
                m = m_safe
        good = (l > 0).to(torch.float32)
        out[r] = ((o / torch.where(l > 0, l, torch.ones_like(l))) * good).to(out.dtype)
        if lse is not None:
            l1 = l.squeeze(-1)
            m1 = m.squeeze(-1)
            lse[r] = torch.where(
                l1 > 0,
                m1 + torch.log(torch.where(l1 > 0, l1, torch.ones_like(l1))),
                torch.full((num_q_heads,), -float("inf"), dtype=torch.float32, device=q.device),
            ).to(lse.dtype)


def decode_ref_numpy_one_row(
    q_row: np.ndarray,
    cache: np.ndarray,
    block_row: np.ndarray,
    seq_len: int,
    page_size: int,
    scale: float,
    out_dim: int,
) -> tuple[np.ndarray, float]:
    """Numpy paged decode for a single query row (used by CPU tests)."""
    if seq_len <= 0:
        # Kernel convention: zero-length sequence -> zeros + lse = -inf.
        return np.zeros((q_row.shape[0], out_dim), dtype=q_row.dtype), -math.inf
    pos = np.arange(seq_len, dtype=np.int64)
    kv = cache[block_row[pos // page_size], pos % page_size]
    out, lse = naive_mla_attention(q_row, kv, scale, out_dim)
    return out, float(lse[0]) if seq_len > 0 else -math.inf
