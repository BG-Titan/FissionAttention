# SPDX-License-Identifier: Apache-2.0
"""Per-step MLA prefill orchestration (torch path, no vLLM imports).

Split-prefill contract (see :mod:`fissionattn.impl`): for request ``r`` the new
tokens are queries ``[context_len_r, total_len_r)`` of the sequence, the
already-cached K/V lives in the paged latent slab, and the new tokens' own
K/V comes from ``kv_b_proj`` applied to their latent part.  Each request is
processed independently — the loop is over *requests*, not tokens, and each
iteration touches at most ``chunk_keys`` keys at a time.

This is the documented-v0 implementation: correctness-first, CUDA-core
friendly (plain matmuls), and it never materializes a full-sequence K/V.
"""

from __future__ import annotations

import torch

from .prefill import OnlineMHA, expand_latent_context, gather_latent_rows

__all__ = ["mla_prefill_torch"]


def mla_prefill_torch(
    q: torch.Tensor,
    latent_new: torch.Tensor,
    kv_b_proj,
    cache3: torch.Tensor,
    block_table_rows: list[torch.Tensor],
    context_lens: list[int],
    query_lens: list[int],
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    page_size: int,
    scale: float,
    chunk_keys: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absorbed MLA prefill over new tokens.

    Args:
        q: ``[n, num_heads, qk_nope + qk_rope]`` absorbed queries for the new
            tokens of all requests, concatenated in request order.
        latent_new: ``[n, kv_lora_rank + qk_rope_head_dim]`` — the same latent
            rows the caller is about to store into ``cache3`` (their K/V come
            from here, so the cache need not be written first).
        kv_b_proj: callable ``[c, kv_lora_rank] -> ([c, num_heads*(nope+v)], _)``.
        cache3: paged latent slab ``[num_blocks, page, kv_lora_rank + rope]``.
        block_table_rows: per-request ``[max_blocks]`` int block tables.
        context_lens / query_lens: cached / new token counts per request.
        num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
        kv_lora_rank, page_size, scale: geometry.
        chunk_keys: keys consumed per online-softmax step.

    Returns:
        out ``[n, num_heads, v_head_dim]`` (dtype of ``q``) and lse
        ``[n, num_heads]`` fp32, for the merge with a decode-style split.
    """
    dev = q.device
    out = torch.zeros(q.size(0), num_heads, v_head_dim, dtype=q.dtype, device=dev)
    lse = torch.full((q.size(0), num_heads), -float("inf"), dtype=torch.float32, device=dev)
    head_size = qk_nope_head_dim + qk_rope_head_dim

    tok = 0
    for r, (clen, qlen) in enumerate(zip(context_lens, query_lens)):
        if qlen <= 0:
            continue
        q_r = q[tok : tok + qlen]
        total = clen + qlen
        att = OnlineMHA(q_r, scale, key_total=total)
        # 1) cached context: gather latent rows -> kv_b_proj -> k/v chunks
        for gidx, rows in gather_latent_rows(
            cache3, block_table_rows[r], 0, clen, page_size, chunk_keys
        ):
            k, v = expand_latent_context(
                rows,
                kv_b_proj,
                num_heads,
                qk_nope_head_dim,
                v_head_dim,
                kv_lora_rank,
            )
            att.add_keys(k, v, gidx)
        # 2) the new tokens themselves (causal within the block)
        for s in range(0, qlen, chunk_keys):
            rows = latent_new[tok + s : tok + min(s + chunk_keys, qlen)]
            k, v = expand_latent_context(
                rows,
                kv_b_proj,
                num_heads,
                qk_nope_head_dim,
                v_head_dim,
                kv_lora_rank,
            )
            att.add_keys(k, v, clen + s)
        o, l = att.finish(q.dtype)
        if o.size(1) != num_heads:  # kv_b_proj gave per-head rows; shape check
            raise RuntimeError(f"fissionattn: prefill head mismatch {tuple(o.shape)}")
        out[tok : tok + qlen] = o
        lse[tok : tok + qlen] = l
        tok += qlen
    if tok != q.size(0):
        raise RuntimeError(
            f"fissionattn: prefill consumed {tok} queries, layer passed {q.size(0)} "
            f"(sum of query_lens must equal num query rows; head_size={head_size})"
        )
    return out, lse
