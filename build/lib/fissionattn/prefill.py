# SPDX-License-Identifier: Apache-2.0
"""Torch-matmul MLA prefill (chunked, CUDA-core friendly).

vLLM's MLA backends get their prefill from an ``MLAPrefillBackend`` (FA /
FlashInfer / TensorRT-LLM ragged / CPU SDPA).  None of the GPU ones run on
Pascal, and the absorbed MLA formulation means the *context* KV has to be
expanded through ``kv_b_proj`` before it can be attended to at all::

    cached row  = [latent (kv_lora_rank) | rope (qk_rope_head_dim)]
    k_context   = concat(latent @ W_ck_nope, rope)      # per q-head
    v_context   = latent @ W_cv

This module implements that path with plain ``torch.matmul`` and an online
softmax over key chunks, so peak memory is ``chunk_keys * num_heads * dim``
instead of the full sequence.  It is deliberately written against plain
tensors (no vLLM imports) so the CPU tests can drive it directly.

Numerics: fp32 accumulation, ``exp`` base e — matches the decode kernel, so
the two halves of a chunked-prefill step merge consistently.
"""

from __future__ import annotations

from typing import Iterator

import torch

__all__ = ["OnlineMHA", "expand_latent_context", "gather_latent_rows"]


class OnlineMHA:
    """Online-softmax multi-head attention fed with key/value chunks.

    Args:
        q: ``[Q, H, Dq]`` queries of ONE request (any float dtype; upcast).
        scale: softmax scale.
        key_total: total number of keys this query block will see (context +
            new tokens); used for the causal offset ``key_total - Q``.
    """

    def __init__(self, q: torch.Tensor, scale: float, key_total: int) -> None:
        self.q = q.to(torch.float32)
        self.H = q.size(1)
        self.Q = q.size(0)
        self.scale = float(scale)
        self.key_total = int(key_total)
        self.dev = q.device
        self.out = torch.zeros(self.Q, self.H, q.size(2), dtype=torch.float32, device=self.dev)
        self.vdim = 0
        self.m = torch.full((self.Q, self.H, 1), -float("inf"), dtype=torch.float32, device=self.dev)
        self.l = torch.zeros(self.Q, self.H, 1, dtype=torch.float32, device=self.dev)
        self._row = torch.arange(self.Q, device=self.dev, dtype=torch.int64)

    def add_keys(self, k: torch.Tensor, v: torch.Tensor, key_start: int, causal: bool = True) -> None:
        """Consume keys ``[C, H, Dk]`` / values ``[C, H, Dv]`` starting at global index.

        Args:
            k: keys for this chunk.
            v: values for this chunk.
            key_start: global index of ``k[0]`` in the request's sequence.
            causal: apply the causal mask (query ``i`` sees keys
                ``<= i + key_total - Q``).
        """
        C = k.size(0)
        if C == 0:
            return
        if self.vdim == 0:
            self.vdim = v.size(2)
            self.out = torch.zeros(self.Q, self.H, self.vdim, dtype=torch.float32, device=self.dev)
        kf = k.to(torch.float32).transpose(0, 1)  # [H, C, Dk]
        vf = v.to(torch.float32).transpose(0, 1)  # [H, C, Dv]
        qh = self.q.transpose(0, 1)  # [H, Q, Dq]
        logits = torch.bmm(qh, kf.transpose(1, 2)) * self.scale  # [H, Q, C]
        allow = None
        if causal:
            offset = self.key_total - self.Q
            j = torch.arange(key_start, key_start + C, device=self.dev, dtype=torch.int64)
            # allowed iff j <= i + offset  (i = query index within this block)
            allow = j[None, :] <= (self._row + offset)[:, None]  # [Q, C]
        out = self.out.transpose(0, 1)  # [H, Q, Dv] view
        m_new = torch.maximum(self.m.transpose(0, 1), logits.max(dim=-1, keepdim=True).values)
        m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
        corr = torch.exp(self.m.transpose(0, 1) - m_safe)
        p = torch.exp(logits - m_safe)
        if causal:
            p = p * allow[None].to(p.dtype)
        self.l = self.l.transpose(0, 1) * corr + p.sum(dim=-1, keepdim=True)
        out = out * corr + torch.bmm(p, vf)
        self.m = m_safe.transpose(0, 1).contiguous()
        self.l = self.l.transpose(0, 1).contiguous()
        self.out = out.transpose(0, 1).contiguous()

    def finish(self, out_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([Q, H, Dv], [Q, H] lse)``; empty rows come out as zeros/-inf."""
        l = self.l
        safe = torch.where(l > 0, l, torch.ones_like(l))
        out = (self.out / safe) * (l > 0).to(l.dtype)
        lse = torch.where(l > 0, self.m + torch.log(safe), torch.full_like(self.m, -float("inf")))
        return out.to(out_dtype), lse.squeeze(-1).to(torch.float32)


def gather_latent_rows(
    cache3: torch.Tensor,
    block_table_row: torch.Tensor,
    start: int,
    length: int,
    page_size: int,
    chunk: int,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(global_index, [c, D])`` latent||rope rows of one request.

    ``cache3`` is ``[num_blocks, page, D]`` (a strided view is fine — rows are
    selected by index, nothing is made contiguous).
    """
    if length <= 0:
        return
    pos = torch.arange(start, start + length, device=cache3.device, dtype=torch.int64)
    for s in range(0, length, chunk):
        pp = pos[s : s + chunk]
        blk = block_table_row[(pp // page_size).to(block_table_row.dtype)]
        off = (pp % page_size).to(torch.int64)
        yield start + s, cache3[blk, off]


def expand_latent_context(
    rows: torch.Tensor,
    kv_b_proj,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[c, kv_lora_rank + rope]`` cache rows -> ``k [c,H,nope+rope]``, ``v [c,H,Dv]``.

    ``kv_b_proj`` is vLLM's ``ColumnParallelLinear`` returning
    ``[c, num_heads * (nope + v)]``; calling it per chunk is what keeps the
    prefill memory bounded.
    """
    latent = rows[..., :kv_lora_rank]
    rope = rows[..., kv_lora_rank:]
    kv = kv_b_proj(latent)[0].view(-1, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, v = kv.split([qk_nope_head_dim, v_head_dim], dim=-1)
    if rope.size(-1):
        k = torch.cat([k_nope, rope.unsqueeze(1).expand(-1, num_heads, -1)], dim=-1)
    else:
        k = k_nope
    return k, v
