"""FISSION_ATTN decode implementation (MLA absorbed MQA on CUDA cores).

``MLACommonImpl`` provides everything except ``forward_mqa``: the absorbed
query built by ``MLAAttention.forward_impl`` (``q_nope @ W_UK`` concatenated
with ``q_pe``) arrives here as ``[rows, num_heads, kv_lora_rank +
qk_rope_head_dim]``, together with the single-head latent cache
``[num_blocks, page, 1, kv_lora_rank + qk_rope_head_dim]``.  The returned
latent attention output is fed to ``_v_up_proj`` (``W_UV``) by the caller, so
this file never touches ``kv_b_proj``/``q_b_proj`` weights.

Decode runs one warp per (row, head): the warp keeps the query in registers,
walks the paged latent rows, and folds the online softmax into an
fp32 accumulator over the first ``kv_lora_rank`` dims of the same row (V is
the latent prefix of K in MLA).  No tensor cores, no shared memory: the
kernel is bandwidth-bound and shape-agnostic, which is what a Pascal-class
device can actually run.  When the native library is unavailable the same
math is executed by :mod:`fissionattn.ref` in pure torch.
"""

from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.v1.attention.backend import AttentionType

from . import ops

MAX_HEAD_SIZE = 768
MAX_OUT_DIM = 512


def _latent_cache(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``[blocks, 1, page, D]``/``[blocks, page, D]`` -> (3-D view, page)."""
    if cache.dim() == 4:
        if cache.size(1) != 1:
            raise NotImplementedError(
                f"FISSION_ATTN expects a single latent KV head, got {cache.size(1)}"
            )
        cache = cache.squeeze(2)
    if cache.dim() != 3:
        raise ValueError(f"FISSION_ATTN: unexpected latent cache shape {tuple(cache.shape)}")
    return cache, cache.size(1)


class FissionAttentionImpl(MLACommonImpl[MLACommonMetadata]):
    """CUDA-core absorbed MLA decode; prefill rides the shared dense-MHA path."""

    can_return_lse_for_decode: bool = False
    supports_dcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "FISSION_ATTN supports none of alibi_slopes / sliding_window / "
                "logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"FISSION_ATTN implements decoder self-attention only, got {attn_type}"
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                f"FISSION_ATTN supports fp16/bf16 KV cache, got {kv_cache_dtype}"
            )
        if self.dcp_world_size != 1:
            raise NotImplementedError("FISSION_ATTN does not support decode context parallelism")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("FISSION_ATTN does not support KV sharing")
        if self.kv_lora_rank > MAX_OUT_DIM:
            raise NotImplementedError(
                f"FISSION_ATTN kernel accumulator covers kv_lora_rank <= {MAX_OUT_DIM}, "
                f"model has {self.kv_lora_rank}"
            )
        if self.qk_head_dim + self.qk_rope_head_dim > MAX_HEAD_SIZE:
            raise NotImplementedError(
                f"FISSION_ATTN kernel covers absorbed head_size <= {MAX_HEAD_SIZE}"
            )
        # Sparse MLA (DeepSeek-V3.2 style indexer) must never reach this impl.
        if getattr(self, "is_sparse", False):
            raise NotImplementedError("FISSION_ATTN does not implement sparse MLA")

        self._torch_decode = os.environ.get("FISSION_ATTN_TORCH_DECODE", "0") == "1"

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)
        if q.stride(2) != 1:
            q = q.contiguous()

        cache, page_size = _latent_cache(kv_c_and_k_pe_cache)
        if cache.dtype != q.dtype:
            raise NotImplementedError(
                f"FISSION_ATTN expects the latent cache in the query dtype "
                f"({q.dtype}), got {cache.dtype}"
            )

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        rows = q.size(0)
        num_rows = seq_lens.shape[0]

        # A uniform multi-token block (speculative decode / uniform chunked
        # batch) arrives as query_len flattened rows over num_rows requests:
        # repeat each request's block table and express intra-block causality
        # as per-row extents.  Non-uniform (mixed prefill+decode) batches do
        # not come here — MLAAttention routes those through forward_mha.
        query_len, rest = divmod(rows, num_rows) if num_rows else (1, 0)
        if rest:
            raise RuntimeError(
                f"FISSION_ATTN: {rows} query rows over {num_rows} requests is not a "
                "uniform decode block"
            )
        if query_len > 1:
            block_table = block_table.repeat_interleave(query_len, dim=0)
            if attn_metadata.causal:
                offsets = torch.arange(
                    1 - query_len, 1, device=seq_lens.device, dtype=seq_lens.dtype
                )
                seq_lens = (seq_lens.unsqueeze(1) + offsets).flatten().clamp(min=0)
            else:
                seq_lens = seq_lens.repeat_interleave(query_len)

        out = torch.empty(
            rows, q.size(1), self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        if self._torch_decode or q.dtype not in (torch.float16, torch.bfloat16):
            from .ref import decode_ref_torch

            decode_ref_torch(
                q,
                out,
                cache,
                block_table,
                seq_lens,
                num_q_heads=q.size(1),
                head_size=q.size(2),
                out_dim=self.kv_lora_rank,
                page_size=page_size,
                scale=self.scale,
            )
        else:
            ops.decode(
                q,
                out,
                cache,
                block_table,
                seq_lens,
                num_q_heads=q.size(1),
                head_size=q.size(2),
                out_dim=self.kv_lora_rank,
                page_size=page_size,
                scale=self.scale,
            )
        return out, None
