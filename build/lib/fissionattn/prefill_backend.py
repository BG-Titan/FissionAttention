"""MLA prefill backend for GPUs without a tensor-core attention library.

vLLM selects an ``MLAPrefillBackend`` for the dense-MHA half of an MLA step
(``MLACommonImpl.forward_mha`` -> ``run_prefill_new_tokens`` for the new-token
block, ``_compute_prefill_context`` -> ``run_prefill_context_chunk`` for each
workspace-sized slice of the cached prefix).  The in-tree GPU choices are
FlashAttention, FlashInfer and TensorRT-LLM ragged — none of which validate on
Pascal (sm_61/6.1 has no bf16 and no flash-attn build) — and the CPU SDPA
backend is only auto-selected on CPU devices.

This backend computes the same thing with plain ``torch.matmul`` in fp32 and an
online softmax over key chunks, so peak memory is
``num_heads * query_chunk * key_chunk`` rather than the full
``heads x Q x K`` score matrix.  No tensor cores, no custom kernels: it is the
CUDA-core counterpart of the decode kernel, and it is the reason an MLA model
can be booted end to end on 2x8G Pascal cards.

Registered by :func:`fissionattn.plugin.register` as
``MLAPrefillBackendEnum.FISSION_ATTN``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )


def _default_chunk_keys() -> int:
    try:
        return max(64, int(os.environ.get(
            "FISSION_ATTN_PREFILL_CHUNK_KEYS", "512")))
    except ValueError:
        return 512


class FissionPrefillBackend(MLAPrefillBackend):
    """Dense per-request MLA prefill in torch (fp32 accumulate, chunked keys)."""

    @staticmethod
    def get_name() -> str:
        return "FISSION_ATTN_PREFILL"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def supports_out(self) -> bool:
        return True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chunk_keys = _default_chunk_keys()

    def _ragged_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_cu_seq_lens: torch.Tensor,
        kv_cu_seq_lens: torch.Tensor,
        causal: bool,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attention over a ragged batch, one request at a time.

        ``q``/``k``/``v`` are ``[num_tokens, num_heads, head_dim]`` and the two
        ``cu_seq_lens`` give the per-request row offsets (they differ for a
        context chunk, where keys are the cached prefix).
        """
        from .prefill import OnlineMHA

        num_reqs = q_cu_seq_lens.numel() - 1
        assert num_reqs == kv_cu_seq_lens.numel() - 1

        out_chunks: list[torch.Tensor] = []
        lse_chunks: list[torch.Tensor] = []
        for r in range(num_reqs):
            q_start, q_end = int(q_cu_seq_lens[r]), int(q_cu_seq_lens[r + 1])
            kv_start, kv_end = int(kv_cu_seq_lens[r]), int(kv_cu_seq_lens[r + 1])
            qlen = q_end - q_start
            if qlen <= 0:
                continue
            klen = kv_end - kv_start
            att = OnlineMHA(
                q[q_start:q_end],
                self.scale,
                key_total=qlen if causal else klen,
            )
            for s in range(0, klen, self.chunk_keys):
                e = min(s + self.chunk_keys, klen)
                att.add_keys(
                    k[kv_start + s : kv_start + e],
                    v[kv_start + s : kv_start + e],
                    key_start=s,
                    causal=causal,
                )
            out_r, lse_r = att.finish(out_dtype=v.dtype)
            out_chunks.append(out_r)  # [Sq, H, Dv]
            if return_softmax_lse:
                lse_chunks.append(lse_r.transpose(0, 1))  # [H, Sq]

        if not out_chunks:
            empty = q.new_zeros((0, q.size(1), v.size(2)))
            if not return_softmax_lse:
                return empty
            return empty, torch.full(
                (q.size(1), 0), -float("inf"), dtype=torch.float32, device=q.device
            )

        out = torch.cat(out_chunks, dim=0)
        if not return_softmax_lse:
            return out
        lse = torch.cat(lse_chunks, dim=1)  # [num_heads, total_queries]
        return out, lse

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert output_scale is None, "FISSION_ATTN prefill does not write quantized output"
        meta = self._prefill_metadata  # set by prepare_metadata
        out_t = self._ragged_attention(
            q,
            k,
            v,
            meta.query_start_loc,
            meta.query_start_loc,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )
        if out is not None and not return_softmax_lse:
            assert isinstance(out_t, torch.Tensor)
            out.copy_(out_t)
            return out
        return out_t

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out is None, "FISSION_ATTN context prefill writes a fresh output"
        result = self._ragged_attention(
            q,
            k,
            v,
            chunk.query_start_loc,
            chunk.cu_seq_lens,
            causal=False,
            return_softmax_lse=True,
        )
        assert isinstance(result, tuple)
        return result

FissionPrefillBackend = FissionPrefillBackend  # pre-rename alias
