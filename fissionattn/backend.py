"""FISSION_ATTN: attention backend for MLA models on CUDA-core GPUs.

Registered as ``AttentionBackendEnum.FISSION_ATTN`` by
:func:`fissionattn.plugin.register`; select it with
``--attention-backend FISSION_ATTN``.  Because it derives from
``MLACommonBackend`` (``is_mla() == True``), vLLM's own
``AttentionBackend.validate_configuration`` rejects it for non-MLA models and
rejects non-MLA backends for MLA models — no extra plumbing needed.

Decode uses the CUDA-core warp-per-(row, head) kernel in ``csrc/mla.cu``;
prefill uses the shared dense-MHA path of ``MLACommonImpl`` with the
torch-matmul prefill backend registered alongside
(``MLAPrefillBackendEnum.FISSION_ATTN``).
"""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    MultipleOf,
)

from .impl import FissionAttentionImpl


class FissionMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    """Default MLA metadata; the kernel needs no per-split workspace.

    Like the Triton MLA backend, multi-token query blocks are flattened to
    one decode row per query token, so causality is expressed as per-row
    sequence extents and only uniform query lengths are supported.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    supports_non_causal_multi_token_decode: ClassVar[bool] = True


class FissionAttentionBackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "FISSION_ATTN"

    @staticmethod
    def get_impl_cls() -> type[FissionAttentionImpl]:
        return FissionAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FissionMetadataBuilder]:
        return FissionMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Absorbed MLA head = kv_lora_rank + qk_rope_head_dim.  The kernel is
        # shape-agnostic up to 768; these are the dimensions shipped models
        # use (DeepSeek-V2/V3, Kimi K2, MiniMax 512+64=576).
        return [320, 512, 576]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # One warp per (row, head), keys folded in order: the reduction does
        # not depend on batch size or row placement.
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

# Pre-rename compatibility aliases (brand: FissionAttention -> FissionAttention).
FissionAttentionBackend = FissionAttentionBackend
FissionMetadataBuilder = FissionMetadataBuilder
