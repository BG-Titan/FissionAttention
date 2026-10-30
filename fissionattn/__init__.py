"""FuseMoeMLA: CUDA-core MLA attention for vLLM — runs on Pascal sm_61.

DeepSeek-style MLA keeps ONE latent KV head of ``kv_lora_rank +
qk_rope_head_dim`` (576 for DeepSeek-V2/V3) instead of per-head K/V, and
decodes by attending with an *absorbed* query against that latent cache.  The
shape is ideal for CUDA cores: one warp per (row, head) keeps the 576-wide
query in registers and folds the online softmax into an fp32 accumulator, with
no tensor cores and no shared memory.  vLLM's MLA backends all need tensor
cores (FlashMLA sm_90, FlashInfer sm_75+, CUTLASS sm_80+) or Triton (sm_70+),
so on 2x8G Pascal cards there is nothing to select — this package supplies it.

The kernel is a plain nvcc-built ``.so`` loaded through ``ctypes`` (no torch
headers, ABI checked via ``dlopen``), so it does not depend on the installed
torch version.  When the library is missing, the identical math runs in pure
torch (:mod:`fissionattn.ref`), which is also what the CPU tests exercise.

Register through the ``vllm.general_plugins`` entry point
(``fissionattn.plugin:load``), then ``--attention-backend FISSION_ATTN``.
"""

import logging
import os

FISSIONATTN_BACKEND_NAME = "FISSION_ATTN"
FISSIONATTN_PREFILL_NAME = "FISSION_ATTN"
__version__ = "0.1.0"


def _setup_logging() -> None:
    level = (os.environ.get("FISSION_ATTN_LOG") or "").upper()
    if not level:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("fissionattn %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger("fissionattn")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(handler)


_setup_logging()
