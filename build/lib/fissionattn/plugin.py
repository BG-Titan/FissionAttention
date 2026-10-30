"""vLLM general-plugin entry: registers FissionAttention (MLA decode + prefill).

The user-facing backend name is FISSION_ATTN.

pyproject entry point:
    vllm.general_plugins  fissionattn = "fissionattn.plugin:register"

(:func:`load` is kept as the setuptools-style alias.)  The vLLM plugin loader
runs this early (before ``AttentionConfig`` validation), so ``--attention-backend FISSION_ATTN`` parses afterwards::

    VLLM_PLUGINS=fissionattn vllm serve deepseek-ai/DeepSeek-V2-Lite \\
        --attention-backend FISSION_ATTN

Two registrations happen here:

1. ``AttentionBackendEnum.FISSION_ATTN`` -> :class:`fissionattn.backend.FissionAttentionBackend`
   (CUDA-core absorbed decode).
2. ``MLAPrefillBackendEnum.FISSION_ATTN`` ->
   :class:`fissionattn.prefill_backend.FissionPrefillBackend` (torch-matmul
   dense prefill).  vLLM has no MLA prefill backend that validates on Pascal,
   so by default the platform auto-selection is wrapped to fall back to it
   instead of raising.  ``FISSION_ATTN_NO_PREFILL_FALLBACK=1`` disables that.

``FISSION_ATTN_AUTO=1`` additionally appends FISSION_ATTN to the CUDA platform's
MLA backend priority list, so MLA models pick it up without the explicit flag.
"""

from __future__ import annotations

import logging
import os

_REGISTERED = False
_logger = logging.getLogger("fissionattn.plugin")

BACKEND_CLASS_PATH = "fissionattn.backend.FissionAttentionBackend"
PREFILL_CLASS_PATH = "fissionattn.prefill_backend.FissionPrefillBackend"
BACKEND_NAME = "FISSION_ATTN"
PREFILL_ENUM_NAME = "FISSION_ATTN"


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if BACKEND_NAME not in AttentionBackendEnum._member_map_:
        AttentionBackendEnum.register(BACKEND_NAME, BACKEND_CLASS_PATH)

    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
    )

    if PREFILL_ENUM_NAME not in MLAPrefillBackendEnum._member_map_:
        MLAPrefillBackendEnum.register(PREFILL_ENUM_NAME, PREFILL_CLASS_PATH)

    _REGISTERED = True

    no_fb = os.environ.get("FISSION_ATTN_NO_PREFILL_FALLBACK", "")
    if no_fb != "1":
        _install_prefill_fallback()
    auto = os.environ.get("FISSION_ATTN_AUTO", "1")
    if auto == "1":
        _install_auto_backend()


def _install_prefill_fallback() -> None:
    """Fall back to the torch MLA prefill when auto-selection has no candidate.

    Every in-tree GPU MLA prefill backend requires tensor cores, so on sm_61
    the platform selection raises and an MLA model cannot boot at all.  The
    patch only changes the failure path.
    """
    try:
        from vllm.platforms.interface import Platform
    except Exception:  # pragma: no cover - no vllm
        return
    desc = Platform.__dict__.get("get_mla_prefill_backend_cls")
    func = getattr(desc, "__func__", desc)
    if func is None or getattr(func, "_fissionattn_patched", False):
        return

    def patched(cls, mla_selector_config):
        try:
            return func(cls, mla_selector_config)
        except Exception as exc:
            from vllm.v1.attention.backends.mla.prefill.registry import (
                MLAPrefillBackendEnum,
            )

            _logger.info(
                "fissionattn: MLA prefill auto-selection failed (%s); "
                "using the torch FissionAttention prefill backend",
                exc,
            )
            return MLAPrefillBackendEnum.FISSION_ATTN.get_class()

    patched._fissionattn_patched = True  # type: ignore[attr-defined]
    Platform.get_mla_prefill_backend_cls = classmethod(patched)  # type: ignore[method-assign]


def _install_auto_backend() -> None:
    """Append FISSION_ATTN to the CUDA MLA priority list (tail = last resort)."""
    try:
        import vllm.platforms.cuda as cuda_mod
    except Exception:  # pragma: no cover - non-CUDA install
        return
    orig = getattr(cuda_mod, "_get_backend_priorities", None)
    if orig is None or getattr(orig, "_fissionattn_patched", False):
        return

    def patched(*args, **kwargs):
        prios = list(orig(*args, **kwargs))
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            use_mla = kwargs.get("use_mla")
            if use_mla is None and args:
                use_mla = args[0]
            if use_mla and AttentionBackendEnum.FISSION_ATTN not in prios:
                prios.append(AttentionBackendEnum.FISSION_ATTN)
        except Exception:
            pass
        return prios

    patched._fissionattn_patched = True  # type: ignore[attr-defined]
    cuda_mod._get_backend_priorities = patched
    _logger.info("fissionattn: MLA auto-fallback priority injection installed")


def load() -> None:
    register()
