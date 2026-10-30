# SPDX-License-Identifier: Apache-2.0
"""GPU parity test: the native CUDA-core MLA decode kernel vs the torch ref.

Skipped without a CUDA device or without the built ``_mla*.so``.  Run with
``CUDA_VISIBLE_DEVICES=0`` (the sandbox has a free GTX 1070 / sm_61).
``FISSIONATTN_REQUIRE_NATIVE=1`` makes the test fail loudly instead of silently
falling back to the torch path — the point is to test the kernel.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fissionattn import ops  # noqa: E402
from fissionattn.ref import decode_ref_torch  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device (sm_61+)"
)


def _setup(seed, rows, heads, latent, rope, page, nb, seq_lens, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    D = latent + rope
    sl = torch.tensor(seq_lens, dtype=torch.int64)
    nblk_needed = max(-(-int(s) // page) for s in seq_lens)
    assert nblk_needed <= nb
    perm = torch.randperm(nb, generator=g)[:nblk_needed]
    bt = torch.zeros(rows, max(1, nblk_needed), dtype=torch.int64)
    bt[:, :nblk_needed] = perm.unsqueeze(0)
    cache4 = torch.randn(nb, 1, page, D, generator=g).to(dtype).cuda()
    q = torch.randn(rows, heads, D, generator=g).to(dtype).cuda()
    return q, cache4, bt.cuda(), sl.cuda(), D, latent


@needs_cuda
def test_native_library_is_loadable():
    assert ops.native_available(), (
        "build it first: FISSION_ATTN_CUDA_ARCHS=6.1 python setup.py build_py "
        "--build-lib .buildlib (or copy the .so next to the package)"
    )


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page", [1, 16])
def test_decode_kernel_matches_torch_reference(dtype, page, monkeypatch):
    monkeypatch.setenv("FISSIONATTN_REQUIRE_NATIVE", "1")
    seq_lens = [1, 17, 200, 577, 1024, 63, 129, 4096]
    q, cache4, bt, sl, D, latent = _setup(
        seed=101 + page, rows=len(seq_lens), heads=16, latent=512, rope=64,
        page=page, nb=4096, seq_lens=seq_lens, dtype=dtype,
    )
    scale = D**-0.5
    out = torch.zeros_like(q[:, :, :latent])
    lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32, device=q.device)
    ops.decode(q, out, cache4, bt, sl, q.size(1), D, latent, page, scale, lse=lse)

    ref_out = torch.zeros_like(out)
    ref_lse = torch.zeros_like(lse)
    decode_ref_torch(q, ref_out, cache4, bt, sl, q.size(1), D, latent, page, scale,
                     lse=ref_lse)
    tol = 1e-2 if dtype == torch.bfloat16 else 3e-3
    err = (out.float() - ref_out.float()).abs().max().item()
    assert err < tol, f"max |Δout| = {err} (dtype {dtype}, page {page})"
    lerr = (lse - ref_lse).abs().max().item()
    assert lerr < tol * 4, f"max |Δlse| = {lerr} (dtype {dtype}, page {page})"


@needs_cuda
def test_decode_kernel_real_geometry_small(monkeypatch):
    """DeepSeek-V2 geometry, 32 requests, ~1k context: the decode microbench."""
    monkeypatch.setenv("FISSIONATTN_REQUIRE_NATIVE", "1")
    rows, heads, latent, rope, page, nb = 32, 16, 512, 64, 32, 2048
    D = latent + rope
    seq_lens = [1 + (i * 137) % 1024 for i in range(rows)]
    q, cache4, bt, sl, D, latent = _setup(
        seed=7, rows=rows, heads=heads, latent=latent, rope=rope, page=page,
        nb=nb, seq_lens=seq_lens, dtype=torch.float16,
    )
    scale = D**-0.5
    out = torch.zeros(rows, heads, latent, dtype=torch.float16, device=q.device)
    lse = torch.zeros(rows, heads, dtype=torch.float32, device=q.device)
    ops.decode(q, out, cache4, bt, sl, heads, D, latent, page, scale, lse=lse)
    ref = torch.zeros_like(out)
    decode_ref_torch(q, ref, cache4, bt, sl, heads, D, latent, page, scale)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 3e-3, f"max |Δout| = {err}"

    # timing is informational only (1070 vs 4090 differ by ~an order of magnitude)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        ops.decode(q, out, cache4, bt, sl, heads, D, latent, page, scale, lse=lse)
    torch.cuda.synchronize()
    print(f"\nfissionattn decode: {rows} rows x {heads} heads x D={D}, "
          f"avg {int(sum(seq_lens) / rows)} ctx -> "
          f"{(time.perf_counter() - t0) / 5 * 1e3:.3f} ms/step")


@needs_cuda
def test_decode_rejects_bad_inputs(monkeypatch):
    monkeypatch.setenv("FISSIONATTN_REQUIRE_NATIVE", "1")
    q, cache4, bt, sl, D, latent = _setup(
        seed=1, rows=2, heads=4, latent=32, rope=8, page=4, nb=8,
        seq_lens=[4, 9], dtype=torch.float16,
    )
    out = torch.zeros(2, 4, 32, dtype=torch.float16, device=q.device)
    with pytest.raises(ValueError, match="seq_lens"):
        ops.decode(q, out, cache4, bt, sl[:1], 4, D, latent, 4, 0.1)
