# SPDX-License-Identifier: Apache-2.0
"""Torch-CPU tests for the MLA decode entry point (:func:`fissionattn.ops.decode`).

On CPU ``ops.decode`` runs the torch implementation of the exact same contract
(no CUDA device, no .so needed), so these tests cover argument marshalling,
the paged layouts vLLM hands us (4-D ``[num_blocks, 1, page, D]`` slabs and
non-compact strided views), dtypes and the empty-sequence convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fissionattn import ops  # noqa: E402


def _ref_fp64(q64, cache64, bt, seq_lens, out_dim, page, scale):
    """Independent float64 reference over a logical [nb, page, D] cache."""
    rows, H, _ = q64.shape
    out = torch.zeros(rows, H, out_dim, dtype=torch.float64)
    lse = torch.full((rows, H), -float("inf"), dtype=torch.float64)
    for r in range(rows):
        length = int(seq_lens[r])
        if length <= 0:
            continue
        pos = torch.arange(length, dtype=torch.int64)
        blk = bt[r][pos // page].long()
        kv = cache64[blk, pos % page]
        logits = (q64[r] @ kv.T) * scale
        w = torch.softmax(logits, dim=-1)
        out[r] = w @ kv[:, :out_dim]
        lse[r] = torch.logsumexp(logits, dim=-1)
    return out, lse


def _case(seed=0, rows=5, heads=3, latent=48, rope=16, page=4, nb=12,
          dtype=torch.float16, idx=torch.int64, strided=False):
    g = torch.Generator().manual_seed(seed)
    D = latent + rope
    seq_lens = torch.tensor([0, 1, 7, 16, 23, 33, 5, 61][:rows], dtype=torch.int64)
    nblk_needed = max(-(-int(s) // page) for s in seq_lens.tolist())
    nb = max(nb, nblk_needed)
    perm = torch.randperm(nb, generator=g)[:nblk_needed].to(idx)
    bt = torch.zeros(rows, max(4, nblk_needed), dtype=idx)
    bt[:, :nblk_needed] = perm.unsqueeze(0)
    if strided:
        big = torch.randn(nb, 2, page, D, generator=g, dtype=torch.float32)
        cache_logical = big[:, 0].clone()
        cache = big[:, 0]  # non-compact: stride(0) == 2 * page * D
    else:
        cache_logical = torch.randn(nb, page, D, generator=g, dtype=torch.float32)
        cache = cache_logical
    q = torch.randn(rows, heads, D, generator=g, dtype=torch.float32).to(dtype)
    return q, cache, cache_logical, bt, seq_lens, D, latent, rope


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("page", [1, 4, 16])
def test_decode_matches_reference(dtype, page):
    q, cache, cache_logical, bt, seq_lens, D, latent, rope = _case(
        seed=11 + page, dtype=dtype, page=page
    )
    scale = 1.0 / (D**0.5)
    out = torch.zeros(q.size(0), q.size(1), latent, dtype=dtype)
    lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32)
    ops.decode(
        q, out, cache, bt, seq_lens,
        num_q_heads=q.size(1), head_size=D, out_dim=latent,
        page_size=page, scale=scale, lse=lse,
    )
    ref_out, ref_lse = _ref_fp64(
        q.double(), cache_logical.double(), bt, seq_lens, latent, page, scale
    )
    tol = 1e-5 if dtype == torch.float32 else 6e-3
    assert torch.allclose(out.double(), ref_out, atol=tol, rtol=tol), (
        (out.double() - ref_out).abs().max().item()
    )
    ok = ref_lse > float("-inf")
    assert torch.allclose(lse.double()[ok], ref_lse[ok], atol=tol * 4, rtol=tol * 4)
    assert torch.all(torch.isneginf(lse.double()[~ok]))


def test_decode_vllm_4d_slab_and_strided_view():
    """The engine hands a [num_blocks, 1, page, D] slab; strides must be honoured."""
    q, cache3, cache_logical, bt, seq_lens, D, latent, rope = _case(seed=21, strided=True)
    cache4 = cache3.unsqueeze(1)  # [nb, 1, page, D] — vLLM's MLA slab layout
    assert cache4.stride(0) == 2 * cache3.size(1) * D
    scale = 1.0 / (D**0.5)
    out_a = torch.zeros(q.size(0), q.size(1), latent, dtype=q.dtype)
    out_b = torch.zeros_like(out_a)
    ops.decode(q, out_a, cache3, bt, seq_lens, q.size(1), D, latent, cache3.size(1), scale)
    ops.decode(q, out_b, cache4, bt, seq_lens, q.size(1), D, latent, cache3.size(1), scale)
    assert torch.equal(out_a, out_b)
    ref_out, _ = _ref_fp64(
        q.double(), cache_logical.double(), bt, seq_lens, latent, cache3.size(1), scale
    )
    assert torch.allclose(out_a.double(), ref_out, atol=6e-3, rtol=6e-3)


def test_decode_int32_block_table_matches_int64():
    q, cache, _, bt64, seq_lens, D, latent, rope = _case(seed=31, idx=torch.int64)
    _, _, _, bt32, _, _, _, _ = _case(seed=31, idx=torch.int32)
    scale = 1.0 / (D**0.5)
    out_a = torch.zeros(q.size(0), q.size(1), latent, dtype=q.dtype)
    out_b = torch.zeros_like(out_a)
    ops.decode(q, out_a, cache, bt64, seq_lens, q.size(1), D, latent, cache.size(1), scale)
    ops.decode(q, out_b, cache, bt32, seq_lens, q.size(1), D, latent, cache.size(1), scale)
    assert torch.equal(out_a, out_b)


def test_decode_rope_only_out_dim_zero():
    q, cache, cache_logical, bt, seq_lens, D, latent, rope = _case(seed=41)
    scale = 1.0 / (D**0.5)
    out = torch.zeros(q.size(0), q.size(1), 0, dtype=q.dtype)
    lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32)
    ops.decode(q, out, cache, bt, seq_lens, q.size(1), D, 0, cache.size(1), scale, lse=lse)
    _, ref_lse = _ref_fp64(q.double(), cache_logical.double(), bt, seq_lens, 0, cache.size(1), scale)
    ok = torch.isfinite(ref_lse)
    assert torch.allclose(lse.double()[ok], ref_lse[ok], atol=2e-2, rtol=2e-2)


def test_cache_strides_validation():
    nb, page, D = 4, 8, 16
    slab = torch.zeros(nb, 1, page, D)
    assert ops.cache_strides(slab) == (slab.stride(0), slab.stride(2))
    squeezed = slab.squeeze(1)
    assert ops.cache_strides(squeezed) == (squeezed.stride(0), squeezed.stride(1))
    with pytest.raises(ValueError, match="num_kv_heads == 1"):
        ops.cache_strides(torch.zeros(nb, 2, page, D))
    with pytest.raises(ValueError, match="2-D|3-D"):
        ops.cache_strides(torch.zeros(nb, page))
    # last dim must be contiguous
    transposed = torch.zeros(nb, D, page).transpose(1, 2)
    with pytest.raises(ValueError, match="contiguous"):
        ops.cache_strides(transposed)


def test_decode_empty_rows_only_is_all_zero():
    q, cache, _, bt, _, D, latent, rope = _case(seed=51, rows=3)
    seq_lens = torch.zeros(3, dtype=torch.int64)
    out = torch.full((3, q.size(1), latent), 7.0, dtype=q.dtype)
    lse = torch.zeros(3, q.size(1), dtype=torch.float32)
    ops.decode(q, out, cache, bt, seq_lens, q.size(1), D, latent, cache.size(1),
               0.1, lse=lse)
    assert not out.any()
    assert torch.all(torch.isneginf(lse.double()))
