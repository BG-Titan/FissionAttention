# SPDX-License-Identifier: Apache-2.0
"""CPU-only (numpy) tests for the MLA absorbed-decode math.

These are the tests that stand in for GPU validation of the softmax/dot logic
in ``csrc/mla.cu``: the kernel implements exactly these identities (D-chunked
dot product, online softmax merge, paged gather, latent||rope split).  No
torch and no CUDA device needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fissionattn.ref import (  # noqa: E402
    decode_ref_numpy_one_row,
    naive_mla_attention,
)


def _two_pass(q: np.ndarray, kv: np.ndarray, scale: float, out_dim: int):
    """Independent float64 reference (deliberately not sharing ref.py code)."""
    q = np.asarray(q, dtype=np.float64)
    kv = np.asarray(kv, dtype=np.float64)
    H, D = q.shape
    q64 = q.astype(np.float64)
    kv64 = kv.astype(np.float64)
    out = np.zeros((H, out_dim), dtype=np.float64)
    lse = np.full(H, -np.inf, dtype=np.float64)
    for h in range(H):
        s = np.array([float(np.dot(q64[h], kv64[t])) * scale for t in range(kv64.shape[0])])
        if s.size == 0:
            continue
        mx = float(s.max())
        acc = np.zeros(out_dim, dtype=np.float64)
        total = 0.0
        for t in range(kv64.shape[0]):
            w = float(np.exp(s[t] - mx))
            total += w
            acc += w * kv64[t, :out_dim]
        out[h] = acc / total
        lse[h] = mx + float(np.log(total))
    return out, lse


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_naive_matches_independent_two_pass():
    rng = _rng(0)
    H, D, out_dim, S = 3, 72, 64, 37
    q = rng.standard_normal((H, D)).astype(np.float32)
    kv = rng.standard_normal((S, D)).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    out, lse = naive_mla_attention(q, kv, scale, out_dim)
    ref_out, ref_lse = _two_pass(q, kv, scale, out_dim)
    assert np.allclose(out, ref_out, atol=2e-5, rtol=2e-5)
    assert np.allclose(lse, ref_lse, atol=2e-4, rtol=2e-4)


def test_lse_is_the_normaliser():
    """out must equal sum_i exp(s_i - lse) * v_i (the split-KV combine identity)."""
    rng = _rng(1)
    H, D, out_dim, S = 2, 40, 32, 11
    q = rng.standard_normal((H, D)).astype(np.float32)
    kv = rng.standard_normal((S, D)).astype(np.float32)
    scale = 0.125
    out, lse = naive_mla_attention(q, kv, scale, out_dim)
    logits = (q.astype(np.float64) @ kv[:, :D].astype(np.float64).T) * scale
    for h in range(H):
        w = np.exp(logits[h] - lse[h])
        assert abs(float(w.sum()) - 1.0) < 1e-6
        assert np.allclose(out[h], w @ kv[:, :out_dim].astype(np.float64), atol=2e-5)


def test_single_token_is_the_value_and_logit():
    rng = _rng(2)
    H, D, out_dim = 4, 96, 64
    q = rng.standard_normal((H, D)).astype(np.float32)
    kv = rng.standard_normal((1, D)).astype(np.float32)
    out, lse = naive_mla_attention(q, kv, 0.3, out_dim)
    assert np.allclose(out, np.broadcast_to(kv[0, :out_dim], (H, out_dim)), atol=1e-6)
    expect = (q.astype(np.float64) @ kv[0, :].astype(np.float64)) * 0.3
    assert np.allclose(lse, expect, atol=1e-4)


def test_empty_and_rope_only_extremes():
    rng = _rng(3)
    H, D = 2, 32
    q = rng.standard_normal((H, D)).astype(np.float32)
    kv = np.zeros((0, D), dtype=np.float32)
    out, lse = naive_mla_attention(q, kv, 0.2, 16)
    assert out.shape == (H, 16) and not out.any()
    assert np.all(np.isneginf(lse))
    # out_dim == 0: rope-only attention (kernel path for the second MLA dot)
    kv2 = rng.standard_normal((5, D)).astype(np.float32)
    out0, lse0 = naive_mla_attention(q, kv2, 0.2, 0)
    assert out0.shape == (H, 0)
    assert np.all(np.isfinite(lse0))


def test_d_chunked_dot_equals_full_dot():
    """The kernel accumulates the dot in D-chunks; the sum must be the same."""
    rng = _rng(4)
    D, CHUNK = 576, 256
    a = rng.standard_normal(D).astype(np.float32)
    b = rng.standard_normal(D).astype(np.float32)
    full = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    acc = 0.0
    for s in range(0, D, CHUNK):
        e = min(s + CHUNK, D)
        acc += float(np.dot(a[s:e].astype(np.float64), b[s:e].astype(np.float64)))
    assert abs(full - acc) < 1e-9


def test_online_softmax_merge_equals_two_pass():
    """Chunked (m, l, acc) merge — the exact recursion the kernel runs."""
    rng = _rng(5)
    H, D, out_dim, S = 3, 64, 48, 50
    q = rng.standard_normal((H, D)).astype(np.float64)
    kv = rng.standard_normal((S, D)).astype(np.float64)
    scale = 1.0 / np.sqrt(D)
    m = np.full(H, -np.inf)
    l = np.zeros(H)
    acc = np.zeros((H, out_dim))
    for s in range(0, S, 7):
        blk = kv[s : min(s + 7, S)]
        logits = (q @ blk.T) * scale
        m_new = np.maximum(m, logits.max(axis=-1))
        m_safe = np.where(np.isfinite(m_new), m_new, 0.0)
        corr = np.exp(m - m_safe)
        p = np.exp(logits - m_safe[:, None])
        l = l * corr + p.sum(axis=-1)
        acc = acc * corr[:, None] + p @ blk[:, :out_dim]
        m = m_safe
    out = acc / l[:, None]
    ref_out, ref_lse = _two_pass(q, kv, scale, out_dim)
    assert np.allclose(out, ref_out, atol=1e-12, rtol=1e-12)
    assert np.allclose(m + np.log(l), ref_lse, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_paged_decode_gathers_the_right_rows(page_size):
    """Block-table gather + softmax, incl. pages that are out of order."""
    rng = _rng(6 + page_size)
    H, D, out_dim = 2, 48, 32
    seq_len = 13
    needed = max(1, -(-seq_len // page_size))
    nb = max(9, needed + 2)
    cache = rng.standard_normal((nb, page_size, D)).astype(np.float32)
    perm = rng.permutation(nb)[:needed]
    block_row = np.zeros(max(4, needed), dtype=np.int64)
    block_row[: len(perm)] = perm
    scale = 0.2
    q = rng.standard_normal((H, D)).astype(np.float32)
    out, lse = decode_ref_numpy_one_row(
        q,
        cache,
        block_row,
        seq_len,
        page_size,
        scale,
        out_dim,
    )
    pos = np.arange(seq_len)
    kv = cache[block_row[pos // page_size], pos % page_size]
    ref_out, ref_lse = _two_pass(q, kv, scale, out_dim)
    assert np.allclose(out, ref_out, atol=2e-5, rtol=2e-5)
    assert abs(lse - ref_lse[0]) < 2e-4
    # rows must NOT be in block order: with a permuted table the naive
    # "first seq_len rows of the flattened cache" would differ.
    naive_flat = cache.reshape(nb * page_size, D)[:seq_len]
    bad_out, _ = _two_pass(q, naive_flat, scale, out_dim)
    if seq_len > 1 and len(set(perm.tolist())) > 1:
        assert not np.allclose(out, bad_out, atol=1e-3)


def test_paged_decode_empty_sequence():
    rng = _rng(7)
    H, D, out_dim = 2, 24, 16
    cache = rng.standard_normal((3, 4, D)).astype(np.float32)
    out, lse = decode_ref_numpy_one_row(
        rng.standard_normal((H, D)).astype(np.float32),
        cache, np.zeros(3, dtype=np.int64), 0, 4, 0.2, out_dim,
    )
    assert not out.any()
    assert lse == -np.inf
