# SPDX-License-Identifier: Apache-2.0
"""Torch-CPU tests for the torch-matmul MLA prefill path.

``fissionattn.prefill_forward.mla_prefill_torch`` is the prefill the
``FISSION_ATTN_PREFILL`` backend runs on GPUs without tensor cores.  These tests
check it against an explicit dense (non-chunked, float64) causal attention
reference, including the cached-context prefix, the causal boundary between
context and new tokens, and invariance to the online-softmax chunk size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fissionattn.prefill_forward import mla_prefill_torch  # noqa: E402

H = 4
NOPE = 16
ROPE = 8
VDIM = 16
LORA = 32
PAGE = 4


class Proj:
    """Stand-in for vLLM's ``kv_b_proj`` ColumnParallelLinear."""

    def __init__(self, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.w = (
            torch.randn(H * (NOPE + VDIM), LORA, generator=g, dtype=torch.float32)
            * (LORA**-0.5)
        )

    def __call__(self, x: torch.Tensor):
        return x.to(self.w.dtype) @ self.w.T, None


def _build_case(seed, context_lens, query_lens):
    g = torch.Generator().manual_seed(seed)
    D = LORA + ROPE
    reqs = []
    latent_rows = []
    for clen, qlen in zip(context_lens, query_lens):
        total = clen + qlen
        rows = torch.randn(total, D, generator=g, dtype=torch.float32)
        latent_rows.append(rows)
        nblk = max(1, -(-total // PAGE))
        reqs.append(nblk)
    nb = sum(reqs) + 2
    cache = torch.zeros(nb, PAGE, D, dtype=torch.float32)
    tables = []
    blk = 0
    for i, (clen, qlen) in enumerate(zip(context_lens, query_lens)):
        total = clen + qlen
        nblk = max(1, -(-total // PAGE))
        tbl = torch.arange(blk, blk + nblk, dtype=torch.int64)
        blk += nblk
        # write the FULL row set (context + new); prefill must only read the
        # first ``clen`` of them for the cached part.
        pos = torch.arange(total)
        cache[tbl[pos // PAGE], pos % PAGE] = latent_rows[i]
        tables.append(
            torch.cat([tbl, torch.zeros(max(0, 8 - nblk), dtype=torch.int64)])
        )
    n = sum(query_lens)
    q = torch.randn(n, H, NOPE + ROPE, generator=g, dtype=torch.float32)
    latent_new = torch.cat([rows[clen:] for rows, (clen, _) in zip(latent_rows,
                                                                    zip(context_lens, query_lens))])
    return q, latent_new, cache, tables, latent_rows


def _reference(q, latent_rows, proj, context_lens, query_lens, scale):
    """Explicit dense causal attention in float64: returns out, lse."""
    outs, lses = [], []
    w = proj.w.double()
    tok = 0
    for rows, (clen, qlen) in zip(latent_rows, zip(context_lens, query_lens)):
        total = clen + qlen
        latent = rows.double()
        kv = latent[:, :LORA] @ w.T  # [T, H*(nope+v)]
        kv = kv.view(total, H, NOPE + VDIM)
        k_nope, v = kv.split([NOPE, VDIM], dim=-1)
        rope = latent[:, LORA:]  # [T, ROPE]
        k = torch.cat([k_nope, rope.unsqueeze(1).expand(-1, H, -1)], dim=-1)  # [T,H,nope+rope]
        for i in range(qlen):
            pos = clen + i
            q_r = q[tok + i].double()  # [H, nope+rope]
            logits = torch.einsum("thd,hd->th", k[: pos + 1], q_r) * scale  # [<=T, H]
            wgt = torch.softmax(logits, dim=0)
            outs.append((wgt.unsqueeze(-1) * v[: pos + 1]).sum(0))
            lses.append(torch.logsumexp(logits, dim=0))
        tok += qlen
    return torch.stack(outs).view(-1, H, VDIM), torch.stack(lses).view(-1, H)


@pytest.mark.parametrize(
    "context_lens,query_lens",
    [
        ([0], [3]),
        ([5], [1]),
        ([0, 5, 13], [3, 1, 6]),
        ([64, 1], [1, 17]),
    ],
)
@pytest.mark.parametrize("chunk_keys", [1, 3, 16, 4096])
def test_prefill_matches_dense_reference(context_lens, query_lens, chunk_keys):
    scale = (NOPE + ROPE) ** -0.5
    q, latent_new, cache, tables, latent_rows = _build_case(
        3 + len(context_lens) + chunk_keys, context_lens, query_lens
    )
    proj = Proj()
    out, lse = mla_prefill_torch(
        q,
        latent_new,
        proj,
        cache,
        tables,
        list(context_lens),
        list(query_lens),
        num_heads=H,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=VDIM,
        kv_lora_rank=LORA,
        page_size=PAGE,
        scale=scale,
        chunk_keys=chunk_keys,
    )
    ref_out, ref_lse = _reference(q, latent_rows, proj, context_lens, query_lens, scale)
    assert out.shape == (sum(query_lens), H, VDIM)
    assert lse.shape == (sum(query_lens), H)
    err = (out.double() - ref_out).abs().max().item()
    lerr = (lse.double() - ref_lse).abs().max().item()
    assert err < 2e-4, f"out err {err} (chunk {chunk_keys}, ctx {context_lens})"
    assert lerr < 2e-3, f"lse err {lerr} (chunk {chunk_keys}, ctx {context_lens})"


def test_prefill_lse_is_the_normaliser():
    scale = (NOPE + ROPE) ** -0.5
    context_lens, query_lens = [0, 7], [4, 2]
    q, latent_new, cache, tables, latent_rows = _build_case(5, context_lens, query_lens)
    proj = Proj()
    out, lse = mla_prefill_torch(
        q, latent_new, proj, cache, tables, context_lens, query_lens,
        num_heads=H, qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VDIM,
        kv_lora_rank=LORA, page_size=PAGE, scale=scale, chunk_keys=3,
    )
    ref_out, ref_lse = _reference(q, latent_rows, proj, context_lens, query_lens, scale)
    # lse must be finite everywhere (every query attends to itself)
    assert torch.all(torch.isfinite(lse))
    assert torch.allclose(lse.double(), ref_lse, atol=2e-3)
    assert torch.allclose(out.double(), ref_out, atol=2e-4)


def test_prefill_empty_query_rows_are_skipped():
    """A request with 0 new tokens consumes no query rows and does not shift
    the others (chunked prefill with a fully-cached request)."""
    scale = (NOPE + ROPE) ** -0.5
    q, latent_new, cache, tables, latent_rows = _build_case(9, [6, 0, 4], [2, 0, 1])
    proj = Proj()
    out, lse = mla_prefill_torch(
        q, latent_new, proj, cache, tables, [6, 0, 4], [2, 0, 1],
        num_heads=H, qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VDIM,
        kv_lora_rank=LORA, page_size=PAGE, scale=scale, chunk_keys=2,
    )
    assert out.size(0) == 3
    ref_out, _ = _reference(q, latent_rows, proj, [6, 0, 4], [2, 0, 1], scale)
    assert torch.allclose(out.double(), ref_out, atol=2e-4)
