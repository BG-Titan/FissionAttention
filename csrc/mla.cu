// SPDX-License-Identifier: Apache-2.0
// FuseMoEMLA: MLA absorbed-decode paged attention on CUDA cores (no Tensor
// Cores), for GPUs vLLM's MLA backends cannot use (Pascal sm_61, etc.).
//
// One warp per (query row, q-head).  Online softmax (FA2 style) over the
// request's KV positions; the D=576 dot product is done as a strided
// lane-parallel loop (18 elements/lane for 576), the 512-wide latent
// accumulator lives in 16 registers/lane.  Correctness-first v0: CUDA cores,
// __ldg-only loads, no smem tiling, no split-KV.
//
// ABI contract (see fissionattn/ops.py):
//   extern "C" int fissionattn_abi(void)          -> 1
//   extern "C" int fissionattn_has_gpu(void)      -> 1 if a CUDA device exists
//   extern "C" int fissionattn_decode(...)        -> cudaError_t (0 = ok)
//
// NO torch headers here on purpose: this file is compiled by plain `nvcc
// -shared` and loaded with ctypes (mirrors fuse-dense/fusionattn).
//
// Element layout assumptions (validated python-side in ops.py):
//   q          : [rows, H, D] contiguous fp16/bf16, D = kv_lora_rank + rope
//   out        : [rows, H, Ov] contiguous (Ov = kv_lora_rank), same dtype
//   lse        : [rows, H] fp32 or NULL
//   kv cache   : last dim contiguous; block / token strides passed in
//                (the [num_blocks, 1, page, D] MLA slab viewed/squeezed)
//   block_table: int32/int64 [rows, stride] physical block ids
//   seq_lens   : int32/int64 [rows] full KV extent of each row (causality is
//                already folded in: vLLM's MLA decode gives one row per
//                request, and multi-token decode rows are flattened
//                python-side with per-row clamped lengths)

#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define FISSIONATTN_ABI_VERSION 1

// Register budget: q supports D <= FMLA_MAX_D (576 today), the latent
// accumulator supports Ov <= FMLA_MAX_OV (512 today).
#define FMLA_MAX_D 768
#define FMLA_MAX_OV 512
#define FMLA_MAX_DREG (FMLA_MAX_D / 32)     // 24
#define FMLA_MAX_OVREG (FMLA_MAX_OV / 32)   // 16

namespace {

template <typename T>
__device__ __forceinline__ float fmla_ld(const T *p);
template <>
__device__ __forceinline__ float fmla_ld<__half>(const __half *p) {
    return __half2float(*p);
}
template <>
__device__ __forceinline__ float fmla_ld<__nv_bfloat16>(const __nv_bfloat16 *p) {
    return __bfloat162float(*p);
}
template <typename T>
__device__ __forceinline__ void fmla_st(T *p, float v);
template <>
__device__ __forceinline__ void fmla_st<__half>(__half *p, float v) {
    *p = __float2half_rn(v);
}
template <>
__device__ __forceinline__ void fmla_st<__nv_bfloat16>(__nv_bfloat16 *p, float v) {
    *p = __float2bfloat16_rn(v);
}

template <typename T, typename IDX>
__global__ void fmla_decode_kernel(const T *__restrict__ q, T *__restrict__ out,
                                   float *__restrict__ lse, const T *__restrict__ cache,
                                   const long long sb, const long long st,
                                   const IDX *__restrict__ block_table,
                                   const long long bt_stride,
                                   const IDX *__restrict__ seq_lens, const int rows,
                                   const int H, const int D, const int Ov,
                                   const int page, const float scale) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    // The grid is a 1-D array of warps covering rows * H (row-major); the
    // launch may round up, so the row count has to come in as a parameter.
    const int total = rows * H;
    if (warp >= total) return;
    const int row = warp / H;
    const int head = warp - row * H;

    const IDX slen = seq_lens[row];
    const T *qrow = q + ((long long)row * H + head) * (long long)D;

    // ---- load the query row into registers (lane-strided) ----
    const int nreg = (D + 31) >> 5;
    float qreg[FMLA_MAX_DREG];
#pragma unroll 1
    for (int i = 0; i < nreg; ++i) {
        const int d = lane + (i << 5);
        qreg[i] = (d < D) ? fmla_ld<T>(qrow + d) : 0.0f;
    }

    const int nacc = (Ov + 31) >> 5;
    float acc[FMLA_MAX_OVREG];
#pragma unroll
    for (int i = 0; i < FMLA_MAX_OVREG; ++i) acc[i] = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    const IDX *bt = block_table + (long long)row * bt_stride;
    const int nblocks = ((int)slen + page - 1) / page;
    for (int b = 0; b < nblocks; ++b) {
        const long long base = (long long)bt[b] * sb;
        IDX left = slen - (IDX)b * (IDX)page;
        const int cnt = (left < (IDX)page) ? (int)left : page;
        const T *kp = cache + base;
        for (int j = 0; j < cnt; ++j, kp += st) {
            // q . k  (k = the full D-wide latent||rope row)
            float dot = 0.0f;
#pragma unroll 1
            for (int i = 0; i < nreg; ++i) {
                const int d = lane + (i << 5);
                if (d < D) dot += qreg[i] * fmla_ld<T>(kp + d);
            }
#pragma unroll
            for (int off = 16; off; off >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, off);
            dot *= scale;

            const float m_new = (dot > m) ? dot : m;
            const float corr = __expf(m - m_new);
            const float p = __expf(dot - m_new);
            l = l * corr + p;
            m = m_new;

            // v = first Ov dims of the same latent row
            const T *vp = kp;
#pragma unroll 1
            for (int i = 0; i < nacc; ++i) {
                const int d = lane + (i << 5);
                if (d < Ov) acc[i] = acc[i] * corr + fmla_ld<T>(vp + d) * p;
            }
        }
    }

    T *orow = out + ((long long)row * H + head) * (long long)Ov;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll 1
    for (int i = 0; i < nacc; ++i) {
        const int d = lane + (i << 5);
        if (d < Ov) fmla_st<T>(orow + d, acc[i] * inv);
    }
    if (lse != nullptr && lane == 0) {
        lse[(long long)row * H + head] = (l > 0.0f) ? (m + __logf(l)) : -INFINITY;
    }
}

template <typename T, typename IDX>
int fmla_launch(const void *q, void *out, float *lse, const void *cache, long long sb,
                long long st, const void *block_table, long long bt_stride,
                const void *seq_lens, int rows, int H, int D, int Ov, int page,
                float scale, void *stream) {
    const long long warps = (long long)rows * H;
    if (warps <= 0 || warps > (1ll << 30)) return -1001;
    if (D > FMLA_MAX_D || Ov > FMLA_MAX_OV) return -1002;
    const int threads = 256;
    const long long blocks = (warps * 32 + threads - 1) / threads;
    if (blocks > (1ll << 30)) return -1003;
    fmla_decode_kernel<T, IDX><<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const T *)q, (T *)out, lse, (const T *)cache, sb, st, (const IDX *)block_table,
        bt_stride, (const IDX *)seq_lens, rows, H, D, Ov, page, scale);
    return (int)cudaGetLastError();
}

}  // namespace

extern "C" int fissionattn_abi(void) { return FISSIONATTN_ABI_VERSION; }

extern "C" int fissionattn_has_gpu(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n > 0 ? 1 : 0;
}

// dtype_code: 0 = fp16, 1 = bf16.  idx_code: 0 = int32, 1 = int64.
// Returns 0 on success, a cudaError_t (>0) or a -1000s argument error.
extern "C" int fissionattn_decode(const void *q, void *out, float *lse, const void *cache,
                                 long long cache_block_stride,
                                 long long cache_token_stride, const void *block_table,
                                 long long block_table_stride, const void *seq_lens,
                                 int rows, int num_q_heads, int head_size, int out_dim,
                                 int page_size, float scale, int dtype_code,
                                 int idx_code, void *stream) {
    if (q == nullptr || out == nullptr || cache == nullptr || block_table == nullptr ||
        seq_lens == nullptr)
        return -1010;
    if (rows <= 0 || num_q_heads <= 0 || head_size <= 0 || out_dim < 0 ||
        out_dim > head_size || page_size <= 0)
        return -1011;
    if (dtype_code != 0 && dtype_code != 1) return -1012;
    if (idx_code != 0 && idx_code != 1) return -1013;

    if (dtype_code == 0) {
        if (idx_code == 0)
            return fmla_launch<__half, int>(q, out, lse, cache, cache_block_stride,
                                            cache_token_stride, block_table,
                                            block_table_stride, seq_lens, rows,
                                            num_q_heads, head_size, out_dim, page_size,
                                            scale, stream);
        return fmla_launch<__half, long long>(
            q, out, lse, cache, cache_block_stride, cache_token_stride, block_table,
            block_table_stride, seq_lens, rows, num_q_heads, head_size, out_dim,
            page_size, scale, stream);
    }
    if (idx_code == 0)
        return fmla_launch<__nv_bfloat16, int>(
            q, out, lse, cache, cache_block_stride, cache_token_stride, block_table,
            block_table_stride, seq_lens, rows, num_q_heads, head_size, out_dim,
            page_size, scale, stream);
    return fmla_launch<__nv_bfloat16, long long>(
        q, out, lse, cache, cache_block_stride, cache_token_stride, block_table,
        block_table_stride, seq_lens, rows, num_q_heads, head_size, out_dim, page_size,
        scale, stream);
}
