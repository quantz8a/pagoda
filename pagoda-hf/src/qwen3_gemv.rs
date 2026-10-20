// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Decode GEMV for MLX affine 8-bit weights: `y = (bias + scale * code) · x`.
//!
//! Scheduling (step 1): many GEMVs share one command buffer / one wait.
//!
//! Kernel: MLX `affine_qmv_fast` shape, widened so one x-load feeds 8 rows
//! and weights move as `uint2` (8 codes). 2 simdgroups × 8 rows, K blocks
//! of 256. Tail threadgroups (n_out not a multiple of 16) take the masked path.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    NSUInteger,
};

const TG_SIZE: u32 = 64; // 2 simdgroups × 32
const ROWS_PER_TG: u32 = 8; // 2 × 4

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant int VALUES_PER_THREAD = 8;
constant int NUM_SIMDGROUPS = 2;
constant int RESULTS_PER_SG = 4;
constant int BLOCK_SIZE = 256;
constant int GROUP_SIZE = 64;

struct GemvParams {
    int n_in;
    int n_out;
};

inline float bf16(ushort u) {
    return as_type<float>(uint(u) << 16);
}

inline float dot8(uint2 ww, float4 x0, float4 x1) {
    float c0 = float(ww.x & 255u);
    float c1 = float((ww.x >> 8) & 255u);
    float c2 = float((ww.x >> 16) & 255u);
    float c3 = float((ww.x >> 24) & 255u);
    float c4 = float(ww.y & 255u);
    float c5 = float((ww.y >> 8) & 255u);
    float c6 = float((ww.y >> 16) & 255u);
    float c7 = float((ww.y >> 24) & 255u);
    return fma(c0, x0[0], fma(c1, x0[1], fma(c2, x0[2], fma(c3, x0[3],
        fma(c4, x1[0], fma(c5, x1[1], fma(c6, x1[2], c7 * x1[3])))))));
}

kernel void affine_gemv(
    device const uint* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant GemvParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int n_in = params.n_in;
    const int n_out = params.n_out;
    const int out_row = int(tid.y) * (NUM_SIMDGROUPS * RESULTS_PER_SG)
        + int(simd_gid) * RESULTS_PER_SG;
    if (out_row >= n_out) {
        return;
    }
    const int last_row = min(RESULTS_PER_SG - 1, n_out - 1 - out_row);
    const int groups = n_in / GROUP_SIZE;

    device const uchar* ws = (device const uchar*)packed;
    ws += out_row * n_in + int(simd_lid) * VALUES_PER_THREAD;
    device const ushort* sc = scales + out_row * groups + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);
    device const ushort* bc = biases + out_row * groups + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);
    device const float* xv = x + int(simd_lid) * VALUES_PER_THREAD;

    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;

    if (last_row == 3) {
        for (int k = 0; k < n_in; k += BLOCK_SIZE) {
            float4 x0 = *(device const float4*)xv;
            float4 x1 = *(device const float4*)(xv + 4);
            float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
            uint2 w0 = *(device const uint2*)(ws);
            uint2 w1 = *(device const uint2*)(ws + n_in);
            uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
            uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
            float s0 = bf16(sc[0]), b0 = bf16(bc[0]);
            float s1 = bf16(sc[groups]), b1 = bf16(bc[groups]);
            float s2 = bf16(sc[2 * groups]), b2 = bf16(bc[2 * groups]);
            float s3 = bf16(sc[3 * groups]), b3 = bf16(bc[3 * groups]);
            r0 = fma(s0, dot8(w0, x0, x1), fma(xsum, b0, r0));
            r1 = fma(s1, dot8(w1, x0, x1), fma(xsum, b1, r1));
            r2 = fma(s2, dot8(w2, x0, x1), fma(xsum, b2, r2));
            r3 = fma(s3, dot8(w3, x0, x1), fma(xsum, b3, r3));
            ws += BLOCK_SIZE;
            sc += BLOCK_SIZE / GROUP_SIZE;
            bc += BLOCK_SIZE / GROUP_SIZE;
            xv += BLOCK_SIZE;
        }
    } else {
        for (int k = 0; k < n_in; k += BLOCK_SIZE) {
            float4 x0 = *(device const float4*)xv;
            float4 x1 = *(device const float4*)(xv + 4);
            float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
            int s1 = min(1, last_row), s2 = min(2, last_row), s3 = min(3, last_row);
            uint2 w0 = *(device const uint2*)(ws);
            uint2 w1 = *(device const uint2*)(ws + s1 * n_in);
            uint2 w2 = *(device const uint2*)(ws + s2 * n_in);
            uint2 w3 = *(device const uint2*)(ws + s3 * n_in);
            r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
            r1 = fma(bf16(sc[s1 * groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[s1 * groups]), r1));
            r2 = fma(bf16(sc[s2 * groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[s2 * groups]), r2));
            r3 = fma(bf16(sc[s3 * groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[s3 * groups]), r3));
            ws += BLOCK_SIZE;
            sc += BLOCK_SIZE / GROUP_SIZE;
            bc += BLOCK_SIZE / GROUP_SIZE;
            xv += BLOCK_SIZE;
        }
    }

    r0 = simd_sum(r0);
    r1 = simd_sum(r1);
    r2 = simd_sum(r2);
    r3 = simd_sum(r3);
    if (simd_lid == 0u) {
        y[out_row] = r0;
        if (last_row >= 1) { y[out_row + 1] = r1; }
        if (last_row >= 2) { y[out_row + 2] = r2; }
        if (last_row >= 3) { y[out_row + 3] = r3; }
    }
}

// One launch, up to four same-K GEMVs. Matrices run back-to-back so register
// pressure stays the same as a single GEMV; the win is fewer serial launches.
struct ManyParams {
    int n_in;
    int n_out0;
    int n_out1;
    int n_out2;
    int n_out3;
    int count;
};

inline void gemv_one(
    device const uchar* ws0,
    device const ushort* sc0,
    device const ushort* bc0,
    device const float* x,
    device float* y,
    device const float* res,
    int mode,
    int n_in,
    int n_out,
    int out_row,
    int simd_lid
) {
    if (out_row >= n_out) { return; }
    const int last_row = min(RESULTS_PER_SG - 1, n_out - 1 - out_row);
    const int groups = n_in / GROUP_SIZE;
    device const uchar* ws = ws0 + out_row * n_in + simd_lid * VALUES_PER_THREAD;
    device const ushort* sc = sc0 + out_row * groups + simd_lid / (GROUP_SIZE / VALUES_PER_THREAD);
    device const ushort* bc = bc0 + out_row * groups + simd_lid / (GROUP_SIZE / VALUES_PER_THREAD);
    device const float* xv = x + simd_lid * VALUES_PER_THREAD;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    if (last_row == 3) {
        for (int k = 0; k < n_in; k += BLOCK_SIZE) {
            float4 x0 = *(device const float4*)xv;
            float4 x1 = *(device const float4*)(xv + 4);
            float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
            uint2 w0 = *(device const uint2*)(ws);
            uint2 w1 = *(device const uint2*)(ws + n_in);
            uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
            uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
            r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
            r1 = fma(bf16(sc[groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[groups]), r1));
            r2 = fma(bf16(sc[2 * groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[2 * groups]), r2));
            r3 = fma(bf16(sc[3 * groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[3 * groups]), r3));
            ws += BLOCK_SIZE; sc += BLOCK_SIZE / GROUP_SIZE; bc += BLOCK_SIZE / GROUP_SIZE; xv += BLOCK_SIZE;
        }
    } else {
        for (int k = 0; k < n_in; k += BLOCK_SIZE) {
            float4 x0 = *(device const float4*)xv;
            float4 x1 = *(device const float4*)(xv + 4);
            float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
            int s1 = min(1, last_row), s2 = min(2, last_row), s3 = min(3, last_row);
            uint2 w0 = *(device const uint2*)(ws);
            uint2 w1 = *(device const uint2*)(ws + s1 * n_in);
            uint2 w2 = *(device const uint2*)(ws + s2 * n_in);
            uint2 w3 = *(device const uint2*)(ws + s3 * n_in);
            r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
            r1 = fma(bf16(sc[s1 * groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[s1 * groups]), r1));
            r2 = fma(bf16(sc[s2 * groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[s2 * groups]), r2));
            r3 = fma(bf16(sc[s3 * groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[s3 * groups]), r3));
            ws += BLOCK_SIZE; sc += BLOCK_SIZE / GROUP_SIZE; bc += BLOCK_SIZE / GROUP_SIZE; xv += BLOCK_SIZE;
        }
    }
    r0 = simd_sum(r0); r1 = simd_sum(r1); r2 = simd_sum(r2); r3 = simd_sum(r3);
    if (simd_lid == 0u) {
        if (mode == 1) {
            r0 += res[out_row];
            if (last_row >= 1) { r1 += res[out_row + 1]; }
            if (last_row >= 2) { r2 += res[out_row + 2]; }
            if (last_row >= 3) { r3 += res[out_row + 3]; }
        }
        y[out_row] = r0;
        if (last_row >= 1) { y[out_row + 1] = r1; }
        if (last_row >= 2) { y[out_row + 2] = r2; }
        if (last_row >= 3) { y[out_row + 3] = r3; }
    }
}

kernel void affine_gemv_n(
    device const float* x [[buffer(0)]],
    device const uint* p0 [[buffer(1)]],
    device const ushort* s0 [[buffer(2)]],
    device const ushort* b0 [[buffer(3)]],
    device float* y0 [[buffer(4)]],
    device const uint* p1 [[buffer(5)]],
    device const ushort* s1 [[buffer(6)]],
    device const ushort* b1 [[buffer(7)]],
    device float* y1 [[buffer(8)]],
    device const uint* p2 [[buffer(9)]],
    device const ushort* s2 [[buffer(10)]],
    device const ushort* b2 [[buffer(11)]],
    device float* y2 [[buffer(12)]],
    device const uint* p3 [[buffer(13)]],
    device const ushort* s3 [[buffer(14)]],
    device const ushort* b3 [[buffer(15)]],
    device float* y3 [[buffer(16)]],
    constant ManyParams& params [[buffer(17)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int out_row = int(tid.y) * (NUM_SIMDGROUPS * RESULTS_PER_SG)
        + int(simd_gid) * RESULTS_PER_SG;
    const int n_in = params.n_in;
    gemv_one((device const uchar*)p0, s0, b0, x, y0, y0, 0, n_in, params.n_out0, out_row, int(simd_lid));
    if (params.count > 1) {
        gemv_one((device const uchar*)p1, s1, b1, x, y1, y1, 0, n_in, params.n_out1, out_row, int(simd_lid));
    }
    if (params.count > 2) {
        gemv_one((device const uchar*)p2, s2, b2, x, y2, y2, 0, n_in, params.n_out2, out_row, int(simd_lid));
    }
    if (params.count > 3) {
        gemv_one((device const uchar*)p3, s3, b3, x, y3, y3, 0, n_in, params.n_out3, out_row, int(simd_lid));
    }
}

// gate GEMV, up GEMV, then mid = silu(gate) * up. One launch.
kernel void affine_gemv2_silu(
    device const float* x [[buffer(0)]],
    device const uint* gp [[buffer(1)]],
    device const ushort* gs [[buffer(2)]],
    device const ushort* gb [[buffer(3)]],
    device float* gy [[buffer(4)]],
    device const uint* up [[buffer(5)]],
    device const ushort* us [[buffer(6)]],
    device const ushort* ub [[buffer(7)]],
    device float* uy [[buffer(8)]],
    device float* mid [[buffer(9)]],
    constant GemvParams& params [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int n_in = params.n_in;
    const int n_out = params.n_out;
    const int out_row = int(tid.y) * (NUM_SIMDGROUPS * RESULTS_PER_SG)
        + int(simd_gid) * RESULTS_PER_SG;
    gemv_one((device const uchar*)gp, gs, gb, x, gy, gy, 0, n_in, n_out, out_row, int(simd_lid));
    gemv_one((device const uchar*)up, us, ub, x, uy, uy, 0, n_in, n_out, out_row, int(simd_lid));
    if (out_row < n_out && simd_lid == 0u) {
        int last_row = min(RESULTS_PER_SG - 1, n_out - 1 - out_row);
        for (int i = 0; i <= last_row; i++) {
            float g = gy[out_row + i];
            float sig = 1.0f / (1.0f + exp(-g));
            mid[out_row + i] = g * sig * uy[out_row + i];
        }
    }
}

// y = residual + GEMV(x), or y += GEMV(x) when residual is y.
kernel void affine_gemv_res(
    device const uint* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* residual [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant GemvParams& params [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int out_row = int(tid.y) * (NUM_SIMDGROUPS * RESULTS_PER_SG)
        + int(simd_gid) * RESULTS_PER_SG;
    gemv_one(
        (device const uchar*)packed, scales, biases, x, y, residual, 1,
        params.n_in, params.n_out, out_row, int(simd_lid));
}

// Split K across 4 threadgroups so an 89 MB MLP tile fills the GPU.
// partial layout: [split][row]
kernel void affine_gemv_s4(
    device const uint* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* partial [[buffer(4)]],
    constant GemvParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int n_in = params.n_in;
    const int n_out = params.n_out;
    const int split = int(tid.x);
    const int out_row = int(tid.y) * (NUM_SIMDGROUPS * RESULTS_PER_SG)
        + int(simd_gid) * RESULTS_PER_SG;
    if (split >= 4 || out_row >= n_out) { return; }
    const int k_lo = split * (n_in / 4);
    const int k_hi = k_lo + n_in / 4;
    const int last_row = min(RESULTS_PER_SG - 1, n_out - 1 - out_row);
    const int groups = n_in / GROUP_SIZE;
    device const uchar* ws = (device const uchar*)packed;
    ws += out_row * n_in + k_lo + int(simd_lid) * VALUES_PER_THREAD;
    device const ushort* sc = scales + out_row * groups + k_lo / GROUP_SIZE
        + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);
    device const ushort* bc = biases + out_row * groups + k_lo / GROUP_SIZE
        + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);
    device const float* xv = x + k_lo + int(simd_lid) * VALUES_PER_THREAD;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    for (int k = k_lo; k < k_hi; k += BLOCK_SIZE) {
        float4 x0 = *(device const float4*)xv;
        float4 x1 = *(device const float4*)(xv + 4);
        float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
        if (last_row == 3) {
            uint2 w0 = *(device const uint2*)(ws);
            uint2 w1 = *(device const uint2*)(ws + n_in);
            uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
            uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
            r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
            r1 = fma(bf16(sc[groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[groups]), r1));
            r2 = fma(bf16(sc[2 * groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[2 * groups]), r2));
            r3 = fma(bf16(sc[3 * groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[3 * groups]), r3));
        } else {
            int s1 = min(1, last_row), s2 = min(2, last_row), s3 = min(3, last_row);
            uint2 w0 = *(device const uint2*)(ws);
            uint2 w1 = *(device const uint2*)(ws + s1 * n_in);
            uint2 w2 = *(device const uint2*)(ws + s2 * n_in);
            uint2 w3 = *(device const uint2*)(ws + s3 * n_in);
            r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
            r1 = fma(bf16(sc[s1 * groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[s1 * groups]), r1));
            r2 = fma(bf16(sc[s2 * groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[s2 * groups]), r2));
            r3 = fma(bf16(sc[s3 * groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[s3 * groups]), r3));
        }
        ws += BLOCK_SIZE; sc += BLOCK_SIZE / GROUP_SIZE; bc += BLOCK_SIZE / GROUP_SIZE; xv += BLOCK_SIZE;
    }
    r0 = simd_sum(r0); r1 = simd_sum(r1); r2 = simd_sum(r2); r3 = simd_sum(r3);
    if (simd_lid == 0u) {
        device float* dst = partial + split * n_out + out_row;
        dst[0] = r0;
        if (last_row >= 1) { dst[1] = r1; }
        if (last_row >= 2) { dst[2] = r2; }
        if (last_row >= 3) { dst[3] = r3; }
    }
}

kernel void reduce_s4(
    device const float* partial [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant int& n_out [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (int(i) >= n_out) { return; }
    y[i] = partial[i] + partial[n_out + i] + partial[2 * n_out + i] + partial[3 * n_out + i];
}

// One embedding row: y[i] = bias + scale * code. 256 threads, hidden=5120.
kernel void dequant_row(
    device const uint* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    constant uint& token [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& n_in [[buffer(5)]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    int groups = n_in / 64;
    device const uchar* row = ((device const uchar*)packed) + token * n_in;
    device const ushort* sc = scales + token * groups;
    device const ushort* bc = biases + token * groups;
    for (uint i = lid; i < uint(n_in); i += lsize) {
        uint g = i >> 6u;
        y[i] = bf16(bc[g]) + bf16(sc[g]) * float(row[i]);
    }
}

// Small-batch GEMM. Each batch row uses the same per-group fma as affine_gemv,
// so one weight read verifies up to 4 tokens. xs, ys are [batch, n_in/n_out].
struct GemmParams {
    int n_in;
    int n_out;
    int batch;
};

kernel void affine_gemm4(
    device const uint* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* xs [[buffer(3)]],
    device float* ys [[buffer(4)]],
    constant GemmParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int n_in = params.n_in;
    const int n_out = params.n_out;
    const int batch = params.batch;
    const int out_row = int(tid.y) * (NUM_SIMDGROUPS * RESULTS_PER_SG)
        + int(simd_gid) * RESULTS_PER_SG;
    if (out_row >= n_out) {
        return;
    }
    const int last_row = min(RESULTS_PER_SG - 1, n_out - 1 - out_row);
    const int groups = n_in / GROUP_SIZE;
    device const uchar* ws = (device const uchar*)packed;
    ws += out_row * n_in + int(simd_lid) * VALUES_PER_THREAD;
    device const ushort* sc = scales + out_row * groups + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);
    device const ushort* bc = biases + out_row * groups + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);

    float r0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float r1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float r2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float r3[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < n_in; k += BLOCK_SIZE) {
        int s1 = min(1, last_row), s2 = min(2, last_row), s3 = min(3, last_row);
        uint2 w0 = *(device const uint2*)(ws);
        uint2 w1 = *(device const uint2*)(ws + s1 * n_in);
        uint2 w2 = *(device const uint2*)(ws + s2 * n_in);
        uint2 w3 = *(device const uint2*)(ws + s3 * n_in);
        float s0 = bf16(sc[0]), b0 = bf16(bc[0]);
        float ss1 = bf16(sc[s1 * groups]), bb1 = bf16(bc[s1 * groups]);
        float ss2 = bf16(sc[s2 * groups]), bb2 = bf16(bc[s2 * groups]);
        float ss3 = bf16(sc[s3 * groups]), bb3 = bf16(bc[s3 * groups]);
        for (int b = 0; b < batch; ++b) {
            device const float* xv = xs + b * n_in + k + int(simd_lid) * VALUES_PER_THREAD;
            float4 x0 = *(device const float4*)xv;
            float4 x1 = *(device const float4*)(xv + 4);
            float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
            r0[b] = fma(s0, dot8(w0, x0, x1), fma(xsum, b0, r0[b]));
            r1[b] = fma(ss1, dot8(w1, x0, x1), fma(xsum, bb1, r1[b]));
            r2[b] = fma(ss2, dot8(w2, x0, x1), fma(xsum, bb2, r2[b]));
            r3[b] = fma(ss3, dot8(w3, x0, x1), fma(xsum, bb3, r3[b]));
        }
        ws += BLOCK_SIZE;
        sc += BLOCK_SIZE / GROUP_SIZE;
        bc += BLOCK_SIZE / GROUP_SIZE;
    }

    for (int b = 0; b < batch; ++b) {
        r0[b] = simd_sum(r0[b]);
        r1[b] = simd_sum(r1[b]);
        r2[b] = simd_sum(r2[b]);
        r3[b] = simd_sum(r3[b]);
        if (simd_lid == 0u) {
            device float* y = ys + b * n_out;
            y[out_row] = r0[b];
            if (last_row >= 1) { y[out_row + 1] = r1[b]; }
            if (last_row >= 2) { y[out_row + 2] = r2[b]; }
            if (last_row >= 3) { y[out_row + 3] = r3[b]; }
        }
    }
}

// batch == 4 and n_out % 8 == 0. Four independent dots, weights loaded once.
kernel void affine_gemm4u(
    device const uint* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* xs [[buffer(3)]],
    device float* ys [[buffer(4)]],
    constant GemmParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int n_in = params.n_in;
    const int n_out = params.n_out;
    const int out_row = int(tid.y) * (NUM_SIMDGROUPS * RESULTS_PER_SG)
        + int(simd_gid) * RESULTS_PER_SG;
    if (out_row >= n_out) {
        return;
    }
    const int groups = n_in / GROUP_SIZE;
    device const uchar* ws = (device const uchar*)packed;
    ws += out_row * n_in + int(simd_lid) * VALUES_PER_THREAD;
    device const ushort* sc = scales + out_row * groups + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);
    device const ushort* bc = biases + out_row * groups + int(simd_lid) / (GROUP_SIZE / VALUES_PER_THREAD);
    float a0=0, a1=0, a2=0, a3=0;
    float b0=0, b1=0, b2=0, b3=0;
    float c0=0, c1=0, c2=0, c3=0;
    float d0=0, d1=0, d2=0, d3=0;
    for (int k = 0; k < n_in; k += BLOCK_SIZE) {
        uint2 w0 = *(device const uint2*)(ws);
        uint2 w1 = *(device const uint2*)(ws + n_in);
        uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
        uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
        float s0 = bf16(sc[0]), z0 = bf16(bc[0]);
        float s1 = bf16(sc[groups]), z1 = bf16(bc[groups]);
        float s2 = bf16(sc[2 * groups]), z2 = bf16(bc[2 * groups]);
        float s3 = bf16(sc[3 * groups]), z3 = bf16(bc[3 * groups]);
        for (int b = 0; b < 4; ++b) {
            device const float* xv = xs + b * n_in + k + int(simd_lid) * VALUES_PER_THREAD;
            float4 x0 = *(device const float4*)xv;
            float4 x1 = *(device const float4*)(xv + 4);
            float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
            float d = dot8(w0, x0, x1);
            float e = dot8(w1, x0, x1);
            float f = dot8(w2, x0, x1);
            float g = dot8(w3, x0, x1);
            if (b == 0) {
                a0 = fma(s0, d, fma(xsum, z0, a0));
                b0 = fma(s1, e, fma(xsum, z1, b0));
                c0 = fma(s2, f, fma(xsum, z2, c0));
                d0 = fma(s3, g, fma(xsum, z3, d0));
            } else if (b == 1) {
                a1 = fma(s0, d, fma(xsum, z0, a1));
                b1 = fma(s1, e, fma(xsum, z1, b1));
                c1 = fma(s2, f, fma(xsum, z2, c1));
                d1 = fma(s3, g, fma(xsum, z3, d1));
            } else if (b == 2) {
                a2 = fma(s0, d, fma(xsum, z0, a2));
                b2 = fma(s1, e, fma(xsum, z1, b2));
                c2 = fma(s2, f, fma(xsum, z2, c2));
                d2 = fma(s3, g, fma(xsum, z3, d2));
            } else {
                a3 = fma(s0, d, fma(xsum, z0, a3));
                b3 = fma(s1, e, fma(xsum, z1, b3));
                c3 = fma(s2, f, fma(xsum, z2, c3));
                d3 = fma(s3, g, fma(xsum, z3, d3));
            }
        }
        ws += BLOCK_SIZE;
        sc += BLOCK_SIZE / GROUP_SIZE;
        bc += BLOCK_SIZE / GROUP_SIZE;
    }
    a0 = simd_sum(a0); b0 = simd_sum(b0); c0 = simd_sum(c0); d0 = simd_sum(d0);
    a1 = simd_sum(a1); b1 = simd_sum(b1); c1 = simd_sum(c1); d1 = simd_sum(d1);
    a2 = simd_sum(a2); b2 = simd_sum(b2); c2 = simd_sum(c2); d2 = simd_sum(d2);
    a3 = simd_sum(a3); b3 = simd_sum(b3); c3 = simd_sum(c3); d3 = simd_sum(d3);
    if (simd_lid == 0u) {
        ys[out_row] = a0; ys[out_row + 1] = b0; ys[out_row + 2] = c0; ys[out_row + 3] = d0;
        ys[n_out + out_row] = a1; ys[n_out + out_row + 1] = b1; ys[n_out + out_row + 2] = c1; ys[n_out + out_row + 3] = d1;
        ys[2 * n_out + out_row] = a2; ys[2 * n_out + out_row + 1] = b2; ys[2 * n_out + out_row + 2] = c2; ys[2 * n_out + out_row + 3] = d2;
        ys[3 * n_out + out_row] = a3; ys[3 * n_out + out_row + 1] = b3; ys[3 * n_out + out_row + 2] = c3; ys[3 * n_out + out_row + 3] = d3;
    }
}
"#;

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    bits.saturating_add(round).wrapping_shr(16) as u16
}

struct Pipe {
    device: Device,
    queue: metal::CommandQueue,
    state: ComputePipelineState,
    state_n: ComputePipelineState,
    state_silu: ComputePipelineState,
    state_res: ComputePipelineState,
    state_gemm: ComputePipelineState,
    state_gemmu: ComputePipelineState,
    #[allow(dead_code)]
    state_s4: ComputePipelineState,
    #[allow(dead_code)]
    state_red: ComputePipelineState,
    state_row: ComputePipelineState,
}

fn pipe() -> Result<&'static Pipe> {
    static PIPE: OnceLock<Pipe> = OnceLock::new();
    if let Some(p) = PIPE.get() {
        return Ok(p);
    }
    let device = Device::system_default().context("no Metal device")?;
    let lib = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .map_err(|e| {
            for (i, line) in SRC.lines().enumerate() {
                eprintln!("{:>3} {line}", i + 1);
            }
            anyhow::anyhow!("metal compile: {e}")
        })?;
    let make = |name: &str| -> Result<ComputePipelineState> {
        let func = lib
            .get_function(name, None)
            .map_err(|e| anyhow::anyhow!("metal function {name}: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("metal pipeline {name}: {e}"))
    };
    let state = make("affine_gemv")?;
    let state_n = make("affine_gemv_n")?;
    let state_silu = make("affine_gemv2_silu")?;
    let state_res = make("affine_gemv_res")?;
    let state_gemm = make("affine_gemm4")?;
    let state_gemmu = make("affine_gemm4u")?;
    let state_s4 = make("affine_gemv_s4")?;
    let state_red = make("reduce_s4")?;
    let state_row = make("dequant_row")?;
    let queue = device.new_command_queue();
    let _ = PIPE.set(Pipe {
        device,
        queue,
        state,
        state_n,
        state_silu,
        state_res,
        state_gemm,
        state_gemmu,
        state_s4,
        state_red,
        state_row,
    });
    Ok(PIPE.get().unwrap())
}

#[derive(Clone)]
pub struct QuantMatrix {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
    n_out: u32,
    n_in: u32,
}

impl QuantMatrix {
    pub fn n_out(&self) -> usize {
        self.n_out as usize
    }

    pub fn n_in(&self) -> usize {
        self.n_in as usize
    }

    pub fn upload(packed: &[u32], scales: &[f32], biases: &[f32], n_out: usize, n_in: usize) -> Result<Self> {
        let p = pipe()?;
        let groups = n_in / 64;
        anyhow::ensure!(scales.len() == n_out * groups && biases.len() == n_out * groups, "scale shape");
        anyhow::ensure!(packed.len() == n_out * (n_in / 4), "packed shape");
        anyhow::ensure!(n_in % 64 == 0, "n_in multiple of 64");
        anyhow::ensure!(n_in % 256 == 0, "n_in multiple of qmv block 256");
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate
            | MTLResourceOptions::HazardTrackingModeUntracked;
        let scales_h: Vec<u16> = scales.iter().copied().map(f32_to_bf16).collect();
        let biases_h: Vec<u16> = biases.iter().copied().map(f32_to_bf16).collect();
        let packed_s = p.device.new_buffer_with_data(
            packed.as_ptr() as *const _,
            (packed.len() * 4) as NSUInteger,
            shared,
        );
        let scales_s = p.device.new_buffer_with_data(
            scales_h.as_ptr() as *const _,
            (scales_h.len() * 2) as NSUInteger,
            shared,
        );
        let biases_s = p.device.new_buffer_with_data(
            biases_h.as_ptr() as *const _,
            (biases_h.len() * 2) as NSUInteger,
            shared,
        );
        // Private + untracked keeps the weight stream off the CPU coherency path.
        // On M3 Ultra that is the difference between ~600 GB/s and ~760 GB/s.
        let packed_b = p.device.new_buffer(packed_s.length(), private);
        let scales_b = p.device.new_buffer(scales_s.length(), private);
        let biases_b = p.device.new_buffer(biases_s.length(), private);
        let cb = p.queue.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer(&packed_s, 0, &packed_b, 0, packed_s.length());
        blit.copy_from_buffer(&scales_s, 0, &scales_b, 0, scales_s.length());
        blit.copy_from_buffer(&biases_s, 0, &biases_b, 0, biases_s.length());
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(Self {
            packed: packed_b,
            scales: scales_b,
            biases: biases_b,
            n_out: n_out as u32,
            n_in: n_in as u32,
        })
    }

    pub fn gemv(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = gemv_batch(&[(self, x)])?;
        Ok(out.remove(0))
    }

    pub fn gemv_shared(mats: &[&QuantMatrix], x: &[f32]) -> Result<Vec<Vec<f32>>> {
        if mats.is_empty() {
            return Ok(Vec::new());
        }
        for m in mats {
            anyhow::ensure!(x.len() == m.n_in as usize, "gemv x");
        }
        let p = pipe()?;
        let opts = MTLResourceOptions::StorageModeShared;
        let x_b = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as NSUInteger,
            opts,
        );
        let ys: Vec<Buffer> = mats
            .iter()
            .map(|m| p.device.new_buffer((m.n_out as NSUInteger) * 4, opts))
            .collect();
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for (m, y_b) in mats.iter().zip(&ys) {
            encode_gemv(enc, &p.state, m, &x_b, y_b);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(ys
            .iter()
            .zip(mats.iter())
            .map(|(y_b, m)| read_f32(y_b, m.n_out as usize))
            .collect())
    }
}

pub fn gemv_batch(jobs: &[(&QuantMatrix, &[f32])]) -> Result<Vec<Vec<f32>>> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    for (m, x) in jobs {
        anyhow::ensure!(x.len() == m.n_in as usize, "gemv x");
    }
    let p = pipe()?;
    let opts = MTLResourceOptions::StorageModeShared;
    let mut x_bufs: Vec<(*const f32, Buffer)> = Vec::new();
    let mut resolve_x = |x: &[f32]| -> Buffer {
        let key = x.as_ptr();
        if let Some((_, b)) = x_bufs.iter().find(|(ptr, _)| *ptr == key) {
            return b.clone();
        }
        let b = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as NSUInteger,
            opts,
        );
        x_bufs.push((key, b.clone()));
        b
    };
    let ys: Vec<Buffer> = jobs
        .iter()
        .map(|(m, _)| p.device.new_buffer((m.n_out as NSUInteger) * 4, opts))
        .collect();
    let cb = p.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    for ((m, x), y_b) in jobs.iter().zip(&ys) {
        let x_b = resolve_x(x);
        encode_gemv(enc, &p.state, m, &x_b, y_b);
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(jobs
        .iter()
        .zip(&ys)
        .map(|((m, _), y_b)| read_f32(y_b, m.n_out as usize))
        .collect())
}

fn encode_gemv(
    enc: &ComputeCommandEncoderRef,
    state: &ComputePipelineState,
    m: &QuantMatrix,
    x: &Buffer,
    y: &Buffer,
) {
    enc.set_compute_pipeline_state(state);
    enc.set_buffer(0, Some(&m.packed), 0);
    enc.set_buffer(1, Some(&m.scales), 0);
    enc.set_buffer(2, Some(&m.biases), 0);
    enc.set_buffer(3, Some(x), 0);
    enc.set_buffer(4, Some(y), 0);
    let params = [m.n_in as i32, m.n_out as i32];
    enc.set_bytes(5, 8, params.as_ptr() as *const _);
    let tg = MTLSize {
        width: TG_SIZE as NSUInteger,
        height: 1,
        depth: 1,
    };
    // tid.y indexes output row blocks of ROWS_PER_TG; tid.x unused (single vector).
    let row_blocks = ((m.n_out + ROWS_PER_TG - 1) / ROWS_PER_TG) as NSUInteger;
    let grid = MTLSize {
        width: 1,
        height: row_blocks,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Batched GEMM, `batch` in 1..=4. `xs` is `[batch, n_in]`, `ys` is `[batch, n_out]`.
/// Each row uses the same group fma as [`encode_into`].
pub fn encode_gemm4(
    enc: &ComputeCommandEncoderRef,
    m: &QuantMatrix,
    xs: &Buffer,
    ys: &Buffer,
    batch: u32,
) -> Result<()> {
    anyhow::ensure!((1..=4).contains(&batch), "gemm batch 1..=4");
    let p = pipe()?;
    let fast = batch == 4 && m.n_out % 8 == 0 && m.n_in % 256 == 0;
    enc.set_compute_pipeline_state(if fast { &p.state_gemmu } else { &p.state_gemm });
    enc.set_buffer(0, Some(&m.packed), 0);
    enc.set_buffer(1, Some(&m.scales), 0);
    enc.set_buffer(2, Some(&m.biases), 0);
    enc.set_buffer(3, Some(xs), 0);
    enc.set_buffer(4, Some(ys), 0);
    let params = [m.n_in as i32, m.n_out as i32, batch as i32];
    enc.set_bytes(5, 12, params.as_ptr() as *const _);
    dispatch_rows(enc, m.n_out);
    Ok(())
}

/// Encode a GEMV into an existing compute encoder (no commit / wait).
pub fn encode_into(
    enc: &ComputeCommandEncoderRef,
    m: &QuantMatrix,
    x: &Buffer,
    y: &Buffer,
) -> Result<()> {
    encode_into_at(enc, m, x, 0, y, 0)
}

pub fn encode_into_at(
    enc: &ComputeCommandEncoderRef,
    m: &QuantMatrix,
    x: &Buffer,
    x_byte_off: NSUInteger,
    y: &Buffer,
    y_byte_off: NSUInteger,
) -> Result<()> {
    let p = pipe()?;
    enc.set_compute_pipeline_state(&p.state);
    enc.set_buffer(0, Some(&m.packed), 0);
    enc.set_buffer(1, Some(&m.scales), 0);
    enc.set_buffer(2, Some(&m.biases), 0);
    enc.set_buffer(3, Some(x), x_byte_off);
    enc.set_buffer(4, Some(y), y_byte_off);
    let params = [m.n_in as i32, m.n_out as i32];
    enc.set_bytes(5, 8, params.as_ptr() as *const _);
    dispatch_rows(enc, m.n_out);
    Ok(())
}

/// Up to four same-K GEMVs in one launch. `n_in` must match and `x` is shared.
pub fn encode_many(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    jobs: &[(&QuantMatrix, &Buffer)],
) -> Result<()> {
    anyhow::ensure!((1..=4).contains(&jobs.len()), "encode_many len");
    let n_in = jobs[0].0.n_in;
    for (m, _) in jobs {
        anyhow::ensure!(m.n_in == n_in, "encode_many k");
    }
    let p = pipe()?;
    let mut packed = [jobs[0].0, jobs[0].0, jobs[0].0, jobs[0].0];
    let mut ys = [jobs[0].1, jobs[0].1, jobs[0].1, jobs[0].1];
    let mut n_out = [0i32; 4];
    for (i, (m, y)) in jobs.iter().enumerate() {
        packed[i] = m;
        ys[i] = y;
        n_out[i] = m.n_out as i32;
    }
    enc.set_compute_pipeline_state(&p.state_n);
    enc.set_buffer(0, Some(x), 0);
    for i in 0..4 {
        enc.set_buffer((1 + i * 4) as u64, Some(&packed[i].packed), 0);
        enc.set_buffer((2 + i * 4) as u64, Some(&packed[i].scales), 0);
        enc.set_buffer((3 + i * 4) as u64, Some(&packed[i].biases), 0);
        enc.set_buffer((4 + i * 4) as u64, Some(ys[i]), 0);
    }
    let params = [
        n_in as i32,
        n_out[0],
        n_out[1],
        n_out[2],
        n_out[3],
        jobs.len() as i32,
    ];
    enc.set_bytes(17, (params.len() * 4) as NSUInteger, params.as_ptr() as *const _);
    let max_out = n_out.into_iter().max().unwrap_or(0).max(0) as u32;
    dispatch_rows(enc, max_out);
    Ok(())
}

/// `mid = silu(gate(x)) * up(x)` in one launch. `gate` and `up` share K and N.
pub fn encode_swiglu(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    gate: &QuantMatrix,
    up: &QuantMatrix,
    gate_y: &Buffer,
    up_y: &Buffer,
    mid: &Buffer,
) -> Result<()> {
    anyhow::ensure!(gate.n_in == up.n_in && gate.n_out == up.n_out, "swiglu shape");
    let p = pipe()?;
    enc.set_compute_pipeline_state(&p.state_silu);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(&gate.packed), 0);
    enc.set_buffer(2, Some(&gate.scales), 0);
    enc.set_buffer(3, Some(&gate.biases), 0);
    enc.set_buffer(4, Some(gate_y), 0);
    enc.set_buffer(5, Some(&up.packed), 0);
    enc.set_buffer(6, Some(&up.scales), 0);
    enc.set_buffer(7, Some(&up.biases), 0);
    enc.set_buffer(8, Some(up_y), 0);
    enc.set_buffer(9, Some(mid), 0);
    let params = [gate.n_in as i32, gate.n_out as i32];
    enc.set_bytes(10, 8, params.as_ptr() as *const _);
    dispatch_rows(enc, gate.n_out);
    Ok(())
}

/// `y = residual + W x`. `residual` may alias `y` (in-place add).
pub fn encode_residual(
    enc: &ComputeCommandEncoderRef,
    m: &QuantMatrix,
    x: &Buffer,
    residual: &Buffer,
    y: &Buffer,
) -> Result<()> {
    let p = pipe()?;
    enc.set_compute_pipeline_state(&p.state_res);
    enc.set_buffer(0, Some(&m.packed), 0);
    enc.set_buffer(1, Some(&m.scales), 0);
    enc.set_buffer(2, Some(&m.biases), 0);
    enc.set_buffer(3, Some(x), 0);
    enc.set_buffer(4, Some(residual), 0);
    enc.set_buffer(5, Some(y), 0);
    let params = [m.n_in as i32, m.n_out as i32];
    enc.set_bytes(6, 8, params.as_ptr() as *const _);
    dispatch_rows(enc, m.n_out);
    Ok(())
}

/// Write one dequantized embedding row into `y` (length `m.n_in`).
pub fn encode_dequant_row(
    enc: &ComputeCommandEncoderRef,
    m: &QuantMatrix,
    token: u32,
    y: &Buffer,
) -> Result<()> {
    let p = pipe()?;
    enc.set_compute_pipeline_state(&p.state_row);
    enc.set_buffer(0, Some(&m.packed), 0);
    enc.set_buffer(1, Some(&m.scales), 0);
    enc.set_buffer(2, Some(&m.biases), 0);
    enc.set_bytes(3, 4, &token as *const u32 as *const _);
    enc.set_buffer(4, Some(y), 0);
    let n_in = m.n_in as i32;
    enc.set_bytes(5, 4, &n_in as *const i32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
    Ok(())
}

pub fn encode_dequant_row_at(
    enc: &ComputeCommandEncoderRef,
    m: &QuantMatrix,
    token: u32,
    y: &Buffer,
    y_byte_off: NSUInteger,
) -> Result<()> {
    let p = pipe()?;
    enc.set_compute_pipeline_state(&p.state_row);
    enc.set_buffer(0, Some(&m.packed), 0);
    enc.set_buffer(1, Some(&m.scales), 0);
    enc.set_buffer(2, Some(&m.biases), 0);
    enc.set_bytes(3, 4, &token as *const u32 as *const _);
    enc.set_buffer(4, Some(y), y_byte_off);
    let n_in = m.n_in as i32;
    enc.set_bytes(5, 4, &n_in as *const i32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
    Ok(())
}

fn dispatch_rows(enc: &ComputeCommandEncoderRef, n_out: u32) {
    let tg = MTLSize {
        width: TG_SIZE as NSUInteger,
        height: 1,
        depth: 1,
    };
    let row_blocks = ((n_out + ROWS_PER_TG - 1) / ROWS_PER_TG) as NSUInteger;
    let grid = MTLSize {
        width: 1,
        height: row_blocks.max(1),
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_matches_bias_plus_scale_times_code() {
        // n_in must be multiple of 256 for the qmv block; use 256 with 4 groups.
        let n_in = 256usize;
        let mut packed = vec![0u32; n_in / 4];
        for i in 0..n_in {
            let code = (i % 256) as u32;
            packed[i / 4] |= code << (8 * (i % 4));
        }
        let groups = n_in / 64;
        let scales = vec![0.5f32; groups];
        let biases = vec![-1.0f32; groups];
        let x = vec![1.0f32; n_in];
        let m = QuantMatrix::upload(&packed, &scales, &biases, 1, n_in).unwrap();
        let y = m.gemv(&x).unwrap();
        let mut expect = 0.0f32;
        for i in 0..n_in {
            let code = (i % 256) as f32;
            expect += (-1.0 + 0.5 * code) * 1.0;
        }
        assert!((y[0] - expect).abs() < 1e-4, "{} vs {expect}", y[0]);
    }

    #[test]
    fn gemv_shared_matches_two_gemv() {
        let n_out = 128usize;
        let n_in = 256usize;
        let words = n_in / 4;
        let groups = n_in / 64;
        let mut packed_a = vec![0u32; n_out * words];
        let mut packed_b = vec![0u32; n_out * words];
        for i in 0..packed_a.len() {
            packed_a[i] = 0x01020304;
            packed_b[i] = 0x04030201;
        }
        let scales = vec![0.01f32; n_out * groups];
        let biases = vec![0.1f32; n_out * groups];
        let x = vec![0.5f32; n_in];
        let a = QuantMatrix::upload(&packed_a, &scales, &biases, n_out, n_in).unwrap();
        let b = QuantMatrix::upload(&packed_b, &scales, &biases, n_out, n_in).unwrap();
        let ya = a.gemv(&x).unwrap();
        let yb = b.gemv(&x).unwrap();
        let both = QuantMatrix::gemv_shared(&[&a, &b], &x).unwrap();
        for i in 0..n_out {
            assert!((both[0][i] - ya[i]).abs() < 1e-4);
            assert!((both[1][i] - yb[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn gemv_multi_row_values() {
        let n_out = 8usize;
        let n_in = 256usize;
        let mut packed = vec![0u32; n_out * (n_in / 4)];
        for r in 0..n_out {
            for i in 0..n_in {
                let code = ((r * 3 + i) % 256) as u32;
                packed[r * (n_in / 4) + i / 4] |= code << (8 * (i % 4));
            }
        }
        let groups = n_in / 64;
        let scales = vec![0.25f32; n_out * groups];
        let biases = vec![1.0f32; n_out * groups];
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.01).collect();
        let m = QuantMatrix::upload(&packed, &scales, &biases, n_out, n_in).unwrap();
        let y = m.gemv(&x).unwrap();
        for r in 0..n_out {
            let mut expect = 0.0f32;
            for i in 0..n_in {
                let code = ((r * 3 + i) % 256) as f32;
                let g = i / 64;
                expect += (biases[r * groups + g] + scales[r * groups + g] * code) * x[i];
            }
            assert!((y[r] - expect).abs() < 1e-1, "row {r}: {} vs {expect}", y[r]);
        }
    }

    #[test]
    fn gemm4_matches_serial_gemv() {
        let n_out = 8usize;
        let n_in = 256usize;
        let mut packed = vec![0u32; n_out * (n_in / 4)];
        for r in 0..n_out {
            for i in 0..n_in {
                let code = ((r * 13 + i * 5) % 251) as u32;
                packed[r * (n_in / 4) + i / 4] |= code << (8 * (i % 4));
            }
        }
        let groups = n_in / 64;
        let scales: Vec<f32> = (0..n_out * groups)
            .map(|i| 0.02 * ((i % 7) as f32) - 0.05)
            .collect();
        let biases: Vec<f32> = (0..n_out * groups)
            .map(|i| 0.01 * ((i % 5) as f32) - 0.02)
            .collect();
        let m = QuantMatrix::upload(&packed, &scales, &biases, n_out, n_in).unwrap();
        let p = pipe().unwrap();
        let opts = MTLResourceOptions::StorageModeShared;
        for batch in [1usize, 3, 4] {
            let mut xs = Vec::new();
            for b in 0..batch {
                xs.extend((0..n_in).map(|i| ((i + b * 17) % 11) as f32 * 0.03 - 0.1));
            }
            let serial: Vec<Vec<f32>> = (0..batch)
                .map(|b| m.gemv(&xs[b * n_in..(b + 1) * n_in]).unwrap())
                .collect();
            let x_b = p.device.new_buffer_with_data(
                xs.as_ptr() as *const _,
                (xs.len() * 4) as NSUInteger,
                opts,
            );
            let y_b = p.device.new_buffer((batch * n_out * 4) as NSUInteger, opts);
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode_gemm4(enc, &m, &x_b, &y_b, batch as u32).unwrap();
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got = read_f32(&y_b, batch * n_out);
            for b in 0..batch {
                for r in 0..n_out {
                    let g = got[b * n_out + r];
                    let e = serial[b][r];
                    assert!(
                        (g - e).abs() == 0.0,
                        "batch {batch} b{b} row {r}: gemm {g} vs gemv {e}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "MLP-shaped batch GEMM bandwidth lab"]
    fn gemm4_bandwidth_mlp() {
        let n_out = 17408usize;
        let n_in = 5120usize;
        let words = n_in / 4;
        let groups = n_in / 64;
        let packed = vec![0x01020304u32; n_out * words];
        let scales = vec![0.01f32; n_out * groups];
        let biases = vec![0.0f32; n_out * groups];
        let m = QuantMatrix::upload(&packed, &scales, &biases, n_out, n_in).unwrap();
        let p = pipe().unwrap();
        let opts = MTLResourceOptions::StorageModeShared;
        for batch in [1u32, 2, 4] {
            let xs = vec![0.01f32; n_in * batch as usize];
            let x_b = p.device.new_buffer_with_data(
                xs.as_ptr() as *const _,
                (xs.len() * 4) as NSUInteger,
                opts,
            );
            let y_b = p.device.new_buffer((n_out * batch as usize * 4) as NSUInteger, opts);
            let time = |gemm: bool| {
                for _ in 0..2 {
                    let cb = p.queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    if gemm {
                        encode_gemm4(enc, &m, &x_b, &y_b, batch).unwrap();
                    } else {
                        for b in 0..batch {
                            let x_off = (b as usize * n_in * 4) as NSUInteger;
                            let y_off = (b as usize * n_out * 4) as NSUInteger;
                            enc.set_compute_pipeline_state(&p.state);
                            enc.set_buffer(0, Some(&m.packed), 0);
                            enc.set_buffer(1, Some(&m.scales), 0);
                            enc.set_buffer(2, Some(&m.biases), 0);
                            enc.set_buffer(3, Some(&x_b), x_off);
                            enc.set_buffer(4, Some(&y_b), y_off);
                            let params = [m.n_in as i32, m.n_out as i32];
                            enc.set_bytes(5, 8, params.as_ptr() as *const _);
                            dispatch_rows(enc, m.n_out);
                        }
                    }
                    enc.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                }
                let n_dispatch = 20u32;
                let cb = p.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for _ in 0..n_dispatch {
                    if gemm {
                        encode_gemm4(enc, &m, &x_b, &y_b, batch).unwrap();
                    } else {
                        for b in 0..batch {
                            let x_off = (b as usize * n_in * 4) as NSUInteger;
                            let y_off = (b as usize * n_out * 4) as NSUInteger;
                            enc.set_compute_pipeline_state(&p.state);
                            enc.set_buffer(0, Some(&m.packed), 0);
                            enc.set_buffer(1, Some(&m.scales), 0);
                            enc.set_buffer(2, Some(&m.biases), 0);
                            enc.set_buffer(3, Some(&x_b), x_off);
                            enc.set_buffer(4, Some(&y_b), y_off);
                            let params = [m.n_in as i32, m.n_out as i32];
                            enc.set_bytes(5, 8, params.as_ptr() as *const _);
                            dispatch_rows(enc, m.n_out);
                        }
                    }
                }
                enc.end_encoding();
                let t = std::time::Instant::now();
                cb.commit();
                cb.wait_until_completed();
                t.elapsed().as_secs_f64() / n_dispatch as f64
            };
            let t_serial = time(false);
            let t_gemm = time(true);
            eprintln!(
                "batch {batch}: serial {t_serial:.5}s  gemm {t_gemm:.5}s  speedup {:.2}x",
                t_serial / t_gemm
            );
        }
    }

    #[test]
    fn gemv_bandwidth_mlp_shape() {
        let n_out = 17408usize;
        let n_in = 5120usize;
        let words = n_in / 4;
        let groups = n_in / 64;
        let packed = vec![0x01020304u32; n_out * words];
        let scales = vec![0.01f32; n_out * groups];
        let biases = vec![0.0f32; n_out * groups];
        let x = vec![0.01f32; n_in];
        let m = QuantMatrix::upload(&packed, &scales, &biases, n_out, n_in).unwrap();
        let p = pipe().unwrap();
        let opts = MTLResourceOptions::StorageModeShared;
        let x_b = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as NSUInteger,
            opts,
        );
        let y_b = p.device.new_buffer((n_out as NSUInteger) * 4, opts);

        for _ in 0..3 {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode_gemv(enc, &p.state, &m, &x_b, &y_b);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let n_dispatch = 40u32;
        let t_enc = std::time::Instant::now();
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..n_dispatch {
            encode_gemv(enc, &p.state, &m, &x_b, &y_b);
        }
        enc.end_encoding();
        let enc_s = t_enc.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let wall = t.elapsed().as_secs_f64();
        let sec = wall / n_dispatch as f64;
        let gb = (n_out * n_in) as f64 / 1e9;
        let gbs = gb / sec;
        eprintln!(
            "mlp gemv {sec:.5}s  {gbs:.1} GB/s  encode {enc_s:.4}s ({:.1} us/dispatch)  gpu-wall {wall:.4}s",
            enc_s / n_dispatch as f64 * 1e6
        );
    }

    /// 7-bit codes, little-endian bitstream, group 64 affine. Not the default:
    /// requantizing the 8-bit checkpoint moves a GEMV by about 1%.
    #[test]
    #[ignore = "MLP-shaped bandwidth lab for a 7-bit unpack"]
    fn gemv_q7_bandwidth() {
        const Q7: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant int GROUP_SIZE = 64;
constant int BLOCK_SIZE = 256;
inline float bf16(ushort u) { return as_type<float>(uint(u) << 16); }
inline float dotq(ulong v, float4 x0, float4 x1) {
    float c0 = float(v & 127ul);
    float c1 = float((v >> 7) & 127ul);
    float c2 = float((v >> 14) & 127ul);
    float c3 = float((v >> 21) & 127ul);
    float c4 = float((v >> 28) & 127ul);
    float c5 = float((v >> 35) & 127ul);
    float c6 = float((v >> 42) & 127ul);
    float c7 = float((v >> 49) & 127ul);
    return fma(c0, x0[0], fma(c1, x0[1], fma(c2, x0[2], fma(c3, x0[3],
        fma(c4, x1[0], fma(c5, x1[1], fma(c6, x1[2], c7 * x1[3])))))));
}
inline ulong load56(device const uchar* p) {
    ulong v = ulong(p[0]);
    v |= ulong(p[1]) << 8;
    v |= ulong(p[2]) << 16;
    v |= ulong(p[3]) << 24;
    v |= ulong(p[4]) << 32;
    v |= ulong(p[5]) << 40;
    v |= ulong(p[6]) << 48;
    return v;
}
inline ulong load56_tg(threadgroup const uint* w, uint byte) {
    uint i = byte >> 2u;
    uint sh = (byte & 3u) * 8u;
    uint got = 32u - sh;
    ulong v = ulong(w[i]) >> sh;
    v |= ulong(w[i + 1u]) << got;
    if (got + 32u < 56u) {
        v |= ulong(w[i + 2u]) << (got + 32u);
    }
    return v;
}
struct GemvParams { int n_in; int n_out; };
kernel void q7_bytes(
    device const uchar* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant GemvParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    int n_in = params.n_in;
    int n_out = params.n_out;
    int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
    if (out_row >= n_out) { return; }
    int groups = n_in / GROUP_SIZE;
    int row_bytes = n_in * 7 / 8;
    device const ushort* sc = scales + out_row * groups + int(simd_lid) / 8;
    device const ushort* bc = biases + out_row * groups + int(simd_lid) / 8;
    device const float* xv = x + int(simd_lid) * 8;
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    for (int k = 0; k < n_in; k += BLOCK_SIZE) {
        float4 x0 = *(device const float4*)xv;
        float4 x1 = *(device const float4*)(xv + 4);
        float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
        int off = (k * 7) / 8 + int(simd_lid) * 7;
        ulong v0 = load56(packed + out_row * row_bytes + off);
        ulong v1 = load56(packed + (out_row + 1) * row_bytes + off);
        ulong v2 = load56(packed + (out_row + 2) * row_bytes + off);
        ulong v3 = load56(packed + (out_row + 3) * row_bytes + off);
        r0 = fma(bf16(sc[0]), dotq(v0, x0, x1), fma(xsum, bf16(bc[0]), r0));
        r1 = fma(bf16(sc[groups]), dotq(v1, x0, x1), fma(xsum, bf16(bc[groups]), r1));
        r2 = fma(bf16(sc[2 * groups]), dotq(v2, x0, x1), fma(xsum, bf16(bc[2 * groups]), r2));
        r3 = fma(bf16(sc[3 * groups]), dotq(v3, x0, x1), fma(xsum, bf16(bc[3 * groups]), r3));
        sc += BLOCK_SIZE / GROUP_SIZE;
        bc += BLOCK_SIZE / GROUP_SIZE;
        xv += BLOCK_SIZE;
    }
    r0 = simd_sum(r0); r1 = simd_sum(r1); r2 = simd_sum(r2); r3 = simd_sum(r3);
    if (simd_lid == 0u) {
        y[out_row] = r0; y[out_row + 1] = r1; y[out_row + 2] = r2; y[out_row + 3] = r3;
    }
}
kernel void q7_staged(
    device const uchar* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant GemvParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    int n_in = params.n_in;
    int n_out = params.n_out;
    int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
    if (out_row >= n_out) { return; }
    int groups = n_in / GROUP_SIZE;
    int row_bytes = n_in * 7 / 8;
    device const ushort* sc = scales + out_row * groups + int(simd_lid) / 8;
    device const ushort* bc = biases + out_row * groups + int(simd_lid) / 8;
    device const float* xv = x + int(simd_lid) * 8;
    threadgroup uint tile[2][4][56];
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    for (int k = 0; k < n_in; k += BLOCK_SIZE) {
        int block_off = (k * 7) / 8;
        if (simd_lid < 28u) {
            for (int r = 0; r < 4; ++r) {
                device const uint2* src = (device const uint2*)(packed
                    + (out_row + r) * row_bytes + block_off + int(simd_lid) * 8);
                uint2 w = *src;
                tile[simd_gid][r][simd_lid * 2u] = w.x;
                tile[simd_gid][r][simd_lid * 2u + 1u] = w.y;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 x0 = *(device const float4*)xv;
        float4 x1 = *(device const float4*)(xv + 4);
        float xsum = x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
        uint byte = simd_lid * 7u;
        ulong v0 = load56_tg(tile[simd_gid][0], byte);
        ulong v1 = load56_tg(tile[simd_gid][1], byte);
        ulong v2 = load56_tg(tile[simd_gid][2], byte);
        ulong v3 = load56_tg(tile[simd_gid][3], byte);
        r0 = fma(bf16(sc[0]), dotq(v0, x0, x1), fma(xsum, bf16(bc[0]), r0));
        r1 = fma(bf16(sc[groups]), dotq(v1, x0, x1), fma(xsum, bf16(bc[groups]), r1));
        r2 = fma(bf16(sc[2 * groups]), dotq(v2, x0, x1), fma(xsum, bf16(bc[2 * groups]), r2));
        r3 = fma(bf16(sc[3 * groups]), dotq(v3, x0, x1), fma(xsum, bf16(bc[3 * groups]), r3));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sc += BLOCK_SIZE / GROUP_SIZE;
        bc += BLOCK_SIZE / GROUP_SIZE;
        xv += BLOCK_SIZE;
    }
    r0 = simd_sum(r0); r1 = simd_sum(r1); r2 = simd_sum(r2); r3 = simd_sum(r3);
    if (simd_lid == 0u) {
        y[out_row] = r0; y[out_row + 1] = r1; y[out_row + 2] = r2; y[out_row + 3] = r3;
    }
}
"#;

        fn pack7(n_out: usize, n_in: usize, code_at: impl Fn(usize, usize) -> u8) -> Vec<u8> {
            let row_bytes = n_in * 7 / 8;
            let mut dst = vec![0u8; n_out * row_bytes];
            for r in 0..n_out {
                let mut acc = 0u64;
                let mut nbits = 0u32;
                let mut at = r * row_bytes;
                for i in 0..n_in {
                    acc |= u64::from(code_at(r, i) & 127) << nbits;
                    nbits += 7;
                    while nbits >= 8 {
                        dst[at] = (acc & 255) as u8;
                        at += 1;
                        acc >>= 8;
                        nbits -= 8;
                    }
                }
                if nbits > 0 {
                    dst[at] = (acc & 255) as u8;
                }
            }
            dst
        }

        let dev = Device::system_default().unwrap();
        let lib = dev
            .new_library_with_source(Q7, &metal::CompileOptions::new())
            .expect("q7 metal");
        let mk = |name: &str| {
            let f = lib.get_function(name, None).unwrap();
            dev.new_compute_pipeline_state_with_function(&f).unwrap()
        };
        let q7_bytes = mk("q7_bytes");
        let q7_staged = mk("q7_staged");
        let queue = dev.new_command_queue();
        let private = MTLResourceOptions::StorageModePrivate
            | MTLResourceOptions::HazardTrackingModeUntracked;
        let shared = MTLResourceOptions::StorageModeShared;

        let check_out = 8usize;
        let check_in = 256usize;
        let check_groups = check_in / 64;
        let packed = pack7(check_out, check_in, |r, i| ((r * 3 + i) % 127) as u8);
        let scales = vec![0.5f32; check_out * check_groups];
        let biases = vec![-1.0f32; check_out * check_groups];
        let x = vec![1.0f32; check_in];
        let scales_h: Vec<u16> = scales.iter().copied().map(f32_to_bf16).collect();
        let biases_h: Vec<u16> = biases.iter().copied().map(f32_to_bf16).collect();
        let y = dev.new_buffer((check_out * 4) as NSUInteger, shared);
        let x_b = dev.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 4) as _, shared);
        let p_b = dev.new_buffer_with_data(packed.as_ptr() as *const _, packed.len() as _, shared);
        let s_b = dev.new_buffer_with_data(scales_h.as_ptr() as *const _, (scales_h.len() * 2) as _, shared);
        let b_b = dev.new_buffer_with_data(biases_h.as_ptr() as *const _, (biases_h.len() * 2) as _, shared);
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&q7_staged);
        enc.set_buffer(0, Some(&p_b), 0);
        enc.set_buffer(1, Some(&s_b), 0);
        enc.set_buffer(2, Some(&b_b), 0);
        enc.set_buffer(3, Some(&x_b), 0);
        enc.set_buffer(4, Some(&y), 0);
        let params = [check_in as i32, check_out as i32];
        enc.set_bytes(5, 8, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 64, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = read_f32(&y, check_out);
        for r in 0..check_out {
            let mut expect = 0.0f32;
            for i in 0..check_in {
                let code = ((r * 3 + i) % 127) as f32;
                expect += (-1.0 + 0.5 * code) * 1.0;
            }
            assert!((got[r] - expect).abs() < 1e-2, "q7 row {r}: {} vs {expect}", got[r]);
        }

        let n_out = 17408usize;
        let n_in = 5120usize;
        let groups = n_in / 64;
        let big = pack7(n_out, n_in, |_, i| (i % 127) as u8);
        let scales = vec![0.01f32; n_out * groups];
        let biases = vec![0.0f32; n_out * groups];
        let x = vec![0.01f32; n_in];
        let scales_h: Vec<u16> = scales.iter().copied().map(f32_to_bf16).collect();
        let biases_h: Vec<u16> = biases.iter().copied().map(f32_to_bf16).collect();
        let stage = |bytes: &[u8]| {
            let src = dev.new_buffer_with_data(bytes.as_ptr() as *const _, bytes.len() as _, shared);
            let dst = dev.new_buffer(src.length(), private);
            let cb = queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            blit.copy_from_buffer(&src, 0, &dst, 0, src.length());
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            dst
        };
        let packed_b = stage(&big);
        let scales_bytes = unsafe {
            std::slice::from_raw_parts(scales_h.as_ptr() as *const u8, scales_h.len() * 2)
        };
        let biases_bytes = unsafe {
            std::slice::from_raw_parts(biases_h.as_ptr() as *const u8, biases_h.len() * 2)
        };
        let scales_b = stage(scales_bytes);
        let biases_b = stage(biases_bytes);
        let x_b = dev.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 4) as _, shared);
        let y_b = dev.new_buffer((n_out * 4) as NSUInteger, shared);
        let time_q7 = |state: &ComputePipelineState, tag: &str| {
            for _ in 0..3 {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(state);
                enc.set_buffer(0, Some(&packed_b), 0);
                enc.set_buffer(1, Some(&scales_b), 0);
                enc.set_buffer(2, Some(&biases_b), 0);
                enc.set_buffer(3, Some(&x_b), 0);
                enc.set_buffer(4, Some(&y_b), 0);
                let params = [n_in as i32, n_out as i32];
                enc.set_bytes(5, 8, params.as_ptr() as *const _);
                enc.dispatch_thread_groups(
                    MTLSize { width: 1, height: (n_out / 8) as NSUInteger, depth: 1 },
                    MTLSize { width: 64, height: 1, depth: 1 },
                );
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let n_dispatch = 40u32;
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for _ in 0..n_dispatch {
                enc.set_compute_pipeline_state(state);
                enc.set_buffer(0, Some(&packed_b), 0);
                enc.set_buffer(1, Some(&scales_b), 0);
                enc.set_buffer(2, Some(&biases_b), 0);
                enc.set_buffer(3, Some(&x_b), 0);
                enc.set_buffer(4, Some(&y_b), 0);
                let params = [n_in as i32, n_out as i32];
                enc.set_bytes(5, 8, params.as_ptr() as *const _);
                enc.dispatch_thread_groups(
                    MTLSize { width: 1, height: (n_out / 8) as NSUInteger, depth: 1 },
                    MTLSize { width: 64, height: 1, depth: 1 },
                );
            }
            enc.end_encoding();
            let t = std::time::Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let sec = t.elapsed().as_secs_f64() / n_dispatch as f64;
            let bytes = (n_out * n_in * 7 / 8) as f64;
            eprintln!("{tag} {sec:.5}s  {:.1} GB/s", bytes / 1e9 / sec);
            sec
        };
        let t_bytes = time_q7(&q7_bytes, "q7-byte");
        let t_staged = time_q7(&q7_staged, "q7-staged");
        let m = QuantMatrix::upload(
            &vec![0x01020304u32; n_out * (n_in / 4)],
            &scales,
            &biases,
            n_out,
            n_in,
        )
        .unwrap();
        let p = pipe().unwrap();
        for _ in 0..3 {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode_gemv(enc, &p.state, &m, &x_b, &y_b);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let n_dispatch = 40u32;
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..n_dispatch {
            encode_gemv(enc, &p.state, &m, &x_b, &y_b);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let t8 = t.elapsed().as_secs_f64() / n_dispatch as f64;
        eprintln!(
            "q8 {t8:.5}s  q7/q8 byte {:.3}  staged {:.3}",
            t_bytes / t8,
            t_staged / t8
        );
    }

    #[test]
    #[ignore = "allocates several GB and is a bandwidth lab"]
    fn gemv_private_and_load_ceiling() {
        let n_out = 17408usize;
        let n_in = 5120usize;
        let words = n_in / 4;
        let groups = n_in / 64;
        let packed = vec![0x01020304u32; n_out * words];
        let scales = vec![0.01f32; n_out * groups];
        let biases = vec![0.0f32; n_out * groups];
        let x = vec![0.01f32; n_in];
        let shared = QuantMatrix::upload(&packed, &scales, &biases, n_out, n_in).unwrap();
        let p = pipe().unwrap();
        let opts_priv = MTLResourceOptions::StorageModePrivate
            | MTLResourceOptions::HazardTrackingModeUntracked;
        let bytes = |b: &Buffer| b.length();
        let pk = p.device.new_buffer(bytes(&shared.packed), opts_priv);
        let sc = p.device.new_buffer(bytes(&shared.scales), opts_priv);
        let bi = p.device.new_buffer(bytes(&shared.biases), opts_priv);
        let x_shared = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        let x_priv = p.device.new_buffer(x_shared.length(), opts_priv);
        let y = p.device.new_buffer((n_out * 4) as NSUInteger, opts_priv);
        let cb = p.queue.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer(&shared.packed, 0, &pk, 0, shared.packed.length());
        blit.copy_from_buffer(&shared.scales, 0, &sc, 0, shared.scales.length());
        blit.copy_from_buffer(&shared.biases, 0, &bi, 0, shared.biases.length());
        blit.copy_from_buffer(&x_shared, 0, &x_priv, 0, x_shared.length());
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let private = QuantMatrix {
            packed: pk,
            scales: sc,
            biases: bi,
            n_out: n_out as u32,
            n_in: n_in as u32,
        };
        let n_dispatch = 40u32;
        for _ in 0..2 {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode_gemv(enc, &p.state, &private, &x_priv, &y);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..n_dispatch {
            encode_gemv(enc, &p.state, &private, &x_priv, &y);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let sec = t.elapsed().as_secs_f64() / n_dispatch as f64;
        let gbs = (n_out * n_in) as f64 / 1e9 / sec;
        eprintln!("private gemv {sec:.5}s  {gbs:.1} GB/s");

        // Eight distinct MLP tiles, bigger than a last-level cache, one pass.
        let n_mat = 8usize;
        let mlp_out = 17408usize;
        let mlp_in = 5120usize;
        let mut mlps = Vec::new();
        for i in 0..n_mat {
            let packed = vec![0x01020304u32.wrapping_mul(i as u32 + 1); mlp_out * (mlp_in / 4)];
            let scales = vec![0.01f32; mlp_out * (mlp_in / 64)];
            let biases = vec![0.0f32; mlp_out * (mlp_in / 64)];
            mlps.push(QuantMatrix::upload(&packed, &scales, &biases, mlp_out, mlp_in).unwrap());
        }
        let y_m = p.device.new_buffer((mlp_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let x_m = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (mlp_in * 4) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        for _ in 0..2 {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for m in &mlps {
                encode_gemv(enc, &p.state, m, &x_m, &y_m);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for m in &mlps {
            encode_gemv(enc, &p.state, m, &x_m, &y_m);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let sec = t.elapsed().as_secs_f64() / n_mat as f64;
        let gbs = (mlp_out * mlp_in) as f64 / 1e9 / sec;
        eprintln!("8x mlp tiles {sec:.5}s each  {gbs:.1} GB/s");
        let gy = p.device.new_buffer((mlp_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let uy = p.device.new_buffer((mlp_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let midb = p.device.new_buffer((mlp_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let run_sw = || {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for i in (0..n_mat).step_by(2) {
                encode_swiglu(enc, &x_m, &mlps[i], &mlps[i + 1], &gy, &uy, &midb).unwrap();
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        run_sw();
        run_sw();
        let t = std::time::Instant::now();
        run_sw();
        let pair_s = t.elapsed().as_secs_f64() / (n_mat as f64 / 2.0);
        eprintln!(
            "interleaved gate+up {:.5}s/pair  {:.1} GB/s (both mats)",
            pair_s,
            (2 * mlp_out * mlp_in) as f64 / 1e9 / pair_s
        );

        let partial = p.device.new_buffer(
            (mlp_out * 4 * 4) as NSUInteger,
            MTLResourceOptions::StorageModePrivate | MTLResourceOptions::HazardTrackingModeUntracked,
        );
        let y_p = p.device.new_buffer(
            (mlp_out * 4) as NSUInteger,
            MTLResourceOptions::StorageModePrivate | MTLResourceOptions::HazardTrackingModeUntracked,
        );
        let dispatch_s4 = |enc: &ComputeCommandEncoderRef, m: &QuantMatrix| {
            enc.set_compute_pipeline_state(&p.state_s4);
            enc.set_buffer(0, Some(&m.packed), 0);
            enc.set_buffer(1, Some(&m.scales), 0);
            enc.set_buffer(2, Some(&m.biases), 0);
            enc.set_buffer(3, Some(&x_m), 0);
            enc.set_buffer(4, Some(&partial), 0);
            let params = [m.n_in as i32, m.n_out as i32];
            enc.set_bytes(5, 8, params.as_ptr() as *const _);
            let rows = ((mlp_out as u32 + ROWS_PER_TG - 1) / ROWS_PER_TG) as NSUInteger;
            enc.dispatch_thread_groups(
                MTLSize { width: 4, height: rows, depth: 1 },
                MTLSize { width: TG_SIZE as NSUInteger, height: 1, depth: 1 },
            );
            enc.set_compute_pipeline_state(&p.state_red);
            enc.set_buffer(0, Some(&partial), 0);
            enc.set_buffer(1, Some(&y_p), 0);
            let n = mlp_out as i32;
            enc.set_bytes(2, 4, &n as *const i32 as *const _);
            enc.dispatch_threads(
                MTLSize { width: mlp_out as NSUInteger, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        };
        for _ in 0..2 {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for m in &mlps {
                dispatch_s4(enc, m);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for m in &mlps {
            dispatch_s4(enc, m);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let sec = t.elapsed().as_secs_f64() / n_mat as f64;
        let gbs = (mlp_out * mlp_in) as f64 / 1e9 / sec;
        eprintln!("8x split-k4 {sec:.5}s each  {gbs:.1} GB/s");

        let one = mlps[0].gemv(&x[..mlp_in]).unwrap();
        let y_chk = p.device.new_buffer((mlp_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let cb = p.queue.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        // run one split into shared y via a shared partial
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let partial_s = p.device.new_buffer((mlp_out * 4 * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&p.state_s4);
        enc.set_buffer(0, Some(&mlps[0].packed), 0);
        enc.set_buffer(1, Some(&mlps[0].scales), 0);
        enc.set_buffer(2, Some(&mlps[0].biases), 0);
        enc.set_buffer(3, Some(&x_m), 0);
        enc.set_buffer(4, Some(&partial_s), 0);
        let params = [mlp_in as i32, mlp_out as i32];
        enc.set_bytes(5, 8, params.as_ptr() as *const _);
        let rows = ((mlp_out as u32 + ROWS_PER_TG - 1) / ROWS_PER_TG) as NSUInteger;
        enc.dispatch_thread_groups(
            MTLSize { width: 4, height: rows, depth: 1 },
            MTLSize { width: TG_SIZE as NSUInteger, height: 1, depth: 1 },
        );
        enc.set_compute_pipeline_state(&p.state_red);
        enc.set_buffer(0, Some(&partial_s), 0);
        enc.set_buffer(1, Some(&y_chk), 0);
        let n = mlp_out as i32;
        enc.set_bytes(2, 4, &n as *const i32 as *const _);
        enc.dispatch_threads(
            MTLSize { width: mlp_out as NSUInteger, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = read_f32(&y_chk, mlp_out);
        let mut worst = 0.0f32;
        for (a, b) in got.iter().zip(&one) {
            worst = worst.max((a - b).abs());
        }
        eprintln!("split-k max abs diff {worst}");
        assert!(worst < 1e-2, "split-k diff {worst}");

        // Specialized K=5120, fully unrolled, to see if the short K loop is the limit.
        let spec = r#"
            #include <metal_stdlib>
            using namespace metal;
            inline float bf16(ushort u) { return as_type<float>(uint(u) << 16); }
            inline float dot8(uint2 ww, float4 x0, float4 x1) {
                float c0 = float(ww.x & 255u);
                float c1 = float((ww.x >> 8) & 255u);
                float c2 = float((ww.x >> 16) & 255u);
                float c3 = float((ww.x >> 24) & 255u);
                float c4 = float(ww.y & 255u);
                float c5 = float((ww.y >> 8) & 255u);
                float c6 = float((ww.y >> 16) & 255u);
                float c7 = float((ww.y >> 24) & 255u);
                return fma(c0, x0[0], fma(c1, x0[1], fma(c2, x0[2], fma(c3, x0[3],
                    fma(c4, x1[0], fma(c5, x1[1], fma(c6, x1[2], c7 * x1[3])))))));
            }
            kernel void gemv_5120(
                device const uint* packed [[buffer(0)]],
                device const ushort* scales [[buffer(1)]],
                device const ushort* biases [[buffer(2)]],
                device const float* x [[buffer(3)]],
                device float* y [[buffer(4)]],
                constant int& n_out [[buffer(5)]],
                uint3 tid [[threadgroup_position_in_grid]],
                uint simd_gid [[simdgroup_index_in_threadgroup]],
                uint simd_lid [[thread_index_in_simdgroup]]
            ) {
                const int n_in = 5120;
                const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
                if (out_row >= n_out) { return; }
                const int groups = n_in / 64;
                device const uchar* ws = ((device const uchar*)packed) + out_row * n_in + int(simd_lid) * 8;
                device const ushort* sc = scales + out_row * groups + int(simd_lid) / 8;
                device const ushort* bc = biases + out_row * groups + int(simd_lid) / 8;
                device const float* xv = x + int(simd_lid) * 8;
                float r0=0, r1=0, r2=0, r3=0;
                for (int k = 0; k < n_in; k += 256) {
                    float4 x0 = *(device const float4*)xv;
                    float4 x1 = *(device const float4*)(xv + 4);
                    float xsum = x0[0]+x0[1]+x0[2]+x0[3]+x1[0]+x1[1]+x1[2]+x1[3];
                    uint2 w0 = *(device const uint2*)(ws);
                    uint2 w1 = *(device const uint2*)(ws + n_in);
                    uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
                    uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
                    r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
                    r1 = fma(bf16(sc[groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[groups]), r1));
                    r2 = fma(bf16(sc[2*groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[2*groups]), r2));
                    r3 = fma(bf16(sc[3*groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[3*groups]), r3));
                    ws += 256; sc += 4; bc += 4; xv += 256;
                }
                r0 = simd_sum(r0); r1 = simd_sum(r1); r2 = simd_sum(r2); r3 = simd_sum(r3);
                if (simd_lid == 0u) {
                    y[out_row]=r0; y[out_row+1]=r1; y[out_row+2]=r2; y[out_row+3]=r3;
                }
            }
        "#;
        let lib = p.device.new_library_with_source(spec, &metal::CompileOptions::new()).unwrap();
        let f = lib.get_function("gemv_5120", None).unwrap();
        let st = p.device.new_compute_pipeline_state_with_function(&f).unwrap();
        let y_s = p.device.new_buffer((mlp_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let run = || {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for m in &mlps {
                enc.set_compute_pipeline_state(&st);
                enc.set_buffer(0, Some(&m.packed), 0);
                enc.set_buffer(1, Some(&m.scales), 0);
                enc.set_buffer(2, Some(&m.biases), 0);
                enc.set_buffer(3, Some(&x_m), 0);
                enc.set_buffer(4, Some(&y_s), 0);
                let n = mlp_out as i32;
                enc.set_bytes(5, 4, &n as *const i32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize { width: 1, height: (mlp_out as NSUInteger) / 8, depth: 1 },
                    MTLSize { width: 64, height: 1, depth: 1 },
                );
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        run();
        run();
        let t = std::time::Instant::now();
        run();
        let sec = t.elapsed().as_secs_f64() / n_mat as f64;
        eprintln!(
            "8x spec5120 {:.5}s each  {:.1} GB/s",
            sec,
            (mlp_out * mlp_in) as f64 / 1e9 / sec
        );
    }

    #[test]
    fn gemv_many_swiglu_and_residual_match_single() {
        let n_in = 256usize;
        let outs = [16usize, 8, 32, 4];
        let mut mats = Vec::new();
        let mut ys_expect = Vec::new();
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.01).collect();
        for (j, &n_out) in outs.iter().enumerate() {
            let mut packed = vec![0u32; n_out * (n_in / 4)];
            for r in 0..n_out {
                for i in 0..n_in {
                    let code = ((r + j + i) % 17) as u32;
                    packed[r * (n_in / 4) + i / 4] |= code << (8 * (i % 4));
                }
            }
            let groups = n_in / 64;
            let scales = vec![0.5f32; n_out * groups];
            let biases = vec![-1.0f32; n_out * groups];
            let m = QuantMatrix::upload(&packed, &scales, &biases, n_out, n_in).unwrap();
            ys_expect.push(m.gemv(&x).unwrap());
            mats.push(m);
        }
        let p = pipe().unwrap();
        let opts = MTLResourceOptions::StorageModeShared;
        let x_b = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as NSUInteger,
            opts,
        );
        let yb: Vec<Buffer> = outs
            .iter()
            .map(|n| p.device.new_buffer((*n as NSUInteger) * 4, opts))
            .collect();
        let jobs: Vec<(&QuantMatrix, &Buffer)> = mats.iter().zip(yb.iter()).collect();
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        encode_many(enc, &x_b, &jobs).unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        for (i, y) in yb.iter().enumerate() {
            let got = read_f32(y, outs[i]);
            for (a, b) in got.iter().zip(&ys_expect[i]) {
                assert!((a - b).abs() < 1e-2, "many {i}: {a} vs {b}");
            }
        }

        let gate = &mats[0];
        let up = QuantMatrix::upload(
            &vec![0x01020304u32; outs[0] * (n_in / 4)],
            &vec![0.5f32; outs[0] * (n_in / 64)],
            &vec![0.0f32; outs[0] * (n_in / 64)],
            outs[0],
            n_in,
        )
        .unwrap();
        let up_y = up.gemv(&x).unwrap();
        let gy = p.device.new_buffer((outs[0] * 4) as NSUInteger, opts);
        let uy = p.device.new_buffer((outs[0] * 4) as NSUInteger, opts);
        let mid = p.device.new_buffer((outs[0] * 4) as NSUInteger, opts);
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        encode_swiglu(enc, &x_b, gate, &up, &gy, &uy, &mid).unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = read_f32(&mid, outs[0]);
        for i in 0..outs[0] {
            let g = ys_expect[0][i];
            let u = up_y[i];
            let expect = g * (1.0 / (1.0 + (-g).exp())) * u;
            assert!((got[i] - expect).abs() < 2e-2, "silu {i}: {} vs {expect}", got[i]);
        }

        let res: Vec<f32> = (0..outs[2]).map(|i| i as f32).collect();
        let res_b = p.device.new_buffer_with_data(
            res.as_ptr() as *const _,
            (res.len() * 4) as NSUInteger,
            opts,
        );
        let y = p.device.new_buffer((outs[2] * 4) as NSUInteger, opts);
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        encode_residual(enc, &mats[2], &x_b, &res_b, &y).unwrap();
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = read_f32(&y, outs[2]);
        for i in 0..outs[2] {
            let expect = res[i] + ys_expect[2][i];
            assert!((got[i] - expect).abs() < 1e-2, "res {i}");
        }
    }

    #[test]
    #[ignore = "allocates several GB and is a bandwidth lab"]
    fn gemv_shape_sweep() {
        fn time_shape(n_out: usize, n_in: usize, split4: bool) -> f64 {
            let copies = if n_out * n_in < 200_000_000 { 4 } else { 1 };
            let groups = n_in / 64;
            let x = vec![0.01f32; n_in];
            let mut ms = Vec::new();
            for c in 0..copies {
                let packed = vec![0x01020304u32.wrapping_add(c as u32); n_out * (n_in / 4)];
                let scales = vec![0.02f32; n_out * groups];
                let biases = vec![0.0f32; n_out * groups];
                ms.push(QuantMatrix::upload(&packed, &scales, &biases, n_out, n_in).unwrap());
            }
            let p = pipe().unwrap();
            let opts = MTLResourceOptions::StorageModeShared;
            let x_b = p.device.new_buffer_with_data(
                x.as_ptr() as *const _,
                (n_in * 4) as NSUInteger,
                opts,
            );
            let y_b = p.device.new_buffer((n_out * 4) as NSUInteger, opts);
            let partial = if split4 {
                Some(p.device.new_buffer(
                    (n_out * 16) as NSUInteger,
                    MTLResourceOptions::StorageModePrivate
                        | MTLResourceOptions::HazardTrackingModeUntracked,
                ))
            } else {
                None
            };
            let once = |enc: &ComputeCommandEncoderRef, m: &QuantMatrix| {
                if split4 {
                    enc.set_compute_pipeline_state(&p.state_s4);
                    enc.set_buffer(0, Some(&m.packed), 0);
                    enc.set_buffer(1, Some(&m.scales), 0);
                    enc.set_buffer(2, Some(&m.biases), 0);
                    enc.set_buffer(3, Some(&x_b), 0);
                    enc.set_buffer(4, Some(partial.as_ref().unwrap()), 0);
                    let params = [n_in as i32, n_out as i32];
                    enc.set_bytes(5, 8, params.as_ptr() as *const _);
                    let rows = ((n_out as u32 + ROWS_PER_TG - 1) / ROWS_PER_TG) as NSUInteger;
                    enc.dispatch_thread_groups(
                        MTLSize { width: 4, height: rows, depth: 1 },
                        MTLSize { width: TG_SIZE as NSUInteger, height: 1, depth: 1 },
                    );
                    enc.set_compute_pipeline_state(&p.state_red);
                    enc.set_buffer(0, Some(partial.as_ref().unwrap()), 0);
                    enc.set_buffer(1, Some(&y_b), 0);
                    let n = n_out as i32;
                    enc.set_bytes(2, 4, &n as *const i32 as *const _);
                    enc.dispatch_threads(
                        MTLSize { width: n_out as NSUInteger, height: 1, depth: 1 },
                        MTLSize { width: 256, height: 1, depth: 1 },
                    );
                } else {
                    encode_gemv(enc, &p.state, m, &x_b, &y_b);
                }
            };
            for _ in 0..2 {
                let cb = p.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for mat in &ms {
                    once(enc, mat);
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let reps = 4u32;
            for _ in 0..reps {
                for mat in &ms {
                    once(enc, mat);
                }
            }
            enc.end_encoding();
            let t = std::time::Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let n_call = (reps as usize) * copies;
            let sec = t.elapsed().as_secs_f64() / n_call as f64;
            let gbs = (n_out * n_in) as f64 / 1e9 / sec;
            eprintln!(
                "  {n_out}x{n_in} split4={split4} copies={copies}  {sec:.5}s  {gbs:.1} GB/s"
            );
            gbs
        }
        eprintln!("shape sweep");
        time_shape(17408, 5120, false);
        time_shape(34816, 5120, false);
        time_shape(52224, 5120, false);
        time_shape(10240, 5120, false);
        time_shape(5120, 17408, false);
        time_shape(5120, 17408, true);
        time_shape(17408, 5120, true);
        // split-2: twice the threadgroups, each still walks half of K.
        let n_out = 17408usize;
        let n_in = 5120usize;
        let src = r#"
            #include <metal_stdlib>
            using namespace metal;
            inline float bf16(ushort u) { return as_type<float>(uint(u) << 16); }
            inline float dot8(uint2 ww, float4 x0, float4 x1) {
                float c0 = float(ww.x & 255u);
                float c1 = float((ww.x >> 8) & 255u);
                float c2 = float((ww.x >> 16) & 255u);
                float c3 = float((ww.x >> 24) & 255u);
                float c4 = float(ww.y & 255u);
                float c5 = float((ww.y >> 8) & 255u);
                float c6 = float((ww.y >> 16) & 255u);
                float c7 = float((ww.y >> 24) & 255u);
                return fma(c0, x0[0], fma(c1, x0[1], fma(c2, x0[2], fma(c3, x0[3],
                    fma(c4, x1[0], fma(c5, x1[1], fma(c6, x1[2], c7 * x1[3])))))));
            }
            kernel void gemv_s2(
                device const uint* packed [[buffer(0)]],
                device const ushort* scales [[buffer(1)]],
                device const ushort* biases [[buffer(2)]],
                device const float* x [[buffer(3)]],
                device float* partial [[buffer(4)]],
                constant int& n_in [[buffer(5)]],
                constant int& n_out [[buffer(6)]],
                uint3 tid [[threadgroup_position_in_grid]],
                uint simd_gid [[simdgroup_index_in_threadgroup]],
                uint simd_lid [[thread_index_in_simdgroup]]
            ) {
                int split = int(tid.x);
                int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
                if (split >= 2 || out_row >= n_out) { return; }
                int k_lo = split * (n_in / 2);
                int k_hi = k_lo + n_in / 2;
                int groups = n_in / 64;
                device const uchar* ws = ((device const uchar*)packed) + out_row * n_in + k_lo + int(simd_lid) * 8;
                device const ushort* sc = scales + out_row * groups + k_lo / 64 + int(simd_lid) / 8;
                device const ushort* bc = biases + out_row * groups + k_lo / 64 + int(simd_lid) / 8;
                device const float* xv = x + k_lo + int(simd_lid) * 8;
                float r0=0, r1=0, r2=0, r3=0;
                for (int k = k_lo; k < k_hi; k += 256) {
                    float4 x0 = *(device const float4*)xv;
                    float4 x1 = *(device const float4*)(xv + 4);
                    float xsum = x0[0]+x0[1]+x0[2]+x0[3]+x1[0]+x1[1]+x1[2]+x1[3];
                    uint2 w0 = *(device const uint2*)(ws);
                    uint2 w1 = *(device const uint2*)(ws + n_in);
                    uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
                    uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
                    r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
                    r1 = fma(bf16(sc[groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[groups]), r1));
                    r2 = fma(bf16(sc[2*groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[2*groups]), r2));
                    r3 = fma(bf16(sc[3*groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[3*groups]), r3));
                    ws += 256; sc += 4; bc += 4; xv += 256;
                }
                r0=simd_sum(r0); r1=simd_sum(r1); r2=simd_sum(r2); r3=simd_sum(r3);
                if (simd_lid == 0u) {
                    device float* dst = partial + split * n_out + out_row;
                    dst[0]=r0; dst[1]=r1; dst[2]=r2; dst[3]=r3;
                }
            }
            kernel void red2(device const float* partial [[buffer(0)]], device float* y [[buffer(1)]],
                             constant int& n_out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
                if (int(i) >= n_out) return;
                y[i] = partial[i] + partial[n_out + i];
            }
        "#;
        let p = pipe().unwrap();
        let lib = p.device.new_library_with_source(src, &metal::CompileOptions::new()).unwrap();
        let st = p.device.new_compute_pipeline_state_with_function(&lib.get_function("gemv_s2", None).unwrap()).unwrap();
        let red = p.device.new_compute_pipeline_state_with_function(&lib.get_function("red2", None).unwrap()).unwrap();
        let packed = vec![0x01020304u32; n_out * (n_in / 4)];
        let scales = vec![0.02f32; n_out * (n_in / 64)];
        let biases = vec![0.0f32; n_out * (n_in / 64)];
        let x = vec![0.01f32; n_in];
        let mats: Vec<_> = (0..4)
            .map(|c| {
                let pk = vec![0x01020304u32.wrapping_add(c); n_out * (n_in / 4)];
                QuantMatrix::upload(&pk, &scales, &biases, n_out, n_in).unwrap()
            })
            .collect();
        let _ = packed;
        let x_b = p.device.new_buffer_with_data(x.as_ptr() as *const _, (n_in * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let y_b = p.device.new_buffer((n_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let partial = p.device.new_buffer((n_out * 8) as NSUInteger, MTLResourceOptions::StorageModePrivate | MTLResourceOptions::HazardTrackingModeUntracked);
        let run = || {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for m in &mats {
                enc.set_compute_pipeline_state(&st);
                enc.set_buffer(0, Some(&m.packed), 0);
                enc.set_buffer(1, Some(&m.scales), 0);
                enc.set_buffer(2, Some(&m.biases), 0);
                enc.set_buffer(3, Some(&x_b), 0);
                enc.set_buffer(4, Some(&partial), 0);
                let ni = n_in as i32;
                let no = n_out as i32;
                enc.set_bytes(5, 4, &ni as *const i32 as *const _);
                enc.set_bytes(6, 4, &no as *const i32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize { width: 2, height: (n_out as NSUInteger) / 8, depth: 1 },
                    MTLSize { width: 64, height: 1, depth: 1 },
                );
                enc.set_compute_pipeline_state(&red);
                enc.set_buffer(0, Some(&partial), 0);
                enc.set_buffer(1, Some(&y_b), 0);
                enc.set_bytes(2, 4, &no as *const i32 as *const _);
                enc.dispatch_threads(
                    MTLSize { width: n_out as NSUInteger, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        run();
        let t = std::time::Instant::now();
        run();
        let sec = t.elapsed().as_secs_f64() / 4.0;
        eprintln!("split2 mlp {sec:.5}s  {:.1} GB/s", (n_out * n_in) as f64 / 1e9 / sec);
    }

    #[test]
    fn gemv_pipeline_variants() {
        let variants = [
            ("pipe", r#"
                #include <metal_stdlib>
                using namespace metal;
                inline float bf16(ushort u) { return as_type<float>(uint(u) << 16); }
                inline float dot8(uint2 ww, float4 x0, float4 x1) {
                    float c0 = float(ww.x & 255u);
                    float c1 = float((ww.x >> 8) & 255u);
                    float c2 = float((ww.x >> 16) & 255u);
                    float c3 = float((ww.x >> 24) & 255u);
                    float c4 = float(ww.y & 255u);
                    float c5 = float((ww.y >> 8) & 255u);
                    float c6 = float((ww.y >> 16) & 255u);
                    float c7 = float((ww.y >> 24) & 255u);
                    return fma(c0, x0[0], fma(c1, x0[1], fma(c2, x0[2], fma(c3, x0[3],
                        fma(c4, x1[0], fma(c5, x1[1], fma(c6, x1[2], c7 * x1[3])))))));
                }
                kernel void gemv(
                    device const uint* packed [[buffer(0)]],
                    device const ushort* scales [[buffer(1)]],
                    device const ushort* biases [[buffer(2)]],
                    device const float* x [[buffer(3)]],
                    device float* y [[buffer(4)]],
                    constant int& n_in [[buffer(5)]],
                    constant int& n_out [[buffer(6)]],
                    uint3 tid [[threadgroup_position_in_grid]],
                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                    uint simd_lid [[thread_index_in_simdgroup]]
                ) {
                    int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
                    if (out_row >= n_out) return;
                    int groups = n_in / 64;
                    device const uchar* ws = ((device const uchar*)packed) + out_row * n_in + int(simd_lid) * 8;
                    device const ushort* sc = scales + out_row * groups + int(simd_lid) / 8;
                    device const ushort* bc = biases + out_row * groups + int(simd_lid) / 8;
                    device const float* xv = x + int(simd_lid) * 8;
                    float r0=0, r1=0, r2=0, r3=0;
                    float4 x0 = *(device const float4*)xv;
                    float4 x1 = *(device const float4*)(xv + 4);
                    uint2 w0 = *(device const uint2*)ws;
                    uint2 w1 = *(device const uint2*)(ws + n_in);
                    uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
                    uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
                    float s0 = bf16(sc[0]), b0 = bf16(bc[0]);
                    float s1 = bf16(sc[groups]), b1 = bf16(bc[groups]);
                    float s2 = bf16(sc[2*groups]), b2 = bf16(bc[2*groups]);
                    float s3 = bf16(sc[3*groups]), b3 = bf16(bc[3*groups]);
                    int k = 0;
                    for (; k + 256 < n_in; k += 256) {
                        float4 nx0 = *(device const float4*)(xv + 256);
                        float4 nx1 = *(device const float4*)(xv + 260);
                        uint2 nw0 = *(device const uint2*)(ws + 256);
                        uint2 nw1 = *(device const uint2*)(ws + n_in + 256);
                        uint2 nw2 = *(device const uint2*)(ws + 2 * n_in + 256);
                        uint2 nw3 = *(device const uint2*)(ws + 3 * n_in + 256);
                        float ns0 = bf16(sc[4]), nb0 = bf16(bc[4]);
                        float ns1 = bf16(sc[groups+4]), nb1 = bf16(bc[groups+4]);
                        float ns2 = bf16(sc[2*groups+4]), nb2 = bf16(bc[2*groups+4]);
                        float ns3 = bf16(sc[3*groups+4]), nb3 = bf16(bc[3*groups+4]);
                        float xsum = x0[0]+x0[1]+x0[2]+x0[3]+x1[0]+x1[1]+x1[2]+x1[3];
                        r0 = fma(s0, dot8(w0, x0, x1), fma(xsum, b0, r0));
                        r1 = fma(s1, dot8(w1, x0, x1), fma(xsum, b1, r1));
                        r2 = fma(s2, dot8(w2, x0, x1), fma(xsum, b2, r2));
                        r3 = fma(s3, dot8(w3, x0, x1), fma(xsum, b3, r3));
                        x0=nx0; x1=nx1; w0=nw0; w1=nw1; w2=nw2; w3=nw3;
                        s0=ns0; s1=ns1; s2=ns2; s3=ns3; b0=nb0; b1=nb1; b2=nb2; b3=nb3;
                        xv += 256; ws += 256; sc += 4; bc += 4;
                    }
                    float xsum = x0[0]+x0[1]+x0[2]+x0[3]+x1[0]+x1[1]+x1[2]+x1[3];
                    r0 = fma(s0, dot8(w0, x0, x1), fma(xsum, b0, r0));
                    r1 = fma(s1, dot8(w1, x0, x1), fma(xsum, b1, r1));
                    r2 = fma(s2, dot8(w2, x0, x1), fma(xsum, b2, r2));
                    r3 = fma(s3, dot8(w3, x0, x1), fma(xsum, b3, r3));
                    r0=simd_sum(r0); r1=simd_sum(r1); r2=simd_sum(r2); r3=simd_sum(r3);
                    if (simd_lid == 0u) { y[out_row]=r0; y[out_row+1]=r1; y[out_row+2]=r2; y[out_row+3]=r3; }
                }
            "#),
            ("ilp2", r#"
                #include <metal_stdlib>
                using namespace metal;
                inline float bf16(ushort u) { return as_type<float>(uint(u) << 16); }
                inline float dot8(uint2 ww, float4 x0, float4 x1) {
                    float c0 = float(ww.x & 255u);
                    float c1 = float((ww.x >> 8) & 255u);
                    float c2 = float((ww.x >> 16) & 255u);
                    float c3 = float((ww.x >> 24) & 255u);
                    float c4 = float(ww.y & 255u);
                    float c5 = float((ww.y >> 8) & 255u);
                    float c6 = float((ww.y >> 16) & 255u);
                    float c7 = float((ww.y >> 24) & 255u);
                    return fma(c0, x0[0], fma(c1, x0[1], fma(c2, x0[2], fma(c3, x0[3],
                        fma(c4, x1[0], fma(c5, x1[1], fma(c6, x1[2], c7 * x1[3])))))));
                }
                kernel void gemv(
                    device const uint* packed [[buffer(0)]],
                    device const ushort* scales [[buffer(1)]],
                    device const ushort* biases [[buffer(2)]],
                    device const float* x [[buffer(3)]],
                    device float* y [[buffer(4)]],
                    constant int& n_in [[buffer(5)]],
                    constant int& n_out [[buffer(6)]],
                    uint3 tid [[threadgroup_position_in_grid]],
                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                    uint simd_lid [[thread_index_in_simdgroup]]
                ) {
                    int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
                    if (out_row >= n_out) return;
                    int groups = n_in / 64;
                    device const uchar* ws = ((device const uchar*)packed) + out_row * n_in + int(simd_lid) * 8;
                    device const ushort* sc = scales + out_row * groups + int(simd_lid) / 8;
                    device const ushort* bc = biases + out_row * groups + int(simd_lid) / 8;
                    device const float* xv = x + int(simd_lid) * 8;
                    float a0=0, a1=0, a2=0, a3=0, b0=0, b1=0, b2=0, b3=0;
                    for (int k = 0; k < n_in; k += 512) {
                        float4 x0 = *(device const float4*)xv;
                        float4 x1 = *(device const float4*)(xv + 4);
                        float4 y0 = *(device const float4*)(xv + 256);
                        float4 y1 = *(device const float4*)(xv + 260);
                        float xs = x0[0]+x0[1]+x0[2]+x0[3]+x1[0]+x1[1]+x1[2]+x1[3];
                        float ys = y0[0]+y0[1]+y0[2]+y0[3]+y1[0]+y1[1]+y1[2]+y1[3];
                        uint2 w0 = *(device const uint2*)ws;
                        uint2 w1 = *(device const uint2*)(ws + n_in);
                        uint2 w2 = *(device const uint2*)(ws + 2*n_in);
                        uint2 w3 = *(device const uint2*)(ws + 3*n_in);
                        uint2 z0 = *(device const uint2*)(ws + 256);
                        uint2 z1 = *(device const uint2*)(ws + n_in + 256);
                        uint2 z2 = *(device const uint2*)(ws + 2*n_in + 256);
                        uint2 z3 = *(device const uint2*)(ws + 3*n_in + 256);
                        a0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xs, bf16(bc[0]), a0));
                        a1 = fma(bf16(sc[groups]), dot8(w1, x0, x1), fma(xs, bf16(bc[groups]), a1));
                        a2 = fma(bf16(sc[2*groups]), dot8(w2, x0, x1), fma(xs, bf16(bc[2*groups]), a2));
                        a3 = fma(bf16(sc[3*groups]), dot8(w3, x0, x1), fma(xs, bf16(bc[3*groups]), a3));
                        b0 = fma(bf16(sc[4]), dot8(z0, y0, y1), fma(ys, bf16(bc[4]), b0));
                        b1 = fma(bf16(sc[groups+4]), dot8(z1, y0, y1), fma(ys, bf16(bc[groups+4]), b1));
                        b2 = fma(bf16(sc[2*groups+4]), dot8(z2, y0, y1), fma(ys, bf16(bc[2*groups+4]), b2));
                        b3 = fma(bf16(sc[3*groups+4]), dot8(z3, y0, y1), fma(ys, bf16(bc[3*groups+4]), b3));
                        xv += 512; ws += 512; sc += 8; bc += 8;
                    }
                    float r0=simd_sum(a0+b0), r1=simd_sum(a1+b1), r2=simd_sum(a2+b2), r3=simd_sum(a3+b3);
                    if (simd_lid == 0u) { y[out_row]=r0; y[out_row+1]=r1; y[out_row+2]=r2; y[out_row+3]=r3; }
                }
            "#),
        ];
        let n_out = 17408usize;
        let n_in = 5120usize;
        let x = vec![0.01f32; n_in];
        let scales = vec![0.02f32; n_out * (n_in / 64)];
        let biases = vec![0.0f32; n_out * (n_in / 64)];
        let mats: Vec<_> = (0..4)
            .map(|c| {
                let pk = vec![0x01020304u32.wrapping_add(c); n_out * (n_in / 4)];
                QuantMatrix::upload(&pk, &scales, &biases, n_out, n_in).unwrap()
            })
            .collect();
        let p = pipe().unwrap();
        let x_b = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (n_in * 4) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        let y_b = p.device.new_buffer((n_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        let base = || {
            let cb = p.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for m in &mats {
                encode_gemv(enc, &p.state, m, &x_b, &y_b);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        base();
        let t = std::time::Instant::now();
        base();
        let sec = t.elapsed().as_secs_f64() / 4.0;
        eprintln!("baseline {sec:.5}s  {:.1} GB/s", (n_out * n_in) as f64 / 1e9 / sec);
        for (name, src) in variants {
            let lib = match p.device.new_library_with_source(src, &metal::CompileOptions::new()) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{name} compile fail {e}");
                    continue;
                }
            };
            let st = p
                .device
                .new_compute_pipeline_state_with_function(&lib.get_function("gemv", None).unwrap())
                .unwrap();
            let run = || {
                let cb = p.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for m in &mats {
                    enc.set_compute_pipeline_state(&st);
                    enc.set_buffer(0, Some(&m.packed), 0);
                    enc.set_buffer(1, Some(&m.scales), 0);
                    enc.set_buffer(2, Some(&m.biases), 0);
                    enc.set_buffer(3, Some(&x_b), 0);
                    enc.set_buffer(4, Some(&y_b), 0);
                    let ni = n_in as i32;
                    let no = n_out as i32;
                    enc.set_bytes(5, 4, &ni as *const i32 as *const _);
                    enc.set_bytes(6, 4, &no as *const i32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize { width: 1, height: (n_out / 8) as NSUInteger, depth: 1 },
                        MTLSize { width: 64, height: 1, depth: 1 },
                    );
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            run();
            let t = std::time::Instant::now();
            run();
            let sec = t.elapsed().as_secs_f64() / 4.0;
            let got = read_f32(&y_b, 4);
            let expect = mats[3].gemv(&x).unwrap();
            let err = (got[0] - expect[0]).abs().max((got[1] - expect[1]).abs());
            eprintln!(
                "{name} {sec:.5}s  {:.1} GB/s  err {err}",
                (n_out * n_in) as f64 / 1e9 / sec
            );
        }
    }

    #[test]
    fn gemv_occupancy_sweep() {
        fn shader(nsg: i32, rows: i32) -> String {
            format!(r#"
                #include <metal_stdlib>
                using namespace metal;
                constant int NSG = {nsg};
                constant int ROWS = {rows};
                inline float bf16(ushort u) {{ return as_type<float>(uint(u) << 16); }}
                inline float dot8(uint2 ww, float4 x0, float4 x1) {{
                    float c0 = float(ww.x & 255u);
                    float c1 = float((ww.x >> 8) & 255u);
                    float c2 = float((ww.x >> 16) & 255u);
                    float c3 = float((ww.x >> 24) & 255u);
                    float c4 = float(ww.y & 255u);
                    float c5 = float((ww.y >> 8) & 255u);
                    float c6 = float((ww.y >> 16) & 255u);
                    float c7 = float((ww.y >> 24) & 255u);
                    return fma(c0, x0[0], fma(c1, x0[1], fma(c2, x0[2], fma(c3, x0[3],
                        fma(c4, x1[0], fma(c5, x1[1], fma(c6, x1[2], c7 * x1[3])))))));
                }}
                kernel void gemv(
                    device const uint* packed [[buffer(0)]],
                    device const ushort* scales [[buffer(1)]],
                    device const ushort* biases [[buffer(2)]],
                    device const float* x [[buffer(3)]],
                    device float* y [[buffer(4)]],
                    constant int& n_in [[buffer(5)]],
                    constant int& n_out [[buffer(6)]],
                    uint3 tid [[threadgroup_position_in_grid]],
                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                    uint simd_lid [[thread_index_in_simdgroup]]
                ) {{
                    if (int(simd_gid) >= NSG) return;
                    int out_row = int(tid.y) * (NSG * ROWS) + int(simd_gid) * ROWS;
                    if (out_row >= n_out) return;
                    int groups = n_in / 64;
                    device const uchar* ws = ((device const uchar*)packed) + out_row * n_in + int(simd_lid) * 8;
                    device const ushort* sc = scales + out_row * groups + int(simd_lid) / 8;
                    device const ushort* bc = biases + out_row * groups + int(simd_lid) / 8;
                    device const float* xv = x + int(simd_lid) * 8;
                    float r0=0, r1=0, r2=0, r3=0;
                    for (int k = 0; k < n_in; k += 256) {{
                        float4 x0 = *(device const float4*)xv;
                        float4 x1 = *(device const float4*)(xv + 4);
                        float xsum = x0[0]+x0[1]+x0[2]+x0[3]+x1[0]+x1[1]+x1[2]+x1[3];
                        uint2 w0 = *(device const uint2*)(ws);
                        r0 = fma(bf16(sc[0]), dot8(w0, x0, x1), fma(xsum, bf16(bc[0]), r0));
                        if (ROWS > 1) {{
                            uint2 w1 = *(device const uint2*)(ws + n_in);
                            r1 = fma(bf16(sc[groups]), dot8(w1, x0, x1), fma(xsum, bf16(bc[groups]), r1));
                        }}
                        if (ROWS > 2) {{
                            uint2 w2 = *(device const uint2*)(ws + 2 * n_in);
                            r2 = fma(bf16(sc[2*groups]), dot8(w2, x0, x1), fma(xsum, bf16(bc[2*groups]), r2));
                        }}
                        if (ROWS > 3) {{
                            uint2 w3 = *(device const uint2*)(ws + 3 * n_in);
                            r3 = fma(bf16(sc[3*groups]), dot8(w3, x0, x1), fma(xsum, bf16(bc[3*groups]), r3));
                        }}
                        ws += 256; sc += 4; bc += 4; xv += 256;
                    }}
                    r0=simd_sum(r0); r1=simd_sum(r1); r2=simd_sum(r2); r3=simd_sum(r3);
                    if (simd_lid == 0u) {{
                        y[out_row]=r0;
                        if (ROWS > 1) y[out_row+1]=r1;
                        if (ROWS > 2) y[out_row+2]=r2;
                        if (ROWS > 3) y[out_row+3]=r3;
                    }}
                }}
            "#)
        }
        let n_out = 17408usize;
        let n_in = 5120usize;
        let x = vec![0.01f32; n_in];
        let scales = vec![0.02f32; n_out * (n_in / 64)];
        let biases = vec![0.0f32; n_out * (n_in / 64)];
        let mats: Vec<_> = (0..4)
            .map(|c| {
                let pk = vec![0x01020304u32.wrapping_add(c); n_out * (n_in / 4)];
                QuantMatrix::upload(&pk, &scales, &biases, n_out, n_in).unwrap()
            })
            .collect();
        let p = pipe().unwrap();
        let x_b = p.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (n_in * 4) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        let y_b = p.device.new_buffer((n_out * 4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        for (nsg, rows) in [(2, 4), (4, 4), (4, 2), (8, 2), (2, 2), (8, 4)] {
            let src = shader(nsg, rows);
            let lib = p.device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
            let st = p
                .device
                .new_compute_pipeline_state_with_function(&lib.get_function("gemv", None).unwrap())
                .unwrap();
            let rows_per_tg = (nsg * rows) as usize;
            let run = || {
                let cb = p.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                for m in &mats {
                    enc.set_compute_pipeline_state(&st);
                    enc.set_buffer(0, Some(&m.packed), 0);
                    enc.set_buffer(1, Some(&m.scales), 0);
                    enc.set_buffer(2, Some(&m.biases), 0);
                    enc.set_buffer(3, Some(&x_b), 0);
                    enc.set_buffer(4, Some(&y_b), 0);
                    let ni = n_in as i32;
                    let no = n_out as i32;
                    enc.set_bytes(5, 4, &ni as *const i32 as *const _);
                    enc.set_bytes(6, 4, &no as *const i32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize {
                            width: 1,
                            height: (n_out / rows_per_tg) as NSUInteger,
                            depth: 1,
                        },
                        MTLSize {
                            width: (nsg as u64) * 32,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            run();
            let t = std::time::Instant::now();
            run();
            let sec = t.elapsed().as_secs_f64() / 4.0;
            eprintln!(
                "nsg={nsg} rows={rows} {sec:.5}s  {:.1} GB/s",
                (n_out * n_in) as f64 / 1e9 / sec
            );
        }
    }

    #[test]
    fn kernel_launch_overhead() {
        let p = pipe().unwrap();
        let src = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void touch(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) {
                if (i == 0u) { y[0] += 1.0f; }
            }
        "#;
        let lib = p.device.new_library_with_source(src, &metal::CompileOptions::new()).unwrap();
        let f = lib.get_function("touch", None).unwrap();
        let state = p.device.new_compute_pipeline_state_with_function(&f).unwrap();
        let y = p.device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let n = 800u32;
        let cb = p.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let tg = MTLSize { width: 32, height: 1, depth: 1 };
        let grid = MTLSize { width: 1, height: 1, depth: 1 };
        for _ in 0..n {
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(&y), 0);
            enc.dispatch_thread_groups(grid, tg);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let us = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        eprintln!("tiny kernel launch {us:.2} us  ({n} dispatches)");
    }
}
