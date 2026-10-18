// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Fused Metal decode ops for Qwen3.5: RMSNorm, SwiGLU mid, residual, argmax,
//! causal-conv step, Gated Delta, and a single command-buffer linear layer
//! (input norm → projections → conv → delta → out → SwiGLU) so decode does
//! not bounce to the CPU between kernels.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    NSUInteger,
};

use crate::qwen3_gemv::{self, QuantMatrix};

const OPS_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    // 256 threads, hidden=5120: five aligned float4s per lane, one simd fold.
    threadgroup float part[8];
    float local = 0.0f;
    if (n == 5120u && lsize == 256u) {
        for (uint blk = 0u; blk < 5u; blk++) {
            uint base = blk * 1024u + lid * 4u;
            float4 v = *(device const float4*)(x + base);
            local += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        }
    } else {
        for (uint i = lid; i < n; i += lsize) {
            float v = x[i];
            local += v * v;
        }
    }
    float s = simd_sum(local);
    if ((lid & 31u) == 0u) { part[lid >> 5u] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((part[0] + part[1] + part[2] + part[3] + part[4] + part[5] + part[6] + part[7]) / float(n) + 1e-6f);
    if (n == 5120u && lsize == 256u) {
        for (uint blk = 0u; blk < 5u; blk++) {
            uint base = blk * 1024u + lid * 4u;
            float4 v = *(device const float4*)(x + base);
            float4 ww = *(device const float4*)(w + base);
            float4 o = float4(v[0] * inv * ww[0], v[1] * inv * ww[1], v[2] * inv * ww[2], v[3] * inv * ww[3]);
            *(device float4*)(y + base) = o;
        }
    } else {
        for (uint i = lid; i < n; i += lsize) {
            y[i] = x[i] * inv * w[i];
        }
    }
}

kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* mid [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    float g = gate[i];
    mid[i] = (g / (1.0f + exp(-g))) * up[i];
}

kernel void add_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    a[i] += b[i];
}

kernel void add2(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    out[i] = a[i] + b[i];
}

kernel void argmax_f32(
    device const float* x [[buffer(0)]],
    device uint* best_i [[buffer(1)]],
    device float* best_v [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]]
) {
    threadgroup float tv[256];
    threadgroup uint ti[256];
    uint start = tgid * lsize + lid;
    uint stride = ntg * lsize;
    float mv = -INFINITY;
    uint mi = 0u;
    for (uint i = start; i < n; i += stride) {
        float v = x[i];
        if (v > mv) { mv = v; mi = i; }
    }
    tv[lid] = mv;
    ti[lid] = mi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = lsize >> 1u; s > 0u; s >>= 1u) {
        if (lid < s && tv[lid + s] > tv[lid]) {
            tv[lid] = tv[lid + s];
            ti[lid] = ti[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        best_i[tgid] = ti[0];
        best_v[tgid] = tv[0];
    }
}

// Depthwise causal conv, decode step (ksize=4, hist=3): y = silu(W*window), then shift state.
kernel void causal_conv1d_step(
    device const float* x [[buffer(0)]],
    device float* state [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& ksize [[buffer(5)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= channels) { return; }
    uint hist = ksize - 1u;
    float acc = 0.0f;
    for (uint k = 0u; k < hist; k++) {
        acc += state[c * hist + k] * weight[c * ksize + k];
    }
    acc += x[c] * weight[c * ksize + hist];
    y[c] = acc / (1.0f + exp(-acc));
    for (uint k = 0u; k + 1u < hist; k++) {
        state[c * hist + k] = state[c * hist + k + 1u];
    }
    if (hist > 0u) {
        state[c * hist + hist - 1u] = x[c];
    }
}

inline float softplus_f(float x) {
    if (x > 20.0f) { return x; }
    if (x < -20.0f) { return exp(x); }
    return log(1.0f + exp(x));
}

struct GatedDeltaParams {
    uint k_heads;
    uint v_heads;
    uint k_dim;
    uint v_dim;
    uint repeat;
    uint key_dim;
    uint value_dim;
};

// One threadgroup per value head. threads_per_tg == v_dim (≤256).
kernel void gated_delta_step(
    device const float* conv [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device const float* z [[buffer(3)]],
    device const float* a_log [[buffer(4)]],
    device const float* dt_bias [[buffer(5)]],
    device const float* norm [[buffer(6)]],
    device float* state [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant GatedDeltaParams& p [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    if (h >= p.v_heads) { return; }
    threadgroup float q_tg[256];
    threadgroup float k_tg[256];
    threadgroup float red[256];

    uint kh = h / p.repeat;
    if (lid < p.k_dim) {
        q_tg[lid] = conv[kh * p.k_dim + lid];
        k_tg[lid] = conv[p.key_dim + kh * p.k_dim + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qss = 0.0f;
    float kss = 0.0f;
    if (lid < p.k_dim) {
        float qv = q_tg[lid];
        float kv = k_tg[lid];
        qss = qv * qv;
        kss = kv * kv;
    }
    qss = simd_sum(qss);
    kss = simd_sum(kss);
    // 128 threads = 4 simdgroups. Fold those four partials.
    if ((lid & 31u) == 0u) {
        red[lid >> 5u] = qss;
        red[4u + (lid >> 5u)] = kss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qinv = rsqrt(red[0] + red[1] + red[2] + red[3] + 1e-6f) / sqrt(float(p.k_dim));
    float kinv = rsqrt(red[4] + red[5] + red[6] + red[7] + 1e-6f);
    if (lid < p.k_dim) {
        q_tg[lid] *= qinv;
        k_tg[lid] *= kinv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float g = -exp(a_log[h]) * softplus_f(a[h] + dt_bias[h]);
    float decay = exp(g);
    float beta = 1.0f / (1.0f + exp(-b[h]));

    device float* st = state + h * p.k_dim * p.v_dim;
    device const float* v = conv + 2u * p.key_dim + h * p.v_dim;
    uint j = lid;
    if (j >= p.v_dim) { return; }

    float mem = 0.0f;
    for (uint i = 0u; i + 7u < p.k_dim; i += 8u) {
        uint stride = p.v_dim;
        uint b = i * stride + j;
        float s0 = st[b] * decay;
        float s1 = st[b + stride] * decay;
        float s2 = st[b + 2u * stride] * decay;
        float s3 = st[b + 3u * stride] * decay;
        float s4 = st[b + 4u * stride] * decay;
        float s5 = st[b + 5u * stride] * decay;
        float s6 = st[b + 6u * stride] * decay;
        float s7 = st[b + 7u * stride] * decay;
        st[b] = s0; st[b + stride] = s1; st[b + 2u * stride] = s2; st[b + 3u * stride] = s3;
        st[b + 4u * stride] = s4; st[b + 5u * stride] = s5; st[b + 6u * stride] = s6; st[b + 7u * stride] = s7;
        mem += s0 * k_tg[i] + s1 * k_tg[i + 1u] + s2 * k_tg[i + 2u] + s3 * k_tg[i + 3u]
            + s4 * k_tg[i + 4u] + s5 * k_tg[i + 5u] + s6 * k_tg[i + 6u] + s7 * k_tg[i + 7u];
    }
    float delta = (v[j] - mem) * beta;
    float o = 0.0f;
    for (uint i = 0u; i + 7u < p.k_dim; i += 8u) {
        uint stride = p.v_dim;
        uint b = i * stride + j;
        float s0 = st[b] + k_tg[i] * delta;
        float s1 = st[b + stride] + k_tg[i + 1u] * delta;
        float s2 = st[b + 2u * stride] + k_tg[i + 2u] * delta;
        float s3 = st[b + 3u * stride] + k_tg[i + 3u] * delta;
        float s4 = st[b + 4u * stride] + k_tg[i + 4u] * delta;
        float s5 = st[b + 5u * stride] + k_tg[i + 5u] * delta;
        float s6 = st[b + 6u * stride] + k_tg[i + 6u] * delta;
        float s7 = st[b + 7u * stride] + k_tg[i + 7u] * delta;
        st[b] = s0; st[b + stride] = s1; st[b + 2u * stride] = s2; st[b + 3u * stride] = s3;
        st[b + 4u * stride] = s4; st[b + 5u * stride] = s5; st[b + 6u * stride] = s6; st[b + 7u * stride] = s7;
        o += s0 * q_tg[i] + s1 * q_tg[i + 1u] + s2 * q_tg[i + 2u] + s3 * q_tg[i + 3u]
            + s4 * q_tg[i + 4u] + s5 * q_tg[i + 5u] + s6 * q_tg[i + 6u] + s7 * q_tg[i + 7u];
    }
    // gated RMSNorm over the head (reduce across v_dim). 4 simdgroups.
    float oss = simd_sum(o * o);
    if ((lid & 31u) == 0u) { red[lid >> 5u] = oss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((red[0] + red[1] + red[2] + red[3]) / float(p.v_dim) + 1e-6f);
    float gate = z[h * p.v_dim + j];
    float silu_g = gate / (1.0f + exp(-gate));
    out[h * p.v_dim + j] = o * inv * norm[j] * silu_g;
}

struct FullMeta {
    uint heads;
    uint kv_heads;
    uint head_dim;
    uint rotary_dim;
    uint groups;
    uint kv_len;
    uint kv_cap;
    float scale_inv;
};

// Split interleaved q/gate, RMSNorm q, RoPE on first rotary_dim.
kernel void full_q_prep(
    device const float* q_raw [[buffer(0)]],
    device const float* q_norm [[buffer(1)]],
    device const float* cos [[buffer(2)]],
    device const float* sin [[buffer(3)]],
    device float* q [[buffer(4)]],
    device float* gate [[buffer(5)]],
    constant FullMeta& p [[buffer(6)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    if (h >= p.heads) { return; }
    threadgroup float part[8];
    uint span = p.head_dim * 2u;
    device const float* src = q_raw + h * span;
    float ss = 0.0f;
    for (uint i = lid; i < p.head_dim; i += lsize) {
        float v = src[i];
        q[h * p.head_dim + i] = v;
        gate[h * p.head_dim + i] = src[p.head_dim + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if ((lid & 31u) == 0u) { part[lid >> 5u] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((part[0] + part[1] + part[2] + part[3] + part[4] + part[5] + part[6] + part[7]) / float(p.head_dim) + 1e-6f);
    for (uint i = lid; i < p.head_dim; i += lsize) {
        q[h * p.head_dim + i] *= inv * q_norm[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rd2 = p.rotary_dim / 2u;
    for (uint i = lid; i < rd2; i += lsize) {
        uint base = h * p.head_dim;
        float x0 = q[base + i];
        float x1 = q[base + rd2 + i];
        float c0 = cos[i], c1 = cos[rd2 + i];
        float s0 = sin[i], s1 = sin[rd2 + i];
        q[base + i] = x0 * c0 + (-x1) * s0;
        q[base + rd2 + i] = x1 * c1 + x0 * s1;
    }
}

kernel void full_kv_prep(
    device const float* k_raw [[buffer(0)]],
    device const float* v_raw [[buffer(1)]],
    device const float* k_norm [[buffer(2)]],
    device const float* cos [[buffer(3)]],
    device const float* sin [[buffer(4)]],
    device float* k_cache [[buffer(5)]],
    device float* v_cache [[buffer(6)]],
    constant FullMeta& p [[buffer(7)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    if (h >= p.kv_heads) { return; }
    threadgroup float part[8];
    uint slot = p.kv_len - 1u;
    device float* kdst = k_cache + (slot * p.kv_heads + h) * p.head_dim;
    device float* vdst = v_cache + (slot * p.kv_heads + h) * p.head_dim;
    device const float* ksrc = k_raw + h * p.head_dim;
    device const float* vsrc = v_raw + h * p.head_dim;
    float ss = 0.0f;
    for (uint i = lid; i < p.head_dim; i += lsize) {
        float v = ksrc[i];
        kdst[i] = v;
        vdst[i] = vsrc[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if ((lid & 31u) == 0u) { part[lid >> 5u] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((part[0] + part[1] + part[2] + part[3] + part[4] + part[5] + part[6] + part[7]) / float(p.head_dim) + 1e-6f);
    for (uint i = lid; i < p.head_dim; i += lsize) {
        kdst[i] *= inv * k_norm[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rd2 = p.rotary_dim / 2u;
    for (uint i = lid; i < rd2; i += lsize) {
        float x0 = kdst[i];
        float x1 = kdst[rd2 + i];
        float c0 = cos[i], c1 = cos[rd2 + i];
        float s0 = sin[i], s1 = sin[rd2 + i];
        kdst[i] = x0 * c0 + (-x1) * s0;
        kdst[rd2 + i] = x1 * c1 + x0 * s1;
    }
}

// One simdgroup (32 threads) per head. Each lane owns head_dim/32 dims and
// the score is a simd_sum. Online softmax, one pass over KV.
kernel void full_attn_heads(
    device const float* q [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* k_cache [[buffer(2)]],
    device const float* v_cache [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant FullMeta& p [[buffer(5)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_simdgroup]]
) {
    if (h >= p.heads) { return; }
    uint kv_h = h / p.groups;
    device const float* qh = q + h * p.head_dim;
    // Consecutive 8 dims per lane so K/V loads are float4 and coalesce.
    uint base = lid * 8u;
    float4 q0 = *(device const float4*)(qh + base);
    float4 q1 = *(device const float4*)(qh + base + 4u);
    float4 a0 = float4(0.0f);
    float4 a1 = float4(0.0f);
    float max_s = -INFINITY;
    float sum = 0.0f;
    for (uint pos = 0u; pos < p.kv_len; pos++) {
        device const float* kh = k_cache + (pos * p.kv_heads + kv_h) * p.head_dim + base;
        device const float* vh = v_cache + (pos * p.kv_heads + kv_h) * p.head_dim + base;
        float4 k0 = *(device const float4*)kh;
        float4 k1 = *(device const float4*)(kh + 4);
        float4 v0 = *(device const float4*)vh;
        float4 v1 = *(device const float4*)(vh + 4);
        float partial = q0[0]*k0[0] + q0[1]*k0[1] + q0[2]*k0[2] + q0[3]*k0[3]
            + q1[0]*k1[0] + q1[1]*k1[1] + q1[2]*k1[2] + q1[3]*k1[3];
        float dot = simd_sum(partial) * p.scale_inv;
        float m = max(max_s, dot);
        float alpha = exp(max_s - m);
        float e = exp(dot - m);
        a0 = a0 * alpha + v0 * e;
        a1 = a1 * alpha + v1 * e;
        sum = sum * alpha + e;
        max_s = m;
    }
    float inv = 1.0f / sum;
    device float* oh = out + h * p.head_dim + base;
    device const float* gh = gate + h * p.head_dim + base;
    float4 g0 = *(device const float4*)gh;
    float4 g1 = *(device const float4*)(gh + 4);
    float4 o0 = float4(
        a0[0] * inv / (1.0f + exp(-g0[0])),
        a0[1] * inv / (1.0f + exp(-g0[1])),
        a0[2] * inv / (1.0f + exp(-g0[2])),
        a0[3] * inv / (1.0f + exp(-g0[3])));
    float4 o1 = float4(
        a1[0] * inv / (1.0f + exp(-g1[0])),
        a1[1] * inv / (1.0f + exp(-g1[1])),
        a1[2] * inv / (1.0f + exp(-g1[2])),
        a1[3] * inv / (1.0f + exp(-g1[3])));
    *(device float4*)oh = o0;
    *(device float4*)(oh + 4) = o1;
}
"#;

struct Ops {
    device: Device,
    queue: metal::CommandQueue,
    rmsnorm: ComputePipelineState,
    silu_mul: ComputePipelineState,
    add_inplace: ComputePipelineState,
    add2: ComputePipelineState,
    argmax: ComputePipelineState,
    causal_conv: ComputePipelineState,
    gated_delta: ComputePipelineState,
    full_q_prep: ComputePipelineState,
    full_kv_prep: ComputePipelineState,
    full_attn: ComputePipelineState,
}

fn ops() -> Result<&'static Ops> {
    static CELL: OnceLock<Ops> = OnceLock::new();
    if let Some(o) = CELL.get() {
        return Ok(o);
    }
    let device = Device::system_default().context("no Metal device")?;
    let lib = device
        .new_library_with_source(OPS_SRC, &metal::CompileOptions::new())
        .map_err(|e| anyhow::anyhow!("metal fuse compile: {e}"))?;
    let pipe = |name: &str| -> Result<ComputePipelineState> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| anyhow::anyhow!("metal function {name}: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| anyhow::anyhow!("metal pipeline {name}: {e}"))
    };
    let o = Ops {
        queue: device.new_command_queue(),
        rmsnorm: pipe("rmsnorm")?,
        silu_mul: pipe("silu_mul")?,
        add_inplace: pipe("add_inplace")?,
        add2: pipe("add2")?,
        argmax: pipe("argmax_f32")?,
        causal_conv: pipe("causal_conv1d_step")?,
        gated_delta: pipe("gated_delta_step")?,
        full_q_prep: pipe("full_q_prep")?,
        full_kv_prep: pipe("full_kv_prep")?,
        full_attn: pipe("full_attn_heads")?,
        device,
    };
    let _ = CELL.set(o);
    Ok(CELL.get().unwrap())
}

fn buf_f32(data: &[f32]) -> Result<Buffer> {
    buf_f32_pub(data)
}

pub fn buf_f32_pub(data: &[f32]) -> Result<Buffer> {
    let o = ops()?;
    Ok(o.device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    ))
}

/// Process-wide final-norm weight buffer (one model per process in the example).
pub fn cached_final_norm(w: &[f32]) -> Result<&'static Buffer> {
    static CELL: OnceLock<Buffer> = OnceLock::new();
    if let Some(b) = CELL.get() {
        return Ok(b);
    }
    let b = buf_f32_pub(w)?;
    let _ = CELL.set(b);
    Ok(CELL.get().unwrap())
}

fn buf_zeros(n: usize) -> Result<Buffer> {
    let o = ops()?;
    Ok(o.device
        .new_buffer((n * 4) as NSUInteger, MTLResourceOptions::StorageModeShared))
}

fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Persistent GPU buffers for one linear-attention layer's recurrent state.
pub struct GpuRecurrent {
    pub conv: Buffer,
    pub state: Buffer,
    conv_len: usize,
    state_len: usize,
}

impl GpuRecurrent {
    pub fn new(conv: &[f32], state: &[f32]) -> Result<Self> {
        Ok(Self {
            conv: buf_f32(conv)?,
            state: buf_f32(state)?,
            conv_len: conv.len(),
            state_len: state.len(),
        })
    }

    pub fn sync_to_cpu(&self, conv: &mut [f32], state: &mut [f32]) {
        if conv.len() == self.conv_len {
            conv.copy_from_slice(unsafe {
                std::slice::from_raw_parts(self.conv.contents() as *const f32, self.conv_len)
            });
        }
        if state.len() == self.state_len {
            state.copy_from_slice(unsafe {
                std::slice::from_raw_parts(self.state.contents() as *const f32, self.state_len)
            });
        }
    }
}

/// Constant weights for linear attention, uploaded once.
#[derive(Clone)]
pub struct LinearAttnAux {
    pub conv_w: Buffer,
    pub a_log: Buffer,
    pub dt_bias: Buffer,
    pub norm: Buffer,
}

impl LinearAttnAux {
    pub fn upload(conv_w: &[f32], a_log: &[f32], dt_bias: &[f32], norm: &[f32]) -> Result<Self> {
        Ok(Self {
            conv_w: buf_f32(conv_w)?,
            a_log: buf_f32(a_log)?,
            dt_bias: buf_f32(dt_bias)?,
            norm: buf_f32(norm)?,
        })
    }
}

/// GPU KV cache for one full-attention layer. Layout `[cap, kv_heads, head_dim]`.
pub struct GpuKv {
    pub k: Buffer,
    pub v: Buffer,
    pub len: usize,
    cap: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl GpuKv {
    pub fn new(kv_heads: usize, head_dim: usize, cap: usize) -> Result<Self> {
        let n = cap * kv_heads * head_dim;
        Ok(Self {
            k: buf_zeros(n)?,
            v: buf_zeros(n)?,
            len: 0,
            cap,
            kv_heads,
            head_dim,
        })
    }

    pub fn ensure_cap(&mut self, need: usize) -> Result<()> {
        if need <= self.cap {
            return Ok(());
        }
        let new_cap = need.max(self.cap * 2).max(64);
        let n = new_cap * self.kv_heads * self.head_dim;
        let nk = buf_zeros(n)?;
        let nv = buf_zeros(n)?;
        let old_n = self.len * self.kv_heads * self.head_dim;
        if old_n > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.k.contents() as *const f32,
                    nk.contents() as *mut f32,
                    old_n,
                );
                std::ptr::copy_nonoverlapping(
                    self.v.contents() as *const f32,
                    nv.contents() as *mut f32,
                    old_n,
                );
            }
        }
        self.k = nk;
        self.v = nv;
        self.cap = new_cap;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct FullLayerMeta {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

pub struct FullLayerOwned {
    pub q: QuantMatrix,
    pub k: QuantMatrix,
    pub v: QuantMatrix,
    pub o: QuantMatrix,
    pub gate: QuantMatrix,
    pub up: QuantMatrix,
    pub down: QuantMatrix,
    pub input_norm: Buffer,
    pub post_norm: Buffer,
    pub q_norm: Buffer,
    pub k_norm: Buffer,
    pub meta: FullLayerMeta,
}

#[repr(C)]
struct FullMeta {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    groups: u32,
    kv_len: u32,
    kv_cap: u32,
    scale_inv: f32,
}

#[repr(C)]
struct GatedDeltaParams {
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    repeat: u32,
    key_dim: u32,
    value_dim: u32,
}

fn encode_rmsnorm(enc: &ComputeCommandEncoderRef, x: &Buffer, w: &Buffer, y: &Buffer, n: u32) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.rmsnorm);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(w), 0);
    enc.set_buffer(2, Some(y), 0);
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, tg);
}

fn encode_silu_mul(enc: &ComputeCommandEncoderRef, gate: &Buffer, up: &Buffer, mid: &Buffer, n: u32) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.silu_mul);
    enc.set_buffer(0, Some(gate), 0);
    enc.set_buffer(1, Some(up), 0);
    enc.set_buffer(2, Some(mid), 0);
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: n as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_add_inplace(enc: &ComputeCommandEncoderRef, a: &Buffer, b: &Buffer, n: u32) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.add_inplace);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_bytes(2, 4, &n as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: n as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_add2(enc: &ComputeCommandEncoderRef, a: &Buffer, b: &Buffer, out: &Buffer, n: u32) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.add2);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: n as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_causal_conv(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    state: &Buffer,
    weight: &Buffer,
    y: &Buffer,
    channels: u32,
    kernel: u32,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.causal_conv);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(state), 0);
    enc.set_buffer(2, Some(weight), 0);
    enc.set_buffer(3, Some(y), 0);
    enc.set_bytes(4, 4, &channels as *const u32 as *const _);
    enc.set_bytes(5, 4, &kernel as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize {
            width: channels as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_gated_delta(
    enc: &ComputeCommandEncoderRef,
    conv: &Buffer,
    a: &Buffer,
    b: &Buffer,
    z: &Buffer,
    aux: &LinearAttnAux,
    state: &Buffer,
    out: &Buffer,
    params: &GatedDeltaParams,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.gated_delta);
    enc.set_buffer(0, Some(conv), 0);
    enc.set_buffer(1, Some(a), 0);
    enc.set_buffer(2, Some(b), 0);
    enc.set_buffer(3, Some(z), 0);
    enc.set_buffer(4, Some(&aux.a_log), 0);
    enc.set_buffer(5, Some(&aux.dt_bias), 0);
    enc.set_buffer(6, Some(&aux.norm), 0);
    enc.set_buffer(7, Some(state), 0);
    enc.set_buffer(8, Some(out), 0);
    enc.set_bytes(
        9,
        std::mem::size_of::<GatedDeltaParams>() as NSUInteger,
        params as *const GatedDeltaParams as *const _,
    );
    let tg = MTLSize {
        width: params.v_dim as NSUInteger,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(
        MTLSize {
            width: params.v_heads as NSUInteger,
            height: 1,
            depth: 1,
        },
        tg,
    );
}

/// After attention: `h = x + mixed`, then SwiGLU MLP, residual into `h`.
pub fn fused_residual_mlp(
    x: &[f32],
    mixed: &[f32],
    post_norm: &[f32],
    gate: &QuantMatrix,
    up: &QuantMatrix,
    down: &QuantMatrix,
) -> Result<Vec<f32>> {
    let hidden = x.len();
    anyhow::ensure!(mixed.len() == hidden && post_norm.len() == hidden, "hidden shape");
    anyhow::ensure!(gate.n_in() == hidden && up.n_in() == hidden, "gate/up in");
    anyhow::ensure!(gate.n_out() == up.n_out(), "gate/up out");
    anyhow::ensure!(down.n_in() == gate.n_out() && down.n_out() == hidden, "down shape");

    let o = ops()?;
    let mid_n = gate.n_out();
    let x_b = buf_f32(x)?;
    let mixed_b = buf_f32(mixed)?;
    let h_b = buf_zeros(hidden)?;
    let w_b = buf_f32(post_norm)?;
    let normed = buf_zeros(hidden)?;
    let gate_b = buf_zeros(mid_n)?;
    let up_b = buf_zeros(mid_n)?;
    let mid_b = buf_zeros(mid_n)?;
    let down_b = buf_zeros(hidden)?;

    let cb = o.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let n_h = hidden as u32;
    let n_m = mid_n as u32;
    encode_add2(enc, &x_b, &mixed_b, &h_b, n_h);
    encode_rmsnorm(enc, &h_b, &w_b, &normed, n_h);
    qwen3_gemv::encode_into(enc, gate, &normed, &gate_b)?;
    qwen3_gemv::encode_into(enc, up, &normed, &up_b)?;
    encode_silu_mul(enc, &gate_b, &up_b, &mid_b, n_m);
    qwen3_gemv::encode_into(enc, down, &mid_b, &down_b)?;
    encode_add_inplace(enc, &h_b, &down_b, n_h);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(read_f32(&h_b, hidden))
}

pub struct LinearLayerMats<'a> {
    pub qkv: &'a QuantMatrix,
    pub z: &'a QuantMatrix,
    pub a: &'a QuantMatrix,
    pub b: &'a QuantMatrix,
    pub out: &'a QuantMatrix,
    pub gate: &'a QuantMatrix,
    pub up: &'a QuantMatrix,
    pub down: &'a QuantMatrix,
}

#[derive(Clone, Copy)]
pub struct LinearLayerMeta {
    pub hidden: usize,
    pub conv_dim: usize,
    pub kernel: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
}

/// Owned matrices + aux for one linear layer encode (cloneable Buffer handles).
pub struct LinearLayerOwned {
    pub qkv: QuantMatrix,
    pub z: QuantMatrix,
    pub a: QuantMatrix,
    pub b: QuantMatrix,
    pub out: QuantMatrix,
    pub gate: QuantMatrix,
    pub up: QuantMatrix,
    pub down: QuantMatrix,
    pub aux: LinearAttnAux,
    pub input_norm: Buffer,
    pub post_norm: Buffer,
    pub meta: LinearLayerMeta,
}

fn encode_linear_layer(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    y: &Buffer,
    layer: &LinearLayerOwned,
    gpu: &GpuRecurrent,
    scratch: &LinearScratch,
) -> Result<()> {
    let meta = &layer.meta;
    let h = meta.hidden;
    let key_dim = meta.k_heads * meta.k_dim;
    let value_dim = meta.v_heads * meta.v_dim;
    let repeat = meta.v_heads / meta.k_heads;
    let n_h = h as u32;
    let gd = GatedDeltaParams {
        k_heads: meta.k_heads as u32,
        v_heads: meta.v_heads as u32,
        k_dim: meta.k_dim as u32,
        v_dim: meta.v_dim as u32,
        repeat: repeat as u32,
        key_dim: key_dim as u32,
        value_dim: value_dim as u32,
    };

    encode_rmsnorm(enc, x, &layer.input_norm, &scratch.normed, n_h);
    qwen3_gemv::encode_many(
        enc,
        &scratch.normed,
        &[
            (&layer.qkv, &scratch.qkv),
            (&layer.z, &scratch.z),
            (&layer.a, &scratch.a),
            (&layer.b, &scratch.b),
        ],
    )?;
    encode_causal_conv(
        enc,
        &scratch.qkv,
        &gpu.conv,
        &layer.aux.conv_w,
        &scratch.conv_y,
        meta.conv_dim as u32,
        meta.kernel as u32,
    );
    encode_gated_delta(
        enc,
        &scratch.conv_y,
        &scratch.a,
        &scratch.b,
        &scratch.z,
        &layer.aux,
        &gpu.state,
        &scratch.delta_out,
        &gd,
    );
    qwen3_gemv::encode_residual(enc, &layer.out, &scratch.delta_out, x, y)?;
    encode_rmsnorm(enc, y, &layer.post_norm, &scratch.post_n, n_h);
    qwen3_gemv::encode_swiglu(
        enc,
        &scratch.post_n,
        &layer.gate,
        &layer.up,
        &scratch.gate,
        &scratch.up,
        &scratch.mid,
    )?;
    qwen3_gemv::encode_residual(enc, &layer.down, &scratch.mid, y, y)?;
    Ok(())
}

struct LinearScratch {
    normed: Buffer,
    qkv: Buffer,
    z: Buffer,
    a: Buffer,
    b: Buffer,
    conv_y: Buffer,
    delta_out: Buffer,
    mixed: Buffer,
    post_n: Buffer,
    gate: Buffer,
    up: Buffer,
    mid: Buffer,
    down: Buffer,
    /// Full-attn temps (sized for Qwen3.8-27B decode).
    fq_raw: Buffer,
    fk_raw: Buffer,
    fv_raw: Buffer,
    fq: Buffer,
    fgate: Buffer,
    fattn: Buffer,
    rope_cos: Buffer,
    rope_sin: Buffer,
}

impl LinearScratch {
    fn new(meta: &LinearLayerMeta, mid_n: usize, full: &FullLayerMeta) -> Result<Self> {
        let key_dim = meta.k_heads * meta.k_dim;
        let value_dim = meta.v_heads * meta.v_dim;
        let q_raw_n = full.heads * full.head_dim * 2;
        let kv_n = full.kv_heads * full.head_dim;
        let q_n = full.heads * full.head_dim;
        Ok(Self {
            normed: buf_zeros(meta.hidden)?,
            qkv: buf_zeros(meta.conv_dim)?,
            z: buf_zeros(value_dim)?,
            a: buf_zeros(meta.v_heads)?,
            b: buf_zeros(meta.v_heads)?,
            conv_y: buf_zeros(meta.conv_dim)?,
            delta_out: buf_zeros(value_dim)?,
            mixed: buf_zeros(meta.hidden)?,
            post_n: buf_zeros(meta.hidden)?,
            gate: buf_zeros(mid_n)?,
            up: buf_zeros(mid_n)?,
            mid: buf_zeros(mid_n)?,
            down: buf_zeros(meta.hidden)?,
            fq_raw: buf_zeros(q_raw_n)?,
            fk_raw: buf_zeros(kv_n)?,
            fv_raw: buf_zeros(kv_n)?,
            fq: buf_zeros(q_n)?,
            fgate: buf_zeros(q_n)?,
            fattn: buf_zeros(q_n)?,
            rope_cos: buf_zeros(full.rotary_dim)?,
            rope_sin: buf_zeros(full.rotary_dim)?,
        })
    }
}

fn encode_full_layer(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    y: &Buffer,
    layer: &FullLayerOwned,
    kv: &GpuKv,
    scratch: &LinearScratch,
) -> Result<()> {
    let meta = &layer.meta;
    let h = meta.hidden;
    let n_h = h as u32;
    let mid_n = layer.gate.n_out() as u32;
    let groups = meta.heads / meta.kv_heads;
    let fm = FullMeta {
        heads: meta.heads as u32,
        kv_heads: meta.kv_heads as u32,
        head_dim: meta.head_dim as u32,
        rotary_dim: meta.rotary_dim as u32,
        groups: groups as u32,
        kv_len: kv.len as u32,
        kv_cap: kv.cap as u32,
        scale_inv: (meta.head_dim as f32).sqrt().recip(),
    };
    let o = ops()?;
    let tg = MTLSize {
        width: meta.head_dim as NSUInteger,
        height: 1,
        depth: 1,
    };

    encode_rmsnorm(enc, x, &layer.input_norm, &scratch.normed, n_h);
    qwen3_gemv::encode_many(
        enc,
        &scratch.normed,
        &[
            (&layer.q, &scratch.fq_raw),
            (&layer.k, &scratch.fk_raw),
            (&layer.v, &scratch.fv_raw),
        ],
    )?;

    enc.set_compute_pipeline_state(&o.full_q_prep);
    enc.set_buffer(0, Some(&scratch.fq_raw), 0);
    enc.set_buffer(1, Some(&layer.q_norm), 0);
    enc.set_buffer(2, Some(&scratch.rope_cos), 0);
    enc.set_buffer(3, Some(&scratch.rope_sin), 0);
    enc.set_buffer(4, Some(&scratch.fq), 0);
    enc.set_buffer(5, Some(&scratch.fgate), 0);
    enc.set_bytes(
        6,
        std::mem::size_of::<FullMeta>() as NSUInteger,
        &fm as *const FullMeta as *const _,
    );
    enc.dispatch_thread_groups(
        MTLSize {
            width: meta.heads as NSUInteger,
            height: 1,
            depth: 1,
        },
        tg,
    );

    enc.set_compute_pipeline_state(&o.full_kv_prep);
    enc.set_buffer(0, Some(&scratch.fk_raw), 0);
    enc.set_buffer(1, Some(&scratch.fv_raw), 0);
    enc.set_buffer(2, Some(&layer.k_norm), 0);
    enc.set_buffer(3, Some(&scratch.rope_cos), 0);
    enc.set_buffer(4, Some(&scratch.rope_sin), 0);
    enc.set_buffer(5, Some(&kv.k), 0);
    enc.set_buffer(6, Some(&kv.v), 0);
    enc.set_bytes(
        7,
        std::mem::size_of::<FullMeta>() as NSUInteger,
        &fm as *const FullMeta as *const _,
    );
    enc.dispatch_thread_groups(
        MTLSize {
            width: meta.kv_heads as NSUInteger,
            height: 1,
            depth: 1,
        },
        tg,
    );

    enc.set_compute_pipeline_state(&o.full_attn);
    enc.set_buffer(0, Some(&scratch.fq), 0);
    enc.set_buffer(1, Some(&scratch.fgate), 0);
    enc.set_buffer(2, Some(&kv.k), 0);
    enc.set_buffer(3, Some(&kv.v), 0);
    enc.set_buffer(4, Some(&scratch.fattn), 0);
    enc.set_bytes(
        5,
        std::mem::size_of::<FullMeta>() as NSUInteger,
        &fm as *const FullMeta as *const _,
    );
    enc.dispatch_thread_groups(
        MTLSize {
            width: meta.heads as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    qwen3_gemv::encode_residual(enc, &layer.o, &scratch.fattn, x, y)?;
    encode_rmsnorm(enc, y, &layer.post_norm, &scratch.post_n, n_h);
    qwen3_gemv::encode_swiglu(
        enc,
        &scratch.post_n,
        &layer.gate,
        &layer.up,
        &scratch.gate,
        &scratch.up,
        &scratch.mid,
    )?;
    qwen3_gemv::encode_residual(enc, &layer.down, &scratch.mid, y, y)?;
    let _ = mid_n;
    Ok(())
}

fn bytes(elems: usize) -> NSUInteger {
    (elems * 4) as NSUInteger
}

fn encode_rmsnorm_at(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    x_elems: usize,
    w: &Buffer,
    y: &Buffer,
    y_elems: usize,
    n: u32,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.rmsnorm);
    enc.set_buffer(0, Some(x), bytes(x_elems));
    enc.set_buffer(1, Some(w), 0);
    enc.set_buffer(2, Some(y), bytes(y_elems));
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

fn encode_silu_mul_at(
    enc: &ComputeCommandEncoderRef,
    gate: &Buffer,
    gate_elems: usize,
    up: &Buffer,
    up_elems: usize,
    mid: &Buffer,
    mid_elems: usize,
    n: u32,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.silu_mul);
    enc.set_buffer(0, Some(gate), bytes(gate_elems));
    enc.set_buffer(1, Some(up), bytes(up_elems));
    enc.set_buffer(2, Some(mid), bytes(mid_elems));
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize { width: n as NSUInteger, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

fn encode_add_inplace_at(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_elems: usize,
    b: &Buffer,
    b_elems: usize,
    n: u32,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.add_inplace);
    enc.set_buffer(0, Some(a), bytes(a_elems));
    enc.set_buffer(1, Some(b), bytes(b_elems));
    enc.set_bytes(2, 4, &n as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize { width: n as NSUInteger, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

fn encode_add2_at(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_elems: usize,
    b: &Buffer,
    b_elems: usize,
    out: &Buffer,
    out_elems: usize,
    n: u32,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.add2);
    enc.set_buffer(0, Some(a), bytes(a_elems));
    enc.set_buffer(1, Some(b), bytes(b_elems));
    enc.set_buffer(2, Some(out), bytes(out_elems));
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize { width: n as NSUInteger, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

fn encode_causal_conv_at(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    x_elems: usize,
    state: &Buffer,
    weight: &Buffer,
    y: &Buffer,
    channels: u32,
    kernel: u32,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.causal_conv);
    enc.set_buffer(0, Some(x), bytes(x_elems));
    enc.set_buffer(1, Some(state), 0);
    enc.set_buffer(2, Some(weight), 0);
    enc.set_buffer(3, Some(y), 0);
    enc.set_bytes(4, 4, &channels as *const u32 as *const _);
    enc.set_bytes(5, 4, &kernel as *const u32 as *const _);
    enc.dispatch_threads(
        MTLSize { width: channels as NSUInteger, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

fn encode_gated_delta_at(
    enc: &ComputeCommandEncoderRef,
    conv: &Buffer,
    a: &Buffer,
    a_elems: usize,
    b: &Buffer,
    b_elems: usize,
    z: &Buffer,
    z_elems: usize,
    aux: &LinearAttnAux,
    state: &Buffer,
    out: &Buffer,
    out_elems: usize,
    params: &GatedDeltaParams,
) {
    let o = ops().unwrap();
    enc.set_compute_pipeline_state(&o.gated_delta);
    enc.set_buffer(0, Some(conv), 0);
    enc.set_buffer(1, Some(a), bytes(a_elems));
    enc.set_buffer(2, Some(b), bytes(b_elems));
    enc.set_buffer(3, Some(z), bytes(z_elems));
    enc.set_buffer(4, Some(&aux.a_log), 0);
    enc.set_buffer(5, Some(&aux.dt_bias), 0);
    enc.set_buffer(6, Some(&aux.norm), 0);
    enc.set_buffer(7, Some(state), 0);
    enc.set_buffer(8, Some(out), bytes(out_elems));
    enc.set_bytes(
        9,
        std::mem::size_of::<GatedDeltaParams>() as NSUInteger,
        params as *const GatedDeltaParams as *const _,
    );
    enc.dispatch_thread_groups(
        MTLSize { width: params.v_heads as NSUInteger, height: 1, depth: 1 },
        MTLSize { width: params.v_dim as NSUInteger, height: 1, depth: 1 },
    );
}

fn encode_proj(enc: &ComputeCommandEncoderRef, m: &QuantMatrix, x: &Buffer, y: &Buffer, batch: u32) -> Result<()> {
    if m.n_in() % 256 == 0 {
        qwen3_gemv::encode_gemm4(enc, m, x, y, batch)
    } else {
        for t in 0..batch as usize {
            qwen3_gemv::encode_into_at(
                enc,
                m,
                x,
                bytes(t * m.n_in()),
                y,
                bytes(t * m.n_out()),
            )?;
        }
        Ok(())
    }
}

fn encode_linear_batch(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    y: &Buffer,
    layer: &LinearLayerOwned,
    gpu: &GpuRecurrent,
    s: &VerifyScratch,
    batch: usize,
) -> Result<()> {
    let meta = &layer.meta;
    let h = meta.hidden;
    let b = batch as u32;
    let n_h = h as u32;
    let value_dim = meta.v_heads * meta.v_dim;
    let repeat = meta.v_heads / meta.k_heads;
    let gd = GatedDeltaParams {
        k_heads: meta.k_heads as u32,
        v_heads: meta.v_heads as u32,
        k_dim: meta.k_dim as u32,
        v_dim: meta.v_dim as u32,
        repeat: repeat as u32,
        key_dim: (meta.k_heads * meta.k_dim) as u32,
        value_dim: value_dim as u32,
    };
    for t in 0..batch {
        encode_rmsnorm_at(enc, x, t * h, &layer.input_norm, &s.normed, t * h, n_h);
    }
    encode_proj(enc, &layer.qkv, &s.normed, &s.qkv, b)?;
    encode_proj(enc, &layer.z, &s.normed, &s.z, b)?;
    encode_proj(enc, &layer.a, &s.normed, &s.a, b)?;
    encode_proj(enc, &layer.b, &s.normed, &s.b, b)?;
    for t in 0..batch {
        encode_causal_conv_at(
            enc,
            &s.qkv,
            t * meta.conv_dim,
            &gpu.conv,
            &layer.aux.conv_w,
            &s.conv_y,
            meta.conv_dim as u32,
            meta.kernel as u32,
        );
        encode_gated_delta_at(
            enc,
            &s.conv_y,
            &s.a,
            t * meta.v_heads,
            &s.b,
            t * meta.v_heads,
            &s.z,
            t * value_dim,
            &layer.aux,
            &gpu.state,
            &s.delta_out,
            t * value_dim,
            &gd,
        );
    }
    encode_proj(enc, &layer.out, &s.delta_out, &s.mixed, b)?;
    let mid_n = layer.gate.n_out();
    for t in 0..batch {
        encode_add2_at(enc, x, t * h, &s.mixed, t * h, y, t * h, n_h);
        encode_rmsnorm_at(enc, y, t * h, &layer.post_norm, &s.post_n, t * h, n_h);
    }
    encode_proj(enc, &layer.gate, &s.post_n, &s.gate, b)?;
    encode_proj(enc, &layer.up, &s.post_n, &s.up, b)?;
    for t in 0..batch {
        encode_silu_mul_at(enc, &s.gate, t * mid_n, &s.up, t * mid_n, &s.mid, t * mid_n, mid_n as u32);
    }
    encode_proj(enc, &layer.down, &s.mid, &s.down, b)?;
    for t in 0..batch {
        encode_add_inplace_at(enc, y, t * h, &s.down, t * h, n_h);
    }
    Ok(())
}

fn encode_full_batch(
    enc: &ComputeCommandEncoderRef,
    x: &Buffer,
    y: &Buffer,
    layer: &FullLayerOwned,
    kv: &GpuKv,
    kv_base: usize,
    rope_cos: &Buffer,
    rope_sin: &Buffer,
    s: &VerifyScratch,
    batch: usize,
) -> Result<()> {
    let meta = &layer.meta;
    let h = meta.hidden;
    let n_h = h as u32;
    let b = batch as u32;
    let groups = meta.heads / meta.kv_heads;
    let q_span = meta.heads * meta.head_dim * 2;
    let kv_n = meta.kv_heads * meta.head_dim;
    let q_n = meta.heads * meta.head_dim;
    anyhow::ensure!(layer.o.n_in() == q_n, "attn out width");
    let o = ops()?;
    let head_tg = MTLSize { width: meta.head_dim as NSUInteger, height: 1, depth: 1 };
    for t in 0..batch {
        encode_rmsnorm_at(enc, x, t * h, &layer.input_norm, &s.normed, t * h, n_h);
    }
    encode_proj(enc, &layer.q, &s.normed, &s.fq_raw, b)?;
    encode_proj(enc, &layer.k, &s.normed, &s.fk_raw, b)?;
    encode_proj(enc, &layer.v, &s.normed, &s.fv_raw, b)?;
    for t in 0..batch {
        let fm = FullMeta {
            heads: meta.heads as u32,
            kv_heads: meta.kv_heads as u32,
            head_dim: meta.head_dim as u32,
            rotary_dim: meta.rotary_dim as u32,
            groups: groups as u32,
            kv_len: (kv_base + t + 1) as u32,
            kv_cap: kv.cap as u32,
            scale_inv: (meta.head_dim as f32).sqrt().recip(),
        };
        let rope = bytes(t * meta.rotary_dim);
        enc.set_compute_pipeline_state(&o.full_q_prep);
        enc.set_buffer(0, Some(&s.fq_raw), bytes(t * q_span));
        enc.set_buffer(1, Some(&layer.q_norm), 0);
        enc.set_buffer(2, Some(rope_cos), rope);
        enc.set_buffer(3, Some(rope_sin), rope);
        enc.set_buffer(4, Some(&s.fq), 0);
        enc.set_buffer(5, Some(&s.fgate), 0);
        enc.set_bytes(6, std::mem::size_of::<FullMeta>() as NSUInteger, &fm as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: meta.heads as NSUInteger, height: 1, depth: 1 },
            head_tg,
        );
        enc.set_compute_pipeline_state(&o.full_kv_prep);
        enc.set_buffer(0, Some(&s.fk_raw), bytes(t * kv_n));
        enc.set_buffer(1, Some(&s.fv_raw), bytes(t * kv_n));
        enc.set_buffer(2, Some(&layer.k_norm), 0);
        enc.set_buffer(3, Some(rope_cos), rope);
        enc.set_buffer(4, Some(rope_sin), rope);
        enc.set_buffer(5, Some(&kv.k), 0);
        enc.set_buffer(6, Some(&kv.v), 0);
        enc.set_bytes(7, std::mem::size_of::<FullMeta>() as NSUInteger, &fm as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: meta.kv_heads as NSUInteger, height: 1, depth: 1 },
            head_tg,
        );
        enc.set_compute_pipeline_state(&o.full_attn);
        enc.set_buffer(0, Some(&s.fq), 0);
        enc.set_buffer(1, Some(&s.fgate), 0);
        enc.set_buffer(2, Some(&kv.k), 0);
        enc.set_buffer(3, Some(&kv.v), 0);
        enc.set_buffer(4, Some(&s.fattn), bytes(t * q_n));
        enc.set_bytes(5, std::mem::size_of::<FullMeta>() as NSUInteger, &fm as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: meta.heads as NSUInteger, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
    }
    encode_proj(enc, &layer.o, &s.fattn, &s.mixed, b)?;
    let mid_n = layer.gate.n_out();
    for t in 0..batch {
        encode_add2_at(enc, x, t * h, &s.mixed, t * h, y, t * h, n_h);
        encode_rmsnorm_at(enc, y, t * h, &layer.post_norm, &s.post_n, t * h, n_h);
    }
    encode_proj(enc, &layer.gate, &s.post_n, &s.gate, b)?;
    encode_proj(enc, &layer.up, &s.post_n, &s.up, b)?;
    for t in 0..batch {
        encode_silu_mul_at(enc, &s.gate, t * mid_n, &s.up, t * mid_n, &s.mid, t * mid_n, mid_n as u32);
    }
    encode_proj(enc, &layer.down, &s.mid, &s.down, b)?;
    for t in 0..batch {
        encode_add_inplace_at(enc, y, t * h, &s.down, t * h, n_h);
    }
    Ok(())
}

struct VerifyScratch {
    normed: Buffer,
    qkv: Buffer,
    z: Buffer,
    a: Buffer,
    b: Buffer,
    conv_y: Buffer,
    delta_out: Buffer,
    mixed: Buffer,
    post_n: Buffer,
    gate: Buffer,
    up: Buffer,
    mid: Buffer,
    down: Buffer,
    fq_raw: Buffer,
    fk_raw: Buffer,
    fv_raw: Buffer,
    fq: Buffer,
    fgate: Buffer,
    fattn: Buffer,
}

fn verify_scratch(lin: &LinearLayerMeta, full: &FullLayerMeta, mid_n: usize, batch: usize) -> Result<VerifyScratch> {
    let value_dim = lin.v_heads * lin.v_dim;
    let q_span = full.heads * full.head_dim * 2;
    let kv_n = full.kv_heads * full.head_dim;
    let q_n = full.heads * full.head_dim;
    Ok(VerifyScratch {
        normed: buf_zeros(batch * lin.hidden)?,
        qkv: buf_zeros(batch * lin.conv_dim)?,
        z: buf_zeros(batch * value_dim)?,
        a: buf_zeros(batch * lin.v_heads)?,
        b: buf_zeros(batch * lin.v_heads)?,
        conv_y: buf_zeros(lin.conv_dim)?,
        delta_out: buf_zeros(batch * value_dim)?,
        mixed: buf_zeros(batch * lin.hidden)?,
        post_n: buf_zeros(batch * lin.hidden)?,
        gate: buf_zeros(batch * mid_n)?,
        up: buf_zeros(batch * mid_n)?,
        mid: buf_zeros(batch * mid_n)?,
        down: buf_zeros(batch * lin.hidden)?,
        fq_raw: buf_zeros(batch * q_span)?,
        fk_raw: buf_zeros(batch * kv_n)?,
        fv_raw: buf_zeros(batch * kv_n)?,
        fq: buf_zeros(q_n)?,
        fgate: buf_zeros(q_n)?,
        fattn: buf_zeros(batch * q_n)?,
    })
}

/// Teacher-forced verify of `batch` tokens (1..=4). One weight read per big
/// projection; recurrent and attention steps stay in token order. Logits match
/// `batch` serial greedy steps when the cache starts from the same state.
pub fn fused_verify(
    embed: &QuantMatrix,
    token_ids: &[u32],
    layers: &[DecodeLayer<'_>],
    kv_bases: &[usize],
    rope_cos: &[f32],
    rope_sin: &[f32],
    lin_meta: &LinearLayerMeta,
    full_meta: &FullLayerMeta,
    mid_n: usize,
    final_norm: &Buffer,
    head: &QuantMatrix,
) -> Result<Vec<(u32, f32)>> {
    let batch = token_ids.len();
    anyhow::ensure!((1..=4).contains(&batch), "verify batch 1..=4");
    anyhow::ensure!(rope_cos.len() == batch * full_meta.rotary_dim, "rope");
    anyhow::ensure!(rope_sin.len() == rope_cos.len(), "rope sin");
    let h = lin_meta.hidden;
    let scratch = verify_scratch(lin_meta, full_meta, mid_n, batch)?;
    let ping = buf_zeros(batch * h)?;
    let pong = buf_zeros(batch * h)?;
    let rope_c = buf_f32(rope_cos)?;
    let rope_s = buf_f32(rope_sin)?;
    let vocab = head.n_out();
    let logits = buf_zeros(batch * vocab)?;
    let n_tg = 256usize;
    let idx_b = buf_zeros(batch * n_tg)?;
    let val_b = buf_zeros(batch * n_tg)?;

    let o = ops()?;
    let cb = o.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    for (t, &id) in token_ids.iter().enumerate() {
        qwen3_gemv::encode_dequant_row_at(enc, embed, id, &ping, bytes(t * h))?;
    }
    let mut use_ping = true;
    let mut full_i = 0usize;
    for layer in layers {
        let (src, dst) = if use_ping { (&ping, &pong) } else { (&pong, &ping) };
        match layer {
            DecodeLayer::Linear(owned, gpu) => {
                encode_linear_batch(enc, src, dst, owned, gpu, &scratch, batch)?;
            }
            DecodeLayer::Full(owned, kv) => {
                encode_full_batch(
                    enc,
                    src,
                    dst,
                    owned,
                    kv,
                    kv_bases[full_i],
                    &rope_c,
                    &rope_s,
                    &scratch,
                    batch,
                )?;
                full_i += 1;
            }
        }
        use_ping = !use_ping;
    }
    let hidden = if use_ping { &ping } else { &pong };
    for t in 0..batch {
        encode_rmsnorm_at(enc, hidden, t * h, final_norm, &scratch.normed, t * h, h as u32);
    }
    qwen3_gemv::encode_gemm4(enc, head, &scratch.normed, &logits, batch as u32)?;
    let n = vocab as u32;
    for t in 0..batch {
        enc.set_compute_pipeline_state(&o.argmax);
        enc.set_buffer(0, Some(&logits), bytes(t * vocab));
        enc.set_buffer(1, Some(&idx_b), bytes(t * n_tg));
        enc.set_buffer(2, Some(&val_b), bytes(t * n_tg));
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize { width: n_tg as NSUInteger, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let idxs = read_f32_u32(&idx_b, batch * n_tg);
    let vals = read_f32(&val_b, batch * n_tg);
    let mut out = Vec::with_capacity(batch);
    for t in 0..batch {
        let best = idxs[t * n_tg..(t + 1) * n_tg]
            .iter()
            .zip(vals[t * n_tg..(t + 1) * n_tg].iter())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&i, &v)| (i, v))
            .unwrap();
        out.push(best);
    }
    Ok(out)
}

fn read_f32_u32(buf: &Buffer, n: usize) -> Vec<u32> {
    let ptr = buf.contents() as *const u32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// One layer in a full-model decode encode.
pub enum DecodeLayer<'a> {
    Linear(&'a LinearLayerOwned, &'a GpuRecurrent),
    Full(&'a FullLayerOwned, &'a GpuKv),
}

/// One wait for every layer in the text tower (linear + full).
pub fn fused_decode_all(
    x: &[f32],
    layers: &[DecodeLayer<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    lin_meta: &LinearLayerMeta,
    full_meta: &FullLayerMeta,
    mid_n: usize,
) -> Result<Vec<f32>> {
    let (h, _) = fused_decode_inner(x, layers, rope_cos, rope_sin, lin_meta, full_meta, mid_n, None, None)?;
    Ok(h)
}

/// Tower + final RMSNorm + lm_head GEMV + argmax in one wait.
pub fn fused_decode_greedy(
    x: &[f32],
    layers: &[DecodeLayer<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    lin_meta: &LinearLayerMeta,
    full_meta: &FullLayerMeta,
    mid_n: usize,
    final_norm: &Buffer,
    head: &QuantMatrix,
    embed: Option<(&QuantMatrix, u32)>,
) -> Result<(u32, f32)> {
    let (_, tok) = fused_decode_inner(
        x,
        layers,
        rope_cos,
        rope_sin,
        lin_meta,
        full_meta,
        mid_n,
        Some((final_norm, head)),
        embed,
    )?;
    Ok(tok.unwrap())
}

fn fused_decode_inner(
    x: &[f32],
    layers: &[DecodeLayer<'_>],
    rope_cos: &[f32],
    rope_sin: &[f32],
    lin_meta: &LinearLayerMeta,
    full_meta: &FullLayerMeta,
    mid_n: usize,
    head: Option<(&Buffer, &QuantMatrix)>,
    embed: Option<(&QuantMatrix, u32)>,
) -> Result<(Vec<f32>, Option<(u32, f32)>)> {
    anyhow::ensure!(!layers.is_empty(), "empty");
    let h = lin_meta.hidden;
    if embed.is_none() {
        anyhow::ensure!(x.len() == h, "hidden");
    }
    anyhow::ensure!(rope_cos.len() == full_meta.rotary_dim, "rope");

    let o = ops()?;
    let arena = arena(lin_meta, mid_n, full_meta)?;
    unsafe {
        if embed.is_none() {
            std::ptr::copy_nonoverlapping(x.as_ptr(), arena.ping.contents() as *mut f32, h);
        }
        std::ptr::copy_nonoverlapping(
            rope_cos.as_ptr(),
            arena.scratch.rope_cos.contents() as *mut f32,
            full_meta.rotary_dim,
        );
        std::ptr::copy_nonoverlapping(
            rope_sin.as_ptr(),
            arena.scratch.rope_sin.contents() as *mut f32,
            full_meta.rotary_dim,
        );
    }

    let profile = std::env::var("PAGODA_PROFILE").is_ok();
    let t_enc = std::time::Instant::now();
    let cb = o.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let mut use_ping = true;
    if let Some((embed_m, token)) = embed {
        anyhow::ensure!(embed_m.n_in() == h, "embed width");
        qwen3_gemv::encode_dequant_row(enc, embed_m, token, &arena.ping)?;
    }
    for layer in layers {
        let (src, dst) = if use_ping {
            (&arena.ping, &arena.pong)
        } else {
            (&arena.pong, &arena.ping)
        };
        match layer {
            DecodeLayer::Linear(owned, gpu) => {
                encode_linear_layer(enc, src, dst, owned, gpu, &arena.scratch)?;
            }
            DecodeLayer::Full(owned, kv) => {
                encode_full_layer(enc, src, dst, owned, kv, &arena.scratch)?;
            }
        }
        use_ping = !use_ping;
    }
    let out_buf = if use_ping { &arena.ping } else { &arena.pong };

    let tok = if let Some((final_norm, head_m)) = head {
        let normed = &arena.scratch.normed;
        encode_rmsnorm(enc, out_buf, final_norm, normed, h as u32);
        // Reuse fattn buffer only if vocab fits — allocate logits on arena via mid/gate oversized?
        // Use a dedicated logits path: write into a fresh buffer sized to vocab.
        let logits = head_logits_buf(head_m.n_out())?;
        qwen3_gemv::encode_into(enc, head_m, normed, logits)?;
        let n_tg = 256u32;
        let (idx_b, val_b) = argmax_bufs(n_tg)?;
        let n = head_m.n_out() as u32;
        enc.set_compute_pipeline_state(&o.argmax);
        enc.set_buffer(0, Some(logits), 0);
        enc.set_buffer(1, Some(&idx_b), 0);
        enc.set_buffer(2, Some(&val_b), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize {
                width: n_tg as NSUInteger,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        let enc_s = t_enc.elapsed().as_secs_f64();
        let t_gpu = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        if profile {
            eprintln!(
                "  profile encode {enc_s:.4}s gpu {:.4}s",
                t_gpu.elapsed().as_secs_f64()
            );
        }
        let idxs =
            unsafe { std::slice::from_raw_parts(idx_b.contents() as *const u32, n_tg as usize) };
        let vals =
            unsafe { std::slice::from_raw_parts(val_b.contents() as *const f32, n_tg as usize) };
        let best = idxs
            .iter()
            .zip(vals.iter())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&i, &v)| (i, v))
            .unwrap();
        (Vec::new(), Some(best))
    } else {
        enc.end_encoding();
        let enc_s = t_enc.elapsed().as_secs_f64();
        let t_gpu = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        if profile {
            eprintln!(
                "  profile encode {enc_s:.4}s gpu {:.4}s",
                t_gpu.elapsed().as_secs_f64()
            );
        }
        (read_f32(out_buf, h), None)
    };
    Ok(tok)
}

fn argmax_bufs(n_tg: u32) -> Result<(&'static Buffer, &'static Buffer)> {
    static CELL: OnceLock<(Buffer, Buffer)> = OnceLock::new();
    if let Some(pair) = CELL.get() {
        return Ok((&pair.0, &pair.1));
    }
    let o = ops()?;
    let n = (n_tg * 4) as NSUInteger;
    let pair = (
        o.device.new_buffer(n, MTLResourceOptions::StorageModeShared),
        o.device.new_buffer(n, MTLResourceOptions::StorageModeShared),
    );
    let _ = CELL.set(pair);
    let pair = CELL.get().unwrap();
    Ok((&pair.0, &pair.1))
}

fn head_logits_buf(n_out: usize) -> Result<&'static Buffer> {
    static CELL: OnceLock<Buffer> = OnceLock::new();
    if let Some(b) = CELL.get() {
        // Grow not supported; vocab is fixed for a model.
        anyhow::ensure!(b.length() as usize >= n_out * 4, "logits buf short");
        return Ok(b);
    }
    let b = buf_zeros(n_out)?;
    let _ = CELL.set(b);
    Ok(CELL.get().unwrap())
}

/// One wait for a stack of consecutive linear layers (activations stay on GPU).
pub fn fused_linear_stack(
    x: &[f32],
    layers: &[LinearLayerOwned],
    gpus: &[&GpuRecurrent],
) -> Result<Vec<f32>> {
    let refs: Vec<&LinearLayerOwned> = layers.iter().collect();
    fused_linear_stack_ref(x, &refs, gpus)
}

pub fn fused_linear_stack_ref(
    x: &[f32],
    layers: &[&LinearLayerOwned],
    gpus: &[&GpuRecurrent],
) -> Result<Vec<f32>> {
    anyhow::ensure!(!layers.is_empty(), "empty stack");
    anyhow::ensure!(layers.len() == gpus.len(), "gpu count");
    let meta0 = &layers[0].meta;
    let h = meta0.hidden;
    anyhow::ensure!(x.len() == h, "hidden");
    for layer in layers {
        anyhow::ensure!(layer.meta.hidden == h, "hidden mismatch");
        anyhow::ensure!(layer.meta.kernel >= 2, "kernel");
        anyhow::ensure!(layer.meta.v_dim > 0 && layer.meta.v_dim <= 256, "v_dim");
        anyhow::ensure!(layer.meta.v_dim.count_ones() == 1, "v_dim pot");
        let key_dim = layer.meta.k_heads * layer.meta.k_dim;
        let value_dim = layer.meta.v_heads * layer.meta.v_dim;
        anyhow::ensure!(
            layer.meta.v_heads / layer.meta.k_heads * layer.meta.k_heads == layer.meta.v_heads,
            "repeat"
        );
        anyhow::ensure!(layer.meta.conv_dim == 2 * key_dim + value_dim, "conv_dim");
    }

    // Fallback stack path: synthesize a dummy full meta from linear dims.
    let full = FullLayerMeta {
        hidden: h,
        heads: 24,
        kv_heads: 4,
        head_dim: 256,
        rotary_dim: 64,
    };
    let o = ops()?;
    let mid_n = layers[0].gate.n_out();
    let arena = arena(meta0, mid_n, &full)?;
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), arena.ping.contents() as *mut f32, h);
    }

    let cb = o.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let mut use_ping = true;
    for (layer, gpu) in layers.iter().zip(gpus.iter()) {
        let (src, dst) = if use_ping {
            (&arena.ping, &arena.pong)
        } else {
            (&arena.pong, &arena.ping)
        };
        encode_linear_layer(enc, src, dst, layer, gpu, &arena.scratch)?;
        use_ping = !use_ping;
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let out_buf = if use_ping { &arena.ping } else { &arena.pong };
    Ok(read_f32(out_buf, h))
}

struct Arena {
    ping: Buffer,
    pong: Buffer,
    scratch: LinearScratch,
}

fn arena(meta: &LinearLayerMeta, mid_n: usize, full: &FullLayerMeta) -> Result<&'static Arena> {
    static CELL: OnceLock<Arena> = OnceLock::new();
    if let Some(a) = CELL.get() {
        return Ok(a);
    }
    let a = Arena {
        ping: buf_zeros(meta.hidden)?,
        pong: buf_zeros(meta.hidden)?,
        scratch: LinearScratch::new(meta, mid_n, full)?,
    };
    let _ = CELL.set(a);
    Ok(CELL.get().unwrap())
}

/// One wait: input RMSNorm → linear attn → residual SwiGLU MLP.
pub fn fused_linear_layer(
    x: &[f32],
    input_norm: &[f32],
    post_norm: &[f32],
    mats: &LinearLayerMats<'_>,
    aux: &LinearAttnAux,
    gpu: &GpuRecurrent,
    meta: &LinearLayerMeta,
) -> Result<Vec<f32>> {
    let owned = LinearLayerOwned {
        qkv: mats.qkv.clone(),
        z: mats.z.clone(),
        a: mats.a.clone(),
        b: mats.b.clone(),
        out: mats.out.clone(),
        gate: mats.gate.clone(),
        up: mats.up.clone(),
        down: mats.down.clone(),
        aux: aux.clone(),
        input_norm: buf_f32(input_norm)?,
        post_norm: buf_f32(post_norm)?,
        meta: LinearLayerMeta {
            hidden: meta.hidden,
            conv_dim: meta.conv_dim,
            kernel: meta.kernel,
            k_heads: meta.k_heads,
            v_heads: meta.v_heads,
            k_dim: meta.k_dim,
            v_dim: meta.v_dim,
        },
    };
    fused_linear_stack(x, &[owned], &[gpu])
}

/// GEMV + GPU argmax. Returns `(token_id, logit)`.
pub fn gemv_argmax(m: &QuantMatrix, x: &[f32]) -> Result<(u32, f32)> {
    anyhow::ensure!(x.len() == m.n_in(), "gemv x");
    let o = ops()?;
    let x_b = buf_f32(x)?;
    let logits = buf_zeros(m.n_out())?;
    let n_tg = 256u32;
    let idx_b = o.device.new_buffer(
        (n_tg * 4) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );
    let val_b = o.device.new_buffer(
        (n_tg * 4) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );

    let cb = o.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    qwen3_gemv::encode_into(enc, m, &x_b, &logits)?;
    let n = m.n_out() as u32;
    enc.set_compute_pipeline_state(&o.argmax);
    enc.set_buffer(0, Some(&logits), 0);
    enc.set_buffer(1, Some(&idx_b), 0);
    enc.set_buffer(2, Some(&val_b), 0);
    enc.set_bytes(3, 4, &n as *const u32 as *const _);
    enc.dispatch_thread_groups(
        MTLSize {
            width: n_tg as NSUInteger,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let idxs = unsafe { std::slice::from_raw_parts(idx_b.contents() as *const u32, n_tg as usize) };
    let vals = unsafe { std::slice::from_raw_parts(val_b.contents() as *const f32, n_tg as usize) };
    let (best_i, best_v) = idxs
        .iter()
        .zip(vals.iter())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(&i, &v)| (i, v))
        .unwrap();
    Ok((best_i, best_v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLSize;

    #[test]
    fn delta_and_attn_time() {
        let o = ops().unwrap();
        let conv = buf_f32(&vec![0.01f32; 10240]).unwrap();
        let a = buf_f32(&vec![0.1f32; 48]).unwrap();
        let b = buf_f32(&vec![0.2f32; 48]).unwrap();
        let z = buf_f32(&vec![0.3f32; 48 * 128]).unwrap();
        let aux = LinearAttnAux::upload(
            &vec![0.1f32; 10240 * 4],
            &vec![-0.5f32; 48],
            &vec![0.1f32; 48],
            &vec![1.0f32; 128],
        )
        .unwrap();
        let state = buf_f32(&vec![0.01f32; 48 * 128 * 128]).unwrap();
        let out = buf_zeros(48 * 128).unwrap();
        let gd = GatedDeltaParams {
            k_heads: 16,
            v_heads: 48,
            k_dim: 128,
            v_dim: 128,
            repeat: 3,
            key_dim: 2048,
            value_dim: 6144,
        };
        let n = 48u32;
        for _ in 0..2 {
            let cb = o.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for _ in 0..n {
                encode_gated_delta(enc, &conv, &a, &b, &z, &aux, &state, &out, &gd);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let cb = o.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..n {
            encode_gated_delta(enc, &conv, &a, &b, &z, &aux, &state, &out, &gd);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let us = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        eprintln!("gated delta {us:.1} us  x{n} = {:.2} ms", us * n as f64 / 1000.0);

        let heads = 24u32;
        let kv_heads = 4u32;
        let head_dim = 256u32;
        let q = buf_zeros((heads * head_dim) as usize).unwrap();
        let gate = buf_zeros((heads * head_dim) as usize).unwrap();
        let kv_len = 32u32;
        let k = buf_zeros((kv_len * kv_heads * head_dim) as usize).unwrap();
        let v = buf_zeros((kv_len * kv_heads * head_dim) as usize).unwrap();
        let attn_out = buf_zeros((heads * head_dim) as usize).unwrap();
        let fm = FullMeta {
            heads,
            kv_heads,
            head_dim,
            rotary_dim: 64,
            groups: heads / kv_heads,
            kv_len,
            kv_cap: kv_len,
            scale_inv: (head_dim as f32).sqrt().recip(),
        };
        let n_attn = 16u32;
        let cb = o.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..n_attn {
            enc.set_compute_pipeline_state(&o.full_attn);
            enc.set_buffer(0, Some(&q), 0);
            enc.set_buffer(1, Some(&gate), 0);
            enc.set_buffer(2, Some(&k), 0);
            enc.set_buffer(3, Some(&v), 0);
            enc.set_buffer(4, Some(&attn_out), 0);
            enc.set_bytes(5, std::mem::size_of::<FullMeta>() as u64, &fm as *const _ as *const _);
            enc.dispatch_thread_groups(
                MTLSize { width: heads as u64, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let us = t.elapsed().as_secs_f64() / n_attn as f64 * 1e6;
        eprintln!("full attn kv=32 {us:.1} us  x{n_attn}");

        let n_rms = 128u32;
        let x = buf_f32(&vec![0.1f32; 5120]).unwrap();
        let w = buf_f32(&vec![1.0f32; 5120]).unwrap();
        let y = buf_zeros(5120).unwrap();
        let cb = o.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..n_rms {
            encode_rmsnorm(enc, &x, &w, &y, 5120);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let us = t.elapsed().as_secs_f64() / n_rms as f64 * 1e6;
        eprintln!("rmsnorm {us:.1} us  x{n_rms} = {:.2} ms", us * n_rms as f64 / 1000.0);

        let conv_x = buf_f32(&vec![0.1f32; 10240]).unwrap();
        let conv_state = buf_f32(&vec![0.1f32; 10240 * 3]).unwrap();
        let conv_w = buf_f32(&vec![0.1f32; 10240 * 4]).unwrap();
        let conv_y = buf_zeros(10240).unwrap();
        let n_conv = 48u32;
        let cb = o.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..n_conv {
            encode_causal_conv(enc, &conv_x, &conv_state, &conv_w, &conv_y, 10240, 4);
        }
        enc.end_encoding();
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let us = t.elapsed().as_secs_f64() / n_conv as f64 * 1e6;
        eprintln!("conv {us:.1} us  x{n_conv} = {:.2} ms", us * n_conv as f64 / 1000.0);
    }
}
