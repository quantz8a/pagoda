// SPDX-License-Identifier: Apache-2.0
//! Minimal CUDA smoke test: device creation, host->device copy, cuBLAS GEMM.
//! Run: PAGODA_DEVICE=cuda cargo run --release --features cuda --example cuda_smoke

use candle_core::{Device, Tensor};

fn main() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    println!("[1/4] device ok: {device:?}");

    let a_cpu = Tensor::randn(0f32, 1.0, (128, 256), &Device::Cpu)?;
    let a = a_cpu.to_device(&device)?;
    println!("[2/4] to_device ok: {a:?}");

    let b = Tensor::randn(0f32, 1.0, (256, 64), &device)?;
    let c = a.matmul(&b)?;
    println!("[3/4] matmul ok: {c:?}");

    let s = c.sum_all()?.to_scalar::<f32>()?;
    println!("[4/4] sum ok: {s:.3}");
    println!("CUDA SMOKE TEST PASSED");
    Ok(())
}
