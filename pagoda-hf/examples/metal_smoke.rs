// SPDX-License-Identifier: Apache-2.0
//! Minimal Metal smoke test: device creation, host->device copy, GEMM.
//! Run: PAGODA_DEVICE=metal cargo run --release --features metal --example metal_smoke

use candle_core::{Device, Tensor};

fn main() -> candle_core::Result<()> {
    let device = Device::new_metal(0)?;
    println!("[1/4] device ok: {device:?}");

    let a_cpu = Tensor::randn(0f32, 1.0, (128, 256), &Device::Cpu)?;
    let a = a_cpu.to_device(&device)?.to_dtype(candle_core::DType::F16)?;
    println!("[2/4] to_device ok: {a:?}");

    let b = Tensor::randn(0f32, 1.0, (256, 64), &device)?.to_dtype(candle_core::DType::F16)?;
    let c = a.matmul(&b)?;
    println!("[3/4] matmul ok: {c:?}");

    let s = c.to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    println!("[4/4] sum ok: {s:.3}");
    println!("METAL SMOKE TEST PASSED");
    Ok(())
}
