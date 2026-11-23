//! Pure Rust `#![no_std]` fixed-point Q1.15 DSP engines for FPU-less microcontrollers
//! (such as ESP32-C6 / `riscv32imac-unknown-none-elf`).
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, zero C/FFI dependencies.

pub mod direct;
pub mod q15;

pub use direct::{
    FixedQ15DirectCapabilities, FixedQ15DirectConfig, FixedQ15DirectConvolver,
    FixedQ15DirectMemoryRequirements,
};
pub use q15::{
    Q15_HALF, Q15_SCALE_F32, Q15_SCALE_F64, Q15_SHIFT, f32_to_q15, f64_to_q15, mul_q15, q15_to_f32,
    q15_to_f64, round_sat_q15_from_i64, sat_i16_from_i32,
};
