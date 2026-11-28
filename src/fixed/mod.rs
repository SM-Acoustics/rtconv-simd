//! Pure Rust `#![no_std]` fixed-point Q1.15 DSP engines for FPU-less microcontrollers
//! (such as ESP32-C6 / `riscv32imac-unknown-none-elf`).
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, zero C/FFI dependencies.

#[cfg(feature = "fixed-fft")]
pub mod block_float;
pub mod direct;
#[cfg(feature = "fixed-fft")]
pub mod fft_q15;
#[cfg(feature = "fixed-fft")]
mod fft_q15_tables;
#[cfg(feature = "fixed-fft")]
pub(crate) mod fft_q31;
#[cfg(feature = "fixed-fft")]
mod fft_q31_tables;
#[cfg(feature = "fixed-nonuniform")]
pub mod nonuniform;
#[cfg(feature = "fixed-fft")]
pub mod partitioned;
pub mod q15;
#[cfg(feature = "fixed-fft")]
pub(crate) mod spectral;

pub use direct::{
    FixedQ15DirectCapabilities, FixedQ15DirectConfig, FixedQ15DirectConvolver,
    FixedQ15DirectMemoryRequirements,
};
#[cfg(feature = "fixed-fft")]
pub use fft_q15::{
    FixedStagedFftState, MAX_FIXED_FFT, fixed_fft_rom_bytes, fixed_fft_scratch_len_i32,
    fixed_fft_staged_steps, fixed_fft_supports_size, irfft_q15_bfp,
    irfft_q15_bfp_step_with_scratch, irfft_q15_bfp_with_scratch, rfft_q15_bfp,
    rfft_q15_bfp_step_with_scratch, rfft_q15_bfp_with_scratch,
};
#[cfg(feature = "fixed-nonuniform")]
pub use nonuniform::{
    FixedQ15NonUniformCapabilities, FixedQ15NonUniformConvolver,
    FixedQ15NonUniformMemoryRequirements,
};
#[cfg(feature = "fixed-fft")]
pub use partitioned::{
    FixedQ15PartitionedCapabilities, FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver,
    FixedQ15PartitionedMemoryRequirements,
};
pub use q15::{
    Q15_HALF, Q15_SCALE_F32, Q15_SCALE_F64, Q15_SHIFT, f32_to_q15, f64_to_q15, mul_q15, q15_to_f32,
    q15_to_f64, round_even_sat_q15_from_i64, round_sat_q15_from_i64, sat_i16_from_i32,
};
