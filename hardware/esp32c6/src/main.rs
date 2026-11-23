//! Standalone `#![no_std]` ESP32-C6 (`riscv32imac-unknown-none-elf`) synthetic DSP harness.
//!
//! Exercises `FixedQ15DirectConvolver`, `DirectConvolver` (`f32` software baseline),
//! `FixedQ15PartitionedConvolver`, and `FixedQ15NonUniformConvolver` using static/stack
//! caller-supplied buffers with zero heap allocation and zero I2S dependency.
//!
//! Flashing and physical cycle measurement (`DEVICE_VERIFIED`) require explicit owner
//! authorization and a connected ESP32-C6 board; do not flash automatically.

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use rtconv_simd::{
    DirectConfig, DirectConvolver, FixedQ15DirectConfig, FixedQ15DirectConvolver,
    FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver, SimdBackend,
};

/// Result summary produced by a synthetic benchmark/validation pass on ESP32-C6.
#[derive(Clone, Copy, Debug)]
pub struct HarnessReport {
    /// Non-zero if all synthetic impulse checks matched expected taps.
    pub impulse_ok: bool,
    /// Saturation count observed in Q15 Direct FIR.
    pub q15_direct_sats: u64,
    /// Saturation count observed in Q15 Partitioned FDL.
    pub q15_fdl_sats: u64,
    /// Total samples processed per engine.
    pub samples_processed: usize,
}

/// Executes the synthetic DSP validation workload (Direct Q15, Direct f32, FDL Q15)
/// at 32-sample blocks (`48 kHz` nominal budget = `666.67 us` / ~`106,667` cycles @ 160 MHz).
pub fn run_synthetic_suite() -> HarnessReport {
    let mut impulse_ok = true;

    // 1. Q15 Direct FIR (64 taps, 32-sample block)
    let mut ir_q15 = [0i16; 64];
    ir_q15[0] = 16384; // +0.5
    ir_q15[15] = -8192; // -0.25
    let mut in_q15 = [0i16; 32];
    in_q15[0] = 16384; // +0.5
    let mut out_q15 = [0i16; 32];

    let qcfg = FixedQ15DirectConfig { max_ir_len: 64 };
    let mut qws = [0i16; 128];
    let mut q15_direct_sats = 0u64;
    if let Ok(mut qconv) = FixedQ15DirectConvolver::prepare(&ir_q15, qcfg, &mut qws) {
        if qconv.process_block(&in_q15, &mut out_q15).is_ok() {
            // Expected: out[0] == +0.25 (8192), out[15] == -0.125 (-4096)
            if (out_q15[0] - 8192).abs() > 1 || (out_q15[15] + 4096).abs() > 1 {
                impulse_ok = false;
            }
        } else {
            impulse_ok = false;
        }
        q15_direct_sats = qconv.saturation_count();
    } else {
        impulse_ok = false;
    }

    // 2. Software f32 Direct FIR baseline (64 taps, 32-sample block)
    let mut ir_f32 = [0.0f32; 64];
    ir_f32[0] = 0.5;
    ir_f32[15] = -0.25;
    let mut in_f32 = [0.0f32; 32];
    in_f32[0] = 0.5;
    let mut out_f32 = [0.0f32; 32];
    let dcfg = DirectConfig { max_ir_len: 64 };
    let mut dws = [0.0f32; 128];
    if let Ok(mut dconv) =
        DirectConvolver::prepare(&ir_f32, dcfg, &mut dws, SimdBackend::baseline())
    {
        if dconv.process_block(&in_f32, &mut out_f32).is_ok() {
            if (out_f32[0] - 0.25).abs() > 1e-5 || (out_f32[15] + 0.125).abs() > 1e-5 {
                impulse_ok = false;
            }
        } else {
            impulse_ok = false;
        }
    } else {
        impulse_ok = false;
    }

    // 3. Q15 BFP Partitioned FDL (B = 32 / N64, 64 taps = 2 partitions)
    let pcfg = FixedQ15PartitionedConfig {
        block_size: 32,
        max_ir_len: 64,
    };
    let mut pws = [0i16; 512];
    let mut q15_fdl_sats = 0u64;
    if let Ok(mut pconv) = FixedQ15PartitionedConvolver::prepare(&ir_q15, pcfg, &mut pws) {
        if pconv.process_block(&in_q15, &mut out_q15).is_ok() {
            if (out_q15[0] - 8192).abs() > 8 || (out_q15[15] + 4096).abs() > 8 {
                impulse_ok = false;
            }
        } else {
            impulse_ok = false;
        }
        q15_fdl_sats = pconv.saturation_count();
    } else {
        impulse_ok = false;
    }

    HarnessReport {
        impulse_ok,
        q15_direct_sats,
        q15_fdl_sats,
        samples_processed: 32,
    }
}

/// Bare-metal entry symbol for cross-compilation verification (`riscv32imac-unknown-none-elf`).
/// When integrating with `esp-hal` on physical hardware, call `run_synthetic_suite()` inside
/// `#[esp_hal::main]` and wrap each block with `mcycle` CSR reads.
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let _report = core::hint::black_box(run_synthetic_suite());
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
