//! Standalone `#![no_std]` ESP32-C6 (`riscv32imac-unknown-none-elf`) synthetic DSP harness.
//!
//! Exercises `FixedQ15DirectConvolver`, `DirectConvolver` (`f32` software baseline),
//! `FixedQ15PartitionedConvolver`, and the zero-configuration `AutoConvolver` (which selects
//! `FixedQ15NonUniformConvolver` on this FPU-less target) loading a raw `f32` `.bin` FIR artifact,
//! using caller-supplied buffers with zero heap allocation and zero I2S dependency.
//!
//! Flashing and physical cycle measurement (`DEVICE_VERIFIED`) require explicit owner
//! authorization and a connected ESP32-C6 board; do not flash automatically.

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use rtconv_simd::auto::{AutoConvolver, AutoEngine, AutoWorkspace, workspace_len};
use rtconv_simd::{
    DirectConfig, DirectConvolver, FixedQ15DirectConfig, FixedQ15DirectConvolver,
    FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver, HAS_HARDWARE_FPU, SimdBackend,
};

/// Taps of the embedded synthetic FIR artifact (`> Q15_DIRECT_MAX_TAPS`, so NUPC is selected).
const FIR_TAPS: usize = 300;

/// Raw little-endian `f32` mono FIR (rePhase `.bin` layout) built at compile time:
/// `h[0] = 0.5`, `h[200] = -0.25`, zeros elsewhere.
const FIR_BIN: [u8; FIR_TAPS * 4] = {
    let mut out = [0u8; FIR_TAPS * 4];
    let mut k = 0;
    while k < FIR_TAPS {
        let v: f32 = if k == 0 {
            0.5
        } else if k == 200 {
            -0.25
        } else {
            0.0
        };
        let b = v.to_le_bytes();
        out[4 * k] = b[0];
        out[4 * k + 1] = b[1];
        out[4 * k + 2] = b[2];
        out[4 * k + 3] = b[3];
        k += 1;
    }
    out
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
    /// `true` if `AutoConvolver` chose the FPU-less Q1.15 non-uniform engine and reproduced
    /// the embedded artifact's impulse response.
    pub auto_ok: bool,
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
    let mut pws = [0i16; 1024];
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

    // 4. Zero-configuration AutoConvolver from the embedded raw f32 artifact.
    let mut auto_ok = !HAS_HARDWARE_FPU;
    let mut ws = AutoWorkspace::<{ workspace_len(FIR_TAPS) }>::new();
    match AutoConvolver::from_bytes(&FIR_BIN, 0, ws.as_mut_slice()) {
        Ok(mut conv) => {
            auto_ok &= conv.info().engine == AutoEngine::FixedNonUniform;
            let mut x = [0i16; 32];
            let mut y = [0i16; 32];
            x[0] = 16384; // +0.5
            for block in 0..8 {
                if conv.process_i16(&x, &mut y).is_err() {
                    auto_ok = false;
                }
                x[0] = 0;
                // Expected: y[0] = 0.25 (8192) and y[200] = -0.125 (-4096).
                if block == 0 && (y[0] - 8192).abs() > 1 {
                    auto_ok = false;
                }
                if block == 6 && (y[200 - 192] + 4096).abs() > 1 {
                    auto_ok = false;
                }
            }
        }
        Err(_) => auto_ok = false,
    }

    HarnessReport {
        impulse_ok,
        q15_direct_sats,
        q15_fdl_sats,
        samples_processed: 32,
        auto_ok,
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
