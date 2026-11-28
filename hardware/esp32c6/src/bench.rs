//! `mcycle` benchmarks for the Q1.31 engine family on ESP32-C6 (`riscv32imac`), compile-only
//! in CI (`DEVICE_VERIFIED = NO`). All stimuli and filters are synthesized with integer
//! arithmetic, so the benchmark itself links no soft-float routine.
//!
//! - **Case A**: 255-tap linear-phase FIR Type I (crossover-like, `‖h‖₁ = 1`) in
//!   `FixedQ31DirectConvolver`, `B = 64` samples @ 48 kHz (`1.333 ms`, `213,333` cycles @
//!   160 MHz), folded (128 MACs/sample) and unfolded (255 MACs/sample), to compare with the C
//!   reference of ~56,000 cycles/block measured on the target board.
//! - **Case B**: rePhase-style long IR (1,024 and 2,048 taps, centred peak, decaying tails) in
//!   `FixedQ31NonUniformConvolver` vs `FixedQ15NonUniformConvolver`, same plan
//!   (`preset_esp32_c6_q15(len, 64)`), `B = 64`: worst and mean cycles per block over 64 blocks
//!   (covers every time-distributed FFT/CMAC/IFFT slot of the plan) and workspace bytes.
//!
//! Cycle counts use the standard RISC-V `mcycle`/`mcycleh` CSRs (M-mode, bare metal). If the
//! silicon exposes its cycle counter only through the Espressif performance-counter CSR
//! (`mpccr`, `0x7E2`), adapt [`mcycle`]; a counter that does not run reads as `0` cycles and is
//! visible as such in [`BenchReport`].
//!
//! Workspaces are local arrays (up to 48 KiB for case B, Q1.31 2,048 taps): run on a stack of
//! at least ~64 KiB, or move them into `static` storage (e.g. `static_cell`) on the device.

use rtconv_simd::{
    FixedQ15NonUniformConvolver, FixedQ31DirectConfig, FixedQ31DirectConvolver,
    FixedQ31NonUniformConvolver, preset_esp32_c6_q15,
};

/// Reads the 64-bit machine cycle counter (`mcycleh:mcycle`, re-read on carry).
#[cfg(target_arch = "riscv32")]
#[inline(always)]
pub fn mcycle() -> u64 {
    loop {
        let (hi, lo, hi2): (u32, u32, u32);
        // SAFETY: `csrr` of the read-only-in-effect counter CSRs `mcycleh`/`mcycle`: no memory
        // access, no stack use, no other architectural side effect. The harness runs in M-mode.
        unsafe {
            core::arch::asm!(
                "csrr {0}, mcycleh",
                "csrr {1}, mcycle",
                "csrr {2}, mcycleh",
                out(reg) hi,
                out(reg) lo,
                out(reg) hi2,
                options(nomem, nostack, preserves_flags)
            );
        }
        if hi == hi2 {
            return ((hi as u64) << 32) | lo as u64;
        }
    }
}

/// Host builds of this crate are not supported; keeps the module type-checkable elsewhere.
#[cfg(not(target_arch = "riscv32"))]
pub fn mcycle() -> u64 {
    0
}

/// Cycle statistics of one benchmark (per processed block).
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockCycles {
    /// Worst block.
    pub max: u32,
    /// Mean over the measured blocks.
    pub mean: u32,
    /// `true` if every call returned `Ok` and no output sample saturated.
    pub ok: bool,
}

/// Results of [`run_benchmarks`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BenchReport {
    /// Case A, folded Type I (128 MACs/sample).
    pub a_q31_direct_folded: BlockCycles,
    /// Case A, unfolded (255 MACs/sample).
    pub a_q31_direct_unfolded: BlockCycles,
    /// MACs per sample of the folded instance (expected 128).
    pub a_macs_folded: usize,
    /// Case B, `[1024 taps, 2048 taps]`, Q1.31 non-uniform.
    pub b_q31_nonuniform: [BlockCycles; 2],
    /// Case B, `[1024 taps, 2048 taps]`, Q1.15 non-uniform.
    pub b_q15_nonuniform: [BlockCycles; 2],
    /// Case B workspace bytes, Q1.31.
    pub b_q31_workspace_bytes: [usize; 2],
    /// Case B workspace bytes, Q1.15.
    pub b_q15_workspace_bytes: [usize; 2],
}

const BLOCK: usize = 64;
const BLOCKS: usize = 64;
const Q31_WS_WORDS: usize = 12_288;
const Q15_WS_WORDS: usize = 13_312;

/// xorshift32 noise, Q1.31 at about −6 dBFS peak.
struct Noise(u32);
impl Noise {
    fn next(&mut self) -> i32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as i32) >> 1
    }
}

fn measure<F: FnMut(&[i32], &mut [i32]) -> bool>(mut f: F) -> BlockCycles {
    let mut noise = Noise(0x1234_5678);
    let mut x = [0i32; BLOCK];
    let mut y = [0i32; BLOCK];
    let (mut max, mut total, mut ok) = (0u64, 0u64, true);
    for _ in 0..BLOCKS {
        for v in x.iter_mut() {
            *v = noise.next();
        }
        let t0 = mcycle();
        ok &= f(&x, &mut y);
        let dt = mcycle().wrapping_sub(t0);
        core::hint::black_box(&y);
        max = max.max(dt);
        total += dt;
    }
    BlockCycles {
        max: max.min(u32::MAX as u64) as u32,
        mean: (total / BLOCKS as u64).min(u32::MAX as u64) as u32,
        ok,
    }
}

/// Triangular linear-phase low-pass, 255 taps, `h[k] = h[254 − k] = (k + 1)·2^17` for
/// `k <= 127`: Type I symmetric, `Σ h = 2^31` (`‖h‖₁ = 1`).
fn type1_tap(k: usize) -> i32 {
    let j = if k <= 127 { k } else { 254 - k };
    ((j as i32) + 1) << 17
}

/// Centred peak (0.5) plus decaying pseudo-random tails, `‖h‖₁ < 0.9` (integer only).
fn rephase_tap(len: usize, k: usize) -> i32 {
    let mid = len / 2;
    if k == mid {
        return 1 << 30;
    }
    let d = k.abs_diff(mid) as u32;
    let mut s = (k as u32).wrapping_mul(0x9E37_79B9) ^ 0x5bd1_e995;
    s ^= s >> 15;
    let r = (s as i32) >> 8; // ±2^23 ≈ ±0.0039
    r >> (d * 12 / mid as u32).min(30)
}

#[inline(never)]
fn case_a(report: &mut BenchReport) {
    let cfg = FixedQ31DirectConfig { max_ir_len: 255 };
    let mut ws = [0i32; 3 * 255];
    if let Ok(mut c) = FixedQ31DirectConvolver::prepare_with(255, type1_tap, cfg, &mut ws) {
        report.a_macs_folded = c.capabilities().macs_per_sample;
        report.a_q31_direct_folded =
            measure(|x, y| c.process_block(x, y).is_ok() && c.saturation_count() == 0);
    }
    let mut ws = [0i32; 3 * 255];
    let mut ir = [0i32; 255];
    for (k, h) in ir.iter_mut().enumerate() {
        *h = type1_tap(k);
    }
    if let Ok(mut c) = FixedQ31DirectConvolver::prepare_unfolded(&ir, cfg, &mut ws) {
        report.a_q31_direct_unfolded =
            measure(|x, y| c.process_block(x, y).is_ok() && c.saturation_count() == 0);
    }
}

#[inline(never)]
fn case_b_q31(report: &mut BenchReport, i: usize, len: usize) {
    let Ok(plan) = preset_esp32_c6_q15(len, BLOCK) else {
        return;
    };
    let Ok(req) = FixedQ31NonUniformConvolver::memory_requirements(&plan) else {
        return;
    };
    report.b_q31_workspace_bytes[i] = req.total_bytes;
    let mut ws = [0i32; Q31_WS_WORDS];
    if req.total_i32 > ws.len() {
        return;
    }
    if let Ok(mut c) =
        FixedQ31NonUniformConvolver::prepare_with(len, |k| rephase_tap(len, k), plan, None, &mut ws)
    {
        report.b_q31_nonuniform[i] =
            measure(|x, y| c.process_block(x, y).is_ok() && c.saturation_count() == 0);
    }
}

#[inline(never)]
fn case_b_q15(report: &mut BenchReport, i: usize, len: usize) {
    let Ok(plan) = preset_esp32_c6_q15(len, BLOCK) else {
        return;
    };
    let Ok(req) = FixedQ15NonUniformConvolver::memory_requirements(&plan) else {
        return;
    };
    report.b_q15_workspace_bytes[i] = req.total_bytes;
    let mut ws = [0i16; Q15_WS_WORDS];
    if req.total_i16 > ws.len() {
        return;
    }
    if let Ok(mut c) = FixedQ15NonUniformConvolver::prepare_q31_with(
        len,
        |k| rephase_tap(len, k),
        plan,
        None,
        &mut ws,
    ) {
        let mut x16 = [0i16; BLOCK];
        let mut y16 = [0i16; BLOCK];
        report.b_q15_nonuniform[i] = measure(|x, y| {
            for (d, &s) in x16.iter_mut().zip(x) {
                *d = (s >> 16) as i16;
            }
            let ok = c.process_block(&x16, &mut y16).is_ok();
            for (d, &s) in y.iter_mut().zip(&y16) {
                *d = (s as i32) << 16;
            }
            ok && c.saturation_count() == 0
        });
    }
}

/// Runs cases A and B and returns the cycle report.
pub fn run_benchmarks() -> BenchReport {
    let mut report = BenchReport::default();
    case_a(&mut report);
    for (i, len) in [1024usize, 2048].into_iter().enumerate() {
        case_b_q31(&mut report, i, len);
        case_b_q15(&mut report, i, len);
    }
    report
}
