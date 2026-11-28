//! Cost-model NUPC partition optimizer (García 2002 / Wefers 2015 DAG + Viterbi DP).
//!
//! Solves the separable minimum-cost non-uniform partitioning problem (Wefers 2015, §6.4–§6.5,
//! Eq. 6.12, 6.14, 6.40–6.46) in `#![no_std]` with **zero heap allocation**, operating
//! strictly inside caller-supplied scratch memory during Class B (`prepare`) setup.
//!
//! ## Formal Derivation of Sample-Time Clearance Offset `c = 1`
//!
//! In `rtconv-simd`, [`crate::nonuniform::NonUniformScheduler`] executes sample-by-sample with zero
//! algorithmic latency and at most **one block in flight per FDL stage**. For an FDL stage `s`
//! with starting tap offset `D_s = ir_offset >= B_s`, block size `B_s`, and causal slack
//! `slack_s = D_s - B_s`:
//!
//! 1. Input block `k >= 0` spans samples `[k * B_s, (k + 1) * B_s - 1]`. Its final sample
//!    arrives at `t_avail(k) = (k + 1) * B_s - 1`, where `on_fdl_sample` triggers
//!    `StartBlockAndCmac` (`samples_since_ready = 0`) — sample `1` of the execution span.
//! 2. In [`crate::NonUniformConvolver::process_sample`] (and `FixedQ15NonUniformConvolver`),
//!    the stage's delay ring is read at the start of sample `t` **before** `on_fdl_sample` runs.
//!    Block `k`'s first output sample is read at global sample `n_due(k) = k * B_s + D_s`, so
//!    the block's IFFT + delay-ring commit must complete no later than sample
//!    `t_last_due(k) = k * B_s + D_s - 1`, giving at most
//!    `t_last_due(k) - t_avail(k) = D_s - B_s = slack_s` follow-up samples.
//! 3. Single-block-in-flight requires block `k` to finish strictly before block `k + 1` arrives
//!    at `t_avail(k + 1) = (k + 2) * B_s - 1`, capping follow-up samples at `B_s - 1`.
//! 4. Combining (2) and (3) gives the scheduler's exact follow-up window
//!    `W_eff = min(slack_s, B_s - 1)` (`scheduler.rs`) and inclusive computation timespan
//!    `timespan_eff(s) = W_eff + 1 = min(D_s - B_s + 1, B_s)`.
//! 5. Enforcing Wefers §6.5.4 Strategy 1 — granting every segment an admissible computation
//!    timespan at least equal to its block length `L_i = B_s` — yields:
//!
//!    ```text
//!    timespan_eff(s) >= B_s
//!      <=> min(D_s - B_s + 1, B_s) >= B_s
//!      <=> D_s - B_s + 1 >= B_s
//!      <=> D_s - B_s >= B_s - 1       (i.e. offset(i) - L_i >= L_i - c  with  c = 1)
//!      <=> offset(i) >= 2 * L_i - 1.
//!    ```
//!
//!    Contrast with Wefers' block-mode formulation (Eq. 6.24, 6.41–6.42, framing latency `B`),
//!    where `timespan = (offset - L_i)/B + 2 >= L_i/B <=> offset >= 2 * L_i - 2 * B` (`c_block = 2B`).
//!    At `D_s = 2 * B_s - 1` (`slack_s = B_s - 1`), block `k` spans `[B_s - 1, 2 * B_s - 2]`
//!    (exactly `B_s` samples), finishes at `t = D_s - 1`, is read at `t = D_s`, and has zero
//!    overlap with block `k + 1` arriving at `t = 2 * B_s - 1` (`deadline_misses() == 0`).
//!    At `D_s = 2 * B_s - 2` (`slack_s = B_s - 2`), only `B_s - 1 < B_s` samples are available.
//!
//! ## Note on Wefers (2015) Thesis §6.12 vs Appendix A.2.4 / A.2.5
//!
//! Primary-source verification against Wefers (2015) revealed two subtle labeling discrepancies
//! between prose and generated tables in the thesis:
//! - §6.12 (p. 191) labels `(128^2 256^4 1024^8 8192^10)` (`240.4` cycles/sample, clearances
//!   `(0, 1, 3, 11)`) as "strategy 1", whereas Appendix A.2.5 (p. 245, row `91392`) confirms it
//!   is **practical variant 2** (`P_0 = 2` for multi-segment partitions, `timespan >= 2` for
//!   `{2B, 4B}`, `>= 4` for `{8B, 16B, 32B}`, `>= 8` for `>= 64B`).
//! - Appendix A.2.4 ("variant 1", row `89856`: `(128^22 1024^21 8192^8)`, clearances
//!   `(0, 15, 127)`, `276.2` cycles/sample) enforced `clearance >= 2 * L_i / B - 1`, whereas
//!   §6.5.4 prose (p. 171) states `clearance >= L_i / B - 1` (which gives `(128^14 1024^14 8192^9)`
//!   at `238.6` cycles/sample with clearances `(0, 7, 63)`).
//!
//! [`optimize_block_mode`] exposes all variants explicitly via [`BlockModePolicy`] so tests
//! can verify both the thesis prose equations and the exact Appendix A.2 tables honestly.

use crate::nonuniform::plan::{
    MAX_NONUNIFORM_STAGES, NonUniformConfig, NonUniformPlan, NonUniformStageConfig,
    NonUniformStageKind,
};
use crate::{ConvError, FftSize};

/// Supported power-of-two FDL block sizes (`32..=8192`, corresponding to `N64..=N16384`).
pub const OPTIMIZER_BLOCK_SIZES: [usize; 9] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// Maximum number of FDL stages allowed alongside a direct head (`MAX_NONUNIFORM_STAGES - 1 = 7`).
pub const MAX_OPTIMIZER_FDL_STAGES: usize = MAX_NONUNIFORM_STAGES - 1;

/// Granularity unit in samples for sample-time DP offsets (equals the smallest FDL block size).
const UNIT_SAMPLES: usize = 32;

/// Sentinel value indicating no back-pointer / skipped stage.
const SKIP_MARK: u32 = u32::MAX;

/// Opaque 8-byte scratch cell supplied by callers to [`NonUniformPlan::optimize`] and
/// [`optimize_block_mode`] so the Viterbi DP executes with zero heap allocation in `#![no_std]`.
///
/// Size your slice with [`optimizer_scratch_len`] (or [`block_mode_scratch_len`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct OptimizerScratchCell(u64);

impl OptimizerScratchCell {
    /// Zero-initialized scratch cell usable in `const`/`static` array initializers.
    pub const ZERO: Self = Self(0);

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        Self(v.to_bits())
    }

    #[inline(always)]
    fn as_f64(self) -> f64 {
        f64::from_bits(self.0)
    }

    #[inline(always)]
    fn from_u32_pair(a: u32, b: u32) -> Self {
        Self(((a as u64) << 32) | (b as u64))
    }

    #[inline(always)]
    fn as_u32_pair(self) -> (u32, u32) {
        ((self.0 >> 32) as u32, self.0 as u32)
    }
}

/// Separable per-operation computational cost model (Wefers 2015, Eq. 6.12, 6.14, 6.43–6.45,
/// Ch. 7).
///
/// Every entry is indexed by `FftSize` level `0..9` corresponding to
/// `block_size = 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192` (`N = 64..=16384`).
/// Costs may be expressed in any consistent positive unit (arithmetic operations, CPU cycles, or
/// nanoseconds); only relative ratios affect the optimal partition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    /// Forward real-to-complex FFT cost `T_FFT(2 * B_k)` for one block of `B_k` samples.
    pub t_fft: [f64; 9],
    /// Inverse complex-to-real FFT cost `T_IFFT(2 * B_k)` for one block of `B_k` samples.
    pub t_ifft: [f64; 9],
    /// First-partition spectral multiply cost `T_CMUL(B_k + 1)` (Wefers Eq. 6.45).
    pub t_cmul: [f64; 9],
    /// Repeat-partition spectral multiply-accumulate cost `T_CMAC(B_k + 1)` (Wefers Eq. 6.44).
    pub t_cmac: [f64; 9],
    /// Overlap-add / delay-ring commit cost `T_ADD(B_k)` for one block of `B_k` samples.
    pub t_add: [f64; 9],
    /// Per-tap cost of the sample-wise direct FIR head (cost of `L_0` taps = `L_0 * t_direct_mac`).
    pub t_direct_mac: f64,
}

#[inline]
const fn block_size_index(block_size: usize) -> Option<usize> {
    match block_size {
        32 => Some(0),
        64 => Some(1),
        128 => Some(2),
        256 => Some(3),
        512 => Some(4),
        1024 => Some(5),
        2048 => Some(6),
        4096 => Some(7),
        8192 => Some(8),
        _ => None,
    }
}

#[inline]
const fn fft_size_index(fft: FftSize) -> usize {
    match fft {
        FftSize::N64 => 0,
        FftSize::N128 => 1,
        FftSize::N256 => 2,
        FftSize::N512 => 3,
        FftSize::N1024 => 4,
        FftSize::N2048 => 5,
        FftSize::N4096 => 6,
        FftSize::N8192 => 7,
        FftSize::N16384 => 8,
    }
}

impl CostModel {
    /// Theoretical arithmetic operation counts (Wefers 2015, Table 7.2):
    /// `T_FFT(N) = T_IFFT(N) = 1.68 * N * log2(N)`, `T_CMUL = 6 * bins`, `T_CMAC = 8 * bins`,
    /// `T_ADD = 1 * B`, `t_direct_mac = 2.0` (1 real multiply + 1 real add per tap).
    pub const fn theoretical() -> Self {
        let mut t_fft = [0.0; 9];
        let mut t_ifft = [0.0; 9];
        let mut t_cmul = [0.0; 9];
        let mut t_cmac = [0.0; 9];
        let mut t_add = [0.0; 9];
        let mut k = 0;
        while k < 9 {
            let b = OPTIMIZER_BLOCK_SIZES[k];
            let n = 2 * b;
            let log2_n = (k + 6) as f64;
            let bins = (b + 1) as f64;
            let fft = 1.68 * (n as f64) * log2_n;
            t_fft[k] = fft;
            t_ifft[k] = fft;
            t_cmul[k] = 6.0 * bins;
            t_cmac[k] = 8.0 * bins;
            t_add[k] = b as f64;
            k += 1;
        }
        Self {
            t_fft,
            t_ifft,
            t_cmul,
            t_cmac,
            t_add,
            t_direct_mac: 2.0,
        }
    }

    /// Host-calibrated `x86-64` SSE2 cost model in **nanoseconds** (`PROVEN`, `HOST_ONLY`).
    ///
    /// Calibrated directly from `benches/fft_backends_bench.rs` on Intel Xeon 2.30 GHz
    /// (`SimdFixedFftBackend(SSE2)`, `rustc 1.100.0`, `opt-level = 3`, `lto = "thin"`).
    /// Because `rtconv-simd` zeroes the spectral accumulator and runs `cmac_simd` for all `P`
    /// partitions, `t_cmul == t_cmac`.
    pub const fn x86_sse2_calibrated() -> Self {
        let t_cmac = [
            24.9, 41.9, 76.0, 144.2, 291.5, 564.1, 1_129.4, 2_261.8, 4_481.5,
        ];
        Self {
            t_fft: [
                318.1, 670.2, 1_410.8, 2_938.0, 6_091.2, 12_637.5, 26_285.5, 56_260.5, 161_735.5,
            ],
            t_ifft: [
                291.6, 601.7, 1_245.9, 2_594.8, 5_385.1, 11_213.5, 23_355.0, 50_251.5, 157_673.5,
            ],
            t_cmul: t_cmac,
            t_cmac,
            t_add: [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1_024.0, 2_048.0],
            t_direct_mac: 0.240,
        }
    }

    /// Host-calibrated `x86-64` AVX2+FMA cost model in **nanoseconds** (`PROVEN`, `HOST_ONLY`).
    ///
    /// Calibrated directly from `benches/fft_backends_bench.rs` on Intel Xeon 2.30 GHz
    /// (`SimdFixedFftBackend(AVX2)`, `rustc 1.100.0`). Captures the L1 cache boundary step in
    /// `cmac_simd` between `bins = 1025` (`0.305 ns/bin`) and `bins = 2049` (`0.473 ns/bin`),
    /// mirroring Wefers Table 7.2.
    pub const fn x86_avx2_calibrated() -> Self {
        let t_cmac = [
            24.4, 32.2, 45.6, 77.6, 144.2, 313.1, 968.9, 1_990.8, 3_960.9,
        ];
        Self {
            t_fft: [
                319.5, 624.4, 1_270.9, 2_600.1, 5_263.4, 10_778.4, 21_953.0, 46_307.5, 121_515.0,
            ],
            t_ifft: [
                315.5, 609.6, 1_221.1, 2_501.8, 5_169.1, 10_676.5, 21_866.5, 46_231.5, 127_369.0,
            ],
            t_cmul: t_cmac,
            t_cmac,
            t_add: [6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6, 819.2, 1_638.4],
            t_direct_mac: 0.234,
        }
    }

    /// Wefers (2015) test-system cost model in **CPU cycles** (Intel Core2 Quad Q6600 @ 2.405 GHz,
    /// Intel IPP 7.0 + FFTW 3.3.3; Ch. 7 Table 7.2 & Appendix A.2).
    ///
    /// Intended for reproducing Wefers §6.5.3, §6.12 (`88,200` taps at `B = 128`), and Appendix A.2
    /// optimal partitions. Per-segment open/repeat costs match A.2.1–A.2.5 within `0.15` cycles/sample.
    pub const fn wefers_2015_test_system() -> Self {
        // Segment open cost (open_i * B_i) split evenly across FFT/IFFT plus Table 7.2 CMUL & ADD,
        // and exact per-repetition CMAC costs derived from Appendix A.2.1 / A.2.2 / A.2.3 / A.2.5.
        // Levels 0..1 (B=32, 64) extrapolate Table 7.2/7.4; levels 2..8 (B=128..8192) match A.2.
        let open_per_sample = [
            54.0,  // B=32 (N=64, extrapolated from Table 7.4)
            47.5,  // B=64 (N=128, extrapolated from Table 7.4)
            43.00, // B=128 (A.2.1 row 128)
            40.00, // B=256 (A.2.2 row 256)
            40.90, // B=512 (A.2.3 row 512)
            43.00, // B=1024 (A.2.1/A.2.5)
            50.20, // B=2048 (A.2.1/A.2.5, L1->L2 boundary)
            54.90, // B=4096 (A.2.2/A.2.5)
            54.90, // B=8192 (A.2.1/A.2.5)
        ];
        let step_per_sample = [
            3.10,  // B=32
            2.95,  // B=64
            2.863, // B=128
            2.686, // B=256
            2.620, // B=512
            2.563, // B=1024
            3.360, // B=2048 (Table 7.2 cache jump)
            3.389, // B=4096
            3.390, // B=8192
        ];
        let mut t_fft = [0.0; 9];
        let mut t_ifft = [0.0; 9];
        let mut t_cmul = [0.0; 9];
        let mut t_cmac = [0.0; 9];
        let mut t_add = [0.0; 9];
        let mut k = 0;
        while k < 9 {
            let b = OPTIMIZER_BLOCK_SIZES[k] as f64;
            let add = (if k <= 5 { 0.69 } else { 1.17 }) * b;
            let cmul = (if k <= 5 { 2.16 } else { 2.82 }) * (b + 1.0);
            let cmac = step_per_sample[k] * b;
            let open_total = open_per_sample[k] * b;
            let transforms = open_total - cmul - add;
            t_fft[k] = 0.5 * transforms;
            t_ifft[k] = 0.5 * transforms;
            t_cmul[k] = cmul;
            t_cmac[k] = cmac;
            t_add[k] = add;
            k += 1;
        }
        Self {
            t_fft,
            t_ifft,
            t_cmul,
            t_cmac,
            t_add,
            t_direct_mac: 1.17,
        }
    }

    /// **ESTIMATE ONLY (`INFERRED`, `DEVICE_VERIFIED = NO`)**: heuristic scalar `f32` cycle model
    /// for ESP32-P4 (`riscv32imafc-unknown-none-elf`, hardware FPU without vector SIMD).
    ///
    /// Replace with measured on-device `mcycle` timings once Issue #24 runs on physical hardware.
    pub const fn esp32_p4_f32_estimate() -> Self {
        let mut t_fft = [0.0; 9];
        let mut t_ifft = [0.0; 9];
        let mut t_cmul = [0.0; 9];
        let mut t_cmac = [0.0; 9];
        let mut t_add = [0.0; 9];
        let mut k = 0;
        while k < 9 {
            let b = OPTIMIZER_BLOCK_SIZES[k];
            let n = 2 * b;
            let log2_n = (k + 6) as f64;
            let bins = (b + 1) as f64;
            // Scalar RV32IMAFC estimate: ~5.5 cyc/(N log2 N) for strided quarter-wave CFFT+untangle,
            // ~14 cyc/bin for split-complex CMAC, ~3 cyc/tap for direct FIR MAC, ~2 cyc/sample add.
            let fft = 5.5 * (n as f64) * log2_n;
            t_fft[k] = fft;
            t_ifft[k] = fft;
            t_cmul[k] = 14.0 * bins;
            t_cmac[k] = 14.0 * bins;
            t_add[k] = 2.0 * (b as f64);
            k += 1;
        }
        Self {
            t_fft,
            t_ifft,
            t_cmul,
            t_cmac,
            t_add,
            t_direct_mac: 3.0,
        }
    }

    /// **ESTIMATE ONLY (`INFERRED`, `DEVICE_VERIFIED = NO`)**: heuristic integer Q1.15/BFP cycle
    /// model for ESP32-C6 (`riscv32imac-unknown-none-elf`, FPU-less).
    ///
    /// Captures the higher integer BFP per-stage scaling and `i64` workspace-packed CMAC cost.
    /// When optimizing for `FixedQ15NonUniformConvolver`, pass `max_block_size = 1024` (`N2048`).
    /// Replace with measured on-device `mcycle` timings once Issue #24 runs on physical hardware.
    pub const fn esp32_c6_q15_estimate() -> Self {
        let mut t_fft = [0.0; 9];
        let mut t_ifft = [0.0; 9];
        let mut t_cmul = [0.0; 9];
        let mut t_cmac = [0.0; 9];
        let mut t_add = [0.0; 9];
        let mut k = 0;
        while k < 9 {
            let b = OPTIMIZER_BLOCK_SIZES[k];
            let n = 2 * b;
            let log2_n = (k + 6) as f64;
            let bins = (b + 1) as f64;
            // Integer RV32IMAC estimate: full N-point BFP radix-2 with conditional rescale + packed
            // workspace load/store (~18 cyc/(N log2 N)), two-pass i64 CMAC (~28 cyc/bin),
            // i64 direct FIR (~6 cyc/tap), Q1.23 delay ring sum (~3 cyc/sample).
            let fft = 18.0 * (n as f64) * log2_n;
            t_fft[k] = fft;
            t_ifft[k] = fft;
            t_cmul[k] = 28.0 * bins;
            t_cmac[k] = 28.0 * bins;
            t_add[k] = 3.0 * (b as f64);
            k += 1;
        }
        Self {
            t_fft,
            t_ifft,
            t_cmul,
            t_cmac,
            t_add,
            t_direct_mac: 6.0,
        }
    }

    /// Default compile-time `f32` cost model for the current compilation target.
    ///
    /// Uses the host-calibrated x86-64 AVX2 model when compiled with `avx2`, the host-calibrated
    /// x86-64 SSE2 model on baseline x86/x86-64, the ESP32-P4 scalar estimate on RISC-V/ARM
    /// targets, and [`Self::theoretical`] elsewhere.
    pub const fn for_target_f32() -> Self {
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx2"
        ))]
        {
            Self::x86_avx2_calibrated()
        }
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(target_feature = "avx2")
        ))]
        {
            Self::x86_sse2_calibrated()
        }
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "arm"))]
        {
            Self::esp32_p4_f32_estimate()
        }
        #[cfg(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "riscv32",
            target_arch = "riscv64",
            target_arch = "arm"
        )))]
        {
            Self::theoretical()
        }
    }

    /// Runtime-matched `f32` cost model for a detected [`crate::SimdBackend`].
    ///
    /// On x86/x86-64 selects [`Self::x86_avx2_calibrated`] when AVX2 or AVX-512 is active and
    /// [`Self::x86_sse2_calibrated`] otherwise; falls back to [`Self::for_target_f32`] on other
    /// architectures.
    pub fn for_backend_f32(backend: crate::SimdBackend) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let lvl = backend.level();
            if !backend.is_scalar() && (lvl.as_avx2().is_some() || lvl.as_avx512().is_some()) {
                return Self::x86_avx2_calibrated();
            }
            Self::x86_sse2_calibrated()
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let _ = backend;
            Self::for_target_f32()
        }
    }

    /// Validates that every entry in the cost model is finite and non-negative, with strictly
    /// positive transform/CMAC/direct costs.
    pub fn validate(&self) -> Result<(), ConvError> {
        if !self.t_direct_mac.is_finite() || self.t_direct_mac <= 0.0 {
            return Err(ConvError::InvalidNonUniformConfig);
        }
        for k in 0..9 {
            let vals = [
                self.t_fft[k],
                self.t_ifft[k],
                self.t_cmul[k],
                self.t_cmac[k],
                self.t_add[k],
            ];
            for v in vals {
                if !v.is_finite() || v < 0.0 {
                    return Err(ConvError::InvalidNonUniformConfig);
                }
            }
            if self.t_fft[k] + self.t_ifft[k] + self.t_cmul[k] <= 0.0 {
                return Err(ConvError::InvalidNonUniformConfig);
            }
        }
        Ok(())
    }

    /// Forward RFFT cost for `size`.
    #[inline]
    pub const fn t_fft_for(&self, size: FftSize) -> f64 {
        self.t_fft[fft_size_index(size)]
    }

    /// Inverse IFFT cost for `size`.
    #[inline]
    pub const fn t_ifft_for(&self, size: FftSize) -> f64 {
        self.t_ifft[fft_size_index(size)]
    }

    /// First-partition CMUL cost for `size`.
    #[inline]
    pub const fn t_cmul_for(&self, size: FftSize) -> f64 {
        self.t_cmul[fft_size_index(size)]
    }

    /// Repeat-partition CMAC cost for `size`.
    #[inline]
    pub const fn t_cmac_for(&self, size: FftSize) -> f64 {
        self.t_cmac[fft_size_index(size)]
    }

    /// Block overlap-add cost for `size`.
    #[inline]
    pub const fn t_add_for(&self, size: FftSize) -> f64 {
        self.t_add[fft_size_index(size)]
    }

    /// Per-sample cost of opening a new FDL segment of length `block_size` with `P = 1` partition
    /// (Wefers Eq. 6.45): `(T_FFT(2B) + T_CMUL(B+1) + T_IFFT(2B) + T_ADD(B)) / B`.
    #[inline]
    pub const fn segment_open_cost_per_sample(&self, block_size: usize) -> Option<f64> {
        let Some(k) = block_size_index(block_size) else {
            return None;
        };
        Some((self.t_fft[k] + self.t_cmul[k] + self.t_ifft[k] + self.t_add[k]) / block_size as f64)
    }

    /// Per-sample cost of adding one additional FDL partition (`P -> P + 1`) to a segment of
    /// length `block_size` (Wefers Eq. 6.44): `T_CMAC(B+1) / B`.
    #[inline]
    pub const fn segment_repeat_cost_per_sample(&self, block_size: usize) -> Option<f64> {
        let Some(k) = block_size_index(block_size) else {
            return None;
        };
        Some(self.t_cmac[k] / block_size as f64)
    }

    /// Total per-sample cost of one FDL segment of `block_size` with `partitions` partitions
    /// (Wefers Eq. 5.6 + Eq. 6.12).
    #[inline]
    pub const fn segment_cost_per_sample(
        &self,
        block_size: usize,
        partitions: usize,
    ) -> Option<f64> {
        if partitions == 0 {
            return Some(0.0);
        }
        let Some(open) = self.segment_open_cost_per_sample(block_size) else {
            return None;
        };
        let Some(rep) = self.segment_repeat_cost_per_sample(block_size) else {
            return None;
        };
        Some(open + (partitions - 1) as f64 * rep)
    }

    /// Direct-form FIR cost per output sample for `ir_len` taps.
    #[inline]
    pub const fn direct_cost_per_sample(&self, ir_len: usize) -> f64 {
        ir_len as f64 * self.t_direct_mac
    }

    /// Zero-latency `HybridConvolver` cost per output sample (`head_len = tail_block_size = B`).
    #[inline]
    pub const fn hybrid_cost_per_sample(&self, ir_len: usize, block_size: usize) -> Option<f64> {
        if ir_len <= block_size {
            return Some(self.direct_cost_per_sample(ir_len));
        }
        let tail_parts = (ir_len - block_size).div_ceil(block_size);
        let Some(tail) = self.segment_cost_per_sample(block_size, tail_parts) else {
            return None;
        };
        Some(self.direct_cost_per_sample(block_size) + tail)
    }

    /// Evaluates the separable per-sample cost (Wefers Eq. 6.12 / 6.14) of any validated
    /// [`NonUniformPlan`].
    pub fn plan_cost_per_sample(&self, plan: &NonUniformPlan) -> Result<f64, ConvError> {
        let mut total = 0.0f64;
        for st in plan.stages() {
            match st.kind {
                NonUniformStageKind::Direct => {
                    total += self.direct_cost_per_sample(st.ir_len);
                }
                NonUniformStageKind::Fdl => {
                    let c = self
                        .segment_cost_per_sample(st.block_size, st.num_partitions)
                        .ok_or(ConvError::UnsupportedFftSize {
                            fft_size: st.fft_size,
                        })?;
                    total += c;
                }
            }
        }
        Ok(total)
    }
}

/// Minimum-clearance constraint policy enforced during sample-time partition optimization.
///
/// See module documentation for the formal derivation of `c = 1` under
/// [`crate::nonuniform::NonUniformScheduler`]'s zero-latency sample-time execution model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearancePolicy {
    /// Strict causality only (`offset(i) >= B_i`, i.e. `slack >= 0`).
    ///
    /// Produces the theoretical minimum average load (analogous to Wefers §6.5.3), typically
    /// leaving zero scheduling clearance on segments.
    MinLoad,
    /// Sample-time Strategy 1 (Gardner/Wefers §6.5.4 principle): every FDL segment receives an
    /// admissible computation span equal to its full block length `B_i`:
    /// `timespan_eff(i) = min(slack_i, B_i - 1) + 1 >= B_i  <=>  offset(i) - B_i >= B_i - 1`
    /// (`c = 1`).
    Strategy1,
    /// Sample-time Strategy 2 (practical Wefers §6.5.4 heuristic adapted to host callback size `H`):
    /// - `B_i <= 4 * H`: `timespan_eff >= min(B_i, 2 * H)`
    /// - `8 * H <= B_i <= 64 * H`: `timespan_eff >= min(B_i, 4 * H)`
    /// - `B_i > 64 * H`: `timespan_eff >= min(B_i, 8 * H)`
    ///
    /// Always less restrictive than (or equal to) [`Self::Strategy1`], while bounding the leading
    /// direct head so the callback stays lightweight when FDL stages are present.
    Strategy2,
    /// Custom minimum slack (`offset(i) - B_i >= min_slack_samples`) for every FDL stage.
    Custom {
        /// Minimum causal slack `D_s - B_s` in samples required for every FDL stage.
        min_slack_samples: usize,
    },
}

impl ClearancePolicy {
    /// Minimum required causal slack `D_s - B_s` in samples for an FDL stage of size `block_size`
    /// under callback size `host_block_max`.
    #[inline]
    pub const fn min_slack_samples(&self, block_size: usize, host_block_max: usize) -> usize {
        let h = if host_block_max == 0 {
            1
        } else {
            host_block_max
        };
        match *self {
            ClearancePolicy::MinLoad => 0,
            ClearancePolicy::Strategy1 => block_size.saturating_sub(1),
            ClearancePolicy::Strategy2 => {
                let cycles = if block_size <= 4 * h {
                    2usize
                } else if block_size <= 64 * h {
                    4usize
                } else {
                    8usize
                };
                let target_span = cycles.saturating_mul(h);
                let eff_span = if target_span < block_size {
                    target_span
                } else {
                    block_size
                };
                eff_span.saturating_sub(1)
            }
            ClearancePolicy::Custom { min_slack_samples } => min_slack_samples,
        }
    }

    /// Minimum required tap offset `D_s = block_size + min_slack_samples` for an FDL stage.
    #[inline]
    pub const fn min_offset_samples(&self, block_size: usize, host_block_max: usize) -> usize {
        block_size.saturating_add(self.min_slack_samples(block_size, host_block_max))
    }

    /// Maximum direct head length `L_0` permitted when at least one FDL stage is present
    /// (`usize::MAX` = unbounded).
    #[inline]
    const fn max_direct_head_with_fdl(&self, host_block_max: usize) -> usize {
        match *self {
            ClearancePolicy::Strategy2 => {
                let h = if host_block_max < 32 {
                    32
                } else {
                    host_block_max
                };
                // Mirror Wefers §6.5.4 `2 <= P_0 <= 4` leading-segment cap (at least 128 taps so
                // B=64/B=128 first FDL stages remain reachable when H=32).
                let cap = h.saturating_mul(4);
                if cap < 128 { 128 } else { cap }
            }
            _ => usize::MAX,
        }
    }
}

/// Summary report returned alongside an optimized [`NonUniformPlan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizationReport {
    /// Predicted average computational cost per output sample under the supplied [`CostModel`].
    pub predicted_cost_per_sample: f64,
    /// Conservative worst-case operation count per callback (`plan.estimated_worst_ops_per_callback()`).
    pub predicted_worst_ops_per_callback: usize,
    /// Exact impulse response length covered (`plan.max_ir_len()`).
    pub covered_ir_len: usize,
    /// Length of the leading direct FIR head (`0` if none; always `>= 1` for causal sample-time plans).
    pub direct_head_len: usize,
    /// Number of FDL stages in the optimized plan (`0..=7`).
    pub fdl_stage_count: usize,
}

#[inline]
const fn num_levels_for_max_block(max_block_size: usize) -> usize {
    let mut k = 0;
    while k < OPTIMIZER_BLOCK_SIZES.len() && OPTIMIZER_BLOCK_SIZES[k] <= max_block_size {
        k += 1;
    }
    k
}

#[inline]
const fn u_max_for_sample_dp(ir_len: usize, max_block_size: usize) -> usize {
    if ir_len == 0 {
        return 0;
    }
    let u_target = ir_len.div_ceil(UNIT_SAMPLES);
    // Allow over-covering by less than one full partition of the largest usable FDL block size
    // (capped at `min(max_block_size, ir_len)` rounded up to a power of two).
    let usable_max_b = if max_block_size < UNIT_SAMPLES {
        UNIT_SAMPLES
    } else if max_block_size > 8192 {
        8192
    } else {
        max_block_size
    };
    let max_step = usable_max_b / UNIT_SAMPLES;
    u_target + max_step
}

/// Exact number of [`OptimizerScratchCell`] elements required by
/// [`NonUniformPlan::optimize_with_max_block_size`] (and [`NonUniformPlan::optimize`] with
/// `max_block_size = 8192`).
///
/// Layout inside `&mut [OptimizerScratchCell]` (`cols = u_max + 1`, `levels <= 9`,
/// `S = MAX_OPTIMIZER_FDL_STAGES = 7`):
/// - `dp[(S + 1) * cols]` (`f64` costs, 1 cell each),
/// - `f_val[cols]` (`f64` running minimum, 1 cell each),
/// - `f_start[cols]` (`u32` packed 2 per cell -> `cols.div_ceil(2)` cells),
/// - `backptr[levels * S * cols]` (`u32` packed 2 per cell -> `(levels * S * cols).div_ceil(2)` cells).
pub const fn optimizer_scratch_len(ir_len: usize, max_block_size: usize) -> usize {
    if ir_len < UNIT_SAMPLES {
        return 0;
    }
    let levels = num_levels_for_max_block(max_block_size);
    if levels == 0 {
        return 0;
    }
    let cols = u_max_for_sample_dp(ir_len, max_block_size) + 1;
    let dp_cells = (MAX_OPTIMIZER_FDL_STAGES + 1) * cols;
    let f_val_cells = cols;
    let f_start_cells = cols.div_ceil(2);
    let bp_entries = levels * MAX_OPTIMIZER_FDL_STAGES * cols;
    let bp_cells = bp_entries.div_ceil(2);
    dp_cells + f_val_cells + f_start_cells + bp_cells
}

#[inline(always)]
fn read_packed_u32(cells: &[OptimizerScratchCell], idx: usize) -> u32 {
    let (hi, lo) = cells[idx / 2].as_u32_pair();
    if idx & 1 == 0 { hi } else { lo }
}

#[inline(always)]
fn write_packed_u32(cells: &mut [OptimizerScratchCell], idx: usize, val: u32) {
    let (hi, lo) = cells[idx / 2].as_u32_pair();
    cells[idx / 2] = if idx & 1 == 0 {
        OptimizerScratchCell::from_u32_pair(val, lo)
    } else {
        OptimizerScratchCell::from_u32_pair(hi, val)
    };
}

/// Candidate stage list reconstructed from one DP pass.
#[derive(Clone, Copy)]
struct CandidatePlan {
    cost: f64,
    stages: [NonUniformStageConfig; MAX_NONUNIFORM_STAGES],
    count: usize,
}

impl CandidatePlan {
    const fn inf() -> Self {
        Self {
            cost: f64::INFINITY,
            stages: [NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 0,
                block_size: 1,
            }; MAX_NONUNIFORM_STAGES],
            count: 0,
        }
    }
}

/// Executes one exact level-by-level Bellman/Viterbi DP pass for direct head residue `rem`
/// (`L_0 = 32 * u0 + rem` for `u0 >= 0`, or `L_0 = 32 * u0 - 1` represented as `rem = 31` with
/// `u0 >= 1`), storing all state inside `scratch`.
#[allow(clippy::too_many_arguments)]
fn run_sample_dp_pass(
    ir_len: usize,
    host_block_max: usize,
    levels: usize,
    max_fdl_stages: usize,
    cost_model: &CostModel,
    policy: ClearancePolicy,
    rem: usize,
    scratch: &mut [OptimizerScratchCell],
) -> CandidatePlan {
    debug_assert!(rem < UNIT_SAMPLES);
    debug_assert!(max_fdl_stages <= MAX_OPTIMIZER_FDL_STAGES);

    // Every offset has the form `off(u) = u * 32 + rem` (note `rem = 31` yields 31, 63, 95, ...).
    // Smallest `u` such that `u * 32 + rem >= ir_len`:
    let u_target = if ir_len <= rem {
        0
    } else {
        (ir_len - rem).div_ceil(UNIT_SAMPLES)
    };
    let max_b = OPTIMIZER_BLOCK_SIZES[levels - 1];
    let u_max = u_target + max_b / UNIT_SAMPLES;
    let cols = u_max + 1;

    let dp_len = (max_fdl_stages + 1) * cols;
    let f_val_len = cols;
    let f_start_cells = cols.div_ceil(2);
    let bp_cells = (levels * max_fdl_stages * cols).div_ceil(2);

    let (dp_cells, rest) = scratch.split_at_mut(dp_len);
    let (f_val_cells, rest) = rest.split_at_mut(f_val_len);
    let (f_start_slice, rest) = rest.split_at_mut(f_start_cells);
    let bp_slice = &mut rest[..bp_cells];

    // Initialize dp[0][u] = direct head cost for L_0 = u * 32 + rem (capped when FDL stages follow).
    let max_l0_fdl = policy.max_direct_head_with_fdl(host_block_max);
    for c in dp_cells.iter_mut() {
        *c = OptimizerScratchCell::from_f64(f64::INFINITY);
    }
    for (u, cell) in dp_cells[..=u_max].iter_mut().enumerate() {
        let l0 = u * UNIT_SAMPLES + rem;
        if l0 > 0 && l0 <= max_l0_fdl {
            *cell = OptimizerScratchCell::from_f64(cost_model.direct_cost_per_sample(l0));
        }
    }

    // Level-by-level exact DP across strictly increasing power-of-two FDL block sizes.
    for (k, &b) in OPTIMIZER_BLOCK_SIZES.iter().enumerate().take(levels) {
        let step = b / UNIT_SAMPLES;
        let open_c = (cost_model.t_fft[k]
            + cost_model.t_cmul[k]
            + cost_model.t_ifft[k]
            + cost_model.t_add[k])
            / (b as f64);
        let rep_c = cost_model.t_cmac[k] / (b as f64);

        // Causality & clearance constraint: off(u0) = u0 * 32 + rem >= min_offset.
        let min_off = policy.min_offset_samples(b, host_block_max);
        let u0_min = if min_off <= rem {
            0
        } else {
            (min_off - rem).div_ceil(UNIT_SAMPLES)
        };

        // Iterate `s` downward so `dp[s - 1]` stays strictly from levels `< k`.
        for s in (1..=max_fdl_stages).rev() {
            let bp_base = (k * max_fdl_stages + (s - 1)) * cols;
            for (u, cell) in f_val_cells[..cols].iter_mut().enumerate() {
                *cell = OptimizerScratchCell::from_f64(f64::INFINITY);
                write_packed_u32(f_start_slice, u, SKIP_MARK);
                write_packed_u32(bp_slice, bp_base + u, SKIP_MARK);
            }

            let prev_row = (s - 1) * cols;
            let cur_row = s * cols;

            for u in step..=u_max {
                let u_prev = u - step;
                let mut best = f64::INFINITY;
                let mut best_start = SKIP_MARK;

                // Option 1: open a new segment of size `b` starting at `u_prev` (P = 1).
                if u_prev >= u0_min {
                    let prev_cost = dp_cells[prev_row + u_prev].as_f64();
                    if prev_cost.is_finite() {
                        let cand = prev_cost + open_c;
                        if cand < best {
                            best = cand;
                            best_start = u_prev as u32;
                        }
                    }
                }

                // Option 2: repeat existing segment of size `b` ending at `u_prev` (P >= 2).
                let f_prev = f_val_cells[u_prev].as_f64();
                if f_prev.is_finite() {
                    let cand = f_prev + rep_c;
                    if cand < best {
                        best = cand;
                        best_start = read_packed_u32(f_start_slice, u_prev);
                    }
                }

                f_val_cells[u] = OptimizerScratchCell::from_f64(best);
                write_packed_u32(f_start_slice, u, best_start);

                let cur_cost = dp_cells[cur_row + u].as_f64();
                if best < cur_cost {
                    dp_cells[cur_row + u] = OptimizerScratchCell::from_f64(best);
                    write_packed_u32(bp_slice, bp_base + u, best_start);
                }
            }
        }
    }

    // Pick best feasible terminal state covering at least `ir_len` taps (`s >= 1`).
    // (Pure Direct-only `0..ir_len` is compared separately by the caller without any cap.)
    let mut best_cost = f64::INFINITY;
    let mut best_s = 0usize;
    let mut best_u = 0usize;
    for s in 1..=max_fdl_stages {
        let row = s * cols;
        for u in u_target..=u_max {
            let c = dp_cells[row + u].as_f64();
            if c < best_cost {
                best_cost = c;
                best_s = s;
                best_u = u;
            }
        }
    }

    if !best_cost.is_finite() {
        return CandidatePlan::inf();
    }

    // Back-track exact FDL stage sequence in reverse, then reverse into ascending offset order.
    let mut fdl_rev = [(0usize, 0usize, 0usize); MAX_OPTIMIZER_FDL_STAGES]; // (offset, end_uncapped, block_size)
    let mut fdl_count = 0usize;
    let mut u = best_u;
    let mut s = best_s;
    for k in (0..levels).rev() {
        if s == 0 {
            break;
        }
        let bp_base = (k * max_fdl_stages + (s - 1)) * cols;
        let u0 = read_packed_u32(bp_slice, bp_base + u);
        if u0 != SKIP_MARK {
            let u0 = u0 as usize;
            let off = u0 * UNIT_SAMPLES + rem;
            let end = u * UNIT_SAMPLES + rem;
            fdl_rev[fdl_count] = (off, end, OPTIMIZER_BLOCK_SIZES[k]);
            fdl_count += 1;
            u = u0;
            s -= 1;
        }
    }

    let l0 = u * UNIT_SAMPLES + rem;
    if l0 == 0 || l0 >= ir_len || fdl_count == 0 {
        return CandidatePlan::inf();
    }

    let mut cand = CandidatePlan {
        cost: best_cost,
        stages: [NonUniformStageConfig {
            ir_offset: 0,
            ir_len: 0,
            block_size: 1,
        }; MAX_NONUNIFORM_STAGES],
        count: 0,
    };
    cand.stages[0] = NonUniformStageConfig {
        ir_offset: 0,
        ir_len: l0,
        block_size: 1,
    };
    cand.count = 1;

    // Emit FDL stages in ascending order, clipping the final stage cleanly to `ir_len`.
    for i in (0..fdl_count).rev() {
        let (off, end_uncapped, b) = fdl_rev[i];
        if off >= ir_len {
            // Trailing stage sits completely beyond `ir_len` (unreachable for strictly positive costs).
            continue;
        }
        let end = end_uncapped.min(ir_len);
        cand.stages[cand.count] = NonUniformStageConfig {
            ir_offset: off,
            ir_len: end - off,
            block_size: b,
        };
        cand.count += 1;
    }

    cand
}

impl NonUniformPlan {
    /// Computes the globally optimal causal sample-time NUPC partition under `cost_model` and
    /// `policy` across all supported `f32` FDL block sizes (`32..=8192`, `N64..=N16384`).
    ///
    /// Runs in Class B (`prepare`) with **zero heap allocation**, using caller-provided `scratch`
    /// of at least [`optimizer_scratch_len`]`(ir_len, 8192)` cells.
    pub fn optimize(
        ir_len: usize,
        host_block_max: usize,
        cost_model: &CostModel,
        policy: ClearancePolicy,
        scratch: &mut [OptimizerScratchCell],
    ) -> Result<(Self, OptimizationReport), ConvError> {
        Self::optimize_with_max_block_size(
            ir_len,
            host_block_max,
            8192,
            cost_model,
            policy,
            scratch,
        )
    }

    /// Computes the globally optimal causal sample-time NUPC partition with an explicit upper cap
    /// `max_block_size` (e.g. `1024` for [`crate::FixedQ15NonUniformConvolver`], whose FFT ceiling
    /// is `N2048`, or `8192` for [`crate::NonUniformConvolver`]).
    ///
    /// Uses caller-provided `scratch` of at least
    /// [`optimizer_scratch_len`]`(ir_len, max_block_size)` cells.
    pub fn optimize_with_max_block_size(
        ir_len: usize,
        host_block_max: usize,
        max_block_size: usize,
        cost_model: &CostModel,
        policy: ClearancePolicy,
        scratch: &mut [OptimizerScratchCell],
    ) -> Result<(Self, OptimizationReport), ConvError> {
        Self::optimize_internal(
            ir_len,
            host_block_max,
            max_block_size,
            MAX_OPTIMIZER_FDL_STAGES,
            cost_model,
            policy,
            scratch,
        )
    }

    pub(crate) fn optimize_internal(
        ir_len: usize,
        host_block_max: usize,
        max_block_size: usize,
        max_fdl_stages: usize,
        cost_model: &CostModel,
        policy: ClearancePolicy,
        scratch: &mut [OptimizerScratchCell],
    ) -> Result<(Self, OptimizationReport), ConvError> {
        if ir_len == 0 || host_block_max == 0 || max_fdl_stages == 0 {
            return Err(ConvError::InvalidNonUniformConfig);
        }
        cost_model.validate()?;

        let max_fdl = max_fdl_stages.min(MAX_OPTIMIZER_FDL_STAGES);

        // Baseline: single Direct FIR stage covering the entire IR (always causal).
        let direct_cost = cost_model.direct_cost_per_sample(ir_len);
        let mut best = CandidatePlan {
            cost: direct_cost,
            stages: [NonUniformStageConfig {
                ir_offset: 0,
                ir_len: 0,
                block_size: 1,
            }; MAX_NONUNIFORM_STAGES],
            count: 1,
        };
        best.stages[0] = NonUniformStageConfig {
            ir_offset: 0,
            ir_len,
            block_size: 1,
        };

        let levels = num_levels_for_max_block(max_block_size);
        if ir_len >= UNIT_SAMPLES && levels > 0 {
            // Also cap `levels` so we don't waste work on block sizes strictly >= ir_len
            // (since causality requires ir_offset >= block_size and ir_len > ir_offset).
            let mut active_levels = 0usize;
            while active_levels < levels && OPTIMIZER_BLOCK_SIZES[active_levels] < ir_len {
                active_levels += 1;
            }
            if active_levels > 0 {
                let required = optimizer_scratch_len(ir_len, max_block_size);
                if scratch.len() < required {
                    return Err(ConvError::WorkspaceTooSmall {
                        required,
                        provided: scratch.len(),
                    });
                }

                // Evaluate up to 3 distinct mod-32 head residues:
                // - `0`: multiples of 32 (standard block alignment),
                // - `31`: `32*u - 1` (captures exact `c = 1` Strategy1/2 boundary `2*B - 1`),
                // - `ir_len % 32`: absorbs trailing partial-block taps into the direct head when
                //   cheaper than adding an extra FDL partition.
                let mut rems = [0usize; 3];
                let mut rem_count = 1usize;
                let r_exact = ir_len % UNIT_SAMPLES;
                if r_exact != 0 {
                    rems[rem_count] = r_exact;
                    rem_count += 1;
                }
                if !matches!(policy, ClearancePolicy::MinLoad) && r_exact != UNIT_SAMPLES - 1 {
                    rems[rem_count] = UNIT_SAMPLES - 1;
                    rem_count += 1;
                }

                for &rem in &rems[..rem_count] {
                    let cand = run_sample_dp_pass(
                        ir_len,
                        host_block_max,
                        active_levels,
                        max_fdl,
                        cost_model,
                        policy,
                        rem,
                        scratch,
                    );
                    if cand.cost < best.cost {
                        best = cand;
                    }
                }
            }
        }

        let plan = NonUniformPlan::validate(NonUniformConfig {
            max_ir_len: ir_len,
            host_block_max,
            stages: &best.stages[..best.count],
        })?;
        let exact_cost = cost_model.plan_cost_per_sample(&plan)?;
        let direct_head_len = plan
            .stages()
            .first()
            .filter(|s| s.kind == NonUniformStageKind::Direct)
            .map_or(0, |s| s.ir_len);
        let fdl_stage_count = plan.stage_count() - usize::from(direct_head_len > 0);
        let report = OptimizationReport {
            predicted_cost_per_sample: exact_cost,
            predicted_worst_ops_per_callback: plan.estimated_worst_ops_per_callback(),
            covered_ir_len: plan.max_ir_len(),
            direct_head_len,
            fdl_stage_count,
        };
        Ok((plan, report))
    }
}

// ------------------------------------------------------------------------------------------
// Workspace-backed zero-stack optimization helpers for `AutoConvolver` (`f32` and `i16`)
// ------------------------------------------------------------------------------------------

/// Required `&mut [f32]` scratch elements for [`optimize_in_f32_scratch`].
#[cfg(feature = "auto")]
pub(crate) const fn optimizer_scratch_len_f32(
    ir_len: usize,
    max_block_size: usize,
    max_fdl_stages: usize,
) -> usize {
    if ir_len < UNIT_SAMPLES || max_block_size < UNIT_SAMPLES {
        return 0;
    }
    let levels = num_levels_for_max_block(max_block_size);
    if levels == 0 {
        return 0;
    }
    let cols = u_max_for_sample_dp(ir_len, max_block_size) + 1;
    // dp[(S+1)*cols] + f_val[cols] + f_start[cols] + bp[levels * S * cols] (all stored as f32;
    // integers up to 2^24 are exact in binary32, and cols <= 4353 << 2^24).
    (max_fdl_stages + 1) * cols + 2 * cols + levels * max_fdl_stages * cols
}

/// Required `&mut [i16]` scratch elements for [`optimize_in_i16_scratch`].
#[cfg(feature = "auto")]
pub(crate) const fn optimizer_scratch_len_i16(
    ir_len: usize,
    max_block_size: usize,
    max_fdl_stages: usize,
) -> usize {
    if ir_len < UNIT_SAMPLES || max_block_size < UNIT_SAMPLES {
        return 0;
    }
    let levels = num_levels_for_max_block(max_block_size);
    if levels == 0 {
        return 0;
    }
    let cols = u_max_for_sample_dp(ir_len, max_block_size) + 1;
    // dp: (S+1)*cols*2 i16, f_val: cols*2 i16, f_start: cols i16, bp: levels*S*cols i16
    (max_fdl_stages + 1) * cols * 2 + 2 * cols + cols + levels * max_fdl_stages * cols
}

/// Runs the exact same Viterbi DP inside an `&mut [f32]` caller buffer (`AutoConvolver` zero-alloc
/// integration, `#![forbid(unsafe_code)]`, zero stack arrays). Hardware-FPU builds only.
#[cfg(all(feature = "auto", not(rtconv_no_fpu)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_in_f32_scratch(
    ir_len: usize,
    host_block_max: usize,
    max_block_size: usize,
    max_fdl_stages: usize,
    cost_model: &CostModel,
    policy: ClearancePolicy,
    scratch: &mut [f32],
) -> Result<(NonUniformPlan, f64), ConvError> {
    if ir_len == 0 || host_block_max == 0 || max_fdl_stages == 0 {
        return Err(ConvError::InvalidNonUniformConfig);
    }
    cost_model.validate()?;

    let max_fdl = max_fdl_stages.min(MAX_OPTIMIZER_FDL_STAGES);
    let mut best = CandidatePlan {
        cost: cost_model.direct_cost_per_sample(ir_len),
        stages: [NonUniformStageConfig {
            ir_offset: 0,
            ir_len: 0,
            block_size: 1,
        }; MAX_NONUNIFORM_STAGES],
        count: 1,
    };
    best.stages[0] = NonUniformStageConfig {
        ir_offset: 0,
        ir_len,
        block_size: 1,
    };

    let levels = num_levels_for_max_block(max_block_size);
    let mut active_levels = 0usize;
    while active_levels < levels
        && policy.min_offset_samples(OPTIMIZER_BLOCK_SIZES[active_levels], host_block_max) < ir_len
    {
        active_levels += 1;
    }

    if ir_len >= UNIT_SAMPLES && active_levels > 0 {
        let u_target = ir_len.div_ceil(UNIT_SAMPLES);
        let max_b = OPTIMIZER_BLOCK_SIZES[active_levels - 1];
        let u_max = u_target + max_b / UNIT_SAMPLES;
        let cols = u_max + 1;
        let needed = (max_fdl + 1) * cols + 2 * cols + active_levels * max_fdl * cols;
        if scratch.len() < needed {
            return Err(ConvError::WorkspaceTooSmall {
                required: needed,
                provided: scratch.len(),
            });
        }

        let (dp, rest) = scratch.split_at_mut((max_fdl + 1) * cols);
        let (f_val, rest) = rest.split_at_mut(cols);
        let (f_start, rest) = rest.split_at_mut(cols);
        let bp = &mut rest[..active_levels * max_fdl * cols];

        const F32_SKIP: f32 = -1.0;
        let max_l0_fdl = policy.max_direct_head_with_fdl(host_block_max);

        dp.fill(f32::INFINITY);
        for (u, slot) in dp[..cols].iter_mut().enumerate() {
            let l0 = u * UNIT_SAMPLES;
            if l0 > 0 && l0 <= max_l0_fdl {
                *slot = cost_model.direct_cost_per_sample(l0) as f32;
            }
        }

        for (k, &b) in OPTIMIZER_BLOCK_SIZES.iter().enumerate().take(active_levels) {
            let step = b / UNIT_SAMPLES;
            let open_c = ((cost_model.t_fft[k]
                + cost_model.t_cmul[k]
                + cost_model.t_ifft[k]
                + cost_model.t_add[k])
                / (b as f64)) as f32;
            let rep_c = (cost_model.t_cmac[k] / (b as f64)) as f32;
            let min_off = policy.min_offset_samples(b, host_block_max);
            let u0_min = min_off.div_ceil(UNIT_SAMPLES);

            for s in (1..=max_fdl).rev() {
                let bp_base = (k * max_fdl + (s - 1)) * cols;
                f_val.fill(f32::INFINITY);
                f_start.fill(F32_SKIP);
                bp[bp_base..bp_base + cols].fill(F32_SKIP);

                let prev_row = (s - 1) * cols;
                let cur_row = s * cols;
                for u in step..=u_max {
                    let u_prev = u - step;
                    let mut best_c = f32::INFINITY;
                    let mut best_st = F32_SKIP;
                    if u_prev >= u0_min && dp[prev_row + u_prev].is_finite() {
                        let cand = dp[prev_row + u_prev] + open_c;
                        if cand < best_c {
                            best_c = cand;
                            best_st = u_prev as f32;
                        }
                    }
                    if f_val[u_prev].is_finite() {
                        let cand = f_val[u_prev] + rep_c;
                        if cand < best_c {
                            best_c = cand;
                            best_st = f_start[u_prev];
                        }
                    }
                    f_val[u] = best_c;
                    f_start[u] = best_st;
                    if best_c < dp[cur_row + u] {
                        dp[cur_row + u] = best_c;
                        bp[bp_base + u] = best_st;
                    }
                }
            }
        }

        let mut best_c = f32::INFINITY;
        let mut best_s = 0usize;
        let mut best_u = 0usize;
        for s in 1..=max_fdl {
            let row = s * cols;
            for u in u_target..=u_max {
                if dp[row + u] < best_c {
                    best_c = dp[row + u];
                    best_s = s;
                    best_u = u;
                }
            }
        }

        if best_c.is_finite() && (best_c as f64) < best.cost {
            let mut fdl_rev = [(0usize, 0usize, 0usize); MAX_OPTIMIZER_FDL_STAGES];
            let mut fdl_count = 0usize;
            let mut u = best_u;
            let mut s = best_s;
            for k in (0..active_levels).rev() {
                if s == 0 {
                    break;
                }
                let bp_base = (k * max_fdl + (s - 1)) * cols;
                let st = bp[bp_base + u];
                if st >= 0.0 {
                    let u0 = st as usize;
                    fdl_rev[fdl_count] = (
                        u0 * UNIT_SAMPLES,
                        u * UNIT_SAMPLES,
                        OPTIMIZER_BLOCK_SIZES[k],
                    );
                    fdl_count += 1;
                    u = u0;
                    s -= 1;
                }
            }
            let l0 = u * UNIT_SAMPLES;
            if l0 > 0 && l0 < ir_len && fdl_count > 0 {
                let mut stages = [NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: 0,
                    block_size: 1,
                }; MAX_NONUNIFORM_STAGES];
                stages[0] = NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: l0,
                    block_size: 1,
                };
                let mut count = 1usize;
                for i in (0..fdl_count).rev() {
                    let (off, end_u, b) = fdl_rev[i];
                    if off < ir_len {
                        stages[count] = NonUniformStageConfig {
                            ir_offset: off,
                            ir_len: end_u.min(ir_len) - off,
                            block_size: b,
                        };
                        count += 1;
                    }
                }
                best.stages = stages;
                best.count = count;
            }
        }
    }

    let plan = NonUniformPlan::validate(NonUniformConfig {
        max_ir_len: ir_len,
        host_block_max,
        stages: &best.stages[..best.count],
    })?;
    let exact_cost = cost_model.plan_cost_per_sample(&plan)?;
    Ok((plan, exact_cost))
}

/// Exact integer image of a [`CostModel`] whose entries are non-negative integers (cycle
/// counts), used by the FPU-less Q1.15 planner so the Viterbi DP runs without floating point.
///
/// Per-sample costs are scaled by [`IntegerCostModel::SCALE`] `= 8192` (the largest optimizer
/// block), so `open(B) = (t_fft + t_cmul + t_ifft + t_add) * 8192 / B`,
/// `repeat(B) = t_cmac * 8192 / B` and `direct(L) = L * t_direct_mac * 8192` are exact
/// integers. Construction is `const` and rejects (at compile time when used in a `const`)
/// models with fractional, negative or oversized entries.
#[cfg(feature = "auto")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IntegerCostModel {
    open: [u32; 9],
    repeat: [u32; 9],
    direct_mac: u32,
}

#[cfg(feature = "auto")]
impl IntegerCostModel {
    /// Fixed-point scale of every per-sample cost.
    pub(crate) const SCALE: u64 = 8192;

    const fn exact_u32(v: f64) -> u32 {
        assert!(v >= 0.0 && v < 4_294_967_296.0, "cost out of integer range");
        let i = v as u64;
        assert!(i as f64 == v, "cost model entry is not an integer");
        i as u32
    }

    /// Converts `model` exactly; panics (a compile error in `const` context) if any entry is
    /// not a non-negative integer or a scaled cost does not fit `u32`.
    pub(crate) const fn from_integral(model: &CostModel) -> Self {
        let mut open = [0u32; 9];
        let mut repeat = [0u32; 9];
        let mut k = 0;
        while k < 9 {
            let mul = Self::SCALE / OPTIMIZER_BLOCK_SIZES[k] as u64;
            let o = Self::exact_u32(model.t_fft[k]) as u64
                + Self::exact_u32(model.t_cmul[k]) as u64
                + Self::exact_u32(model.t_ifft[k]) as u64
                + Self::exact_u32(model.t_add[k]) as u64;
            assert!(o > 0, "transform cost must be positive");
            let o = o * mul;
            let r = Self::exact_u32(model.t_cmac[k]) as u64 * mul;
            assert!(
                o <= u32::MAX as u64 && r <= u32::MAX as u64,
                "scaled cost overflow"
            );
            open[k] = o as u32;
            repeat[k] = r as u32;
            k += 1;
        }
        let d = Self::exact_u32(model.t_direct_mac) as u64 * Self::SCALE;
        assert!(
            d > 0 && d <= u32::MAX as u64,
            "direct MAC cost out of range"
        );
        Self {
            open,
            repeat,
            direct_mac: d as u32,
        }
    }

    /// Scaled direct-FIR cost of `taps` taps (exact, `u64`).
    #[inline]
    const fn direct(&self, taps: usize) -> u64 {
        taps as u64 * self.direct_mac as u64
    }
}

/// [`CostModel::esp32_c6_q15_estimate`] as an exact [`IntegerCostModel`] (evaluated at compile
/// time; the planner never touches `f64` at run time).
#[cfg(feature = "auto")]
pub(crate) const ESP32_C6_Q15_INTEGER_MODEL: IntegerCostModel =
    IntegerCostModel::from_integral(&CostModel::esp32_c6_q15_estimate());

/// Runs the Q1.15 auto planner's Viterbi DP inside an `&mut [i16]` caller buffer with **integer
/// arithmetic only** (`AutoConvolver` Q1.15 integration, `#![forbid(unsafe_code)]`, zero stack
/// arrays, `u32` costs packed as 2 `i16` words).
///
/// Costs are exact multiples of `1 / IntegerCostModel::SCALE` cycles per sample, added with
/// saturation (`u32::MAX` = unreachable). Saturation cannot affect the optimum: every partial
/// cost on an optimal path is bounded by the optimal total, which for any accepted `ir_len`
/// (`u_max < 2^16`) stays far below `2^32 / 8192 = 524,288` cycles per sample (a plan of
/// `B <= 1024` partitions costs about `28` cycles per partition, `< 60,000` for the largest
/// accepted IR). Equal costs keep the first candidate found, in the same loop order as the
/// floating-point DP it replaces, which is verified to choose identical plans in the unit tests.
#[cfg(feature = "auto")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_in_i16_scratch(
    ir_len: usize,
    host_block_max: usize,
    max_block_size: usize,
    max_fdl_stages: usize,
    cost_model: &IntegerCostModel,
    policy: ClearancePolicy,
    scratch: &mut [i16],
) -> Result<NonUniformPlan, ConvError> {
    if ir_len == 0 || host_block_max == 0 || max_fdl_stages == 0 {
        return Err(ConvError::InvalidNonUniformConfig);
    }

    let max_fdl = max_fdl_stages.min(MAX_OPTIMIZER_FDL_STAGES);
    let mut best_stages = [NonUniformStageConfig {
        ir_offset: 0,
        ir_len: 0,
        block_size: 1,
    }; MAX_NONUNIFORM_STAGES];
    best_stages[0] = NonUniformStageConfig {
        ir_offset: 0,
        ir_len,
        block_size: 1,
    };
    let mut best_count = 1usize;
    let best_cost = cost_model.direct(ir_len);

    let levels = num_levels_for_max_block(max_block_size);
    let mut active_levels = 0usize;
    while active_levels < levels
        && policy.min_offset_samples(OPTIMIZER_BLOCK_SIZES[active_levels], host_block_max) < ir_len
    {
        active_levels += 1;
    }

    if ir_len >= UNIT_SAMPLES && active_levels > 0 {
        let u_target = ir_len.div_ceil(UNIT_SAMPLES);
        let max_b = OPTIMIZER_BLOCK_SIZES[active_levels - 1];
        let u_max = u_target + max_b / UNIT_SAMPLES;
        let cols = u_max + 1;
        // Require `u_max < u16::MAX` (true for ir_len up to ~2M taps).
        if u_max >= u16::MAX as usize {
            return Err(ConvError::ArithmeticOverflow);
        }
        // dp: (max_fdl + 1) * cols * 2 i16 (u32), f_val: cols * 2 i16 (u32),
        // f_start: cols i16 (u16), bp: active_levels * max_fdl * cols i16 (u16)
        let dp_words = (max_fdl + 1) * cols * 2;
        let f_val_words = cols * 2;
        let f_start_words = cols;
        let bp_words = active_levels * max_fdl * cols;
        let needed = dp_words + f_val_words + f_start_words + bp_words;
        if scratch.len() < needed {
            return Err(ConvError::WorkspaceTooSmall {
                required: needed,
                provided: scratch.len(),
            });
        }

        const INF: u32 = u32::MAX;
        #[inline(always)]
        fn load_u32(buf: &[i16], idx: usize) -> u32 {
            ((buf[2 * idx] as u16 as u32) << 16) | (buf[2 * idx + 1] as u16 as u32)
        }
        #[inline(always)]
        fn store_u32(buf: &mut [i16], idx: usize, v: u32) {
            buf[2 * idx] = (v >> 16) as u16 as i16;
            buf[2 * idx + 1] = v as u16 as i16;
        }

        let (dp, rest) = scratch.split_at_mut(dp_words);
        let (f_val, rest) = rest.split_at_mut(f_val_words);
        let (f_start, rest) = rest.split_at_mut(f_start_words);
        let bp = &mut rest[..bp_words];

        const U16_SKIP: i16 = -1; // u16::MAX
        let max_l0_fdl = policy.max_direct_head_with_fdl(host_block_max);

        for i in 0..((max_fdl + 1) * cols) {
            store_u32(dp, i, INF);
        }
        for u in 0..cols {
            let l0 = u * UNIT_SAMPLES;
            if l0 > 0 && l0 <= max_l0_fdl {
                let c = cost_model.direct(l0);
                store_u32(dp, u, if c >= INF as u64 { INF } else { c as u32 });
            }
        }

        for (k, &b) in OPTIMIZER_BLOCK_SIZES.iter().enumerate().take(active_levels) {
            let step = b / UNIT_SAMPLES;
            let open_c = cost_model.open[k];
            let rep_c = cost_model.repeat[k];
            let min_off = policy.min_offset_samples(b, host_block_max);
            let u0_min = min_off.div_ceil(UNIT_SAMPLES);

            for s in (1..=max_fdl).rev() {
                let bp_base = (k * max_fdl + (s - 1)) * cols;
                for u in 0..cols {
                    store_u32(f_val, u, INF);
                }
                f_start.fill(U16_SKIP);
                bp[bp_base..bp_base + cols].fill(U16_SKIP);

                let prev_row = (s - 1) * cols;
                let cur_row = s * cols;
                for u in step..=u_max {
                    let u_prev = u - step;
                    let mut best_c = INF;
                    let mut best_st = U16_SKIP;
                    if u_prev >= u0_min {
                        let pv = load_u32(dp, prev_row + u_prev);
                        if pv != INF {
                            let cand = pv.saturating_add(open_c);
                            if cand < best_c {
                                best_c = cand;
                                best_st = (u_prev as u16) as i16;
                            }
                        }
                    }
                    let fv = load_u32(f_val, u_prev);
                    if fv != INF {
                        let cand = fv.saturating_add(rep_c);
                        if cand < best_c {
                            best_c = cand;
                            best_st = f_start[u_prev];
                        }
                    }
                    store_u32(f_val, u, best_c);
                    f_start[u] = best_st;
                    if best_c < load_u32(dp, cur_row + u) {
                        store_u32(dp, cur_row + u, best_c);
                        bp[bp_base + u] = best_st;
                    }
                }
            }
        }

        let mut best_c = INF;
        let mut best_s = 0usize;
        let mut best_u = 0usize;
        for s in 1..=max_fdl {
            let row = s * cols;
            for u in u_target..=u_max {
                let c = load_u32(dp, row + u);
                if c < best_c {
                    best_c = c;
                    best_s = s;
                    best_u = u;
                }
            }
        }

        if best_c != INF && (best_c as u64) < best_cost {
            let mut fdl_rev = [(0usize, 0usize, 0usize); MAX_OPTIMIZER_FDL_STAGES];
            let mut fdl_count = 0usize;
            let mut u = best_u;
            let mut s = best_s;
            for k in (0..active_levels).rev() {
                if s == 0 {
                    break;
                }
                let bp_base = (k * max_fdl + (s - 1)) * cols;
                let st = bp[bp_base + u];
                if st != U16_SKIP {
                    let u0 = (st as u16) as usize;
                    fdl_rev[fdl_count] = (
                        u0 * UNIT_SAMPLES,
                        u * UNIT_SAMPLES,
                        OPTIMIZER_BLOCK_SIZES[k],
                    );
                    fdl_count += 1;
                    u = u0;
                    s -= 1;
                }
            }
            let l0 = u * UNIT_SAMPLES;
            if l0 > 0 && l0 < ir_len && fdl_count > 0 {
                let mut stages = [NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: 0,
                    block_size: 1,
                }; MAX_NONUNIFORM_STAGES];
                stages[0] = NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: l0,
                    block_size: 1,
                };
                let mut count = 1usize;
                for i in (0..fdl_count).rev() {
                    let (off, end_u, b) = fdl_rev[i];
                    if off < ir_len {
                        stages[count] = NonUniformStageConfig {
                            ir_offset: off,
                            ir_len: end_u.min(ir_len) - off,
                            block_size: b,
                        };
                        count += 1;
                    }
                }
                best_stages = stages;
                best_count = count;
            }
        }
    }

    NonUniformPlan::validate(NonUniformConfig {
        max_ir_len: ir_len,
        host_block_max,
        stages: &best_stages[..best_count],
    })
}

/// Test oracle: the `f32` Viterbi DP that [`optimize_in_i16_scratch`] replaced (IEEE `f32` bits
/// packed as 2 `i16` words). Kept only to prove that the integer DP selects identical plans.
#[cfg(all(test, feature = "auto"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_in_i16_scratch_f32_reference(
    ir_len: usize,
    host_block_max: usize,
    max_block_size: usize,
    max_fdl_stages: usize,
    cost_model: &CostModel,
    policy: ClearancePolicy,
    scratch: &mut [i16],
) -> Result<(NonUniformPlan, f64), ConvError> {
    if ir_len == 0 || host_block_max == 0 || max_fdl_stages == 0 {
        return Err(ConvError::InvalidNonUniformConfig);
    }
    cost_model.validate()?;

    let max_fdl = max_fdl_stages.min(MAX_OPTIMIZER_FDL_STAGES);
    let mut best = CandidatePlan {
        cost: cost_model.direct_cost_per_sample(ir_len),
        stages: [NonUniformStageConfig {
            ir_offset: 0,
            ir_len: 0,
            block_size: 1,
        }; MAX_NONUNIFORM_STAGES],
        count: 1,
    };
    best.stages[0] = NonUniformStageConfig {
        ir_offset: 0,
        ir_len,
        block_size: 1,
    };

    let levels = num_levels_for_max_block(max_block_size);
    let mut active_levels = 0usize;
    while active_levels < levels
        && policy.min_offset_samples(OPTIMIZER_BLOCK_SIZES[active_levels], host_block_max) < ir_len
    {
        active_levels += 1;
    }

    if ir_len >= UNIT_SAMPLES && active_levels > 0 {
        let u_target = ir_len.div_ceil(UNIT_SAMPLES);
        let max_b = OPTIMIZER_BLOCK_SIZES[active_levels - 1];
        let u_max = u_target + max_b / UNIT_SAMPLES;
        let cols = u_max + 1;
        // Require `u_max < u16::MAX` (true for ir_len up to ~2M taps).
        if u_max >= u16::MAX as usize {
            return Err(ConvError::ArithmeticOverflow);
        }
        // dp: (max_fdl + 1) * cols * 2 i16 (f32 bits)
        // f_val: cols * 2 i16 (f32 bits)
        // f_start: cols i16 (u16)
        // bp: active_levels * max_fdl * cols i16 (u16)
        let dp_words = (max_fdl + 1) * cols * 2;
        let f_val_words = cols * 2;
        let f_start_words = cols;
        let bp_words = active_levels * max_fdl * cols;
        let needed = dp_words + f_val_words + f_start_words + bp_words;
        if scratch.len() < needed {
            return Err(ConvError::WorkspaceTooSmall {
                required: needed,
                provided: scratch.len(),
            });
        }

        #[inline(always)]
        fn load_f32(buf: &[i16], idx: usize) -> f32 {
            let hi = buf[2 * idx] as u16 as u32;
            let lo = buf[2 * idx + 1] as u16 as u32;
            f32::from_bits((hi << 16) | lo)
        }
        #[inline(always)]
        fn store_f32(buf: &mut [i16], idx: usize, v: f32) {
            let bits = v.to_bits();
            buf[2 * idx] = (bits >> 16) as u16 as i16;
            buf[2 * idx + 1] = bits as u16 as i16;
        }

        let (dp, rest) = scratch.split_at_mut(dp_words);
        let (f_val, rest) = rest.split_at_mut(f_val_words);
        let (f_start, rest) = rest.split_at_mut(f_start_words);
        let bp = &mut rest[..bp_words];

        const U16_SKIP: i16 = -1; // u16::MAX
        let max_l0_fdl = policy.max_direct_head_with_fdl(host_block_max);

        for i in 0..((max_fdl + 1) * cols) {
            store_f32(dp, i, f32::INFINITY);
        }
        for u in 0..cols {
            let l0 = u * UNIT_SAMPLES;
            if l0 > 0 && l0 <= max_l0_fdl {
                store_f32(dp, u, cost_model.direct_cost_per_sample(l0) as f32);
            }
        }

        for (k, &b) in OPTIMIZER_BLOCK_SIZES.iter().enumerate().take(active_levels) {
            let step = b / UNIT_SAMPLES;
            let open_c = ((cost_model.t_fft[k]
                + cost_model.t_cmul[k]
                + cost_model.t_ifft[k]
                + cost_model.t_add[k])
                / (b as f64)) as f32;
            let rep_c = (cost_model.t_cmac[k] / (b as f64)) as f32;
            let min_off = policy.min_offset_samples(b, host_block_max);
            let u0_min = min_off.div_ceil(UNIT_SAMPLES);

            for s in (1..=max_fdl).rev() {
                let bp_base = (k * max_fdl + (s - 1)) * cols;
                for u in 0..cols {
                    store_f32(f_val, u, f32::INFINITY);
                }
                f_start.fill(U16_SKIP);
                bp[bp_base..bp_base + cols].fill(U16_SKIP);

                let prev_row = (s - 1) * cols;
                let cur_row = s * cols;
                for u in step..=u_max {
                    let u_prev = u - step;
                    let mut best_c = f32::INFINITY;
                    let mut best_st = U16_SKIP;
                    if u_prev >= u0_min {
                        let pv = load_f32(dp, prev_row + u_prev);
                        if pv.is_finite() {
                            let cand = pv + open_c;
                            if cand < best_c {
                                best_c = cand;
                                best_st = (u_prev as u16) as i16;
                            }
                        }
                    }
                    let fv = load_f32(f_val, u_prev);
                    if fv.is_finite() {
                        let cand = fv + rep_c;
                        if cand < best_c {
                            best_c = cand;
                            best_st = f_start[u_prev];
                        }
                    }
                    store_f32(f_val, u, best_c);
                    f_start[u] = best_st;
                    if best_c < load_f32(dp, cur_row + u) {
                        store_f32(dp, cur_row + u, best_c);
                        bp[bp_base + u] = best_st;
                    }
                }
            }
        }

        let mut best_c = f32::INFINITY;
        let mut best_s = 0usize;
        let mut best_u = 0usize;
        for s in 1..=max_fdl {
            let row = s * cols;
            for u in u_target..=u_max {
                let c = load_f32(dp, row + u);
                if c < best_c {
                    best_c = c;
                    best_s = s;
                    best_u = u;
                }
            }
        }

        if best_c.is_finite() && (best_c as f64) < best.cost {
            let mut fdl_rev = [(0usize, 0usize, 0usize); MAX_OPTIMIZER_FDL_STAGES];
            let mut fdl_count = 0usize;
            let mut u = best_u;
            let mut s = best_s;
            for k in (0..active_levels).rev() {
                if s == 0 {
                    break;
                }
                let bp_base = (k * max_fdl + (s - 1)) * cols;
                let st = bp[bp_base + u];
                if st != U16_SKIP {
                    let u0 = (st as u16) as usize;
                    fdl_rev[fdl_count] = (
                        u0 * UNIT_SAMPLES,
                        u * UNIT_SAMPLES,
                        OPTIMIZER_BLOCK_SIZES[k],
                    );
                    fdl_count += 1;
                    u = u0;
                    s -= 1;
                }
            }
            let l0 = u * UNIT_SAMPLES;
            if l0 > 0 && l0 < ir_len && fdl_count > 0 {
                let mut stages = [NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: 0,
                    block_size: 1,
                }; MAX_NONUNIFORM_STAGES];
                stages[0] = NonUniformStageConfig {
                    ir_offset: 0,
                    ir_len: l0,
                    block_size: 1,
                };
                let mut count = 1usize;
                for i in (0..fdl_count).rev() {
                    let (off, end_u, b) = fdl_rev[i];
                    if off < ir_len {
                        stages[count] = NonUniformStageConfig {
                            ir_offset: off,
                            ir_len: end_u.min(ir_len) - off,
                            block_size: b,
                        };
                        count += 1;
                    }
                }
                best.stages = stages;
                best.count = count;
            }
        }
    }

    let plan = NonUniformPlan::validate(NonUniformConfig {
        max_ir_len: ir_len,
        host_block_max,
        stages: &best.stages[..best.count],
    })?;
    let exact_cost = cost_model.plan_cost_per_sample(&plan)?;
    Ok((plan, exact_cost))
}

// ------------------------------------------------------------------------------------------
// Wefers (2015) Block-Mode Viterbi Optimizer (for §6.5 / §6.12 / Appendix A.2 validation)
// ------------------------------------------------------------------------------------------

/// Block-mode clearance restriction policy matching Wefers (2015) §6.5.3–§6.5.4 and Appendix A.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockModePolicy {
    /// Minimal-load causal partitions (`clearance(i) >= 0`, Eq. 6.41–6.42; Appendix A.2.1–A.2.3).
    MinLoad,
    /// §6.5.4 prose Strategy 1 (`clearance(i) >= L_i / B - 1`, i.e. `timespan(i) >= L_i / B`).
    Strategy1Prose,
    /// Appendix A.2.4 table variant 1 (`clearance(i) >= 2 * L_i / B - 1`).
    Strategy1TableA24,
    /// §6.5.4 prose Strategy 2 (`2 <= P_0 <= 4` when `N > 4B`, `timespan >= 2 / 4 / 8` for
    /// `{2B, 4B}` / `{8B..=64B}` / `> 64B`).
    Strategy2Prose,
    /// Appendix A.2.5 & §6.12 table variant 2 (`P_0 = 2` for multi-segment partitions `N > 4B`,
    /// `timespan >= 2 / 4 / 8` for `{2B, 4B}` / `{8B, 16B, 32B}` / `>= 64B`). Reproduces
    /// `(128^2 256^4 1024^8 8192^10)` at `240.4` cycles/sample for `N = 88,200, B = 128`.
    Strategy2TableA25,
}

/// One segment in a block-mode NUPOLS partition (Wefers Eq. 6.1–6.4, 6.24).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockModeSegment {
    /// Sub-filter length `L_i = B_i`.
    pub block_size: usize,
    /// Partition multiplicity `P_i >= 1`.
    pub partitions: usize,
    /// Starting tap offset `offset(i)` in samples.
    pub offset_samples: usize,
    /// Block-mode clearance in stream cycles `offset(i)/B - L_i/B + 1` (Eq. 6.24, 6.41).
    pub clearance_cycles: usize,
}

/// Result of [`optimize_block_mode`] (stack-allocated, zero heap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockModePartition {
    segments: [BlockModeSegment; MAX_NONUNIFORM_STAGES],
    segment_count: usize,
    /// Total impulse response taps covered (`>= target_ir_len`).
    pub covered_len: usize,
    /// Stream block length `B`.
    pub stream_block_size: usize,
    /// Total cost per output sample under the supplied [`CostModel`] (Wefers Eq. 6.12).
    pub cost_per_sample: f64,
}

impl BlockModePartition {
    /// Optimal ordered segments `P_opt = (L_0^{P_0}, ..., L_{M-1}^{P_{M-1}})`.
    #[inline]
    pub fn segments(&self) -> &[BlockModeSegment] {
        &self.segments[..self.segment_count]
    }
}

/// Exact number of [`OptimizerScratchCell`] elements required by [`optimize_block_mode`].
pub const fn block_mode_scratch_len(
    ir_len: usize,
    stream_block_size: usize,
    max_block_size: usize,
) -> usize {
    if ir_len == 0 || stream_block_size == 0 || max_block_size < stream_block_size {
        return 0;
    }
    let Some(base_idx) = block_size_index(stream_block_size) else {
        return 0;
    };
    let Some(max_idx) = block_size_index(max_block_size) else {
        return 0;
    };
    let levels = max_idx - base_idx + 1;
    let u_target = ir_len.div_ceil(stream_block_size);
    let u_max = u_target + max_block_size / stream_block_size;
    let cols = u_max + 1;
    // dp[cols] + new_dp[cols] + f_val[cols] + f_start[cols/2] + bp[levels * cols / 2]
    3 * cols + cols.div_ceil(2) + (levels * cols).div_ceil(2)
}

/// Solves Wefers' (2015) block-mode NUPOLS optimization problem (§6.5.2–§6.5.4, Eq. 6.40–6.46)
/// with stream block length `B = stream_block_size` (framing latency `B`), zero heap allocation.
pub fn optimize_block_mode(
    ir_len: usize,
    stream_block_size: usize,
    max_block_size: usize,
    cost_model: &CostModel,
    policy: BlockModePolicy,
    scratch: &mut [OptimizerScratchCell],
) -> Result<BlockModePartition, ConvError> {
    if ir_len == 0 || max_block_size < stream_block_size {
        return Err(ConvError::InvalidNonUniformConfig);
    }
    let base_idx = block_size_index(stream_block_size).ok_or(ConvError::UnsupportedFftSize {
        fft_size: stream_block_size.saturating_mul(2),
    })?;
    let max_idx = block_size_index(max_block_size).ok_or(ConvError::UnsupportedFftSize {
        fft_size: max_block_size.saturating_mul(2),
    })?;
    cost_model.validate()?;

    let levels = max_idx - base_idx + 1;
    let u_target = ir_len.div_ceil(stream_block_size);
    let u_max = u_target + max_block_size / stream_block_size;
    let cols = u_max + 1;

    let required = block_mode_scratch_len(ir_len, stream_block_size, max_block_size);
    if scratch.len() < required {
        return Err(ConvError::WorkspaceTooSmall {
            required,
            provided: scratch.len(),
        });
    }

    let (dp, rest) = scratch.split_at_mut(cols);
    let (new_dp, rest) = rest.split_at_mut(cols);
    let (f_val, rest) = rest.split_at_mut(cols);
    let (f_start, rest) = rest.split_at_mut(cols.div_ceil(2));
    let bp = &mut rest[..(levels * cols).div_ceil(2)];

    for c in dp.iter_mut() {
        *c = OptimizerScratchCell::from_f64(f64::INFINITY);
    }
    for u in 0..cols {
        write_packed_u32(bp, u, SKIP_MARK);
    }

    // Level 0: mandatory leading segment of size `B = stream_block_size` (clearance = 0).
    let open_0 = (cost_model.t_fft[base_idx]
        + cost_model.t_cmul[base_idx]
        + cost_model.t_ifft[base_idx]
        + cost_model.t_add[base_idx])
        / (stream_block_size as f64);
    let rep_0 = cost_model.t_cmac[base_idx] / (stream_block_size as f64);

    let (p0_min, p0_max) = match policy {
        BlockModePolicy::Strategy2Prose if ir_len > 4 * stream_block_size => (2, 4.min(u_max)),
        BlockModePolicy::Strategy2TableA25 if ir_len > 4 * stream_block_size => (2, 2.min(u_max)),
        BlockModePolicy::Strategy2Prose | BlockModePolicy::Strategy2TableA25 => (1, 4.min(u_max)),
        _ => (1, u_max),
    };
    for (offset, cell) in dp[p0_min..=p0_max].iter_mut().enumerate() {
        let p = p0_min + offset;
        *cell = OptimizerScratchCell::from_f64(open_0 + (p - 1) as f64 * rep_0);
        write_packed_u32(bp, p, 0);
    }

    // Levels 1..levels: strictly larger power-of-two sub-filters `L_k = (2^k) * B`.
    for rel in 1..levels {
        let k = base_idx + rel;
        let l_k = OPTIMIZER_BLOCK_SIZES[k];
        let step = l_k / stream_block_size;
        let open_c = (cost_model.t_fft[k]
            + cost_model.t_cmul[k]
            + cost_model.t_ifft[k]
            + cost_model.t_add[k])
            / (l_k as f64);
        let rep_c = cost_model.t_cmac[k] / (l_k as f64);

        // Block-mode clearance: `c = u0 - step + 1 >= c_min  <=>  u0 >= c_min + step - 1`.
        let c_min = match policy {
            BlockModePolicy::MinLoad => 0,
            BlockModePolicy::Strategy1Prose => step - 1,
            BlockModePolicy::Strategy1TableA24 => 2 * step - 1,
            BlockModePolicy::Strategy2Prose => {
                if step <= 4 {
                    1
                } else if step <= 64 {
                    3
                } else {
                    7
                }
            }
            BlockModePolicy::Strategy2TableA25 => {
                if step <= 4 {
                    1
                } else if step <= 32 {
                    3
                } else {
                    7
                }
            }
        };
        let u0_min = c_min + step - 1;

        new_dp.copy_from_slice(dp);
        let bp_base = rel * cols;
        for (u, cell) in f_val[..cols].iter_mut().enumerate() {
            *cell = OptimizerScratchCell::from_f64(f64::INFINITY);
            write_packed_u32(f_start, u, SKIP_MARK);
            write_packed_u32(bp, bp_base + u, SKIP_MARK);
        }

        for u in step..=u_max {
            let u_prev = u - step;
            let mut best = f64::INFINITY;
            let mut best_st = SKIP_MARK;

            if u_prev >= u0_min {
                let pv = dp[u_prev].as_f64();
                if pv.is_finite() {
                    let cand = pv + open_c;
                    if cand < best {
                        best = cand;
                        best_st = u_prev as u32;
                    }
                }
            }
            let fv = f_val[u_prev].as_f64();
            if fv.is_finite() {
                let cand = fv + rep_c;
                if cand < best {
                    best = cand;
                    best_st = read_packed_u32(f_start, u_prev);
                }
            }

            f_val[u] = OptimizerScratchCell::from_f64(best);
            write_packed_u32(f_start, u, best_st);

            if best < new_dp[u].as_f64() {
                new_dp[u] = OptimizerScratchCell::from_f64(best);
                write_packed_u32(bp, bp_base + u, best_st);
            }
        }
        dp.copy_from_slice(new_dp);
    }

    let mut best_u = u_target;
    let mut best_cost = f64::INFINITY;
    for (offset, cell) in dp[u_target..=u_max].iter().enumerate() {
        let c = cell.as_f64();
        if c < best_cost {
            best_cost = c;
            best_u = u_target + offset;
        }
    }
    if !best_cost.is_finite() {
        return Err(ConvError::InvalidNonUniformConfig);
    }

    let mut rev_segs = [BlockModeSegment {
        block_size: 0,
        partitions: 0,
        offset_samples: 0,
        clearance_cycles: 0,
    }; MAX_NONUNIFORM_STAGES];
    let mut count = 0usize;
    let mut u = best_u;
    for rel in (0..levels).rev() {
        let u0 = read_packed_u32(bp, rel * cols + u);
        if u0 != SKIP_MARK {
            let u0 = u0 as usize;
            let l_k = OPTIMIZER_BLOCK_SIZES[base_idx + rel];
            let step = l_k / stream_block_size;
            if count >= MAX_NONUNIFORM_STAGES {
                return Err(ConvError::InvalidNonUniformConfig);
            }
            rev_segs[count] = BlockModeSegment {
                block_size: l_k,
                partitions: (u - u0) / step,
                offset_samples: u0 * stream_block_size,
                clearance_cycles: u0 + 1 - step,
            };
            count += 1;
            u = u0;
        }
    }

    let mut segments = [BlockModeSegment {
        block_size: 0,
        partitions: 0,
        offset_samples: 0,
        clearance_cycles: 0,
    }; MAX_NONUNIFORM_STAGES];
    for i in 0..count {
        segments[i] = rev_segs[count - 1 - i];
    }

    Ok(BlockModePartition {
        segments,
        segment_count: count,
        covered_len: best_u * stream_block_size,
        stream_block_size,
        cost_per_sample: best_cost,
    })
}

// ------------------------------------------------------------------------------------------
// Pre-computed MCU Presets (zero optimizer scratch RAM at runtime)
// ------------------------------------------------------------------------------------------

fn build_clipped_plan(
    ir_len: usize,
    host_block_max: usize,
    template: &[(usize, usize, usize)], // (offset, end_uncapped, block_size)
) -> Result<NonUniformPlan, ConvError> {
    if ir_len == 0 || host_block_max == 0 {
        return Err(ConvError::InvalidNonUniformConfig);
    }
    let mut stages = [NonUniformStageConfig {
        ir_offset: 0,
        ir_len: 0,
        block_size: 1,
    }; MAX_NONUNIFORM_STAGES];
    let mut count = 0usize;
    for &(off, end_uncapped, b) in template {
        if ir_len <= off {
            break;
        }
        let end = end_uncapped.min(ir_len);
        stages[count] = NonUniformStageConfig {
            ir_offset: off,
            ir_len: end - off,
            block_size: b,
        };
        count += 1;
    }
    NonUniformPlan::validate(NonUniformConfig {
        max_ir_len: ir_len,
        host_block_max,
        stages: &stages[..count],
    })
}

/// Pre-computed Strategy-2 NUPC plan for ESP32-P4 (`f32` scalar, `INFERRED / ESTIMATE`,
/// `DEVICE_VERIFIED = NO`), requiring **0 B** of optimizer scratch RAM.
///
/// Covers any `ir_len >= 1` using optimized stage transitions derived from
/// [`CostModel::esp32_p4_f32_estimate`] (`Direct 64 | B=32 | B=256 | B=4096`), clipped cleanly
/// to `ir_len`.
pub fn preset_esp32_p4_f32(
    ir_len: usize,
    host_block_max: usize,
) -> Result<NonUniformPlan, ConvError> {
    if ir_len <= 160 {
        return build_clipped_plan(ir_len, host_block_max, &[(0, usize::MAX, 1)]);
    }
    if ir_len <= 2_048 {
        return build_clipped_plan(ir_len, host_block_max, &[(0, 64, 1), (64, usize::MAX, 32)]);
    }
    if ir_len <= 16_384 {
        return build_clipped_plan(
            ir_len,
            host_block_max,
            &[(0, 64, 1), (64, 384, 32), (384, usize::MAX, 256)],
        );
    }
    build_clipped_plan(
        ir_len,
        host_block_max,
        &[
            (0, 64, 1),
            (64, 384, 32),
            (384, 4_480, 256),
            (4_480, usize::MAX, 4096),
        ],
    )
}

/// Pre-computed Strategy-2 NUPC plan for ESP32-C6 (`Q1.15/BFP`, `B_max <= 1024 / N2048`,
/// `INFERRED / ESTIMATE`, `DEVICE_VERIFIED = NO`), requiring **0 B** of optimizer scratch RAM.
///
/// Derived from [`CostModel::esp32_c6_q15_estimate`], strictly obeying `block_size <= 1024`
/// for [`crate::FixedQ15NonUniformConvolver`].
pub fn preset_esp32_c6_q15(
    ir_len: usize,
    host_block_max: usize,
) -> Result<NonUniformPlan, ConvError> {
    if ir_len <= 128 {
        return build_clipped_plan(ir_len, host_block_max, &[(0, usize::MAX, 1)]);
    }
    if ir_len <= 1_024 {
        return build_clipped_plan(ir_len, host_block_max, &[(0, 64, 1), (64, usize::MAX, 32)]);
    }
    if ir_len <= 4_096 {
        return build_clipped_plan(
            ir_len,
            host_block_max,
            &[(0, 64, 1), (64, 384, 32), (384, usize::MAX, 256)],
        );
    }
    build_clipped_plan(
        ir_len,
        host_block_max,
        &[
            (0, 64, 1),
            (64, 192, 32),
            (192, 1_216, 128),
            (1_216, usize::MAX, 1024),
        ],
    )
}

#[cfg(all(test, feature = "auto"))]
mod integer_planner_tests {
    use super::*;
    use std::vec;

    /// Same block cap as `auto::auto_max_block`.
    fn auto_cap(taps: usize, ceiling: usize) -> usize {
        let cap = ceiling.min(32usize.max(taps / 4));
        let mut best = 0;
        for b in OPTIMIZER_BLOCK_SIZES {
            if b <= cap && ClearancePolicy::Strategy2.min_offset_samples(b, 64) < taps {
                best = b;
            }
        }
        best
    }

    fn both(
        taps: usize,
        max_b: usize,
        fdl: usize,
        policy: ClearancePolicy,
    ) -> (NonUniformPlan, NonUniformPlan) {
        let len = optimizer_scratch_len_i16(taps, max_b, fdl).max(1);
        let mut a = vec![0i16; len];
        let mut b = vec![0i16; len];
        let int = optimize_in_i16_scratch(
            taps,
            64,
            max_b,
            fdl,
            &ESP32_C6_Q15_INTEGER_MODEL,
            policy,
            &mut a,
        )
        .unwrap();
        let (flt, _) = optimize_in_i16_scratch_f32_reference(
            taps,
            64,
            max_b,
            fdl,
            &CostModel::esp32_c6_q15_estimate(),
            policy,
            &mut b,
        )
        .unwrap();
        (int, flt)
    }

    #[test]
    fn integer_model_is_exact_image_of_esp32_c6_estimate() {
        let m = CostModel::esp32_c6_q15_estimate();
        for (k, &b) in OPTIMIZER_BLOCK_SIZES.iter().enumerate() {
            let b = b as f64;
            let open = (m.t_fft[k] + m.t_cmul[k] + m.t_ifft[k] + m.t_add[k]) / b;
            let rep = m.t_cmac[k] / b;
            assert_eq!(ESP32_C6_Q15_INTEGER_MODEL.open[k] as f64, open * 8192.0);
            assert_eq!(ESP32_C6_Q15_INTEGER_MODEL.repeat[k] as f64, rep * 8192.0);
        }
        assert_eq!(
            ESP32_C6_Q15_INTEGER_MODEL.direct(1) as f64,
            m.t_direct_mac * 8192.0
        );
    }

    #[test]
    fn integer_q15_planner_selects_the_same_plans_as_the_f32_dp() {
        // Every auto-planned length up to 6,000 taps, then a sweep to 300k and the extremes.
        let lengths = (1..=6_000)
            .chain((6_001..=300_000).step_by(131))
            .chain([524_288, 1_000_000, 2_000_000]);
        let mut fdl_plans = 0usize;
        for taps in lengths {
            let max_b = auto_cap(taps, 1024);
            if max_b == 0 {
                continue;
            }
            let (int, flt) = both(taps, max_b, 3, ClearancePolicy::Strategy2);
            assert_eq!(int, flt, "taps={taps}");
            fdl_plans += usize::from(int.stage_count() > 1);
        }
        assert!(fdl_plans > 2_000);
        // Other stage limits, caps and policies.
        for taps in (64..=20_000).step_by(37) {
            for (max_b, fdl) in [(32, 1), (128, 2), (1024, 7), (8192, 4)] {
                for policy in [ClearancePolicy::Strategy1, ClearancePolicy::Strategy2] {
                    let (int, flt) = both(taps, max_b, fdl, policy);
                    assert_eq!(int, flt, "taps={taps} max_b={max_b} fdl={fdl} {policy:?}");
                }
            }
        }
    }
}
