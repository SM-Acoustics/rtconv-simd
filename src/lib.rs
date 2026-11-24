//! `rtconv_simd`: zero-latency, allocation-free, `#![no_std]` FIR convolution with
//! `fearless_simd` acceleration, a Q1.15 / block-floating-point stack for FPU-less MCUs, FIR
//! artifact decoding, and a zero-configuration automatic layer.
//!
//! ## Core Guarantees
//! - `#![no_std]` and `#![forbid(unsafe_code)]`; `alloc` is never used (the `std` feature only
//!   enables runtime CPU detection via `fearless_simd::Level::new()`).
//! - All working buffers are caller-supplied slices sized with checked arithmetic (or `const fn`
//!   for [`auto`]); `prepare`, `process_*`, and `reset` never allocate.
//! - Phase-agnostic: any finite causal FIR is applied verbatim (order, gain, phase, delay).
//!   The exception is magnitudes below [`simd_kernels::DENORMAL_GUARD`] (`2^-63`, ≈ −379 dBFS):
//!   `f32` engines flush them to zero to stay out of IEEE 754 subnormal arithmetic, which is up
//!   to two orders of magnitude slower on x86 without FTZ/DAZ.
//!
//! ## Entry points
//! - [`auto::AutoConvolver`] / [`auto::AutoStereoConvolver`] (feature `auto`, default): load a
//!   WAV / raw / text FIR artifact ([`artifact::FirArtifact`]) and let the crate choose the
//!   numeric domain (`f32` with an FPU, Q1.15 without), engine, and partition size.
//! - [`DirectConvolver`]: time-domain FIR (sample and block API).
//! - [`PartitionedConvolver`]: uniform frequency-domain delay line with 50 % overlap-save.
//! - [`HybridConvolver`]: zero-latency direct head + partitioned FFT tail (BURST / SPREAD).
//! - `NonUniformConvolver` (feature `nonuniform`): multi-stage zero-latency NUPC.
//! - `fixed::*` (features `fixed-q15`, `fixed-fft`, `fixed-nonuniform`): Q1.15 / BFP engines.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(test)]
extern crate std;

/// README code blocks compiled and run as doctests so the documentation cannot drift.
#[cfg(all(doctest, feature = "auto"))]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod artifact;
#[cfg(feature = "auto")]
pub mod auto;
pub mod direct;
pub mod error;
pub mod fft;
#[cfg(feature = "fixed-q15")]
pub mod fixed;
pub mod hybrid;
#[cfg(feature = "nonuniform")]
pub mod nonuniform;
pub mod partitioned;
#[cfg(feature = "phastft")]
pub mod phastft_backend;
pub mod simd_fft;
mod simd_fft_tables;
pub mod simd_kernels;

pub use artifact::{ArtifactError, Container, FirArtifact, SampleEncoding};
#[cfg(feature = "auto")]
pub use auto::{
    AutoConvolver, AutoEngine, AutoInfo, AutoStereoConvolver, AutoWorkspace, HAS_HARDWARE_FPU,
    NativeElem, NumericDomain,
};
pub use direct::{DirectConfig, DirectConvolver, DirectMemoryRequirements};
pub use error::ConvError;
pub use fearless_simd::Level;
pub use fft::{FftBackend, FftSize, MicroFftBackend};
#[cfg(feature = "fixed-q15")]
pub use fixed::{
    FixedQ15DirectCapabilities, FixedQ15DirectConfig, FixedQ15DirectConvolver,
    FixedQ15DirectMemoryRequirements,
};
#[cfg(feature = "fixed-nonuniform")]
pub use fixed::{
    FixedQ15NonUniformCapabilities, FixedQ15NonUniformConvolver,
    FixedQ15NonUniformMemoryRequirements,
};
#[cfg(feature = "fixed-fft")]
pub use fixed::{
    FixedQ15PartitionedCapabilities, FixedQ15PartitionedConfig, FixedQ15PartitionedConvolver,
    FixedQ15PartitionedMemoryRequirements,
};
pub use hybrid::{HybridConfig, HybridConvolver, HybridMemoryRequirements};
#[cfg(feature = "nonuniform")]
pub use nonuniform::{
    BlockModePartition, BlockModePolicy, BlockModeSegment, BudgetVerdict, ClearancePolicy,
    CostModel, MAX_NONUNIFORM_STAGES, MAX_OPTIMIZER_FDL_STAGES, NonUniformBudget,
    NonUniformCapabilities, NonUniformConfig, NonUniformConvolver, NonUniformMemoryRequirements,
    NonUniformPlan, NonUniformStageConfig, NonUniformStageKind, NonUniformStagePlan,
    OPTIMIZER_BLOCK_SIZES, OptimizationReport, OptimizerScratchCell, block_mode_scratch_len,
    optimize_block_mode, optimizer_scratch_len, preset_esp32_c6_q15, preset_esp32_p4_f32,
};
pub use partitioned::{PartitionedConfig, PartitionedConvolver, PartitionedMemoryRequirements};
#[cfg(feature = "phastft")]
pub use phastft_backend::PhastFftBackend;
pub use simd_fft::SimdFixedFftBackend;

/// Convolution engine variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// Time-domain direct FIR convolver.
    Direct,
    /// Uniform partitioned FFT overlap-save FDL convolver.
    Partitioned,
    /// Zero-latency hybrid direct head + FFT tail convolver.
    Hybrid,
    /// Zero-latency non-uniform partitioned convolver (NUPC).
    #[cfg(feature = "nonuniform")]
    NonUniform,
}

/// Runtime metadata describing the active engine configuration and backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Active engine architecture.
    pub engine: EngineKind,
    /// Selected SIMD backend label.
    pub simd_backend: &'static str,
    /// Whether vector SIMD instructions are actually executed for FIR/CMAC kernels
    /// (`false` when scalar override or `fearless_simd::Fallback` is active).
    pub simd_active: bool,
    /// Active FFT backend label (or `"none"` for `DirectConvolver`).
    pub fft_backend: &'static str,
    /// Whether the FFT/IFFT transform itself is SIMD-vectorized.
    pub fft_is_simd: bool,
    /// Processing or tail partition block size, if applicable.
    pub block_size: Option<usize>,
    /// FFT length in samples, if applicable.
    pub fft_size: Option<usize>,
    /// Maximum configured impulse response length.
    pub max_ir_len: usize,
    /// Active impulse response length.
    pub active_ir_len: usize,
    /// Algorithmic latency in samples.
    pub latency_samples: usize,
}

/// Pre-validated SIMD backend selection passed during `prepare()`.
///
/// Selection happens strictly at initialization time — never inside the audio callback.
#[derive(Clone, Copy, Debug)]
pub struct SimdBackend {
    level: Level,
    scalar_override: bool,
}

impl SimdBackend {
    /// Forces the pure scalar reference kernel (no SIMD dispatch).
    #[inline]
    pub const fn scalar() -> Self {
        Self {
            level: Level::fallback(),
            scalar_override: true,
        }
    }

    /// Uses the compile-time target baseline (`Level::baseline()`), always available in `no_std`.
    #[inline]
    pub const fn baseline() -> Self {
        Self {
            level: Level::baseline(),
            scalar_override: false,
        }
    }

    /// Uses a caller-supplied `fearless_simd::Level` (e.g. detected externally on `no_std`).
    #[inline]
    pub const fn from_level(level: Level) -> Self {
        Self {
            level,
            scalar_override: false,
        }
    }

    /// Performs one-time runtime CPU feature detection via `fearless_simd::Level::new()`.
    ///
    /// Available when the `std` feature is enabled (or on `wasm32`). Call once at setup,
    /// never inside the real-time audio callback.
    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[inline]
    pub fn detect() -> Self {
        Self {
            level: Level::new(),
            scalar_override: false,
        }
    }

    /// Attempts runtime detection, falling back to [`Self::baseline()`] on `no_std` targets.
    #[inline]
    pub fn detect_or_baseline() -> Self {
        match Level::try_detect() {
            Some(level) => Self {
                level,
                scalar_override: false,
            },
            None => Self::baseline(),
        }
    }

    /// Selects SSE2 explicitly on x86/x86-64 if available in the detected `Level`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn try_sse2() -> Result<Self, ConvError> {
        let lvl = Self::detect_or_baseline().level;
        match lvl.as_sse2() {
            Some(tok) => Ok(Self::from_level(Level::Sse2(tok))),
            None => Err(ConvError::BackendUnavailable),
        }
    }

    /// Selects SSE4.2 (x86-64-v2) explicitly if supported by the current host.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn try_sse4_2() -> Result<Self, ConvError> {
        let lvl = Self::detect_or_baseline().level;
        match lvl.as_sse4_2() {
            Some(tok) => Ok(Self::from_level(Level::Sse4_2(tok))),
            None => Err(ConvError::BackendUnavailable),
        }
    }

    /// Selects AVX2 + FMA (x86-64-v3) explicitly if supported by the current host.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn try_avx2() -> Result<Self, ConvError> {
        let lvl = Self::detect_or_baseline().level;
        match lvl.as_avx2() {
            Some(tok) => Ok(Self::from_level(Level::Avx2(tok))),
            None => Err(ConvError::BackendUnavailable),
        }
    }

    /// Selects AVX-512 (Ice Lake+) explicitly if supported by the current host.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn try_avx512() -> Result<Self, ConvError> {
        let lvl = Self::detect_or_baseline().level;
        match lvl.as_avx512() {
            Some(tok) => Ok(Self::from_level(Level::Avx512(tok))),
            None => Err(ConvError::BackendUnavailable),
        }
    }

    /// Returns `true` if this backend executes the pure scalar reference path.
    #[inline]
    pub const fn is_scalar(&self) -> bool {
        self.scalar_override
    }

    /// Returns `true` if vector SIMD instructions are actually active
    /// (neither explicit scalar override nor `fearless_simd::Fallback`).
    #[inline]
    pub fn is_simd_active(&self) -> bool {
        !self.scalar_override && !self.level.is_fallback()
    }

    /// Returns the underlying `fearless_simd::Level`.
    #[inline]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// Human-readable backend name.
    #[inline]
    pub fn name(&self) -> &'static str {
        if self.scalar_override {
            return "scalar-reference";
        }
        if self.level.is_fallback() {
            return "fearless_simd::Fallback";
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.level.as_avx512().is_some() {
                return "fearless_simd::Avx512";
            }
            if self.level.as_avx2().is_some() {
                return "fearless_simd::Avx2+FMA";
            }
            if self.level.as_sse4_2().is_some() {
                return "fearless_simd::Sse4_2";
            }
            if self.level.as_sse2().is_some() {
                return "fearless_simd::Sse2";
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.level.as_neon().is_some() {
                return "fearless_simd::Neon";
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if self.level.as_wasm_simd128().is_some() {
                return "fearless_simd::WasmSimd128";
            }
        }
        "fearless_simd::Fallback"
    }
}
