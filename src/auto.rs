//! Zero-configuration convolution: load a FIR artifact, get the right engine.
//!
//! [`AutoConvolver`] picks, at `prepare` time and without any user tuning:
//!
//! 1. **Numeric domain** from the workspace element type. [`NativeElem`] resolves at compile
//!    time to `f32` on targets with a hardware FPU (x86/x86-64, AArch64, WASM, ARM `eabihf`,
//!    RISC-V with the `F` extension such as ESP32-P4 `riscv32imafc`) and to `i16` (Q1.15 with
//!    block-floating-point FFTs) on FPU-less targets (RISC-V without `F` such as ESP32-C6
//!    `riscv32imac`, soft-float ARM such as Cortex-M0/M3, AVR, MSP430); see
//!    [`HAS_HARDWARE_FPU`]. Passing an explicit `&mut [f32]` or `&mut [i16]` workspace
//!    overrides the choice.
//! 2. **Engine and partition size** from the IR length with an operation-count cost model:
//!    - `f32`: direct FIR costs `L` MACs/sample; the zero-latency hybrid (direct head of `B` taps
//!      plus a uniformly partitioned overlap-save tail, Gardner 1995 / Garcia 2002) costs about
//!      `B + 4 L / B + 20 log2(2B)` (head MACs + `4` real MACs per complex bin per partition +
//!      forward/inverse FFT); the cheapest of `Direct` and `B = 32..=8192` is selected.
//!    - `Q1.15`: direct FIR up to [`Q15_DIRECT_MAX_TAPS`] taps, otherwise the multi-stage
//!      zero-latency non-uniform convolver (`Direct 32 | B=32 | B=64 | B=128`).
//! 3. **Gain headroom** for Q1.15: filters whose peak tap exceeds `1.0` are scaled by `2^-s`
//!    on load and the output is scaled back by `2^s` with saturation (telemetry in
//!    [`AutoConvolver::saturation_count`]).
//!
//! Every engine has **zero algorithmic latency**; all memory comes from one caller workspace
//! whose size is a `const fn` ([`workspace_len`]), so it can be a `static` array:
//!
//! ```
//! use rtconv_simd::auto::{AutoConvolver, AutoWorkspace, workspace_len};
//!
//! // Typically `include_bytes!("filter.wav")` exported by rePhase (WAV, raw `.bin` or text).
//! let bytes: &[u8] = b"0.5\n0.25\n-0.125\n";
//! let mut ws = AutoWorkspace::<{ workspace_len(1024) }>::new();
//! let mut conv = AutoConvolver::from_bytes(bytes, 0, ws.as_mut_slice()).unwrap();
//! let input = [1.0f32, 0.0, 0.0, 0.0];
//! let mut output = [0.0f32; 4];
//! conv.process_f32(&input, &mut output).unwrap();
//! assert!((output[1] - 0.25).abs() < 1e-3);
//! ```

use crate::artifact::FirArtifact;
use crate::fixed::block_float::q31_to_q15;
use crate::fixed::{
    FixedQ15DirectConfig, FixedQ15DirectConvolver, FixedQ15NonUniformConvolver, q15_to_f32,
};
use crate::nonuniform::NonUniformPlan;
use crate::{
    ConvError, DirectConfig, DirectConvolver, HybridConfig, HybridConvolver, SimdBackend,
    SimdFixedFftBackend,
};

// ------------------------------------------------------------------------------------------
// Compile-time FPU detection
// ------------------------------------------------------------------------------------------

/// `true` when the compilation target executes `f32` arithmetic in hardware.
///
/// Decided by the build script from the target triple and cfgs: FPU-less for RISC-V ISAs
/// without `F`/`D`/`G` (e.g. ESP32-C6 `riscv32imac`, ESP32-C3 `riscv32imc`), 32-bit ARM with
/// a soft-float ABI and no VFP (Cortex-M0/M3, `thumbv7em-none-eabi`), Xtensa ESP32-S2, AVR,
/// and MSP430; hardware FPU otherwise (x86/x86-64, AArch64, WASM, `eabihf` ARM such as
/// Cortex-M4F/M7, RISC-V `imafc`/`gc` such as ESP32-P4). Stable Rust does not expose the RISC-V
/// `F` extension as `cfg(target_feature)`, which is why a build script is used. The
/// `native-q15` feature forces the fixed-point domain on any target.
pub const HAS_HARDWARE_FPU: bool = !cfg!(rtconv_no_fpu);

/// Workspace element of the native numeric domain: `f32` with a hardware FPU, `i16` (Q1.15)
/// without one. See [`HAS_HARDWARE_FPU`].
#[cfg(not(rtconv_no_fpu))]
pub type NativeElem = f32;

/// Workspace element of the native numeric domain: `f32` with a hardware FPU, `i16` (Q1.15)
/// without one. See [`HAS_HARDWARE_FPU`].
#[cfg(rtconv_no_fpu)]
pub type NativeElem = i16;

/// Numeric domain an [`AutoConvolver`] runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericDomain {
    /// IEEE `f32` engines (direct / hybrid, `fearless_simd` where available).
    Float32,
    /// Q1.15 fixed-point engines with block-floating-point FFTs (no FPU required).
    FixedQ15,
}

/// Native numeric domain for the compilation target.
pub const NATIVE_DOMAIN: NumericDomain = if HAS_HARDWARE_FPU {
    NumericDomain::Float32
} else {
    NumericDomain::FixedQ15
};

// ------------------------------------------------------------------------------------------
// Engine selection and const sizing
// ------------------------------------------------------------------------------------------

/// Engine chosen by the automatic planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoEngine {
    /// `f32` direct FIR ([`DirectConvolver`]).
    Direct,
    /// `f32` zero-latency hybrid with SPREAD tail ([`HybridConvolver`]), head = partition = `B`.
    Hybrid {
        /// Head length and tail partition size `B`.
        block_size: usize,
    },
    /// Q1.15 direct FIR ([`FixedQ15DirectConvolver`]).
    FixedDirect,
    /// Q1.15/BFP zero-latency non-uniform convolver ([`FixedQ15NonUniformConvolver`]).
    FixedNonUniform,
}

/// Longest IR handled by the Q1.15 direct engine before switching to the non-uniform engine.
///
/// Heuristic from operation counts (the NUPC head alone costs 32 MACs plus a radix-2 FFT pair
/// per partition block); not yet validated by on-device cycle measurements.
pub const Q15_DIRECT_MAX_TAPS: usize = 128;

const HYBRID_BLOCKS: [usize; 9] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

const fn log2(x: usize) -> usize {
    (usize::BITS - 1 - x.leading_zeros()) as usize
}

/// Operations per output sample for the `f32` hybrid with partition size `b` and `taps` taps.
const fn hybrid_cost(taps: usize, b: usize) -> usize {
    b + 4 * taps / b + 20 * log2(2 * b)
}

/// Selects the `f32` engine for `taps` coefficients (cheapest by the cost model).
pub const fn select_f32(taps: usize) -> AutoEngine {
    let mut best = AutoEngine::Direct;
    let mut best_cost = taps;
    let mut i = 0;
    while i < HYBRID_BLOCKS.len() {
        let b = HYBRID_BLOCKS[i];
        let c = hybrid_cost(taps, b);
        if c < best_cost {
            best_cost = c;
            best = AutoEngine::Hybrid { block_size: b };
        }
        i += 1;
    }
    best
}

/// Selects the Q1.15 engine for `taps` coefficients.
pub const fn select_q15(taps: usize) -> AutoEngine {
    if taps <= Q15_DIRECT_MAX_TAPS {
        AutoEngine::FixedDirect
    } else {
        AutoEngine::FixedNonUniform
    }
}

/// Engine workspace (`f32` elements) required by [`select_f32`]`(taps)` with
/// [`SimdFixedFftBackend`]. Mirrors `DirectConvolver`/`HybridConvolver::memory_requirements`.
pub const fn engine_len_f32(taps: usize) -> usize {
    match select_f32(taps) {
        AutoEngine::Hybrid { block_size: b } => {
            let head = if taps < b { taps } else { b };
            let mut total = 2 * head + 2 * b; // direct head + tail staging
            if taps > b {
                let p = (taps - b).div_ceil(b);
                // ir + fdl spectra, prev block, time frame, accumulator, FFT scratch (N = 2B).
                total += 4 * p * (b + 1) + b + 2 * b + 2 * (b + 1) + 2 * b;
            }
            total
        }
        _ => 2 * taps,
    }
}

/// Engine workspace (`i16` elements) required by [`select_q15`]`(taps)`. Mirrors
/// `FixedQ15DirectConvolver` / `FixedQ15NonUniformConvolver::memory_requirements` for the
/// `NonUniformPlan::default_profile` layout.
pub const fn engine_len_q15(taps: usize) -> usize {
    if taps <= Q15_DIRECT_MAX_TAPS {
        return 2 * taps;
    }
    // (ir_offset, end, block_size) of the default profile FDL stages.
    let stages: [(usize, usize, usize); 3] =
        [(32, 256, 32), (256, 1024, 64), (1024, usize::MAX, 128)];
    let head = if taps < 32 { taps } else { 32 };
    let mut total = 2 * head;
    let mut max_fft = 0;
    let mut max_bins = 0;
    let mut i = 0;
    while i < stages.len() {
        let (off, end, b) = stages[i];
        if taps > off {
            let stop = if taps < end { taps } else { end };
            let p = (stop - off).div_ceil(b);
            let bins = b + 1;
            // overlap(2B) + ir/fdl spectra + exponents + i64 accumulators (8*bins)
            // + Q1.23 delay ring (2 words/sample).
            total += 2 * b + 4 * p * bins + 2 * p + 8 * bins + 2 * (off + b);
            max_fft = 2 * b;
            max_bins = bins;
        }
        i += 1;
    }
    // Shared scratch: spec_re + spec_im + time_scratch + packed FFT butterfly scratch (4*max_fft)
    total + 2 * max_bins + 5 * max_fft
}

const fn max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// Exact maximum `f32` engine workspace across all `1 <= L <= max_taps`.
///
/// Within each fixed engine regime (`Direct` or `Hybrid(b)`), `engine_len_f32(L)` is monotonically
/// non-decreasing in `L`. For `hybrid_cost(L, b) = b + (4 * L) / b + 20 * log2(2 * b)`, transitions
/// between consecutive regimes are monotone and occur at:
/// - `Direct -> Hybrid(32)` at `L = 174` (regime end `173`),
/// - `Hybrid(b) -> Hybrid(2b)` at `L = b * (b + 20) / 2 + b / 4` (regime end `L - 1`).
///
/// Checking `max_taps` and every analytical regime end `< max_taps` finds the exact maximum in
/// `O(1)` time during `const` evaluation even for `max_taps >= 131_072`.
const fn worst_engine_len_f32(max_taps: usize) -> usize {
    let mut worst = engine_len_f32(max_taps);
    if max_taps > 173 {
        worst = max(worst, engine_len_f32(173));
    }
    let mut i = 0;
    while i + 1 < HYBRID_BLOCKS.len() {
        let b = HYBRID_BLOCKS[i];
        let regime_end = b * (b + 20) / 2 + b / 4 - 1;
        if regime_end < max_taps {
            worst = max(worst, engine_len_f32(regime_end));
        }
        i += 1;
    }
    worst
}

/// Total `f32` workspace for any IR of up to `max_taps` taps in the `f32` domain
/// (engine memory for the worst engine choice + `max_taps` IR decode scratch).
pub const fn workspace_len_f32(max_taps: usize) -> usize {
    worst_engine_len_f32(max_taps) + max_taps
}

/// Total `i16` workspace for any IR of up to `max_taps` taps in the Q1.15 domain
/// (engine memory for the worst engine choice + `2 * max_taps` words of Q1.31 IR decode
/// scratch).
pub const fn workspace_len_q15(max_taps: usize) -> usize {
    let direct = if max_taps < Q15_DIRECT_MAX_TAPS {
        max_taps
    } else {
        Q15_DIRECT_MAX_TAPS
    };
    max(engine_len_q15(max_taps), 2 * direct) + 2 * max_taps
}

/// Workspace elements ([`NativeElem`]) for one channel of up to `max_taps` taps on the
/// compilation target.
pub const fn workspace_len(max_taps: usize) -> usize {
    if HAS_HARDWARE_FPU {
        workspace_len_f32(max_taps)
    } else {
        workspace_len_q15(max_taps)
    }
}

/// Statically sized, zero-initialized workspace in the native element type.
///
/// Place it in a `static` (e.g. via a `StaticCell`) on embedded targets; for long IRs it is
/// too large for a typical task stack.
pub struct AutoWorkspace<const N: usize> {
    buf: [NativeElem; N],
}

impl<const N: usize> AutoWorkspace<N> {
    /// Creates a zeroed workspace (usable in `const`/`static` initializers).
    pub const fn new() -> Self {
        Self {
            buf: [0 as NativeElem; N],
        }
    }

    /// Mutable view passed to [`AutoConvolver::from_artifact`].
    pub fn as_mut_slice(&mut self) -> &mut [NativeElem] {
        &mut self.buf
    }
}

impl<const N: usize> Default for AutoWorkspace<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------------------------------------
// Workspace element trait (domain dispatch)
// ------------------------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i16 {}
}

/// Workspace element type selecting the numeric domain (`f32` or `i16`). Sealed.
pub trait AutoElem: sealed::Sealed + Copy + 'static {
    /// Domain selected by this element type.
    const DOMAIN: NumericDomain;

    #[doc(hidden)]
    fn prepare<'a>(
        artifact: &FirArtifact<'_>,
        channel: usize,
        workspace: &'a mut [Self],
    ) -> Result<AutoConvolver<'a>, ConvError>;
}

impl AutoElem for f32 {
    const DOMAIN: NumericDomain = NumericDomain::Float32;

    fn prepare<'a>(
        artifact: &FirArtifact<'_>,
        channel: usize,
        workspace: &'a mut [f32],
    ) -> Result<AutoConvolver<'a>, ConvError> {
        let taps = artifact.frames();
        let engine_len = engine_len_f32(taps);
        let need = engine_len + taps;
        if workspace.len() < need {
            return Err(ConvError::WorkspaceTooSmall {
                required: need,
                provided: workspace.len(),
            });
        }
        let (engine_ws, scratch) = workspace.split_at_mut(engine_len);
        let ir = &mut scratch[..taps];
        artifact.decode_f32(channel, ir)?;
        AutoConvolver::prepare_f32(ir, engine_ws).map(|c| c.with_rate(artifact.sample_rate()))
    }
}

impl AutoElem for i16 {
    const DOMAIN: NumericDomain = NumericDomain::FixedQ15;

    fn prepare<'a>(
        artifact: &FirArtifact<'_>,
        channel: usize,
        workspace: &'a mut [i16],
    ) -> Result<AutoConvolver<'a>, ConvError> {
        let taps = artifact.frames();
        let engine_len = engine_len_q15(taps);
        let need = engine_len + 2 * taps;
        if workspace.len() < need {
            return Err(ConvError::WorkspaceTooSmall {
                required: need,
                provided: workspace.len(),
            });
        }
        let shift = artifact.q15_headroom_shift(channel)?;
        let (engine_ws, scratch) = workspace.split_at_mut(engine_len);
        // Q1.31 taps stored as (hi, lo) `i16` word pairs in the scratch tail, so the FFT
        // partitions get per-partition BFP coefficients from 24/32-bit/float sources.
        let pairs = &mut scratch[..2 * taps];
        artifact.for_each_q31(channel, shift, |i, v| {
            pairs[2 * i] = (v >> 16) as i16;
            pairs[2 * i + 1] = v as i16;
        })?;
        let pairs = &*pairs;
        let tap = |i: usize| ((pairs[2 * i] as i32) << 16) | (pairs[2 * i + 1] as u16 as i32);
        AutoConvolver::prepare_q31_with(taps, tap, shift, engine_ws)
            .map(|c| c.with_rate(artifact.sample_rate()))
    }
}

// ------------------------------------------------------------------------------------------
// AutoConvolver
// ------------------------------------------------------------------------------------------

/// Runtime description of the configuration chosen by [`AutoConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoInfo {
    /// Numeric domain.
    pub domain: NumericDomain,
    /// Selected engine.
    pub engine: AutoEngine,
    /// FIR length in taps.
    pub taps: usize,
    /// Algorithmic latency in samples (always `0`).
    pub latency_samples: usize,
    /// Q1.15 load-time attenuation `2^-gain_shift` compensated at the output (`0` for `f32`).
    pub gain_shift: u32,
    /// Sample rate declared by the artifact (WAV only).
    pub sample_rate: Option<u32>,
}

#[allow(clippy::large_enum_variant)]
enum Engine<'a> {
    Direct(DirectConvolver<'a>),
    Hybrid(HybridConvolver<'a, SimdFixedFftBackend>),
    FixedDirect(FixedQ15DirectConvolver<'a>),
    FixedNonUniform(FixedQ15NonUniformConvolver<'a>),
}

/// Zero-latency, zero-allocation convolver that configures itself from the FIR it is given.
///
/// All signal buffers, FDL spectra, and Q1.15 accumulators live inside the caller-supplied
/// workspace; the `AutoConvolver` value itself holds only compact engine metadata and scheduler
/// state.
pub struct AutoConvolver<'a> {
    engine: Engine<'a>,
    info: AutoInfo,
    conversion_saturations: u64,
}

const IO_CHUNK: usize = 64;

impl<'a> AutoConvolver<'a> {
    /// Parses `bytes` with [`FirArtifact::parse`] and prepares `channel` (see
    /// [`Self::from_artifact`]).
    pub fn from_bytes<E: AutoElem>(
        bytes: &[u8],
        channel: usize,
        workspace: &'a mut [E],
    ) -> Result<Self, ConvError> {
        let artifact = FirArtifact::parse(bytes)?;
        E::prepare(&artifact, channel, workspace)
    }

    /// Prepares `channel` of `artifact` in the numeric domain of `E` (use [`NativeElem`] /
    /// [`AutoWorkspace`] for the target's natural domain). The workspace must hold at least
    /// [`workspace_len_f32`] / [`workspace_len_q15`] of the IR length; its tail is used as IR
    /// decode scratch during preparation.
    pub fn from_artifact<E: AutoElem>(
        artifact: &FirArtifact<'_>,
        channel: usize,
        workspace: &'a mut [E],
    ) -> Result<Self, ConvError> {
        E::prepare(artifact, channel, workspace)
    }

    /// Prepares the `f32` domain from IR taps (`workspace.len() >= engine_len_f32(ir.len())`).
    pub fn prepare_f32(ir: &[f32], workspace: &'a mut [f32]) -> Result<Self, ConvError> {
        if ir.is_empty() {
            return Err(ConvError::EmptyIr);
        }
        let taps = ir.len();
        let backend = SimdBackend::detect_or_baseline();
        let engine_kind = select_f32(taps);
        let engine = match engine_kind {
            AutoEngine::Hybrid { block_size } => {
                let fft = SimdFixedFftBackend::new(backend);
                let cfg = HybridConfig {
                    head_len: block_size,
                    tail_block_size: block_size,
                    max_ir_len: taps,
                };
                Engine::Hybrid(HybridConvolver::prepare_with_tail_spreading(
                    ir, cfg, workspace, backend, fft, true,
                )?)
            }
            _ => Engine::Direct(DirectConvolver::prepare(
                ir,
                DirectConfig { max_ir_len: taps },
                workspace,
                backend,
            )?),
        };
        Ok(Self {
            engine,
            info: AutoInfo {
                domain: NumericDomain::Float32,
                engine: engine_kind,
                taps,
                latency_samples: 0,
                gain_shift: 0,
                sample_rate: None,
            },
            conversion_saturations: 0,
        })
    }

    /// Prepares the Q1.15 domain from Q1.15 taps already scaled by `2^-gain_shift`; the output
    /// is multiplied back by `2^gain_shift` with saturation
    /// (`workspace.len() >= engine_len_q15(ir.len())`).
    pub fn prepare_q15(
        ir: &[i16],
        gain_shift: u32,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        if ir.is_empty() {
            return Err(ConvError::EmptyIr);
        }
        let taps = ir.len();
        let engine_kind = select_q15(taps);
        let engine = match engine_kind {
            AutoEngine::FixedDirect => Engine::FixedDirect(FixedQ15DirectConvolver::prepare(
                ir,
                FixedQ15DirectConfig { max_ir_len: taps },
                workspace,
            )?),
            _ => {
                let plan = NonUniformPlan::default_profile(taps, IO_CHUNK)?;
                Engine::FixedNonUniform(FixedQ15NonUniformConvolver::prepare(ir, plan, workspace)?)
            }
        };
        Ok(Self::fixed(engine, engine_kind, taps, gain_shift))
    }

    /// Prepares the Q1.15 domain from Q1.31 taps (`tap(i) / 2^31`, already scaled by
    /// `2^-gain_shift`). FFT partitions keep per-partition block-floating-point coefficients;
    /// the direct paths round to Q1.15 (`workspace.len() >= engine_len_q15(taps)`).
    pub fn prepare_q31_with<F: Fn(usize) -> i32>(
        taps: usize,
        tap: F,
        gain_shift: u32,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        if taps == 0 {
            return Err(ConvError::EmptyIr);
        }
        let engine_kind = select_q15(taps);
        let engine = match engine_kind {
            AutoEngine::FixedDirect => {
                let mut ir = [0i16; Q15_DIRECT_MAX_TAPS];
                for (i, d) in ir[..taps].iter_mut().enumerate() {
                    *d = q31_to_q15(tap(i));
                }
                Engine::FixedDirect(FixedQ15DirectConvolver::prepare(
                    &ir[..taps],
                    FixedQ15DirectConfig { max_ir_len: taps },
                    workspace,
                )?)
            }
            _ => {
                let plan = NonUniformPlan::default_profile(taps, IO_CHUNK)?;
                Engine::FixedNonUniform(FixedQ15NonUniformConvolver::prepare_q31_with(
                    taps, tap, plan, None, workspace,
                )?)
            }
        };
        Ok(Self::fixed(engine, engine_kind, taps, gain_shift))
    }

    fn fixed(engine: Engine<'a>, engine_kind: AutoEngine, taps: usize, gain_shift: u32) -> Self {
        Self {
            engine,
            info: AutoInfo {
                domain: NumericDomain::FixedQ15,
                engine: engine_kind,
                taps,
                latency_samples: 0,
                gain_shift: gain_shift.min(15),
                sample_rate: None,
            },
            conversion_saturations: 0,
        }
    }

    fn with_rate(mut self, rate: Option<u32>) -> Self {
        self.info.sample_rate = rate;
        self
    }

    /// Configuration chosen at preparation time.
    #[inline]
    pub const fn info(&self) -> AutoInfo {
        self.info
    }

    /// Output samples clipped by fixed-point saturation or I/O format conversion since
    /// preparation or [`Self::reset`].
    pub fn saturation_count(&self) -> u64 {
        let engine = match &self.engine {
            Engine::FixedDirect(c) => c.saturation_count(),
            Engine::FixedNonUniform(c) => c.saturation_count(),
            _ => 0,
        };
        engine.saturating_add(self.conversion_saturations)
    }

    /// Clears all signal state (history, FDL rings, delay lines) keeping the filter.
    pub fn reset(&mut self) {
        match &mut self.engine {
            Engine::Direct(c) => c.reset(),
            Engine::Hybrid(c) => c.reset(),
            Engine::FixedDirect(c) => c.reset(),
            Engine::FixedNonUniform(c) => c.reset(),
        }
        self.conversion_saturations = 0;
    }

    fn check_len(input: usize, output: usize) -> Result<(), ConvError> {
        if input != output {
            return Err(ConvError::MismatchedBufferLength {
                expected: input,
                actual: output,
            });
        }
        Ok(())
    }

    #[inline]
    fn apply_gain_q15(&mut self, block: &mut [i16]) {
        let s = self.info.gain_shift;
        if s == 0 {
            return;
        }
        for y in block.iter_mut() {
            let v = (*y as i32) << s;
            let c = v.clamp(i16::MIN as i32, i16::MAX as i32);
            if c != v {
                self.conversion_saturations = self.conversion_saturations.saturating_add(1);
            }
            *y = c as i16;
        }
    }

    /// Processes `f32` samples (full scale `±1.0`), any block length.
    ///
    /// In the Q1.15 domain samples are converted in chunks of 64 on the stack (software
    /// floating point on FPU-less targets; prefer [`Self::process_i16`] there).
    pub fn process_f32(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), ConvError> {
        Self::check_len(input.len(), output.len())?;
        match &mut self.engine {
            Engine::Direct(c) => return c.process_block(input, output),
            Engine::Hybrid(c) => return c.process_block(input, output),
            _ => {}
        }
        let mut qi = [0i16; IO_CHUNK];
        let mut qo = [0i16; IO_CHUNK];
        for (xi, yo) in input.chunks(IO_CHUNK).zip(output.chunks_mut(IO_CHUNK)) {
            let n = xi.len();
            for (d, &s) in qi[..n].iter_mut().zip(xi) {
                let (q, sat) = f32_to_q15_sat(s);
                if sat {
                    self.conversion_saturations = self.conversion_saturations.saturating_add(1);
                }
                *d = q;
            }
            self.process_q15_engine(&qi[..n], &mut qo[..n])?;
            for (d, &q) in yo.iter_mut().zip(&qo[..n]) {
                *d = q15_to_f32(q);
            }
        }
        Ok(())
    }

    /// Processes Q1.15 / 16-bit PCM samples (e.g. straight from I2S), any block length.
    ///
    /// In the `f32` domain samples are converted in chunks of 64 on the stack and the output is
    /// rounded to nearest with saturation.
    pub fn process_i16(&mut self, input: &[i16], output: &mut [i16]) -> Result<(), ConvError> {
        Self::check_len(input.len(), output.len())?;
        if matches!(
            self.engine,
            Engine::FixedDirect(_) | Engine::FixedNonUniform(_)
        ) {
            return self.process_q15_engine(input, output);
        }
        let mut fi = [0.0f32; IO_CHUNK];
        let mut fo = [0.0f32; IO_CHUNK];
        for (xi, yo) in input.chunks(IO_CHUNK).zip(output.chunks_mut(IO_CHUNK)) {
            let n = xi.len();
            for (d, &s) in fi[..n].iter_mut().zip(xi) {
                *d = q15_to_f32(s);
            }
            match &mut self.engine {
                Engine::Direct(c) => c.process_block(&fi[..n], &mut fo[..n])?,
                Engine::Hybrid(c) => c.process_block(&fi[..n], &mut fo[..n])?,
                _ => unreachable_engine()?,
            }
            for (d, &y) in yo.iter_mut().zip(&fo[..n]) {
                let (q, sat) = f32_to_q15_sat(y);
                if sat {
                    self.conversion_saturations = self.conversion_saturations.saturating_add(1);
                }
                *d = q;
            }
        }
        Ok(())
    }

    fn process_q15_engine(&mut self, input: &[i16], output: &mut [i16]) -> Result<(), ConvError> {
        match &mut self.engine {
            Engine::FixedDirect(c) => c.process_block(input, output)?,
            Engine::FixedNonUniform(c) => c.process_block(input, output)?,
            _ => unreachable_engine()?,
        }
        self.apply_gain_q15(output);
        Ok(())
    }
}

/// Typed refusal for a domain/engine mismatch (cannot occur through the public API).
#[inline]
fn unreachable_engine() -> Result<(), ConvError> {
    Err(ConvError::InvalidStateTransition)
}

/// Rounds `x * 32768` to nearest with saturation, reporting whether it clipped (NaN → 0).
#[inline]
fn f32_to_q15_sat(x: f32) -> (i16, bool) {
    if x.is_nan() {
        return (0, false);
    }
    let s = x * 32768.0;
    let r = if s >= 0.0 { s + 0.5 } else { s - 0.5 };
    if r >= 32767.0 {
        (i16::MAX, r >= 32768.0)
    } else if r <= -32768.0 {
        (i16::MIN, r < -32769.0)
    } else {
        (r as i16, false)
    }
}

// ------------------------------------------------------------------------------------------
// Stereo
// ------------------------------------------------------------------------------------------

/// Two independent [`AutoConvolver`]s for a stereo FIR artifact (`L/R`), or the same mono FIR
/// applied to both channels (dual mono).
pub struct AutoStereoConvolver<'a> {
    left: AutoConvolver<'a>,
    right: AutoConvolver<'a>,
}

/// Workspace elements ([`NativeElem`]) for a stereo pair of up to `max_taps` taps per channel.
pub const fn stereo_workspace_len(max_taps: usize) -> usize {
    2 * workspace_len(max_taps)
}

impl<'a> AutoStereoConvolver<'a> {
    /// Prepares channels 0/1 of a stereo artifact, or channel 0 twice for a mono artifact.
    /// The workspace is split in two equal halves.
    pub fn from_artifact<E: AutoElem>(
        artifact: &FirArtifact<'_>,
        workspace: &'a mut [E],
    ) -> Result<Self, ConvError> {
        let half = workspace.len() / 2;
        let (wl, wr) = workspace.split_at_mut(half);
        let right_ch = if artifact.channels() >= 2 { 1 } else { 0 };
        Ok(Self {
            left: E::prepare(artifact, 0, wl)?,
            right: E::prepare(artifact, right_ch, wr)?,
        })
    }

    /// Parses `bytes` and prepares a stereo pair (see [`Self::from_artifact`]).
    pub fn from_bytes<E: AutoElem>(
        bytes: &[u8],
        workspace: &'a mut [E],
    ) -> Result<Self, ConvError> {
        let artifact = FirArtifact::parse(bytes)?;
        Self::from_artifact(&artifact, workspace)
    }

    /// Left channel convolver.
    pub fn left(&mut self) -> &mut AutoConvolver<'a> {
        &mut self.left
    }

    /// Right channel convolver.
    pub fn right(&mut self) -> &mut AutoConvolver<'a> {
        &mut self.right
    }

    /// Processes planar `f32` stereo blocks.
    pub fn process_f32(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), ConvError> {
        self.left.process_f32(in_l, out_l)?;
        self.right.process_f32(in_r, out_r)
    }

    /// Processes planar Q1.15 stereo blocks.
    pub fn process_i16(
        &mut self,
        in_l: &[i16],
        in_r: &[i16],
        out_l: &mut [i16],
        out_r: &mut [i16],
    ) -> Result<(), ConvError> {
        self.left.process_i16(in_l, out_l)?;
        self.right.process_i16(in_r, out_r)
    }

    /// Processes interleaved `L R L R ...` Q1.15 frames (typical I2S DMA layout) in place.
    pub fn process_interleaved_i16(&mut self, frames: &mut [i16]) -> Result<(), ConvError> {
        if !frames.len().is_multiple_of(2) {
            return Err(ConvError::MismatchedBufferLength {
                expected: frames.len() + 1,
                actual: frames.len(),
            });
        }
        let mut l = [0i16; IO_CHUNK];
        let mut r = [0i16; IO_CHUNK];
        let mut lo = [0i16; IO_CHUNK];
        let mut ro = [0i16; IO_CHUNK];
        for chunk in frames.chunks_mut(2 * IO_CHUNK) {
            let n = chunk.len() / 2;
            for i in 0..n {
                l[i] = chunk[2 * i];
                r[i] = chunk[2 * i + 1];
            }
            self.left.process_i16(&l[..n], &mut lo[..n])?;
            self.right.process_i16(&r[..n], &mut ro[..n])?;
            for i in 0..n {
                chunk[2 * i] = lo[i];
                chunk[2 * i + 1] = ro[i];
            }
        }
        Ok(())
    }

    /// Resets both channels.
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }
}
