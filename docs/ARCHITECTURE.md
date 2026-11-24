# Architecture (`docs/ARCHITECTURE.md`)

## Layering & Independence

In the SM-Acoustics layered model (`SM-ENG-001`, *Proposed v1*), `rtconv-simd` sits at **L1 (Portable DSP Algorithms)**:

- **Zero platform coupling**: No dependency on ESP-IDF, FreeRTOS, RTOS, HAL, OS audio drivers, MIDI, USB, networking, file I/O, or GUI frameworks. (The optional bare-metal ESP32-C6 validation harness lives in an independent excluded package under `hardware/esp32c6/`.)
- **Core invariant**: `#![no_std]` and `#![forbid(unsafe_code)]`, with `default = []` (zero `std`, zero `alloc`).
- **Phase-agnostic semantics**: Any finite real causal FIR impulse response (`&[f32]` or `&[i16]` Q1.15) is processed verbatim without phase classification, conversion, or normalization.

## Three-Plane NUPC Architecture

Both `NonUniformConvolver` (`f32`) and `FixedQ15NonUniformConvolver` (`Q1.15/BFP`) share the exact same planning and streaming contracts with zero logic duplication:

1. **Planning Plane (`src/nonuniform/plan.rs`, `src/nonuniform/optimizer.rs`)**: Representation-independent stage validation (`1..=8` contiguous stages), strict zero-latency causality (`D_s >= B_s` for every FDL stage), `NonUniformBudget` admission control (`WithinKnownBounds` / `Exceeded` / `Unverified`), default multi-stage profile (`Direct 0..32`, `FDL B=32 32..256`, `FDL B=64 256..1024`, `FDL B=128 1024..`), and the zero-allocation `#![no_std]` García/Wefers Viterbi partition optimizer (`CostModel`, `ClearancePolicy`, `NonUniformPlan::optimize`, `optimize_block_mode`, and zero-scratch MCU presets `preset_esp32_p4_f32` / `preset_esp32_c6_q15`).
2. **Streaming Plane (`src/nonuniform/scheduler.rs`)**: Sample-time `NonUniformScheduler` with clearance deferral (`defer_s = slack_s.saturating_sub(B_s - 1)`), periodic ceiling distribution (`target(slot) = ceil((slot + 1) * U / span)`) across RFFT stages, CMAC partition ranges, and IFFT stages (`StageWorkItem::RfftSteps` / `ContinueCmac` / `IfftSteps` / `CombinedSlice`), deterministic per-item operation accounting (`stage_work_item_ops`), power-of-two `2 * B_s` output delay rings (bitmask wrap, zero integer division), a single shared real input history ring (`max_s(max(2 * B_s, D_s + 1))`), and `deadline_misses()` telemetry.
3. **Numeric Plane**:
   - `f32` (`src/nonuniform/stages.rs`, `src/nonuniform/mod.rs`): `Direct` FIR head (reverse-indexed circular buffer + `dot_two_segments_simd`) + `FftBackend` 50% overlap-save FDL stages executing zero-extra-RAM in-place staged RFFT/IFFT via `SimdFixedFftBackend` (with transparent monolithic fallback for `MicroFftBackend` / `PhastFftBackend`).
   - `Q1.15/BFP` (`src/fixed/nonuniform.rs`): `i64`-accumulated Q1.15 direct head + time-distributed in-place BFP radix-2 RFFT/IFFT (`rfft_q15_bfp_ws_step` / `irfft_q15_bfp_wide_ws_step` reusing each stage's `8 * bins >= 4 * N` accumulator buffer) + exponent-aligned `i64` CMAC + power-of-two `2 * B_s` Q1.23 output rings (single final convergent RNE rounding). Per-partition coefficient exponents when prepared from Q1.31 taps (`prepare_q31`).

## Automatic Layer

1. **Artifact plane (`src/artifact.rs`)**: zero-copy `FirArtifact` over RIFF/WAVE, raw binary and text. Decodes to `f32`, Q1.15 or Q1.31 on demand.
2. **Domain plane (`build.rs` → `cfg(rtconv_no_fpu)`)**: compile-time FPU classification. On stable Rust, RISC-V `F` is not exposed as a cfg, so the target ISA string is parsed.
3. **Policy plane (`src/auto.rs`)**:
   - `select_f32` evaluates `Direct`, `Hybrid(B)` (`B ∈ {32..=8192}`), and Strategy-2 `NonUniform` under `CostModel::for_target_f32()`; when `AutoEngine::NonUniform` wins, `AutoConvolver::prepare_f32` runs the Viterbi DP in-place inside the caller's `&mut [f32]` workspace with zero heap allocation and zero stack arrays.
   - `select_q15` uses `FixedDirect` up to 128 taps and cost-optimized `FixedNonUniform` (`B <= 1024 / N2048`, `CostModel::esp32_c6_q15_estimate()`) above, also planned in-place inside the caller's `&mut [i16]` workspace.
   - `const fn` sizing (`workspace_len`, verified up to `131,072` taps in `const` generics): exact for `Direct` / `Hybrid` / `FixedDirect`, and a mathematically proven tight upper bound for `NonUniform` / `FixedNonUniform` (tightened by PR-C's shared input ring and `2 * B_s` output rings) covering both in-place DP scratch and the resulting convolver workspace; Q1.15 gain headroom; `f32` and `i16` I/O adapters; stereo wrapper.

## Module Structure

- `build.rs`: FPU classification (`cfg(rtconv_no_fpu)`) plus build-time generation of single quarter-wave `f64`-derived IEEE 754 RNE twiddle ROMs (`TW_COS_Q: [f32; 4097]`, `16,388 B`; `TWIDDLE_COS_Q_2048: [i16; 513]`, `1,026 B`).
- `src/lib.rs`: Public API, `SimdBackend`, `Capabilities`, `EngineKind`.
- `src/artifact.rs`: `FirArtifact`, `ArtifactError`, `SampleEncoding`, `Container`.
- `src/auto.rs` (`#[cfg(feature = "auto")]`): `AutoConvolver`, `AutoStereoConvolver`, `AutoWorkspace`, sizing and selection `const fn`s.
- `src/error.rs`: Typed `ConvError` implementing `core::error::Error`.
- `src/simd_kernels.rs`: Scalar and `fearless_simd 1.2.0` dot-product and split-complex (SoA) CMAC kernels (including partition sub-range accumulation for time-distributed FDL).
- `src/fft.rs`: `FftBackend` trait (with optional zero-extra-RAM staged transform methods `supports_staged_fft`, `staged_forward_steps`, `staged_inverse_steps`, `rfft_forward_step`, `rfft_inverse_step`), `FftSize` (`N64` through `N16384`, `FftSize::ALL`), and zero-stack `MicroFftBackend` (`microfft 0.6.0`, `size-16384`, caller `N`-float scratch, Hermitian real-FFT inverse identity).
- `src/simd_fft.rs` + `src/simd_fft_tables.rs`: `#![no_std]` `SimdFixedFftBackend` (`N64..=N16384`) using `fearless_simd 1.2.0`, `N/2`-point CFFT for R2C/C2R, radix-4 stages 0–1, register-hoisted `half = 4, 8` and branchless Q1/Q2 `S::f32s` butterflies for `half >= 16`, bit-exact in-place staged forward/inverse steps (`log2(N) - 1` steps per transform, `0 B` extra RAM), single strided quarter-wave `static` ROM (`16,388 B`), and `usize::reverse_bits()` (`0 B` bitrev ROM).
- `src/phastft_backend.rs` (`#[cfg(feature = "phastft")]`): Desktop SIMD `PhastFftBackend` wrapping `phastft 0.4.1` (`default-features = false`) with pre-initialized `PlannerR2c32` for `N64..=N16384`.
- `src/direct.rs`: `DirectConvolver` (reverse-indexed circular buffer, unit-stride forward SIMD dot product).
- `src/partitioned.rs`: `PartitionedConvolver` (50% overlap-save uniform FDL, `block_size ∈ {32..=8192}`, single-shot and time-distributed helpers).
- `src/hybrid.rs`: `HybridConvolver` (direct FIR head + partitioned FFT tail with `head_len == tail_block_size ∈ {32..=8192}`, 0-sample latency, optional `set_tail_spreading(true)`).
- `src/nonuniform/` (`#[cfg(feature = "nonuniform")]`):
  - `plan.rs`: `NonUniformPlan`, `NonUniformConfig`, `NonUniformStageConfig` (FDL `block_size ∈ {32..=8192}`), `NonUniformBudget`, `BudgetVerdict`, alignment-independent time-distributed worst-case op bounds (`estimated_worst_ops_per_callback`, `estimated_worst_ops_for_backend`, `check_budget_for_backend`).
  - `optimizer.rs`: Separable `CostModel` (theoretical, calibrated x86 SSE2/AVX2, Wefers 2015 test system, and `INFERRED` ESP32-P4/C6 estimates), `ClearancePolicy` (formal sample-time `c = 1` Strategy 1 & Strategy 2), zero-alloc `#![no_std]` Viterbi DP (`NonUniformPlan::optimize`), block-mode validator (`optimize_block_mode`), in-place `AutoConvolver` DP helpers, and zero-scratch MCU presets.
  - `scheduler.rs`: Sample-time `NonUniformScheduler`, deferred block triggers, periodic ceiling RFFT/CMAC/IFFT work distribution (`StageWorkItem`), and `stage_work_item_ops`.
  - `memory.rs`: Checked workspace sizing (`NonUniformMemoryRequirements`, single shared real input ring + power-of-two `2 * B_s` delay rings).
  - `stages.rs` + `mod.rs`: `NonUniformConvolver` (`f32`).
- `src/fixed/` (`#[cfg(feature = "fixed-q15")]`):
  - `q15.rs`: Q1.15 conversion, round-half-up and convergent round-to-nearest-even (`round_even_sat_q15_from_i64`), and symmetric saturation primitives.
  - `direct.rs`: `FixedQ15DirectConvolver` (`i16` workspace, `i64` accumulator, `saturation_count()`).
  - `block_float.rs`, `fft_q15.rs`, `fft_q15_tables.rs`, `partitioned.rs` (`#[cfg(feature = "fixed-fft")]`): BFP radix-2 FFT (`N64..=N2048`, `1,026 B` quarter-wave ROM) with input-block normalization, convergent RNE per-stage conditional down-scaling, exact unit twiddles, caller-supplied scratch (`rfft_q15_bfp_with_scratch` / `irfft_q15_bfp_with_scratch`), and bit-exact per-stage time-distributed API (`fixed_fft_staged_steps`, `FixedStagedFftState`, `rfft_q15_bfp_step_with_scratch`, `irfft_q15_bfp_step_with_scratch`); two-pass exponent-aligned `FixedQ15PartitionedConvolver` (`block_size ∈ {32..=1024}`) with workspace-packed `i64` accumulators and FFT scratch (`< 512 B` peak stack); `TapSource` / `Q31Taps` per-partition coefficient quantization.
  - `nonuniform.rs` (`#[cfg(feature = "fixed-nonuniform")]`): `FixedQ15NonUniformConvolver` integrating the shared NUPC Planning/Streaming planes with time-distributed Q1.15/BFP stages (`block_size ∈ {32..=1024}`, shared Q1.15 input ring, power-of-two `2 * B_s` Q1.23 output rings, in-place staged BFP FFT/IFFT inside `acc_re + acc_im` with zero extra FFT scratch, single convergent RNE output rounding).
