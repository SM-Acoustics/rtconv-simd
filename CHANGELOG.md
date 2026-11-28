# Changelog

All notable changes to `rtconv-simd` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Planned work is tracked in `docs/ROADMAP.md` and issue #13 (sub-issues #14–#24).

### FPU-less Code Size: no `f32` auto path, no soft-float, no `dec2flt` — Issue #32 (PR 1)

#### Breaking (FPU-less builds only: `cfg(rtconv_no_fpu)`, i.e. RV32 without `F`, soft-float ARM, and any build with `native-q15`, including `--all-features`)
- **`AutoConvolver` links only the Q1.15 family.** `Engine::{Direct, Hybrid, NonUniform}` and their `prepare`/`process`/`reset` arms are compiled out; `AutoElem` is implemented for `i16` only (passing an `&mut [f32]` workspace is a compile-time error instead of silently running soft-float `f32` engines).
- **`AutoConvolver::prepare_f32(ir: &[f32], workspace: &mut [i16])`** on FPU-less builds: taps are decomposed from `f32::to_bits()` (IEEE Std 754-1985 §3.2.1), the headroom shift is found by exact integer comparison and the taps are rounded to Q1.31 with round-to-nearest-even (§4.1, §5.4) before `prepare_q31_with`. Needs `engine_len_q15(taps)` `i16` words; bit-identical to `from_artifact` on a raw `f32` artifact with the same taps (`tests/auto_convolver.rs::no_fpu_prepare_f32_matches_from_artifact_bit_exactly`). Hardware-FPU builds keep the `f32` signature and behaviour unchanged.
- The explicit engines (`DirectConvolver`, `HybridConvolver`, `NonUniformConvolver`, `SimdFixedFftBackend`) remain public on every target for deliberate soft-float use.

#### Changed
- **Integer-only artifact decoding (`src/artifact.rs`, new `src/numconv.rs`).** Parsing, `q15_headroom_shift`, `decode_q15`, `decode_q31` and the auto-layer Q1.31 load execute no floating-point operation for any container. `f32`/`f64` samples are decoded from their bit patterns; raw-float auto-detection compares bit patterns (identical decisions). Only `decode_f32` and `peak` produce floats and are linked only when called.
- **Text decoder without `str::parse::<f64>` on the fixed-point path.** Grammar `[+-]? (D+ ([.,] D*)? | [.,] D+) ([eE] [+-]? D+)?` plus `inf`/`infinity`/`nan` (same accepted set as `core`'s parser over `.`-tokens, verified exhaustively up to 5 characters). Values are scaled exactly in `u128` and rounded to nearest even with a sticky bit, for **any** number of digits. Note on the 19-digit design: a `u64` mantissa of 19 significant digits plus a sticky flag cannot round Q1.31 ties exactly, because ties `(2q+1)·2^-32` have up to 32 significant digits — `3·2^-32 = 6.9849193096160888671875e-10` would come out `1` instead of `2`; the decoder therefore truncates at the `10^(s-32)` decimal place, where every rounding boundary lies, which is exact (proof in `src/numconv.rs`, counterexample pinned in `numconv::tests::nineteen_digit_truncation_counterexample`).
- **Rounding of IEEE and text sources into Q1.15 / Q1.31 is now round-half-to-even** (IEEE Std 754-1985 §4.1; ABNT NBR 5891:2014 §2.2–§2.4) instead of half-away-from-zero through `f64`. Results differ only on exact ties (±1 LSB). Integer PCM keeps its half-away-from-zero rule (pinned by `pcm_to_q15_is_integer_exact`).
- **Q1.15-domain `AutoConvolver::process_f32`** converts with integer arithmetic on every target (bit-identical host/MCU): input `RNE(x·2^15)` with saturation, output exactly `q/2^15`. This also removes a double-rounding artefact of the former `x * 32768 + 0.5` (e.g. the largest `f32` below half an LSB rounded up).
- **Integer Q1.15 NUPC planner (`src/nonuniform/optimizer.rs`).** The auto layer's in-workspace Viterbi DP for `FixedQ15NonUniformConvolver` now runs on `IntegerCostModel`, an exact compile-time image of `CostModel::esp32_c6_q15_estimate()` (costs × 8192, all integers), with saturating `u32` costs in the same `i16` scratch layout. Unit tests show identical plans to the former `f32` DP for every length `1..=6000`, a sweep to `300,000` taps, `2,000,000` taps, and other stage limits, caps and policies.
- **`hardware/esp32c6`**: the `f32` `DirectConvolver` soft-float comparison moved behind the harness feature `soft-float-baseline` (off by default).
- **CI (`.github/workflows/embedded.yml`, `scripts/check-no-fpu-binary.sh`)**: after `cargo build --release` of the C6 harness, the job fails if `llvm-nm` finds `TW_COS_Q`, `SimdFixedFftBackend`, `POWER_OF_FIVE_128`, `__mulsf3`, `__muldf3` or any other soft-float routine, or if `.text + .rodata` exceeds 81,196 B (0.2.0); `llvm-size -A` sections are written to the job summary. `ci.yml` now runs the full suite under `native-q15`.

#### Measured (`CROSS_COMPILE_VERIFIED`, `rustc 1.97.0`, `hardware/esp32c6` profile)
| Tree | `.text` | `.rodata` | `.text + .rodata` |
| :--- | ---: | ---: | ---: |
| before (`caa0581`) | 82,076 | 37,528 | 119,604 |
| after | 43,362 | 6,656 | **50,018** (budget 81,196) |

### License Transition (`0.3.0+`)

#### Changed
- **License transition for post-`v0.2.0` (`0.3.0+`)**: Changed package license from `MIT OR Apache-2.0` to `GPL-3.0-or-later` (with optional commercial dual-licensing via SM Acoustics) by sole copyright holder decision, aligning `rtconv-simd` with the SM Acoustics organization licensing policy. Historical release tags `v0.1.0` and `v0.2.0` remain available under their original `MIT OR Apache-2.0` terms.

### Time-Distributed FFT/IFFT & Clearance Deferral — Issue #16 (PR-C)

#### Added
- **Non-breaking staged FFT extension on `FftBackend` (`src/fft.rs`)**: `supports_staged_fft(&self) -> bool`, `staged_forward_steps`, `staged_inverse_steps`, `rfft_forward_step`, and `rfft_inverse_step` with monolithic defaults so `MicroFftBackend`, `PhastFftBackend`, and external backends keep full backward compatibility.
- **Bit-exact zero-extra-RAM staged `f32` transforms (`src/simd_fft.rs`)**: `SimdFixedFftBackend` decomposes `N64..=N16384` forward/inverse RFFTs into `log2(N) - 1` (`5..=13`) radix-2 butterfly stages executing in-place inside each FDL stage's newest FDL slot / accumulator (`0 B` extra scratch, bit-identical `to_bits()` output across `Scalar` and `AVX2+FMA`).
- **Bit-exact zero-extra-RAM staged `Q1.15` BFP transforms (`src/fixed/fft_q15.rs`, `src/fixed/mod.rs`)**: `fixed_fft_staged_steps`, `FixedStagedFftState`, `rfft_q15_bfp_step_with_scratch`, and `irfft_q15_bfp_step_with_scratch` (`N64..=N2048`, `log2(N)` = `6..=11` steps), plus workspace-backed `rfft_q15_bfp_ws_step` / `irfft_q15_bfp_wide_ws_step` running in-place inside each stage's `8 * (B_s + 1) >= 4 * N_s` `i64` accumulator buffer.
- **Unified periodic-ceiling scheduler & clearance deferral (`src/nonuniform/scheduler.rs`)**: extended `StageWorkItem` (`RfftSteps`, `ContinueCmac`, `IfftSteps`, `CombinedSlice`, `starts_block`, `commits_output`), added `on_fdl_sample_with_steps`, `stage_work_item_ops`, clearance deferral `defer_s = slack_s.saturating_sub(B_s - 1)`, and periodic-ceiling distribution `target(slot) = ceil((slot + 1) * U / span)` across `[t_start(k), t_fin(k)]` eliminating boundary double-spikes.
- **Backend-aware admission bounds (`src/nonuniform/plan.rs`)**: `direct_sample_ops()`, `estimated_worst_ops_for_backend(&fft)`, and `check_budget_for_backend(budget, bytes, &fft)` using the proven alignment-independent callback bound `q * U + min(U, ceil(rem * U / span))` (`100.0%` tight on Strategy 1 `4k`/`16k`, `2.69×` worst-ops reduction on `64k` Strategy 1, `3.10×` admission-bound tightening on `64k` Strategy 2).
- **Comprehensive verification (`tests/time_distributed_fft.rs`)**: bit-exactness (`SimdFixedFftBackend` `N64..=N16384` Scalar/SIMD, `fft_q15` `N64..=N2048`, full `NonUniformConvolver` vs monolithic wrapper + `f64` oracle `SNR > 100 dB`), deterministic sliding-window op-trace proofs (`1.21×–1.76×` peak/mean on Strategy 1), RAM reduction proofs, fragmented-block zero-deadline-miss tests, Q1.15 AES17 `997 Hz` tests (`91.13 / 90.83 dB` @ `−1 dBFS`, `0` saturations), plus new coverage in `tests/denormal_guard.rs` and `tests/no_alloc_rt.rs`.

#### Changed
- **Shared real input history ring & power-of-two `2 * B_s` output delay rings (`src/nonuniform/{memory,stages}.rs`, `src/fixed/nonuniform.rs`)**: replaced all per-stage `2 * B_s` overlap input buffers with a single shared input ring (`max_s(max(2 * B_s, D_s + 1))`), shrank every FDL output delay ring from `D_s + B_s` to power-of-two `2 * B_s` (`4 * B_s` `i16` words in Q1.23) with branchless bitmask wrap (`pos & delay_ring_mask`, zero integer division in the hot loop), and eliminated the separate `4 * max_fft` Q1.15 shared FFT scratch (`−16.00 KiB` at `N2048`, `−56.9%` to `−75.2%` delay-ring RAM on high-offset plans).
- **Tightened `AutoConvolver` `const fn` NUPC workspace upper bounds (`src/auto.rs`)**: updated `nupc_upper_bound_f32` and `nupc_upper_bound_q15` for `2 * B_s` / `4 * B_s` rings and shared input ring (verified in `tests/auto_convolver.rs` up to `workspace_len(131_072)`).

### Cost-Model NUPC Partition Optimizer — Issue #15 (PR-B)

#### Added
- **Separable `CostModel` (`src/nonuniform/optimizer.rs`)**: `t_fft`, `t_ifft`, `t_cmul`, `t_cmac`, `t_add`, `t_direct_mac` (Wefers 2015 Eq. 6.12, 6.14, 6.43–6.45) with 6 presets: `theoretical()` (Table 7.2), `x86_sse2_calibrated()` / `x86_avx2_calibrated()` (`HOST_ONLY` Xeon PR-A calibration), `wefers_2015_test_system()` (reproducing Ch. 6/Appendix A.2 within `0.15` cycles/sample), `esp32_p4_f32_estimate()` / `esp32_c6_q15_estimate()` (explicitly labelled `INFERRED`, `DEVICE_VERIFIED = NO`), plus `for_target_f32()` and `for_backend_f32(SimdBackend)`.
- **Configurable `ClearancePolicy` with formal `c = 1` sample-time derivation**: `MinLoad`, `Strategy1` (`offset(i) - B_i >= B_i - 1`, derived formally from `NonUniformScheduler`'s `W_eff = min(slack, B_s - 1)` single-block-in-flight contract and verified on boundary `D_s = 2B - 1` vs `2B - 2`), `Strategy2` (host-block-scaled `2/4/8`-cycle spans + bounded direct head), and `Custom { min_slack_samples }`.
- **Zero-allocation `#![no_std]` Viterbi DP (`NonUniformPlan::optimize`, `optimize_with_max_block_size`, `optimizer_scratch_len`, `OptimizerScratchCell`, `OptimizationReport`)**: exact level-by-level Bellman/Viterbi solver in caller-supplied scratch (`0 B` heap, `< 512 B` stack), verified against exhaustive brute-force DFS for small `N`, guaranteeing cost `<= default_profile` and `100%` `NonUniformPlan::validate` compliance.
- **Block-mode Wefers (2015) validator (`optimize_block_mode`, `BlockModePolicy`, `block_mode_scratch_len`)**: reproduces §6.5.3 (`<= 4` segments, uniform optimal up to `~20–30·B`), §6.12 & Appendix A.2.5 `(128^2 256^4 1024^8 8192^10)` at `240.3` vs `240.4` cycles/sample for `88,200` taps (`B = 128`), A.2.1 `(128^7 1024^7 8192^10)` at `204.0` vs `204.1`, A.2.4 `(128^22 1024^21 8192^8)`, and §6.5.4 prose Strategy 1.
- **Zero-scratch MCU presets**: `preset_esp32_p4_f32` and `preset_esp32_c6_q15` (`0 B` optimizer scratch, `B <= 1024 / N2048` on C6).
- **Tests & Benchmarks**: `tests/nupc_optimizer.rs` (validation, brute-force optimality proof, `default_profile` dominance, formal `c = 1` boundary proof, Wefers §6.5.3/§6.12/A.2.5 reproduction, `f64` oracle `SNR > 100 dB` up to 64k, AES17 997 Hz @ `−60` / `−1 dBFS`), zero-alloc coverage in `tests/no_alloc_rt.rs`, and `benches/optimizer_prediction_bench.rs` (Wefers Figs. 7.5 & 7.6 analogue with GUM Type A and ISO 80000-1 / NBR 5891 statistics).

#### Changed
- **`AutoConvolver` (`src/auto.rs`)**: added `AutoEngine::NonUniform`; `select_f32` and `prepare_f32` now select cost-optimized `NonUniformConvolver` automatically whenever Strategy-2 NUPC beats `Hybrid`, running the Viterbi DP in-place inside the caller's `&mut [f32]` workspace with zero extra RAM; `prepare_q15` / `prepare_q31_with` likewise optimize `FixedQ15NonUniformConvolver` (`B <= 1024`) in-place inside `&mut [i16]`. `engine_len_f32` / `engine_len_q15` / `workspace_len*` remain `const fn` (exact for `Direct`/`Hybrid`/`FixedDirect`, mathematically proven upper bound for cost-optimized NUPC, verified at `workspace_len(131_072)` in `const` generics).

### Large FFT Sizes `N512`–`N16384` (`f32`) & up to `N2048` (`Q1.15`) — Issue #14 (PR-A)

#### Added
- **`FftSize` extended to `N512`, `N1024`, `N2048`, `N4096`, `N8192`, `N16384`** (`block_size` `32..=8192`), plus `FftSize::ALL` and `FftSize::log2()`.
- **Build-time single quarter-wave twiddle ROMs (`build.rs`)**: `TW_COS_Q: [f32; 4097]` (`16,388 B`, `f64` → `f32` IEEE 754 RNE) shared across `N64..=N16384` in `SimdFixedFftBackend` (`0 B` bit-reversal ROM via `usize::reverse_bits()`) and `TWIDDLE_COS_Q_2048: [i16; 513]` (`1,026 B`, `f64::round_ties_even`) shared across `N64..=N2048` in `fft_q15`.
- **Zero-stack Q1.15 BFP FFT scratch API (`src/fixed/fft_q15.rs`)**: `rfft_q15_bfp_with_scratch` / `irfft_q15_bfp_with_scratch`, `fixed_fft_scratch_len_i32`, `fixed_fft_supports_size`, `fixed_fft_rom_bytes`, and convergent round-to-nearest-even (`round_even_sat_q15_from_i64`, `round_shift_even_i32/i64`).
- **AES17 & ITU-R BS.1770-5 audio verification suite (`tests/aes17_audio_metrics.rs`)**: 997 Hz tone at `−60 dBFS`, `−1 dBFS`, and `0 dBFS` across `f32` (`N512..=N16384`, SNR `127.9–135.9 dB`, DR `193.4–201.8 dB`) and `Q1.15` (`N64..=N2048`, reporting exact `~2.7 dB/stage` BFP degradation vs the 98.09 dB Q1.15 sine ceiling and `0.00 dB` DR loss at `−60 dBFS`).
- **131,072-tap adversarial `f64` oracle sweep (`tests/adversarial_reference.rs`)**: NUPC up to `N16384` (`SNR = 131.77 dB`, `max_abs = 6.122e-6`, `deadline_misses = 0`), plus large-FFT coverage in `tests/fft_backends_suite.rs` (`> 131.9 dB` across all backends), `tests/denormal_guard.rs`, `tests/no_alloc_rt.rs`, and `tests/auto_convolver.rs` (`workspace_len(131_072)` in `const` generics).
- **GUM Type A + ISO 80000-1 / ABNT NBR 5891 benchmarks (`benches/nonuniform_bench.rs`, `benches/fft_backends_bench.rs`)**: 16k and 64k taps NUPC vs Hybrid (`HOST_ONLY`) and isolated FFT/CMAC/DOT calibration timings for Issue #15.

#### Changed
- **`MicroFftBackend`**: enabled `microfft` `size-16384` (`16,380 B` sine ROM) and replaced stack-allocated `[f32; N]` / `[Complex32; N]` arrays with caller-supplied `N`-float external scratch (`scratch_len_f32 == size.len()`) and a zero-stack Hermitian real-FFT inverse identity (`< 256 B` peak stack across `N64..=N16384`).
- **`SimdFixedFftBackend` & `PhastFftBackend`**: generalized to `N64..=N16384`; `SimdFixedFftBackend` adds register-hoisted `half = 8` and branchless Q1/Q2 `S::f32s` butterflies.
- **`PartitionedConfig`, `HybridConfig`, `NonUniformPlan::validate`, `auto::select_f32`**: extended to `block_size` `32..=8192`; `auto::worst_engine_len_f32` uses O(1) analytical regime endpoints so `workspace_len(131_072)` evaluates cleanly in `const` generics.
- **`FixedQ15PartitionedConvolver` & `FixedQ15NonUniformConvolver`**: extended to `block_size <= 1024` (`N2048`), rejecting `> 1024` with `ConvError::UnsupportedFftSize`; moved per-stage `i64` spectral accumulators and FFT `re`/`im` work arrays into the caller `&mut [i16]` workspace (peak stack `< 512 B`, `FixedQ15NonUniformConvolver::instance_bytes < 4 KiB`, single convergent RNE rounding at final output). Legacy stack-based `rfft_q15_bfp` / `irfft_q15_bfp` remain capped at `N <= 256` and return `Err(ConvError::UnsupportedFftSize)` for `N > 256`.

## [0.2.0] - 2026-11-23

Version bump on `main`. The `v0.2.0` tag, GitHub release and crates.io publication are maintainer actions.

### IEEE 754 subnormal guard (2026-11-23)

#### Fixed
- **Subnormal (denormal) slowdowns in every `f32` engine.** New `simd_kernels::DENORMAL_GUARD = 2^-63`, with
  `flush_tiny` / `copy_flush_tiny` / `flush_tiny_in_place`.
  - What is flushed: input samples, IR coefficients, pre-transformed IR spectra, FDL input slots, spectral
    accumulators and outputs, whenever `|v| < 2^-63` (≈ −379 dBFS).
  - Why `2^-63`: every non-zero product of two guarded values is `≥ 2^-126`, i.e. normal.
  - Measured on x86-64 without FTZ/DAZ (`benches/denormal_bench.rs`, 1024 taps):
    - `DirectConvolver`: 346 → 4.3 µs per 32-sample block for 1e-37 inputs, and 67 → 4.2 µs for an IR whose
      f64 deep tail decoded to subnormal `f32` taps.
    - `HybridConvolver`: 48 → 0.9 µs.
  - No measurable cost for normal signals.
  - Q1.15 engines are integer-only and unchanged.
- Tests: `tests/denormal_guard.rs` checks that tiny or subnormal inputs and coefficients are bit-identical to
  their flushed counterparts, that no output is subnormal, and that a decaying input drains to exact zeros,
  for every engine.

#### Added
- `benches/denormal_bench.rs`.
- `docs/ROADMAP.md` rewritten around Wefers 2015 (*Partitioned convolution algorithms for real-time
  auralization*), heterogeneous CPU/GPU execution, AES69 (SOFA), ISO 3382-1 and IEEE 754. GitHub issues
  #13–#24 opened, each with scope and acceptance criteria.

### Adversarial DSP review (2026-11-23) — see `docs/REVIEW-2026-11-ADVERSARIAL.md`

#### Fixed
- **Q1.15 FFT is now true block floating point** (`rfft_q15_bfp` / `irfft_q15_bfp`): quiet blocks are
  normalized to the full mantissa (negative block exponent) before the per-stage conditional
  down-scaling. `FixedQ15PartitionedConvolver` SNR at 0 / −20 / −40 dBFS input: 47.8 / 46.0 / 30.5 dB →
  **68.7 / 50.4 / 30.5 dB**, within 1.7 dB of the Q1.15 output-rounding ceiling.
- **`FixedQ15NonUniformConvolver` single rounding**: FDL stage outputs are carried at Q1.23 in the delay rings
  (the ring now uses 2 `i16` words per sample) and the direct head is summed at Q2.30, with one final rounding.
  SNR 47.7 / 43.0 / 25.8 dB → **69.4 / 50.4 / 30.5 dB**. The direct head is no longer saturated on its own
  before the sum.
- **Exact unit twiddles** (`W^0`, `W^{N/4}`) in the Q1.15 butterflies (previously `32767/32768`).
- **`NonUniformPlan::estimated_worst_ops_per_callback` is now an actual upper bound.** It counts `ceil(H/B)`
  block triggers per callback, both FFT transforms, and spread completions; before, it counted one block
  whatever the host block size.
- **`FixedQ15NonUniformMemoryRequirements::instance_bytes`** discloses the ~16.5 KiB of `i64` accumulators held
  inside the convolver value. It is included in `NonUniformBudget` memory admission.
- **`NonUniformConvolver` (`f32`) direct head now uses the selected SIMD backend** (it was ignored before).
- Documentation claims corrected: partitioned latency, `fft_q15` stack usage, the competitive matrix,
  harness coverage, and CI scope.

#### Added
- **`artifact` module** (always available, `no_std`, zero-copy, no allocation): `FirArtifact` decodes RIFF/WAVE
  (PCM `u8`/`i16`/`i24`/`i32`, IEEE `f32`/`f64`, `WAVE_FORMAT_EXTENSIBLE`, any channel count), raw `.bin`
  (`f32`/`f64` auto-detected, explicit integer layouts), and numeric text (decimal point or comma, multi-column).
  It decodes to `f32`, Q1.15, or Q1.31; integer→fixed conversion uses integer arithmetic only.
  `ConvError::Artifact(ArtifactError)` added.
- **`auto` module (default feature)**: `AutoConvolver` / `AutoStereoConvolver` / `AutoWorkspace`.
  - The numeric domain is chosen at compile time by `build.rs` from the target ISA: ESP32-C6 `riscv32imac` → Q1.15;
    ESP32-P4 `riscv32imafc` → `f32`. Stable Rust does not expose RISC-V `F` as a cfg.
  - The engine and partition size come from a cost model; Q1.15 gain headroom is automatic.
  - Workspace sizing is a `const fn`, and `f32` / `i16` I/O works in both domains.
- **`native-q15` feature** forces the Q1.15 auto domain on FPU targets.
- **Per-partition BFP coefficients**: `FixedQ15PartitionedConvolver::prepare_q31`, `FixedQ15NonUniformConvolver::prepare_q31`
  / `prepare_q31_with`. A 4096-tap linear-phase FIR measured 62.6 dB before the review, and now 64.2 dB
  (Q1.15 taps) or **70.4 dB** (Q1.31 taps).
- Tests: `tests/adversarial_reference.rs` (cross-engine `f64` oracle sweep), `tests/artifact_formats.rs`,
  `tests/auto_convolver.rs`, `auto` coverage in `tests/no_alloc_rt.rs`, and README code blocks run as doctests.
- `examples/auto_from_artifact.rs`. The ESP32-C6 harness now exercises `AutoConvolver` on a raw `.bin` artifact.
- CI: ESP32-P4 (`riscv32imafc`) and Cortex-M0 (`thumbv6m`) targets; feature builds with `-D warnings` on every
  embedded target; ESP32-C6 harness build; default-feature and `native-q15` test runs.

#### Changed
- `default = ["auto"]` (was `[]`). Every default feature is still `no_std` with no `alloc`; `--no-default-features`
  gives the previous minimal core.

### Added (work orders `WO-RTCONV-004*`)
- **Multi-Stage Non-Uniform Partitioned Convolution (`--features nonuniform`)**: `NonUniformConvolver`, `NonUniformPlan`, `NonUniformStageConfig`, `NonUniformBudget`, `BudgetVerdict`, and sample-time `NonUniformScheduler` supporting up to `MAX_NONUNIFORM_STAGES = 8` causal zero-latency stages (`D_s >= B_s`), `SPREAD` work distribution, and admission-control validation (`#6`).
- **Fixed-Point Q1.15 Direct FIR (`--features fixed-q15`)**: `FixedQ15DirectConvolver` with `i16` caller-managed workspace (50% RAM vs `f32`), `i64` accumulation, round-half-up `(acc + 16384) >> 15`, symmetric saturation, and `saturation_count()` telemetry (`#7`).
- **Fixed-Point Q1.15 / Block Floating Point FFT & FDL (`--features fixed-fft`)**: Conditional per-stage BFP radix-2 FFT (`rfft_q15_bfp` / `irfft_q15_bfp` for `N64/N128/N256`, `512 B` twiddle ROM) and two-pass exponent-aligned `FixedQ15PartitionedConvolver` (`#8`).
- **Integrated Fixed-Point NUPC (`--features fixed-nonuniform`)**: `FixedQ15NonUniformConvolver` reusing `NonUniformPlan`, `NonUniformBudget`, and `NonUniformScheduler` verbatim with Q1.15/BFP numeric stages (`#9`).
- **ESP32-C6 Bare-Metal Validation Harness**: Independent `#![no_std]` `#![no_main]` crate under `hardware/esp32c6/` (`riscv32imac-unknown-none-elf`, excluded from `.crate`), `CROSS_COMPILE_VERIFIED` (`#10`).
- **Agentic Software Assurance Policy & Guard**: `.github/workflows/agentic-assurance.yml`, `scripts/agentic_guard.py`, and `docs/AGENTIC-ASSURANCE.md` (`#4`).
- **Engineering Audit Reports (`WO-RTCONV-004`)**: `docs/WO-RTCONV-004A-CI-AUDIT.md` through `docs/WO-RTCONV-004F-ESP32C6.md` and consolidated `docs/WO-RTCONV-004-FINAL-REPORT.md`.

### Fixed & Changed
- Refactored `tests/no_alloc_rt.rs` to a standalone single-threaded `harness = false` binary with positive allocation control, resolving the macOS `libtest` concurrency flake while preserving Rust `1.89.0` compatibility (`#5`).
- Pinned `dtolnay/rust-toolchain` by commit SHA with explicit `toolchain:` inputs across CI workflows and added a Dependabot `ignore` rule so MSRV `1.89.0` is never auto-bumped (`#5`).
- Upgraded GitHub Actions `actions/checkout` from `v4` to `v7` across all four workflows (`#2`).

## [0.1.0] - 2026-11-22 (GitHub Release `v0.1.0`)

### Added
- Pure Rust `#![no_std]`, `#![forbid(unsafe_code)]` FIR convolution core (`rtconv_simd`) with zero heap allocation in `prepare` and audio processing (`#1`).
- Three convolution engines: `DirectConvolver`, `PartitionedConvolver` (uniform 50% overlap-save FDL), and zero-latency `HybridConvolver`.
- Time-distributed FDL tail spreading in `HybridConvolver` (`prepare_with_tail_spreading`, `set_tail_spreading`) with `ConvError::InvalidStateTransition` protection against mid-cycle mode changes.
- Pluggable `FftBackend` trait with `MicroFftBackend` (`microfft 0.6.0`), `SimdFixedFftBackend` (`fearless_simd 1.2.0`, `no_std`, `N = 64, 128, 256`), and optional desktop `PhastFftBackend` (`phastft 0.4.1`, `--features phastft`).
- Explicit `Capabilities::simd_active` and `SimdBackend::is_simd_active()` distinguishing active vector SIMD from scalar fallback.
- Comprehensive validation suite against an independent `f64` reference, serialized zero-allocation `GlobalAlloc` tests, separated and integrated benchmarks, examples, and GitHub Actions CI workflows.
