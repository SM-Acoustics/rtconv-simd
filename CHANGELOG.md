# Changelog

All notable changes to `rtconv-simd` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Target `0.2.0`

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

## [0.1.0] - 2026-11-22 (GitHub Release)

### Added
- Pure Rust `#![no_std]`, `#![forbid(unsafe_code)]` FIR convolution core (`rtconv_simd`) with zero heap allocation in `prepare` and audio processing (`#1`).
- Three convolution engines: `DirectConvolver`, `PartitionedConvolver` (uniform 50% overlap-save FDL), and zero-latency `HybridConvolver`.
- Time-distributed FDL tail spreading in `HybridConvolver` (`prepare_with_tail_spreading`, `set_tail_spreading`) with `ConvError::InvalidStateTransition` protection against mid-cycle mode changes.
- Pluggable `FftBackend` trait with `MicroFftBackend` (`microfft 0.6.0`), `SimdFixedFftBackend` (`fearless_simd 1.2.0`, `no_std`, `N = 64, 128, 256`), and optional desktop `PhastFftBackend` (`phastft 0.4.1`, `--features phastft`).
- Explicit `Capabilities::simd_active` and `SimdBackend::is_simd_active()` distinguishing active vector SIMD from scalar fallback.
- Comprehensive validation suite against an independent `f64` reference, serialized zero-allocation `GlobalAlloc` tests, separated and integrated benchmarks, examples, and GitHub Actions CI workflows.
