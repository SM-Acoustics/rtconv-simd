# Changelog

All notable changes to `rtconv-simd` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Target `0.2.0`

### Added
- **Multi-Stage Non-Uniform Partitioned Convolution (`--features nonuniform`)**: `NonUniformConvolver`, `NonUniformPlan`, `NonUniformStageConfig`, `NonUniformBudget`, `BudgetVerdict`, and sample-time `NonUniformScheduler` supporting up to `MAX_NONUNIFORM_STAGES = 8` causal zero-latency stages (`D_s >= B_s`), `SPREAD` work distribution, and admission-control validation (`#6`).
- **Fixed-Point Q1.15 Direct FIR (`--features fixed-q15`)**: `FixedQ15DirectConvolver` with `i16` caller-managed workspace (50% RAM vs `f32`), `i64` accumulation, round-half-up `(acc + 16384) >> 15`, symmetric saturation, and `saturation_count()` telemetry (`#7`).
- **Fixed-Point Q1.15 / Block Floating Point FFT & FDL (`--features fixed-fft`)**: Conditional per-stage BFP radix-2 FFT (`rfft_q15_bfp` / `irfft_q15_bfp` for `N64/N128/N256`, `512 B` twiddle ROM) and two-pass exponent-aligned `FixedQ15PartitionedConvolver` (`#8`).
- **Integrated Fixed-Point NUPC (`--features fixed-nonuniform`)**: `FixedQ15NonUniformConvolver` reusing `NonUniformPlan`, `NonUniformBudget`, and `NonUniformScheduler` verbatim with Q1.15/BFP numeric stages (`#9`).
- **ESP32-C6 Bare-Metal Validation Harness**: Independent `#![no_std]` `#![no_main]` crate under `hardware/esp32c6/` (`riscv32imac-unknown-none-elf`, excluded from `.crate`), `CROSS_COMPILE_VERIFIED` (`#10`).
- **Agentic Software Assurance Policy & Guard**: `.github/workflows/agentic-assurance.yml`, `scripts/agentic_guard.py`, and `docs/AGENTIC-ASSURANCE.md` (`#4`).
- **Engineering Audit Reports (`WO-RTCONV-004`)**: `docs/WO-RTCONV-004A-CI-AUDIT.md` through `docs/WO-RTCONV-004F-ESP32C6.md` and consolidated `docs/WO-RTCONV-004-FINAL-REPORT.md`.

### Fixed & Changed
- **License transition for `0.2.0+`**: Changed package license from `MIT OR Apache-2.0` to `GPL-3.0-or-later` (with optional commercial dual-licensing via SM Acoustics) by sole copyright holder decision, aligning `rtconv-simd` with the SM Acoustics organization licensing policy. Historical release `v0.1.0` remains available under its original `MIT OR Apache-2.0` terms.
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
