# WO-RTCONV-004 — Consolidated Engineering Report

**Repository**: `https://github.com/SM-Acoustics/rtconv-simd`  
**Date**: `2026-11-23`  
**Toolchains Verified**: Windows `rustc 1.100.0 (0ef1bc386 2026-11-09)` + MSRV `rustc 1.89.0 (29483883e 2025-08-04)` (`x86_64-pc-windows-gnu`, AMD Ryzen 3 PRO 2300U); WSL Ubuntu 24.04 `rustc 1.99.1 (58cf8f9c3 2026-10-13)` (`x86_64-unknown-linux-gnu` + `riscv32imac-unknown-none-elf`).  
**Release / Registry State**: GitHub Release `v0.1.0` (`0e17e20`) published `2026-11-22T23:57:17Z`; `crates.io` `rtconv-simd` returns HTTP `404` (not yet published).

---

## 1. Initial Baseline & Final HEAD
- **Initial Remote `origin/main`**: `e214f74` (after PR #4 `policy: introduce AI-assisted software assurance and review gates`, ahead of historical reference `b706e71`). Tag `v0.1.0` (`0e17e20`) preserved untouched.
- **Final Branch Stack** (no direct writes or merges to `main`):
  - `fix/rtconv-004a-ci-harness` (`6cba28d`)
  - `feat/rtconv-004b-nonuniform` (`e7168a7`)
  - `feat/rtconv-004c-fixed-q15` (`ce4e354`)
  - `feat/rtconv-004d-fixed-fft` (`14047ac`)
  - `feat/rtconv-004e-fixed-nupc` (`08e6bc2`)
  - `test/rtconv-004f-esp32c6` (this commit)

## 2. Pull Requests Created (No Auto-Merge)
- **PR #5** (`fix/rtconv-004a-ci-harness`): `https://github.com/SM-Acoustics/rtconv-simd/pull/5` — **10/10 GitHub Actions checks PASS** (`macos-latest`, `ubuntu-latest`, `windows-latest`, `MSRV 1.89.0`, `Format/Clippy/Docs`, `agentic-guard`, 4 `no_std` cross-targets).
- **PR #6** (`feat/rtconv-004b-nonuniform`): `https://github.com/SM-Acoustics/rtconv-simd/pull/6`
- **PR #7** (`feat/rtconv-004c-fixed-q15`): `https://github.com/SM-Acoustics/rtconv-simd/pull/7`
- **PR #8** (`feat/rtconv-004d-fixed-fft`): `https://github.com/SM-Acoustics/rtconv-simd/pull/8`
- **PR #9** (`feat/rtconv-004e-fixed-nupc`): `https://github.com/SM-Acoustics/rtconv-simd/pull/9`
- **PR #10** (`test/rtconv-004f-esp32c6`): opened on push of this branch.
- **External Dependabot PRs Preserved**: PR #2 (`actions/checkout` 4→7, open for maintainer review); PR #3 (`dtolnay/rust-toolchain` 1.89→1.120, broken — superseded by PR #5, must not be merged).

## 3. CI Fixes, macOS Root Cause & Dependabot Policy (RTCONV-004A)
- **macOS Root Cause**: `tests/no_alloc_rt.rs` ran under `libtest` (`harness = true`), spawning a worker thread while the main harness thread stayed active. On macOS, concurrent harness/runtime allocations raced with the process-wide `GUARD_ACTIVE: AtomicBool`. Converting to a standalone single-threaded `harness = false` binary with environment warm-up, separated phase assertions, and a deliberate `black_box(Box::new(...))` positive control eliminated the flake while preserving Rust `1.89.0` compatibility (zero TLS inside `GlobalAlloc`).
- **Dependabot Policy**: Pinned `dtolnay/rust-toolchain@df2a88e9fe4098bad09cb42c3dec7187f15861fc` with explicit `toolchain: "1.89.0"` / `toolchain: stable` inputs across `ci.yml`, `embedded.yml`, and `release-check.yml`, and added an `ignore` rule for `dtolnay/rust-toolchain` in `.github/dependabot.yml`.

## 4. NUPC Architecture & Scheduler Contract (RTCONV-004B)
- `src/nonuniform/{plan,scheduler,memory,stages,mod}.rs` (`feature = "nonuniform"`, `#![no_std]`, `#![forbid(unsafe_code)]`).
- Enforces contiguous IR coverage, strict zero-latency causality (`D_s >= B_s`), `NonUniformBudget` admission control (`BudgetVerdict::{WithinKnownBounds, Exceeded, Unverified}`), and sample-time `NonUniformScheduler` with causal delay rings (`ir_offset + block_size`) and `SPREAD` sub-stepping bounded by `min(slack_samples, B_s - 1)`.

## 5. Fixed-Point Q15 Direct, FFT/BFP & Integrated NUPC (RTCONV-004C / 004D / 004E)
- **Q15 Direct (`004C`)**: `FixedQ15DirectConvolver` (`i16` samples/coeffs, `i64` accumulation, round-half-up `(acc + 16384) >> 15`, symmetric saturation + `saturation_count()`).
- **Q15/BFP FFT & FDL (`004D`)**: `rfft_q15_bfp` / `irfft_q15_bfp` (`N64`, `N128`, `N256`, `512 B` twiddle ROM, conditional per-stage `i32` BFP scaling) + `FixedQ15PartitionedConvolver` (two-pass max-exponent alignment in `i64` before BFP renormalization).
- **Integrated Fixed NUPC (`004E`)**: `FixedQ15NonUniformConvolver` reuses `NonUniformPlan`, `NonUniformBudget`, and `NonUniformScheduler` verbatim with Q1.15/BFP numeric stages and pre-computed per-block `acc_max_exp`.

## 6. ESP32-C6 Embedded Harness (RTCONV-004F)
- Independent `#![no_std]` `#![no_main]` crate in `hardware/esp32c6/` (excluded from `.crate`), verified via `cargo check --release --target riscv32imac-unknown-none-elf` (`CROSS_COMPILE_VERIFIED = PASS`).
- Physical hardware execution: `DEVICE_VERIFIED = NO` (`NOT_VERIFIED`; no flashing executed without owner authorization).

## 7. Numerical, Benchmark & Memory Summary (`HOST_ONLY` / `CROSS_COMPILE_VERIFIED`)
- **Numerical Accuracy vs `f64`**:
  - `NonUniformConvolver` (`f32`, 1..2,051 taps, dynamic fragmentation): `max_abs < 1e-4`, `SNR > 95 dB`, `deadline_misses == 0`.
  - `FixedQ15DirectConvolver` (512-tap decaying IR): `max_abs <= 1 LSB (3.05e-5)`, `RMS = 8.911e-6`, `SNR = 69.55 dB`.
  - `rfft_q15_bfp` / `irfft_q15_bfp` (`N64/N128/N256`): forward `SNR > 62 dB`, roundtrip `SNR > 60 dB`, `0` saturations.
  - `FixedQ15PartitionedConvolver` (500 taps, `B=32/64/128`): `SNR = 51.36 dB`, `max_abs = 1.899e-4`, `0` saturations.
  - `FixedQ15NonUniformConvolver` (1,350 taps, 4 stages, fragmented blocks): `SNR = 44.19 dB` (-36 dBFS output), `max_err = 3.137e-4`, `deadline_misses == 0`, `saturation_count == 0`.
- **Host Benchmarks (`48 kHz`, `block = 32`, `666.67 µs` budget, Ryzen 3 PRO 2300U, `rustc 1.100.0`, `x86_64-pc-windows-gnu`)**:
  - `16,383` taps `f32`: `Hybrid(H128/T128) SPREAD` `mean=5.92 µs, p99=8.30 µs (1.25%), ws=267,768 B` vs `NonUniform(D32/32/64/128)` `mean=8.91 µs, p99=20.50 µs (3.08%), ws=275,912 B`.
  - `32..512` taps Direct: `Q15` `1.35..19.34 µs` (`128..2,048 B`, 50% RAM) vs `f32 SIMD` `0.65..4.21 µs` (`256..4,096 B`).
- **Regressions**: Zero regressions across existing `f32` `Direct`, `Partitioned`, `Hybrid`, `MicroFft`, `SimdFixedFft`, and `PhastFft` suites (`--no-default-features` and `--all-features` all PASS).

## 8. Known Limits, Risks & Gate Status Matrix

| Gate | Status | Scope / Notes |
| :--- | :--- | :--- |
| **Gate A (`004A` CI/macOS/Dependabot)** | **`PASS`** | PR #5: 10/10 GitHub Actions checks green (Linux, Windows, macOS, MSRV `1.89.0`). |
| **Gate B (`004B` NUPC `f32`)** | **`PASS`** | Parity, zero-alloc, fragmentation, budget & benchmarks verified (`N64/N128/N256`; `N512/N1024` future). |
| **Gate C (`004C` Q15 Direct)** | **`PASS`** | Numerical `f64` oracle, zero-alloc, RV32IMAC `CROSS_COMPILE_VERIFIED`. |
| **Gate D (`004D` FFT Q15/BFP + FDL)** | **`PASS`** | `N64/N128/N256` validated vs `f64`, zero-alloc, RV32IMAC `CROSS_COMPILE_VERIFIED`. |
| **Gate E (`004E` Fixed NUPC)** | **`PASS`** | Shared scheduler, fragmentation, zero-alloc, RV32IMAC `CROSS_COMPILE_VERIFIED`. |
| **Gate F (`004F` ESP32-C6 Hardware)** | **`CONDITIONAL_PASS`** | `CROSS_COMPILE_VERIFIED = PASS`; `DEVICE_VERIFIED = NOT_VERIFIED` (awaits physical ESP32-C6 authorization). |

## 9. Pending Decisions for `0.2.0`
1. Merge order: review and merge PR #5 (`004A`) first, close PR #3, decide on PR #2, then review PRs #6–#10.
2. Decide whether to add `N512`/`N1024` FFT sizes to `FftBackend` so NUPC outperforms `Hybrid SPREAD` on long IRs (`>= 16k` taps).
3. Authorize physical ESP32-C6 flashing + `mcycle` CSR profiling to upgrade Gate F from `CROSS_COMPILE_VERIFIED` to `DEVICE_VERIFIED`.
4. Decide timing of `crates.io` publication for `0.1.0` vs bumping directly to `0.2.0`.
