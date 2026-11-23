# WO-RTCONV-004E — Integrated Fixed-Point Q1.15/BFP Non-Uniform Partitioned Convolution

> **Superseded numbers (2026-11-23)**: the adversarial review ([`REVIEW-2026-11-ADVERSARIAL.md`](./REVIEW-2026-11-ADVERSARIAL.md)) found that this BFP implementation did not normalize quiet blocks (SNR fell with signal level) and that the fixed NUPC rounded each stage separately. Both were fixed. The SNR figures below are historical records of the work order, not current behaviour.

**Work Order**: `RTCONV-004E`  
**Branch**: `feat/rtconv-004e-fixed-nupc`  
**Feature Gate**: `fixed-nonuniform = ["nonuniform", "fixed-fft"]` (`default = []` preserved; `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap, zero C/FFI)  
**Verification Scope**: `HOST_TESTED` (`rustc 1.100.0` + MSRV `1.89.0`) & `CROSS_COMPILE_VERIFIED` (`riscv32imac-unknown-none-elf`, WSL Ubuntu 24.04 `rustc 1.99.1`); `DEVICE_VERIFIED = NO`.

---

## 1. Shared Three-Plane Architecture

`FixedQ15NonUniformConvolver` (`src/fixed/nonuniform.rs`) strictly separates and reuses the three planes required by `WO-RTCONV-004E`:

- **Planning Plane (Shared verbatim from `004B`)**: `NonUniformPlan`, `NonUniformStageConfig`, `NonUniformBudget`, `BudgetVerdict`, `check_budget()`, and `stage_delay_ring_len()` — zero duplication of stage validation, causal offset bounds (`D_s >= B_s`), or admission-control logic.
- **Streaming Plane (Shared verbatim from `004B`)**: `NonUniformScheduler`, `StageScheduleState`, `StageWorkItem` — identical sample-time triggers, SPREAD sub-stepping, and `deadline_misses()` telemetry.
- **Numeric Plane (Reused from `004C` & `004D`)**: Q1.15 `i64`-accumulated direct head FIR + Q1.15 BFP `rfft_q15_bfp` / `irfft_q15_bfp` + exponent-aligned `i64` CMAC (`bfp_normalize_acc_i64_to_q15`) with pre-computed per-block `acc_max_exp` so spread CMAC steps produce identical spectra to burst execution.

## 2. Memory & Admission Control

- `FixedQ15NonUniformMemoryRequirements` explicitly reports `state_i16`, `spectral_i16`, `delay_i16`, `scratch_i16`, `total_i16`, and `total_bytes` (~50% of equivalent `f32` workspace RAM).
- `prepare_with_budget()` evaluates `NonUniformBudget` upfront and rejects exceeded limits with `ConvError::InvalidNonUniformConfig` (or returns `BudgetVerdict::Unverified` when bounds are omitted, never claiming formal WCET).
- Future shareable immutable FIR spectra (`&[i16]` read-only partition tables) can be layered cleanly on top of the split `ir_re`/`ir_im`/`ir_exp` regions without mutating coefficients during streaming.

## 3. Verification Evidence

- **`tests/cross_backend_equivalence.rs`**: 1,350-tap decaying random IR spanning all 4 NUPC stages (`D32 / B32 / B64 / B128`) under dynamic host block fragmentation (`[1, 7, 13, 32, 19, 64, 5]`): verifies `FixedQ15NonUniformConvolver` vs `NonUniformConvolver` (`f32`) vs `f64` oracle (measured `SNR = 44.19 dB`, `max_err = 3.137e-4` on the -36 dBFS output, `deadline_misses == 0`, `saturation_count == 0`, bit-identical `reset()`).
- **`tests/no_alloc_rt.rs`**: `fixed_q15_nonuniform::{prepare, process_block, reset, metadata}` verified `0` heap allocations/deallocations on `1.100.0` and MSRV `1.89.0`.
- **Cross-Compilation (`CROSS_COMPILE_VERIFIED`)**: `cargo check --lib --no-default-features --features fixed-nonuniform --target riscv32imac-unknown-none-elf` -> `EXIT 0`.
