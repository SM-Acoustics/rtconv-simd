# WO-RTCONV-004D — Fixed-Point Q1.15 / Block Floating Point FFT & FDL Convolver

> **Superseded numbers (2026-11-23)**: the adversarial review ([`REVIEW-2026-11-ADVERSARIAL.md`](./REVIEW-2026-11-ADVERSARIAL.md)) found that this BFP implementation did not normalize quiet blocks (SNR fell with signal level) and that the fixed NUPC rounded each stage separately. Both were fixed. The SNR figures below are historical records of the work order, not current behaviour.

**Work Order**: `RTCONV-004D`  
**Branch**: `feat/rtconv-004d-fixed-fft`  
**Feature Gate**: `fixed-fft = ["fixed-q15"]` (`default = []` preserved; `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap, zero C/FFI)  
**Verification Scope**: `HOST_TESTED` (`rustc 1.100.0` + MSRV `1.89.0`) & `CROSS_COMPILE_VERIFIED` (`riscv32imac-unknown-none-elf`, WSL Ubuntu 24.04 `rustc 1.99.1`); `DEVICE_VERIFIED = NO`.

---

## 1. Numeric Architecture: Block Floating Point (BFP) vs Fixed-Scale Q15

- **Selected Representation**: **Conditional Stage-Scaling Block Floating Point (BFP)** (`src/fixed/block_float.rs`, `src/fixed/fft_q15.rs`):
  $$X[k] = \frac{\text{re}[k] + j\,\text{im}[k]}{32768} \cdot 2^{\text{exponent}}$$
- **Why BFP over Fixed-Scale Q15**: Fixed-scale radix-2 FFTs shift right by 1 on every stage ($\div N$), losing 6–8 bits of SNR on low-energy signals. Conditional BFP computes butterflies in `i32` (zero intermediate overflow), checks `peak = max(|re|, |im|)` after each stage, and downscales only when `peak > 32767`, accumulating the exact right-shift into `exponent: i8`.
- **Twiddle ROM**: `TWIDDLE_COS_256` + `TWIDDLE_SIN_256` (`512 B` total ROM shared across `N64`, `N128`, `N256`).
- **IFFT Normalization**: `irfft_q15_bfp` applies `1 / N` directly in the exponent (`net_exp = exponent + inv_exp - log2(N)`), avoiding extra rounding passes, and returns exact saturation telemetry.
- **FDL Spectral Alignment (`FixedQ15PartitionedConvolver`)**:
  1. Pass 1 computes `max_exp = max_p (fdl_exp[slot(p)] + ir_exp[p])` across all active partitions.
  2. Pass 2 aligns every partition product `(xr * hr - xi * hi)` in `i64` (Q2.30) by `delta = (max_exp - pe).min(62)` before summing — incompatible exponents are never added directly.
  3. `bfp_normalize_acc_i64_to_q15` rescales the `i64` accumulator back to full Q1.15 precision before `irfft_q15_bfp`.

---

## 2. Verification Evidence

- **`tests/fixed_fft_reference.rs`** (`N = 64, 128, 256`; impulse, DC, Nyquist, integer-bin sine, off-bin sine, deterministic noise vs `f64` DFT): forward spectrum `SNR > 62 dB`, roundtrip `SNR > 60 dB`, `0` saturations — **PASS**.
- **`tests/fixed_partitioned.rs`** (500-tap decaying random IR, `B = 32, 64, 128` vs `f64` reference): `SNR > 50 dB` (measured `51.36 dB`, `max_abs = 1.899e-4` at `B = 32`), `0` saturations, bit-identical `reset()` — **PASS**.
- **`tests/no_alloc_rt.rs`**: `fixed_q15_partitioned::{prepare, process_block, reset, metadata}` verified `0` heap allocations/deallocations on `1.100.0` and MSRV `1.89.0` — **PASS**.
- **Cross-Compilation (`CROSS_COMPILE_VERIFIED`)**: `cargo check --lib --no-default-features --features fixed-fft --target riscv32imac-unknown-none-elf` -> `EXIT 0`.
